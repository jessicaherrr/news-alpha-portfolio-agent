"""Runtime Research Agent (Phase 16).

The agent is wired to Claude now that every deterministic tool it relies on
exists (feature registry, reliability validator, experiment registry, failure
memory, duplicate prevention). Its single job: turn a frozen
:class:`ResearchContext` into a schema-valid :class:`HypothesisSpec`.

What it is NOT allowed to do -- enforced by construction, not by prompt wording:

* execute a backtest, or sequence any tool (that is the orchestrator, prompt 18);
* produce official PnL, fills, verdicts, or touch validation thresholds / the
  BH-FDR family definition;
* read the locked 2025 holdout (the context is holdout-guarded; the output is
  holdout-guarded again);
* touch the filesystem or route an order. The only file it reads is its own
  prompt template, at construction.

Retry policy: a retry happens **only** on a schema failure (unparseable JSON or
a ``HypothesisSpec`` validation error). A semantically valid spec that violates
a deterministic guardrail (unknown feature, market outside the approved
universe) is *rejected*, not retried -- silent retries would multiply tests
without registry records (prompt 18).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from alpha_agent.agents.context import ResearchContext
from alpha_agent.agents.llm import DEFAULT_MODEL, LLMClient, LLMResponse
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data
from alpha_agent.schemas.hypothesis import HypothesisSpec

DEFAULT_PROMPT_PATH = Path(__file__).parent / "runtime_prompts" / "research_system.md"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ResearchAgentError(RuntimeError):
    """Base class for Phase 16 research-agent failures."""


class SchemaRetryExhausted(ResearchAgentError):
    """The model never returned a schema-valid HypothesisSpec within the retry
    budget. Carries every attempt for the registry / orchestrator."""

    def __init__(self, attempts: tuple[AttemptRecord, ...]):
        self.attempts = attempts
        last = attempts[-1].error if attempts else "no attempts"
        super().__init__(
            f"no schema-valid HypothesisSpec after {len(attempts)} attempt(s); "
            f"last error: {last}"
        )


class BudgetExceeded(ResearchAgentError):
    """A turn or token budget was hit before a valid spec was produced."""


@dataclass(frozen=True)
class AttemptRecord:
    """One model call and what became of it."""

    index: int
    raw_text: str
    parsed_ok: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


class PromptLog(BaseModel):
    """Prompt / version provenance for one :meth:`ResearchAgent.propose` call.

    Enough to reproduce exactly which instructions and which context produced a
    proposal, without storing anything the agent was not allowed to see.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    prompt_version: str
    prompt_path: str
    system_prompt_sha256: str
    context_sha256: str
    model: str
    temperature: float
    max_output_tokens: int
    n_attempts: int
    total_input_tokens: int
    total_output_tokens: int


class ResearchProposal(BaseModel):
    """The outcome of one proposal round. Either an accepted spec or a typed
    rejection -- never a raw model string leaking upward."""

    model_config = {"frozen": True, "extra": "forbid"}

    accepted: bool
    hypothesis: HypothesisSpec | None = None
    rejection_code: str | None = None
    rejection_detail: str | None = None
    prompt_log: PromptLog
    attempts: tuple[AttemptRecord, ...]


_OUTPUT_CONTRACT = """
## Output contract (Phase 16, enforced deterministically downstream)

Return ONLY a single JSON object, no prose, no markdown fence required.
It must validate against this JSON schema:

{schema}

Hard rules:
- `universe` must be a subset of the approved market universe you were given.
- `required_features` must all be `kind` values from the feature catalog you
  were given. Do not invent feature names.
- Never reference any date on or after 2025-01-01. The 2025 holdout does not
  exist for you.
- You do not compute PnL, fills, verdicts, or p-values, and you do not change
  any validation threshold or multiple-testing family definition.
- If `failure_memory` shows only INVALID_EXECUTION history for a mechanism, that
  is an engineering defect, not a refutation -- you may re-propose it, and say so.
- If `failure_memory` shows a genuine scientific refusal (e.g. insufficient
  event density, FDR/DSR not passed), a re-proposal MUST state in
  `novelty_notes` what is materially different.
- If one or more `research_knowledge_base` snippets genuinely inspired this
  proposal's mechanism, name them in `source_inspirations` (quote enough of
  the snippet's own title/text to identify it -- never invent a citation).
  This is lineage only, informational -- it never substitutes for your own
  independent economic reasoning, and a source's claimed performance is
  NEVER evidence you may cite as if it were your own.
"""


class ResearchAgent:
    """LLM-backed hypothesis proposer with deterministic guardrails."""

    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        model: str = DEFAULT_MODEL,
        max_schema_retries: int = 2,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        token_budget: int | None = 40_000,
        log_sink: Callable[[PromptLog], None] | None = None,
    ):
        if max_schema_retries < 0:
            raise ValueError("max_schema_retries must be >= 0")
        self._client = client
        self._prompt_path = Path(prompt_path)
        self._prompt_template = self._prompt_path.read_text(encoding="utf-8")
        self._model = model
        self._max_schema_retries = max_schema_retries
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._token_budget = token_budget
        self._log_sink = log_sink

        schema_json = json.dumps(HypothesisSpec.model_json_schema(), indent=2, sort_keys=True)
        self._system_prompt = (
            self._prompt_template.rstrip()
            + "\n"
            + _OUTPUT_CONTRACT.format(schema=schema_json)
        )
        self._prompt_version = _sha256(self._prompt_template)[:16]

    # -- public API --------------------------------------------------------

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def propose(self, context: ResearchContext) -> ResearchProposal:
        if not isinstance(context, ResearchContext):
            raise TypeError("context must be a ResearchContext")

        system = self._system_prompt
        messages: list[dict] = [{"role": "user", "content": context.render_user_message()}]
        attempts: list[AttemptRecord] = []
        total_in = 0
        total_out = 0

        for i in range(self._max_schema_retries + 1):
            if self._token_budget is not None and (total_in + total_out) >= self._token_budget:
                raise BudgetExceeded(
                    f"token budget {self._token_budget} reached after {i} attempt(s)"
                )

            resp: LLMResponse = self._client.complete(
                system=system,
                messages=messages,
                model=self._model,
                max_tokens=self._max_output_tokens,
                temperature=self._temperature,
            )
            total_in += resp.input_tokens
            total_out += resp.output_tokens

            error: str | None = None
            spec: HypothesisSpec | None = None
            try:
                payload = _extract_json(resp.text)
                spec = HypothesisSpec.model_validate(payload)
            except (ValueError, ValidationError) as exc:  # JSON or schema failure only
                error = f"{type(exc).__name__}: {exc}"

            attempts.append(
                AttemptRecord(
                    index=i,
                    raw_text=resp.text,
                    parsed_ok=spec is not None,
                    error=error,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    stop_reason=resp.stop_reason,
                )
            )

            if spec is None:
                if i >= self._max_schema_retries:
                    self._emit_log(context, attempts, total_in, total_out)
                    raise SchemaRetryExhausted(tuple(attempts))
                messages.append({"role": "assistant", "content": resp.text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That did not validate against the HypothesisSpec schema:\n"
                            f"{error}\n"
                            "Return only a single corrected JSON object."
                        ),
                    }
                )
                continue

            prompt_log = self._emit_log(context, attempts, total_in, total_out)
            return self._finalize(spec, context, prompt_log, tuple(attempts))

        # unreachable: the loop either returns or raises
        raise ResearchAgentError("proposal loop exited without a result")  # pragma: no cover

    # -- internals --------------------------------------------------------

    def _finalize(
        self,
        spec: HypothesisSpec,
        context: ResearchContext,
        prompt_log: PromptLog,
        attempts: tuple[AttemptRecord, ...],
    ) -> ResearchProposal:
        # Locked holdout: fail loud, never a soft rejection.
        assert_no_holdout_market_data(spec.model_dump(mode="json"), path="$.hypothesis")

        code, detail = _deterministic_guardrails(spec, context)
        if code is not None:
            return ResearchProposal(
                accepted=False,
                hypothesis=None,
                rejection_code=code,
                rejection_detail=detail,
                prompt_log=prompt_log,
                attempts=attempts,
            )
        return ResearchProposal(
            accepted=True,
            hypothesis=spec,
            prompt_log=prompt_log,
            attempts=attempts,
        )

    def _emit_log(
        self,
        context: ResearchContext,
        attempts: list[AttemptRecord],
        total_in: int,
        total_out: int,
    ) -> PromptLog:
        log = PromptLog(
            prompt_version=self._prompt_version,
            prompt_path=str(self._prompt_path),
            system_prompt_sha256=_sha256(self._system_prompt),
            context_sha256=context.sha256(),
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            n_attempts=len(attempts),
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )
        if self._log_sink is not None:
            self._log_sink(log)
        return log


# -- module-level helpers -------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> dict:
    """Pull one JSON object out of a model response. Tolerates a markdown fence
    or leading/trailing prose; raises ``ValueError`` on anything else so the
    caller treats it as a schema failure."""
    if text is None:
        raise ValueError("empty model response")
    candidate = text.strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in model response")
        candidate = candidate[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        # ValueError (not TypeError) on purpose: the caller catches ValueError as
        # a schema failure and retries; a TypeError would escape the retry path.
        raise ValueError(  # noqa: TRY004
            f"expected a JSON object, got {type(obj).__name__}"
        )
    return obj


def _deterministic_guardrails(
    spec: HypothesisSpec, context: ResearchContext
) -> tuple[str | None, str | None]:
    """Non-retryable semantic checks. The agent proposes; these deterministic
    rules decide whether the proposal is admissible at all."""
    universe = set(context.market_universe)
    unknown_markets = [m for m in spec.universe if m not in universe]
    if not spec.universe:
        return "EMPTY_UNIVERSE", "hypothesis names no market"
    if unknown_markets:
        return (
            "MARKET_OUTSIDE_UNIVERSE",
            f"markets not in approved universe: {sorted(unknown_markets)}",
        )

    known_features = context.feature_kinds
    unknown_features = [f for f in spec.required_features if f not in known_features]
    if not spec.required_features:
        return "NO_REQUIRED_FEATURES", "hypothesis lists no required features"
    if unknown_features:
        return (
            "UNKNOWN_FEATURE",
            f"required_features not in catalog: {sorted(unknown_features)}",
        )

    return None, None

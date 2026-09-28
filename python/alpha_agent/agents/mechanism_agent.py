"""Runtime Mechanism/Factor Agent (Phase 1, prompt 1 section 6).

Claude's role in the Observation -> Mechanism -> Factor translation is
strictly narrative: it proposes candidate economic mechanisms, measurable
variables, and factor ideas as a schema-valid
:class:`MechanismTranslationProposal`. It is never the source of
researchability, feature availability, or point-in-time status -- those are
always recomputed deterministically, after this agent returns, by
:func:`alpha_agent.translation.pipeline.build_observation_translation`
against live repository state (the real ``alpha_agent.features.REGISTRY``).

This module mirrors :mod:`alpha_agent.agents.research_agent` closely
(retry-only-on-schema-failure, holdout guard on the output, no filesystem /
registry-write / broker surface) rather than inventing a second agent
pattern.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from alpha_agent.agents.context import FeatureCatalogEntry
from alpha_agent.agents.llm import DEFAULT_MODEL, LLMClient, LLMResponse
from alpha_agent.agents.research_agent import AttemptRecord, BudgetExceeded, PromptLog
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data
from alpha_agent.translation.schemas import Observation

__all__ = [
    "FactorCandidateProposal",
    "MeasurableVariableProposal",
    "MechanismAgent",
    "MechanismAgentError",
    "MechanismCandidateProposal",
    "MechanismProposalResult",
    "MechanismSchemaRetryExhausted",
    "MechanismTranslationProposal",
]

DEFAULT_PROMPT_PATH = Path(__file__).parent / "runtime_prompts" / "mechanism_system.md"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull one JSON object out of a model response. Tolerates a markdown
    fence or leading/trailing prose; raises ``ValueError`` on anything else
    so the caller treats it as a schema failure. Deliberately duplicated
    (not imported) from :mod:`alpha_agent.agents.research_agent`'s private
    helper of the same name -- that module is Phase 16/17/18, frozen and
    approved; this keeps Phase 1 self-contained rather than reaching into
    another module's private surface."""
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
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")  # noqa: TRY004
    return obj


class MechanismAgentError(RuntimeError):
    """Base class for Phase 1 mechanism-agent failures."""


class MechanismSchemaRetryExhausted(MechanismAgentError):
    """The model never returned a schema-valid proposal within the retry budget."""

    def __init__(self, attempts: tuple[AttemptRecord, ...]):
        self.attempts = attempts
        last = attempts[-1].error if attempts else "no attempts"
        super().__init__(
            f"no schema-valid MechanismTranslationProposal after {len(attempts)} attempt(s); last error: {last}"
        )


class MechanismCandidateProposal(BaseModel):
    model_config = {"extra": "forbid"}

    mechanism: EconomicMechanism
    explanation: str
    causal_chain: list[str] = Field(default_factory=list)
    evidence_basis: str = ""


class MeasurableVariableProposal(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    economic_meaning: str
    required_source: str


class FactorCandidateProposal(BaseModel):
    model_config = {"extra": "forbid"}

    concept: str
    mechanism: EconomicMechanism
    transform_or_proxy: str
    proposed_feature_kinds: list[str] = Field(default_factory=list)
    required_external_data: list[str] = Field(default_factory=list)


class MechanismTranslationProposal(BaseModel):
    """Claude's ENTIRE proposal for one Observation -- narrative/reasoning
    only. See the module docstring: nothing here is trusted as a capability
    or availability claim."""

    model_config = {"extra": "forbid"}

    mechanism_candidates: list[MechanismCandidateProposal] = Field(min_length=1)
    measurable_variables: list[MeasurableVariableProposal] = Field(default_factory=list)
    factor_candidates: list[FactorCandidateProposal] = Field(min_length=1)


class MechanismProposalResult(BaseModel):
    """The outcome of one proposal round -- mirrors
    :class:`alpha_agent.agents.research_agent.ResearchProposal`'s shape."""

    model_config = {"frozen": True, "extra": "forbid"}

    accepted: bool
    proposal: MechanismTranslationProposal | None = None
    rejection_code: str | None = None
    rejection_detail: str | None = None
    prompt_log: PromptLog
    attempts: tuple[AttemptRecord, ...]


_OUTPUT_CONTRACT = """
## Output contract (enforced deterministically downstream)

Return ONLY a single JSON object, no prose, no markdown fence required. It
must validate against this JSON schema:

{schema}

The `EconomicMechanism` enum values you may use for `mechanism` are exactly:
{mechanism_values}

The feature catalog you may reference by `kind` in `proposed_feature_kinds` is:
{feature_kinds}
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MechanismAgent:
    """LLM-backed mechanism/factor proposer with deterministic downstream
    reclassification (see module docstring)."""

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
        log_sink=None,
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
        self._prompt_version = _sha256(self._prompt_template)[:16]

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def _system_prompt(self, feature_catalog: tuple[FeatureCatalogEntry, ...]) -> str:
        schema_json = json.dumps(MechanismTranslationProposal.model_json_schema(), indent=2, sort_keys=True)
        mechanism_values = ", ".join(m.value for m in EconomicMechanism)
        feature_kinds = ", ".join(sorted(e.kind for e in feature_catalog)) or "(none registered)"
        return (
            self._prompt_template.rstrip()
            + "\n"
            + _OUTPUT_CONTRACT.format(schema=schema_json, mechanism_values=mechanism_values, feature_kinds=feature_kinds)
        )

    def _user_message(self, observation: Observation) -> str:
        payload = {
            "observation": {
                "event_type": observation.event_type,
                "root_symbol": observation.root_symbol,
                "affected_products": list(observation.affected_products),
                "source": observation.source,
                "summary": observation.summary,
                "structured_attributes": dict(observation.structured_attributes),
            },
        }
        return "Propose mechanism/factor candidates for this observation, as JSON.\n\n" + json.dumps(
            payload, indent=2, sort_keys=True
        )

    def propose(
        self, observation: Observation, *, feature_catalog: tuple[FeatureCatalogEntry, ...]
    ) -> MechanismProposalResult:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")

        system = self._system_prompt(feature_catalog)
        messages: list[dict] = [{"role": "user", "content": self._user_message(observation)}]
        attempts: list[AttemptRecord] = []
        total_in = 0
        total_out = 0

        for i in range(self._max_schema_retries + 1):
            if self._token_budget is not None and (total_in + total_out) >= self._token_budget:
                raise BudgetExceeded(f"token budget {self._token_budget} reached after {i} attempt(s)")

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
            proposal: MechanismTranslationProposal | None = None
            try:
                payload = _extract_json(resp.text)
                proposal = MechanismTranslationProposal.model_validate(payload)
            except (ValueError, ValidationError) as exc:
                error = f"{type(exc).__name__}: {exc}"

            attempts.append(
                AttemptRecord(
                    index=i,
                    raw_text=resp.text,
                    parsed_ok=proposal is not None,
                    error=error,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    stop_reason=resp.stop_reason,
                )
            )

            if proposal is None:
                if i >= self._max_schema_retries:
                    self._emit_log(system, attempts, total_in, total_out)
                    raise MechanismSchemaRetryExhausted(tuple(attempts))
                messages.append({"role": "assistant", "content": resp.text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That did not validate against the MechanismTranslationProposal schema:\n"
                            f"{error}\nReturn only a single corrected JSON object."
                        ),
                    }
                )
                continue

            prompt_log = self._emit_log(system, attempts, total_in, total_out)
            assert_no_holdout_market_data(proposal.model_dump(mode="json"), path="$.mechanism_proposal")
            return MechanismProposalResult(accepted=True, proposal=proposal, prompt_log=prompt_log, attempts=tuple(attempts))

        raise MechanismAgentError("proposal loop exited without a result")  # pragma: no cover

    def _emit_log(self, system: str, attempts: list[AttemptRecord], total_in: int, total_out: int) -> PromptLog:
        log = PromptLog(
            prompt_version=self._prompt_version,
            prompt_path=str(self._prompt_path),
            system_prompt_sha256=_sha256(system),
            context_sha256=_sha256(system),
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

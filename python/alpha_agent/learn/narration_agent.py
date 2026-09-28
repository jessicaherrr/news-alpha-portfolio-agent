"""Phase 4 -- optional Claude narration over an already-built
:class:`~alpha_agent.learn.autopsy.FailureAutopsy` (prompt section 5: "Claude
narrates but never assigns verdict").

This mirrors :mod:`alpha_agent.agents.mechanism_agent`'s safety pattern
(thin transport, holdout-guarded output) but is deliberately simpler: the
model's ENTIRE output is free-text commentary, not a schema the rest of the
system trusts. Every fact `FailureAutopsyNarrator.narrate` shows the model
(verdict, reason codes, gate states) is already-typed, already-committed
evidence the caller built with `alpha_agent.learn.autopsy.build_failure_autopsy`
-- Claude adds READER-FACING color on top, it never supplies a new fact.

`_assert_no_verdict_contradiction` is a deterministic, non-LLM safety net: if
the model's free text claims a DIFFERENT headline verdict than the autopsy's
own committed `verdict`, the narration is rejected rather than shown -- the
one way this module could otherwise let an LLM quietly override science.

Narration is entirely optional. `alpha_agent.ui.views.learn` renders a full,
useful Failure Autopsy with `narrative=None` -- this agent, when a live
`LLMClient` is available, only ever ADDS to that, never gates it.
"""
from __future__ import annotations

import re

from alpha_agent.agents.llm import DEFAULT_MODEL, LLMClient
from alpha_agent.learn.autopsy import FailureAutopsy
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data

__all__ = ["FailureAutopsyNarrator", "NarrationRejected"]

_SYSTEM_PROMPT = """You are explaining ONE already-adjudicated futures research \
experiment to a beginner. You are a narrator, not a judge: every fact below \
(verdict, gate states, reason codes) is already final and committed by a \
deterministic statistical pipeline you cannot see or change. Restate it in \
plain, encouraging, precise language; explain intuitively why the failed gate \
matters; do not invent numbers, do not claim a different verdict than the one \
given, do not suggest the result should be reinterpreted. 3-5 sentences, no \
markdown headers, no bullet lists."""

#: verdict word -> the OTHER verdict words that would contradict it if the
#: model's free text asserts one of them as ITS OWN claim about this result.
_VERDICT_WORDS = ("PASS", "REJECT", "INCONCLUSIVE")


class NarrationRejected(RuntimeError):
    """The model's free-text narration contradicted the autopsy's own
    committed verdict, or otherwise failed the deterministic safety check."""


def _assert_no_verdict_contradiction(text: str, *, verdict: str | None) -> None:
    if verdict is None:
        return
    upper = text.upper()
    for word in _VERDICT_WORDS:
        if word == verdict:
            continue
        # A bare mention of another verdict word is allowed (e.g. contrasting
        # language); an explicit claim-of-verdict pattern is not.
        if re.search(rf"\b(IS|WAS|VERDICT IS|RESULT IS)\s+{word}\b", upper):
            raise NarrationRejected(
                f"narration asserted verdict {word!r} but the committed verdict is {verdict!r}"
            )


class FailureAutopsyNarrator:
    """Thin, free-text-only LLM transport. No schema retry: a malformed or
    unavailable response simply means no narrative (the caller falls back to
    `narrative=None`), never an error surfaced to the beginner reading the
    page."""

    def __init__(self, client: LLMClient, *, model: str = DEFAULT_MODEL, max_output_tokens: int = 400):
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens

    def narrate(self, autopsy: FailureAutopsy) -> str | None:
        payload = {
            "root_symbol": autopsy.root_symbol,
            "strategy_family": autopsy.strategy_family,
            "verdict": autopsy.verdict,
            "reason_codes": list(autopsy.reason_codes),
            "first_failed_gate": autopsy.first_failed_gate,
            "what_looked_promising": list(autopsy.what_looked_promising),
            "what_failed": autopsy.what_failed,
        }
        assert_no_holdout_market_data(payload, path="$.failure_autopsy_narration_input")
        resp = self._client.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Explain this result:\n{payload}"}],
            model=self._model,
            max_tokens=self._max_output_tokens,
            temperature=0.2,
        )
        text = (resp.text or "").strip()
        if not text:
            return None
        _assert_no_verdict_contradiction(text, verdict=autopsy.verdict)
        return text

"""Alpha Discovery campaign, Part M -- LLM cost tracking (task spec section
66: "Track where available: model, tokens, estimated API cost. Do not
fabricate cost values.").

Every `alpha_agent.agents.research_agent.PromptLog` already carries `model`,
`total_input_tokens`, `total_output_tokens` -- real, already-measured
quantities from the actual `LLMResponse`s a proposal round consumed. This
module ADDS one small, honest transform on top: an ESTIMATED USD cost from
those real token counts, using a documented, versioned pricing table.

Two failure modes this module refuses to paper over:

* An unrecognized `model` string (a future model, a typo, a test double's
  `model="scripted-model"`) returns `None` for cost, never a guessed number
  -- "do not fabricate cost values" is absolute, not "unless the number is
  probably fine".
* Pricing drifts. `PRICING_TABLE_AS_OF` records when this table was last
  checked; every returned `EstimatedCost` carries `pricing_as_of` so a
  caller can see how stale the assumption might be, rather than trusting a
  dollar figure that silently rotted.
"""
from __future__ import annotations

from pydantic import BaseModel

#: USD per token (not per million -- pre-divided so callers never repeat that
#: arithmetic). Source: the live Anthropic pricing table as of the date
#: below. Update this table (and `PRICING_TABLE_AS_OF`) when pricing changes;
#: never silently reuse a stale number without moving the date.
PRICING_TABLE_AS_OF = "2026-06-24"

#: (input_usd_per_token, output_usd_per_token).
_PRICING_PER_TOKEN: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00 / 1_000_000, 50.00 / 1_000_000),
    "claude-mythos-5-1": (10.00 / 1_000_000, 50.00 / 1_000_000),
    "claude-fable-5": (10.00 / 1_000_000, 50.00 / 1_000_000),
    "claude-opus-5": (5.00 / 1_000_000, 25.00 / 1_000_000),
    "claude-opus-4-8": (5.00 / 1_000_000, 25.00 / 1_000_000),
    "claude-opus-4-7": (5.00 / 1_000_000, 25.00 / 1_000_000),
    "claude-opus-4-6": (5.00 / 1_000_000, 25.00 / 1_000_000),
    "claude-sonnet-5": (2.00 / 1_000_000, 10.00 / 1_000_000),
    "claude-sonnet-4-6": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-haiku-4-5": (1.00 / 1_000_000, 5.00 / 1_000_000),
}


class EstimatedCost(BaseModel):
    """`usd is None` means "this model is not in the pricing table" -- never
    a zero or a guessed fallback rate."""

    model_config = {"frozen": True, "extra": "forbid"}

    model: str
    input_tokens: int
    output_tokens: int
    usd: float | None
    pricing_as_of: str = PRICING_TABLE_AS_OF
    priced: bool = False


def estimate_cost_usd(*, model: str, input_tokens: int, output_tokens: int) -> EstimatedCost:
    """A pure function of (model, token counts) -- never reads a network
    price feed, never caches a stale guess as a default for an unknown
    model."""
    rates = _PRICING_PER_TOKEN.get(model)
    if rates is None:
        return EstimatedCost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens,
            usd=None, priced=False,
        )
    in_rate, out_rate = rates
    usd = input_tokens * in_rate + output_tokens * out_rate
    return EstimatedCost(
        model=model, input_tokens=input_tokens, output_tokens=output_tokens,
        usd=round(usd, 6), priced=True,
    )


def estimate_cost_for_prompt_log(prompt_log) -> EstimatedCost:
    """Convenience wrapper over an already-real
    `alpha_agent.agents.research_agent.PromptLog` (duck-typed -- no import
    of that module here, to keep this module dependency-light and usable
    from the UI/compiler-agent side too, which has its own equivalent log
    shape)."""
    return estimate_cost_usd(
        model=prompt_log.model,
        input_tokens=prompt_log.total_input_tokens,
        output_tokens=prompt_log.total_output_tokens,
    )


def summarize_campaign_cost(prompt_logs: list) -> dict:
    """Total ESTIMATED cost across several prompt logs (e.g. one Discover
    Strategies pass's research + compiler calls). `priced_calls` /
    `unpriced_calls` count how many logs actually had a recognized model --
    an honest signal for "this total is a lower bound" when any are
    unpriced."""
    total = 0.0
    priced_calls = 0
    unpriced_calls = 0
    per_model: dict[str, float] = {}
    for log in prompt_logs:
        est = estimate_cost_for_prompt_log(log)
        if est.usd is None:
            unpriced_calls += 1
            continue
        priced_calls += 1
        total += est.usd
        per_model[est.model] = per_model.get(est.model, 0.0) + est.usd
    return {
        "total_usd": round(total, 6) if priced_calls else None,
        "priced_calls": priced_calls,
        "unpriced_calls": unpriced_calls,
        "per_model_usd": {k: round(v, 6) for k, v in per_model.items()},
        "pricing_as_of": PRICING_TABLE_AS_OF,
    }

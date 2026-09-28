"""Opportunity V1 -- typed contract + pure, deterministic decision logic.

`OpportunitySnapshot` answers exactly one question: WHAT DESERVES
INVESTIGATION NOW? It is not a BUY/SELL model, not a probability-of-profit or
expected-return model, not a price prediction, and not a Scientific Verdict
(that authority stays with the frozen `ReliabilityPolicy` / registry, see
CLAUDE.md). There is deliberately no confidence percentage, no 0-100 score,
and no expected-return field anywhere on this schema.

Everything in this module is a PURE function of already-computed, typed
`OpportunityInputs` -- no Streamlit import, no network call, no registry
write, no filesystem access. `alpha_agent.ui.opportunity_context` is the
separate, real evidence-assembly boundary that builds `OpportunityInputs`
from the existing observation-plane accessors (`alpha_agent.ui.market_home`,
`market_contracts`, `market_relative`, `market_intel_context`) and read-only
registry evidence (`alpha_agent.ui.services`); this module never imports any
of those, which is what makes `build_opportunity`/`rank_opportunities`
deterministic and unit-testable with no live data.

An internal, undocumented-to-the-user PRIORITY exists only to order the
"Top Opportunities" list -- it is deliberately never a field on
`OpportunitySnapshot` (task spec section 7: "keep it internal, document its
components, do not present it as expected return or confidence"). Its
components are documented on `_priority_key` below.

CURRENT OPPORTUNITY vs. RESEARCH EVIDENCE (Product Acceptance Fix Pass,
strictly separated per explicit product-review direction): `opportunity_state`
(WATCH/INVESTIGATE/WAIT -- renamed from an earlier, semantically wrong
`research_status`; it is triage over CURRENT observational conditions, never
a scientific research status), `why_now`, `risks`, and the internal
`_priority_key` ranking are all computed from CURRENT OBSERVATIONAL EVIDENCE
ONLY (trend, volatility, volume, session position, curve shape, relative-
market confirmation, catalyst timing) -- never from `research_verdict` /
`research_promise_label` / `best_strategy_family`. Those three fields still
exist on `OpportunityInputs` (read-only, unchanged assembly) but feed ONLY
the separately-labeled "Scientific Status" / Research Context / Next Action
surfaces (see `evidence_refs` and `alpha_agent.ui.views.agent`'s "RESEARCH"
card section) -- historical research evidence may explain what to do about a
current opportunity; it may never change whether the current market itself
is interesting. `tests/python/test_opportunity_v1.py`'s "CURRENT OPPORTUNITY
independence" tests hold this apart mechanically: two `OpportunityInputs`
differing ONLY in `research_verdict` must produce identical
`opportunity_state`, `why_now`, `risks`, and ranking.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

__all__ = [
    "CatalystInfo",
    "OpportunityInputs",
    "OpportunitySnapshot",
    "OpportunityState",
    "TrendDirection",
    "build_opportunity",
    "rank_opportunities",
]


class TrendDirection(str, Enum):
    """Mirrors `alpha_agent.ui.market_home.trend_state`'s own return
    vocabulary exactly -- never a second, independently-invented label set."""

    UP = "Up"
    DOWN = "Down"
    SIDEWAYS = "Sideways"
    UNKNOWN = "Insufficient data"


class OpportunityState(str, Enum):
    """OPPORTUNITY TRIAGE over CURRENT OBSERVATIONAL EVIDENCE, not a trade
    instruction and NOT a scientific research status (task spec section 13;
    renamed from `ResearchStatus` in the Product Acceptance Fix Pass -- that
    name was semantically wrong: WATCH/INVESTIGATE/WAIT are not PASS/REJECT/
    INCONCLUSIVE/NOT VALIDATED/NOT CERTIFIED, and the two concepts are never
    merged). Exactly three states, deliberately small."""

    WATCH = "WATCH"
    INVESTIGATE = "INVESTIGATE"
    WAIT = "WAIT"


class CatalystInfo(BaseModel):
    """One real, mapped upcoming scheduled event (`alpha_agent.market_intel
    .event_schemas.ScheduledMarketEvent`), narrowed to exactly what a
    catalyst label needs. Never fabricated -- absent when no connector
    currently supplies one for this product (see
    `alpha_agent.ui.market_intel_context.next_high_impact_event_for_product`)."""

    model_config = {"frozen": True, "extra": "forbid"}

    name: str
    source_name: str
    scheduled_at: datetime
    importance: str


class OpportunityInputs(BaseModel):
    """Already-computed, typed EVIDENCE for one product -- every field is a
    direct read of an existing deterministic accessor. Nothing here is
    fetched by this module; see the module docstring. `observed_at` is the
    wall-clock moment this evidence was assembled (UTC-aware), used only to
    compute a catalyst's time-to-event -- never used to infer a trading
    signal.

    `window_bars`/`window_timeframe` describe the SAME window `window_low`/
    `window_high` were computed over -- carried explicitly so "what changes
    this view" text can state the real window size accurately (e.g. "72
    observed 1h bars") rather than implying a technical support/resistance
    model that does not exist.

    `research_verdict`/`research_promise_label`/`best_strategy_family` are
    RESEARCH EVIDENCE, not current-opportunity evidence -- read by the UI's
    separate "Scientific Status"/Research Context/Next Action surfaces only;
    see the module docstring's "CURRENT OPPORTUNITY vs. RESEARCH EVIDENCE"
    section for why the decision functions below never touch them."""

    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    display_name: str
    trend: TrendDirection
    volatility: str
    volume_context: str
    session_position: str
    window_low: float | None
    window_high: float | None
    window_bars: int
    window_timeframe: str
    curve_shape: str | None
    peer_confirming_count: int
    peer_total_count: int
    news_count_24h: int
    catalyst: CatalystInfo | None
    research_verdict: str | None
    research_promise_label: str | None
    best_strategy_family: str | None
    observed_at: datetime


class OpportunitySnapshot(BaseModel):
    """The ONLY fields this schema carries -- deliberately minimal (task
    spec section 7). No `confidence`, no `probability_of_profit`, no
    `expected_return`, no 0-100 score: see
    `tests/python/test_opportunity_v1.py::test_schema_never_carries_a_forbidden_field`."""

    model_config = {"frozen": True, "extra": "forbid"}

    product: str
    display_name: str
    setup: str
    why_now: tuple[str, ...]
    risks: tuple[str, ...]
    catalyst: str
    strengthen_if: tuple[str, ...]
    invalidate_if: tuple[str, ...]
    opportunity_state: OpportunityState
    next_action: str
    observed_at: datetime
    evidence_refs: tuple[str, ...]


#: Field names this schema must never carry -- a static regression guard
#: (see the test above) against V1 quietly growing a confidence/expected-
#: return/probability field.
FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "confidence", "confidence_score", "probability_of_profit", "win_probability",
        "expected_return", "expected_pnl", "score", "priority_score", "priority",
        "signal", "recommendation", "buy_sell", "action_signal",
        # Names that would re-merge Opportunity triage with Scientific
        # Research Status -- the two are architecturally distinct concepts.
        "research_status", "scientific_verdict", "verdict",
    }
)


# ---------------------------------------------------------------------------
# pure decision logic -- CURRENT OBSERVATIONAL EVIDENCE ONLY below this line
# (never `research_verdict` / `research_promise_label` / `best_strategy_family`)
# ---------------------------------------------------------------------------


def _setup_label(trend: TrendDirection) -> str:
    if trend is TrendDirection.UP:
        return "Uptrend Continuation Watch"
    if trend is TrendDirection.DOWN:
        return "Downtrend Continuation Watch"
    return "Range / Consolidation Watch"


def _why_now(i: OpportunityInputs) -> tuple[str, ...]:
    lines: list[str] = []
    if i.trend in (TrendDirection.UP, TrendDirection.DOWN):
        lines.append(f"{i.trend.value} trend over the observed window")
    if i.volume_context == "Elevated":
        lines.append("Volume elevated relative to the observed window's own average")
    if i.curve_shape and i.curve_shape != "Insufficient Data":
        lines.append(f"Curve: {i.curve_shape}")
    if i.peer_total_count > 0 and i.peer_confirming_count > 0:
        lines.append(f"{i.peer_confirming_count}/{i.peer_total_count} related market(s) confirming")
    if not lines:
        lines.append("No strengthening evidence observed in the current window")
    return tuple(lines)


def _risks(i: OpportunityInputs) -> tuple[str, ...]:
    """CURRENT observational risk only -- never a registry verdict. Whether
    a related strategy is scientifically validated is Research Evidence
    (surfaced separately as "Scientific Status"), never a reason THIS
    market is risky right now."""
    risks: list[str] = []
    if i.volatility == "High":
        risks.append("Elevated realized volatility")
    if i.catalyst is not None and i.catalyst.importance == "HIGH":
        risks.append(f"{i.catalyst.name} approaching")
    if i.peer_total_count > 0 and i.peer_confirming_count == 0:
        risks.append("Related markets are not confirming")
    if not risks:
        risks.append("No elevated risk observed in the current window")
    return tuple(risks)


def _format_time_to(delta_seconds: float) -> str:
    if delta_seconds < 0:
        return "already passed"
    hours, remainder = divmod(int(delta_seconds), 3600)
    minutes = remainder // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"in {days}d {hours}h"
    return f"in {hours}h {minutes}m"


def _catalyst_label(i: OpportunityInputs) -> str:
    if i.catalyst is None:
        return "No major mapped catalyst"
    delta_seconds = (i.catalyst.scheduled_at - i.observed_at).total_seconds()
    return f"{i.catalyst.name} · {_format_time_to(delta_seconds)} · {i.catalyst.importance} IMPACT"


def _fmt_level(value: float | None) -> str:
    return f"{value:,.2f}" if value is not None else "the observed level"


def _window_desc(i: OpportunityInputs) -> str:
    return f"last {i.window_bars} observed {i.window_timeframe} bar(s)"


def _strengthen_invalidate(i: OpportunityInputs) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministic levels ONLY from this window's own observed high/low
    (never invented, and never labeled "support"/"resistance"/"predicted
    level" -- no such model exists here, only a plain observed range over an
    explicitly stated window). `Up`/`Down` frame the SAME level as a
    strengthen/invalidate pair (holds above vs. breaks below, or the
    symmetric case for a downtrend); `Sideways`/unknown frames the whole
    observed range. Peer-confirmation and volume/volatility conditions are
    appended only when that evidence actually exists for this product."""
    strengthen: list[str] = []
    invalidate: list[str] = []
    window = _window_desc(i)
    if i.trend is TrendDirection.UP and i.window_low is not None:
        strengthen.append(f"Price holds above the recent observed range low ({_fmt_level(i.window_low)}, {window})")
        invalidate.append(f"Price breaks below the recent observed range low ({_fmt_level(i.window_low)}, {window})")
    elif i.trend is TrendDirection.DOWN and i.window_high is not None:
        strengthen.append(f"Price holds below the recent observed range high ({_fmt_level(i.window_high)}, {window})")
        invalidate.append(f"Price breaks above the recent observed range high ({_fmt_level(i.window_high)}, {window})")
    elif i.window_low is not None and i.window_high is not None:
        strengthen.append(
            f"Price stays within the recent observed range [{_fmt_level(i.window_low)}, "
            f"{_fmt_level(i.window_high)}] ({window})"
        )
        invalidate.append(
            f"Price breaks out of the recent observed range [{_fmt_level(i.window_low)}, "
            f"{_fmt_level(i.window_high)}] ({window})"
        )

    if i.peer_total_count > 0:
        strengthen.append("Related markets continue confirming")
        invalidate.append("Relative-market confirmation disappears")
    if i.volume_context == "Elevated":
        strengthen.append("Volume remains elevated")
    invalidate.append(
        "Volatility regime changes materially (shifts to High)" if i.volatility != "High"
        else "Volatility regime changes materially"
    )

    if not strengthen:
        strengthen.append("No deterministic strengthening level available for this window")
    if not invalidate:
        invalidate.append("No deterministic invalidation level available for this window")
    return tuple(strengthen), tuple(invalidate)


#: How soon a HIGH-impact catalyst must be for it to argue against immediate
#: investigation (task spec section 13's WAIT state: "timing conditions
#: argue against immediate investigation").
_WAIT_CATALYST_HORIZON_SECONDS = 6 * 3600.0


def _opportunity_state(i: OpportunityInputs) -> OpportunityState:
    """OPPORTUNITY TRIAGE over CURRENT OBSERVATIONAL EVIDENCE ONLY -- never
    `research_verdict` (Product Acceptance Fix Pass: an existing PASS/REJECT/
    Research Promise must never change whether the CURRENT market looks
    interesting; that distinction lives in the separate "Scientific Status"
    surface). Documented rule, checked in this order:

    1. WAIT -- a HIGH-impact catalyst is due within `_WAIT_CATALYST_HORIZON_SECONDS`.
    2. INVESTIGATE -- a directional trend (Up/Down) PLUS at least one
       supporting signal (elevated volume or at least one confirming related
       market).
    3. WATCH -- everything else with any observed evidence at all.
    """
    if i.catalyst is not None and i.catalyst.importance == "HIGH":
        delta_seconds = (i.catalyst.scheduled_at - i.observed_at).total_seconds()
        if 0 <= delta_seconds <= _WAIT_CATALYST_HORIZON_SECONDS:
            return OpportunityState.WAIT

    has_directional_evidence = i.trend in (TrendDirection.UP, TrendDirection.DOWN)
    has_supporting_evidence = i.volume_context == "Elevated" or (
        i.peer_total_count > 0 and i.peer_confirming_count >= 1
    )
    if has_directional_evidence and has_supporting_evidence:
        return OpportunityState.INVESTIGATE
    return OpportunityState.WATCH


def _next_action(i: OpportunityInputs, state: OpportunityState) -> str:
    if state is OpportunityState.WAIT:
        return "Wait for the mapped catalyst to pass before investigating further"
    if i.trend in (TrendDirection.UP, TrendDirection.DOWN):
        return "Investigate trend-continuation mechanism"
    return "Investigate range / mean-reversion mechanism"


def _evidence_refs(i: OpportunityInputs) -> tuple[str, ...]:
    """Inspectability trace. The first six entries are what
    `opportunity_state`/`why_now`/`risks`/ranking actually used; the last
    (`research_verdict`) is RESEARCH EVIDENCE included for the UI's separate
    "Scientific Status" surface only -- it never fed the opportunity
    decision above (see the module docstring)."""
    return (
        f"trend_state={i.trend.value}",
        f"volatility_state={i.volatility}",
        f"volume_context_state={i.volume_context}",
        f"session_position_state={i.session_position}",
        f"curve_shape={i.curve_shape or 'N/A'}",
        f"peer_confirmation={i.peer_confirming_count}/{i.peer_total_count}",
        f"news_count_24h={i.news_count_24h}",
        f"research_verdict={i.research_verdict or 'UNTESTED'}",
    )


def _priority_key(i: OpportunityInputs, state: OpportunityState) -> tuple[int, int, int, int]:
    """INTERNAL ranking only -- never exposed on `OpportunitySnapshot`, never
    rendered as a confidence/probability/expected-return figure (task spec
    section 7/9), and never a function of `research_verdict` (Product
    Acceptance Fix Pass: current-opportunity ranking uses only current
    observational evidence). Sorted descending, in this priority order:

    1. `state_rank` -- INVESTIGATE (a genuine, actionable observational
       setup) outranks WATCH, which outranks WAIT (a timing condition
       argues against acting on this right now).
    2. `directional_rank` -- a directional trend (Up/Down) outranks Sideways.
    3. `volume_rank` -- elevated volume outranks normal/quiet/unavailable.
    4. `peer_confirming_count` -- more confirming related markets outranks fewer.
    """
    state_rank = {OpportunityState.INVESTIGATE: 2, OpportunityState.WATCH: 1, OpportunityState.WAIT: 0}[state]
    directional_rank = 1 if i.trend in (TrendDirection.UP, TrendDirection.DOWN) else 0
    volume_rank = 1 if i.volume_context == "Elevated" else 0
    return (state_rank, directional_rank, volume_rank, i.peer_confirming_count)


def build_opportunity(inputs: OpportunityInputs) -> OpportunitySnapshot:
    """The one pure builder: `OpportunityInputs` in, `OpportunitySnapshot`
    out. Deterministic -- calling this twice with an identical `inputs`
    always produces an identical (`==`) result."""
    state = _opportunity_state(inputs)
    strengthen, invalidate = _strengthen_invalidate(inputs)
    return OpportunitySnapshot(
        product=inputs.root_symbol,
        display_name=inputs.display_name,
        setup=_setup_label(inputs.trend),
        why_now=_why_now(inputs),
        risks=_risks(inputs),
        catalyst=_catalyst_label(inputs),
        strengthen_if=strengthen,
        invalidate_if=invalidate,
        opportunity_state=state,
        next_action=_next_action(inputs, state),
        observed_at=inputs.observed_at,
        evidence_refs=_evidence_refs(inputs),
    )


def rank_opportunities(all_inputs: list[OpportunityInputs]) -> list[OpportunitySnapshot]:
    """Every input built into a snapshot, ordered by the internal priority
    key (see `_priority_key`) -- highest priority first. Ties keep their
    original relative order (`sort` is stable) rather than an arbitrary
    re-ordering."""
    scored = [(_priority_key(i, _opportunity_state(i)), build_opportunity(i)) for i in all_inputs]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [snapshot for _, snapshot in scored]

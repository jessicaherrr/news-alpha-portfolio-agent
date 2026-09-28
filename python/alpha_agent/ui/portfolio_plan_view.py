"""News Alpha Phase G -- renders a `PortfolioPlan` under the signal ranking.

A pure renderer. Selection, statistics, sizing, limits and units come from
`alpha_agent.portfolio` (sizing from the C++ allocator); this module never
sorts, sizes or constrains anything -- it formats. A plan is rebuilt on each
render only from CACHED market snapshots (fast: the C++ allocator runs in
milliseconds); snapshots that are not cached yet are loaded behind an explicit
button, because a futures root is rebuilt offline from the raw store (about a
minute each, once).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import streamlit as st

from alpha_agent.news_alpha import DOMAIN_LABELS, CandidateSignalSet, MandateDomain, ResearchMandate
from alpha_agent.portfolio import (
    EligibilityMode,
    PlanStatus,
    PortfolioConstructionPolicy,
    PortfolioPlan,
    construct_portfolio_plan,
)
from alpha_agent.portfolio.handoff import build_handoff
from alpha_agent.portfolio.policy import ELIGIBILITY_TEXT
from alpha_agent.portfolio.risk_model import InsufficientRiskData, MarketSnapshotStore
from alpha_agent.portfolio.selection import screen_eligibility
from alpha_agent.recommendation.signal_ranking import RankedSignalSet
from alpha_agent.ui import candidate_signal_view

__all__ = [
    "SNAPSHOT_STORE",
    "build_plan",
    "cached_snapshot",
    "constraint_rows",
    "expander_label",
    "exposure_rows",
    "handoff_rows",
    "load_missing",
    "missing_snapshots",
    "position_rows",
    "rejection_rows",
    "render_plan_tabs",
    "render_portfolio_card",
    "screens_for",
    "selected_rows",
    "summary_line",
    "unheld_rows",
]

SNAPSHOT_STORE = MarketSnapshotStore()
_FAILED_KEY = "portfolio-snapshot-failures"

_MODE_LABEL = {EligibilityMode.QUALIFIED: "Qualified", EligibilityMode.EXPLORATORY: "Exploratory preview"}
_STATUS_LABEL = {
    PlanStatus.CONSTRUCTED: "constructed", PlanStatus.NO_EXECUTABLE_POSITION: "no executable position",
    PlanStatus.NO_ELIGIBLE_SIGNAL: "no eligible signal", PlanStatus.NO_ALLOCATABLE_SIGNAL: "nothing allocatable",
    PlanStatus.DEGENERATE_RISK_MODEL: "exposures hedge exactly",
    PlanStatus.RISK_INPUTS_UNAVAILABLE: "risk inputs unavailable",
    PlanStatus.ALLOCATOR_UNAVAILABLE: "C++ allocator not built",
}
_CONSTRAINT_LABEL = {
    "short_restriction": ("Shorting", "Shorting"), "unit_cap": ("Futures contract cap", "Futures contracts"),
    "instrument_concentration": ("Instrument concentration", "Instrument concentration"),
    "liquidity_participation": ("Liquidity participation", "Liquidity participation"),
    "sector_gross": ("Sector gross", "Sector concentration"),
    "asset_class_gross": ("Asset-class gross", "Asset-class concentration"),
    "gross_leverage": ("Gross leverage", "Gross leverage"), "net_exposure": ("Net exposure", "Net exposure"),
    "cluster_risk_share": ("Risk per exposure", "Risk per exposure"),
    "volatility_ceiling": ("Volatility ceiling", "Target volatility"), "turnover": ("Turnover", "Turnover"),
}


def screens_for(candidate_sets: Sequence[CandidateSignalSet]) -> dict:
    screens: dict = {}
    for cset in candidate_sets:
        screens.update(candidate_signal_view.stored_screens(cset))
    return screens


def missing_snapshots(ranked: RankedSignalSet, policy: PortfolioConstructionPolicy) -> list[tuple[MandateDomain, str]]:
    """Eligible instruments whose market snapshot is not cached (and has not already failed to load)."""
    eligible, _ = screen_eligibility(ranked, policy)
    failed = st.session_state.get(_FAILED_KEY, {})
    needed = dict.fromkeys((s.domain, s.instrument) for s in eligible)
    return [k for k in needed if f"{k[0].value}:{k[1]}" not in failed and not SNAPSHOT_STORE.has(*k)]


def cached_snapshot(domain: MandateDomain, symbol: str):
    snap = SNAPSHOT_STORE.load(domain, symbol)
    if snap is None:
        failed = st.session_state.get(_FAILED_KEY, {})
        raise InsufficientRiskData(failed.get(f"{domain.value}:{symbol}", "market data not loaded"))
    return snap


def _load(missing: list[tuple[MandateDomain, str]]) -> None:
    failed = st.session_state.setdefault(_FAILED_KEY, {})
    get = SNAPSHOT_STORE.provider()
    with st.status(f"Loading real 2018-2022 market data for {len(missing)} instrument(s)…", expanded=True) as status:
        for domain, symbol in missing:
            slow = " (rebuilt offline from the raw store, about a minute)" if domain is MandateDomain.FUTURES else ""
            status.write(f"{DOMAIN_LABELS[domain]} {symbol}{slow}")
            try:
                get(domain, symbol)
            except Exception as exc:  # noqa: BLE001 -- one instrument's data gap must not block the others
                failed[f"{domain.value}:{symbol}"] = f"{type(exc).__name__}: {exc}"
        status.update(label="Market data loaded", state="complete", expanded=False)


# ---------------------------------------------------------------------------
# rows (formatting only)
# ---------------------------------------------------------------------------


def _pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def _sector(text: str) -> str:
    return " ".join(w if w == "US" else w.capitalize() for w in text.split("_"))


def _participation(p: float) -> str:
    if p < 0:
        return "not measured"
    return "<0.001% of ADV" if p < 0.00001 else f"{p:.3%} of ADV"


def _symbol(plan: PortfolioPlan, key: str) -> str:
    risk = plan.risk_model.instrument(key)
    return f"{risk.symbol} ({risk.traded_symbol})" if risk.unit == "contract" else risk.symbol


def position_rows(plan: PortfolioPlan) -> list[dict[str, object]]:
    a, risk = plan.allocation, plan.risk_model
    if a is None or risk is None:
        return []
    vol = a.executable.annual_vol
    rows = []
    for i in plan.positions:
        r = risk.instrument(i.instrument_key)
        unit = "contract" if r.unit == "contract" else "share"
        liquidity = _participation(i.adv_participation) + ("*" if r.volume_scope == "PRIMARY_LISTING_ONLY" else "")
        rows.append({
            "Instrument": _symbol(plan, i.instrument_key), "Domain": DOMAIN_LABELS[r.domain],
            "Side": "Long" if i.units > 0 else "Short", "Units": f"{abs(i.units):,} {unit}{'s' if abs(i.units) != 1 else ''}",
            "Price": i.price, "Notional ($)": i.executable_notional_usd,
            "Weight": f"{i.executable_weight:+.1%}", "Volatility": _pct(i.annual_vol),
            "Risk share": _pct(i.risk_contribution / vol, 0) if vol > 0 else "--",
            "Class · sector": f"{i.asset_class.title()} · {_sector(i.sector)}",
            "Liquidity": liquidity,
        })
    return rows


def unheld_rows(plan: PortfolioPlan) -> list[dict[str, str]]:
    a = plan.allocation
    if a is None:
        return []
    rows = []
    for i in a.instruments:
        if i.units != 0:
            continue
        if not i.admitted:
            why = f"Not admitted: {i.reason.replace('_', ' ').lower()}"
        elif i.below_one_unit:
            why = (f"Its target (${abs(i.target_notional_usd):,.0f}) is smaller than one unit (${i.unit_notional_usd:,.0f}); "
                   f"one unit alone would carry {_pct(i.unit_risk_fraction)} of capital in annual volatility")
        elif i.target_weight == 0:
            why = "Its signals net to zero on this instrument"
        else:
            why = "Removed by a limit (see Constraints)"
        rows.append({"Instrument": _symbol(plan, i.instrument_key) if plan.risk_model else i.instrument_key,
                     "Why no position": why})
    return rows


def _labels(plan: PortfolioPlan) -> dict[str, str]:
    return {s.exposure_group_id: s.exposure_label for s in plan.selected}


def exposure_rows(plan: PortfolioPlan) -> list[dict[str, object]]:
    a = plan.allocation
    if a is None:
        return []
    labels = _labels(plan)
    rows = []
    for c in a.clusters:
        members = [plan.selected_signal(sid) for sid in c.signal_ids]
        rows.append({
            "Exposure": labels.get(c.cluster_id, c.cluster_id),
            "Signals": ", ".join(f"#{m.rank} {m.instrument} {m.prediction_horizon_days}D" for m in members) or "--",
            "Budget share (sizing)": (_pct(c.target_risk_contribution / a.target.annual_vol, 0)
                                      if c.allocated and a.target.annual_vol > 0 else "--"),
            "Risk share (held)": (_pct(c.executable_risk_contribution / a.executable.annual_vol, 0)
                                  if c.allocated and a.executable.annual_vol > 0 else "--"),
            "Status": "Allocated" if c.allocated else c.reason.replace("_", " ").lower(),
        })
    return rows


def _constraint_value(name: str, value: float) -> str:
    if name in ("unit_cap",):
        return f"{value:.2f}x of cap"
    if name in ("liquidity_participation",):
        return f"{value:.3%} of ADV"
    if name in ("cluster_risk_share", "volatility_ceiling"):
        return f"{value:.2%} vol"
    return f"{value:.1%} of capital"


def constraint_rows(plan: PortfolioPlan) -> list[dict[str, str]]:
    a = plan.allocation
    if a is None:
        return []
    source = {x.name: x for x in plan.constraints.limits}
    symbol = {i.instrument_key: i.instrument_key.split(":", 1)[1] for i in a.instruments}
    rows = []
    for c in a.constraints:
        if c.status == "NOT_APPLICABLE":
            continue
        label, limit_name = _CONSTRAINT_LABEL.get(c.name, (c.name, c.name))
        scope = _sector(c.scope.split(":", 1)[1]) if ":" in c.scope else c.scope
        set_by = source.get(limit_name)
        rows.append({
            "Constraint": label, "Scope": scope, "Stage": "Sizing" if c.stage == "allocation" else "Unit rounding",
            "Status": c.status.title(), "Limit": _constraint_value(c.name, c.limit),
            "Before": _constraint_value(c.name, c.before), "After": _constraint_value(c.name, c.after),
            "Changed": ", ".join(symbol.get(k, k) for k in c.affected) or "--",
            "Set by": f"{set_by.source.value.replace('_', ' ').lower()} -- {set_by.note}" if set_by else "policy",
        })
    return rows


def selected_rows(plan: PortfolioPlan) -> list[dict[str, object]]:
    a = plan.allocation
    rows = []
    for s in plan.selected:
        out = a.signal(s.candidate_signal_id) if a else None
        rows.append({
            "Rank": s.rank, "Signal": f"{s.instrument} {s.expression} → {s.prediction_horizon_days}D",
            "Exposure": s.exposure_label, "Factor on as-of": s.factor_value,
            "Direction": "Long" if s.direction == 1 else "Short",
            "Held weight": f"{out.executable_weight:+.2%}" if out and out.allocated else "--",
            "Status": ("Allocated" if out is None or (out.allocated and out.executable_weight != 0)
                       else "Allocated, not held (see Positions)" if out.allocated
                       else out.reason.replace("_", " ").lower()),
        })
    return rows


def rejection_rows(plan: PortfolioPlan) -> list[dict[str, object]]:
    return [{"Signal": r.name, "Rank": r.rank, "Stage": r.stage.value.title(),
             "Reason": r.reason.value.replace("_", " ").lower(), "Detail": r.detail} for r in plan.rejected]


def handoff_rows(plan: PortfolioPlan) -> list[dict[str, object]]:
    if plan.status is not PlanStatus.CONSTRUCTED:
        return []
    return [{
        "Decision bar (UTC)": datetime.fromtimestamp(r.ts_event_ns / 1e9, tz=UTC).strftime("%Y-%m-%d %H:%M"),
        "Root": r.root_symbol, "Contract / ticker": r.traded_symbol, "Target units": r.target_units,
    } for r in build_handoff(plan).rows]


def summary_line(plan: PortfolioPlan | None, missing: int, mode: EligibilityMode) -> str:
    if plan is None:
        return (f"Portfolio construction ({_MODE_LABEL[mode]}): market data for {missing} eligible instrument(s) is "
                "not loaded yet.")
    return f"Portfolio construction: {plan.headline()}"


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def _render_plan(plan: PortfolioPlan, *, key: str) -> None:
    a = plan.allocation
    if plan.is_exploratory:
        st.warning(ELIGIBILITY_TEXT[EligibilityMode.EXPLORATORY] + ".", icon=":material/science:")
    if a is None:
        st.caption(plan.headline())
    else:
        cols = st.columns(5)
        cols[0].metric("As of", str(plan.as_of), help="The last common trading day of the 2018-2022 discovery data")
        cols[1].metric("Capital", f"${plan.constraints.capital_usd:,.0f}",
                       help=next(x.note for x in plan.constraints.limits if x.name == "Capital"))
        cols[2].metric("Ex-ante vol / target", f"{_pct(a.executable.annual_vol)} / {_pct(a.target_annual_vol)}",
                       help="Annualized volatility of the held (whole-unit) book vs the target it was sized to")
        cols[3].metric("Gross / net", f"{a.executable.gross_exposure:.2f}x / {a.executable.net_exposure:+.2f}x")
        cols[4].metric("Exposures · positions", f"{sum(c.allocated for c in a.clusters)} · {len(plan.positions)}",
                       help="Independent exposures allocated · instruments actually held")
    tabs = st.tabs(["Positions", "Exposures & risk", "Constraints", "Signals", "Risk model", "Execution handoff"],
                   key=f"{key}-portfolio-tabs")
    with tabs[0]:
        rows = position_rows(plan)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True, key=f"{key}-positions", column_config={
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Notional ($)": st.column_config.NumberColumn(format="dollar")})
        else:
            st.caption("No position.")
        unheld = unheld_rows(plan)
        if unheld:
            st.caption("Targeted but not held:")
            st.dataframe(unheld, width="stretch", hide_index=True, key=f"{key}-unheld")
        st.caption("Units are whole futures contracts or whole shares, truncated toward zero so rounding never breaches "
                   "a limit. Notional = units x price x contract multiplier (1 for a share). ADV = median daily traded "
                   "notional; * = ETF volume from the primary listing venue only, so its liquidity is a lower bound.")
    with tabs[1]:
        rows = exposure_rows(plan)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True, key=f"{key}-exposures")
        st.caption("Each independent exposure (a Phase F exposure group) gets an equal share of the risk budget, "
                   "computed on the covariance of the exposures (equal risk contribution); signals inside one exposure "
                   "share its budget with equal standalone risk, each scaled by its own volatility. Held risk differs "
                   "from the sizing budget when a limit or whole-unit rounding cuts one exposure more than another "
                   "(see Constraints and Positions). Rank never sets a size -- it only decides which exposures and "
                   "members enter first.")
    with tabs[2]:
        rows = constraint_rows(plan)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True, key=f"{key}-constraints")
        st.table([{"Limit": x.name, "Value": x.value, "Set by": x.source.value.replace("_", " ").lower(),
                   "Why": x.note} for x in plan.constraints.limits], hide_index=True, border="horizontal")
    with tabs[3]:
        rows = selected_rows(plan)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True, key=f"{key}-selected", column_config={
                "Factor on as-of": st.column_config.NumberColumn(format="%+.4f")})
        st.caption("Direction = the sign of the signal's own factor on the as-of date x its declared sign. Screening "
                   "statistics never become an expected return or a weight.")
        rejected = rejection_rows(plan)
        if rejected:
            st.caption(f"Not in the portfolio ({len(rejected)}):")
            st.dataframe(rejected, width="stretch", hide_index=True, key=f"{key}-rejected")
    with tabs[4]:
        risk = plan.risk_model
        if risk is None:
            st.caption("No risk model -- " + plan.status_detail)
        else:
            st.caption(f"{risk.estimator}: {risk.n_observations} joint daily returns, {risk.window_start} → "
                       f"{risk.window_end}, annualized x{risk.annualization_days}. " + " ".join(risk.notes))
            labels = {i.key: i.symbol for i in risk.instruments}
            st.dataframe([{"Instrument": i.symbol, "Volatility": _pct(i.annual_vol), "Reference price": i.price,
                           "Traded": i.traded_symbol, "Median daily notional ($)": i.adv_usd,
                           "Class · sector": f"{i.asset_class.value.title()} · {_sector(i.sector)}"}
                          for i in risk.instruments], width="stretch", hide_index=True, key=f"{key}-risk-vols",
                         column_config={"Median daily notional ($)": st.column_config.NumberColumn(format="compact"),
                                        "Reference price": st.column_config.NumberColumn(format="%.2f")})
            if len(risk.instruments) > 1:
                st.dataframe([{"": labels[a], **{labels[b]: risk.correlation(a, b) for b in labels}} for a in labels],
                             width="stretch", hide_index=True, key=f"{key}-risk-corr",
                             column_config={labels[b]: st.column_config.NumberColumn(format="%+.2f") for b in labels})
    with tabs[5]:
        rows = handoff_rows(plan)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True, key=f"{key}-handoff")
            st.caption("The plan's target units per root at its decision bar, with a hard-risk configuration built from "
                       "the same limits -- the input of the existing C++ engine (BacktestEngine + PortfolioRiskManager), "
                       "which resolves the real contract, fills on the next bar and can resize or reject. That next bar "
                       "lies in the 2023-2024 validation window, so replaying the plan belongs to validation (Phase H).")
        else:
            st.caption("No executable targets.")
    st.caption(f"{plan.validation_note} Sizing: C++ {a.method if a else plan.method}. "
               f"Plan {plan.fingerprint()[:26]}…")


def expander_label(plan: PortfolioPlan | None, mode: EligibilityMode) -> str:
    state = "market data to load" if plan is None else _STATUS_LABEL[plan.status]
    return f"Portfolio construction · {_MODE_LABEL[mode]} · {state}"


def build_plan(ranked: RankedSignalSet, candidate_sets: Sequence[CandidateSignalSet], mandate: ResearchMandate,
               mode: EligibilityMode) -> tuple[PortfolioPlan | None, list[tuple[MandateDomain, str]]]:
    """``(plan, missing)``: the canonical plan for ``ranked`` in ``mode`` from
    CACHED snapshots only, or ``(None, missing)`` while an eligible
    instrument's market data is not loaded. The one plan-building call every
    renderer shares -- sizing stays in the C++ allocator."""
    policy = PortfolioConstructionPolicy(eligibility=mode)
    missing = missing_snapshots(ranked, policy)
    if missing:
        return None, missing
    return construct_portfolio_plan(ranked, screens_for(candidate_sets), mandate, snapshots=cached_snapshot,
                                    policy=policy), []


def load_missing(missing: list[tuple[MandateDomain, str]]) -> None:
    """The explicit "load market data" action (never run on render)."""
    _load(missing)


def render_plan_tabs(plan: PortfolioPlan, *, key: str) -> None:
    """The full plan inspector (positions, exposures, constraints, signals,
    risk model, execution handoff) -- for an Advanced details section."""
    _render_plan(plan, key=key)


def render_portfolio_card(ranked: RankedSignalSet, candidate_sets: Sequence[CandidateSignalSet],
                          mandate: ResearchMandate, *, key: str) -> None:
    """Caption + expander: the plan for ``ranked`` under ``mandate``, in the
    eligibility mode last chosen (Qualified by default)."""
    mode_key = f"{key}-portfolio-mode"
    mode = st.session_state.get(mode_key) or EligibilityMode.QUALIFIED
    plan, missing = build_plan(ranked, candidate_sets, mandate, mode)
    st.caption(summary_line(plan, len(missing), mode))
    with st.expander(expander_label(plan, mode), expanded=False, key=f"{key}-portfolio-expander"):
        st.segmented_control(
            "Eligibility", list(EligibilityMode), default=EligibilityMode.QUALIFIED, key=mode_key,
            format_func=_MODE_LABEL.__getitem__,
            help="Qualified: only signals whose screen supports continuing to validation. Exploratory preview: "
                 "signals without screening support whose evidence at least leans the declared way -- to see how the "
                 "allocator treats them, never a qualified portfolio.",
        )
        if plan is None:
            futures = sum(d is MandateDomain.FUTURES for d, _ in missing)
            wait = f" -- about {futures} minute(s) for futures" if futures else ""
            st.caption(f"{len(missing)} eligible instrument(s) need real market data before the plan can be sized: "
                       + ", ".join(f"{DOMAIN_LABELS[d]} {s}" for d, s in missing) + ".")
            if st.button(f"Load market data and build the plan{wait}", key=f"{key}-portfolio-load",
                         icon=":material/account_balance:",
                         help="Reads already-acquired 2018-2022 bars only (no download, no cost), never the 2023-2024 "
                              "validation window or the 2025 holdout. Cached for every later plan."):
                _load(missing)
                st.rerun()
            return
        _render_plan(plan, key=key)

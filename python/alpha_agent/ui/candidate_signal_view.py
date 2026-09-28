"""News Alpha Phase E -- renders one event's candidate signals and their
factor diagnostics in a Research Thread (and the Agent chat).

Every formula shown is `CandidateSignal.expression` -- the render of the
same typed `FactorExpression` the factor series is computed from and the
diagnostics name (`CandidateScreen` refuses to exist otherwise). Nothing
here is a verdict: a screen says whether a candidate deserves deeper
validation, and statuses are shown as text, never as PASS/FAIL colors.

Screening loads real bars (about a minute per futures root, rebuilt offline
from the raw store), so it runs ONLY on an explicit button press; results
are kept in the operational `FactorScreenStore` and shown on every later
render without recomputing -- including on another event's card that
reaches the same structural candidate.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.news_alpha import DOMAIN_LABELS, ExpressionFidelity, PathType
from alpha_agent.news_alpha.candidate_signals import (
    NOTE_TEXT,
    CandidateSignal,
    CandidateSignalSet,
    RefusalReason,
)
from alpha_agent.screening.candidate_signal_screen import (
    CandidateScreen,
    FactorScreenStore,
    discovery_window,
    screen_candidates,
)
from alpha_agent.screening.factor_diagnostics import ScreenStatus
from alpha_agent.ui import charts, components

__all__ = [
    "SCREEN_STORE",
    "candidate_rows",
    "chat_text",
    "diagnostic_rows",
    "expander_label",
    "lineage_markdown",
    "lineage_steps",
    "refusal_rows",
    "render_candidate_signals",
    "render_diagnostic_detail",
    "run_screens",
    "screen_status_of",
    "spec_rows",
    "stored_screens",
    "summary_line",
]

#: Module-level so tests (`tests/python/conftest.py`) point it at a temporary
#: directory -- never the user's real cache.
SCREEN_STORE = FactorScreenStore()

_SCREEN_TEXT = {
    None: "Candidate · not screened",
    ScreenStatus.SCREEN_CONTINUE: "Screened: continue to validation",
    ScreenStatus.NO_SCREEN_SUPPORT: "Screened: no support",
    ScreenStatus.CONTRADICTS_EXPECTED_SIGN: "Screened: contradicts its sign",
    ScreenStatus.INSUFFICIENT_DATA: "Screened: insufficient data",
}
_PATH_TYPE_TEXT = {
    PathType.DIRECT: "Direct", PathType.SUPPLY_CHAIN: "Supply chain", PathType.CROSS_SECTOR: "Cross-sector",
    None: "Unclassified",
}
_FIDELITY_TEXT = {
    ExpressionFidelity.DIRECT_UNDERLYING: "Direct underlying",
    ExpressionFidelity.DIRECT_COMPANY: "Direct company",
    ExpressionFidelity.SEGMENT_EXPOSURE: "One segment",
    ExpressionFidelity.CONSTITUENT_EXPOSURE: "Holds exposed names",
    ExpressionFidelity.ECOSYSTEM_PROXY: "Ecosystem proxy",
    ExpressionFidelity.MACRO_PROXY: "Macro proxy",
}
_REFUSAL_TEXT = {
    RefusalReason.NOT_CANDIDATE_READY: "No usable measurement (Phase D gate)",
    RefusalReason.CONFIRMATION_ONLY_MEASUREMENT: "Confirmation input, not directional",
    RefusalReason.NO_SIGNAL_RULE: "No signal rule for this measurement",
}
_ARROW = {"UP": "↑", "DOWN": "↓"}
_CANDIDATE_COLUMNS = {
    "Signal": st.column_config.TextColumn(width="large"),
    "Factor expression": st.column_config.TextColumn(
        width="medium", help="The exact formula the factor series is computed from and the diagnostics test",
    ),
    "Status": st.column_config.TextColumn(width="medium"),
    "Consequences": st.column_config.TextColumn(width="large"),
}
_DIAGNOSTIC_COLUMNS = {
    "Factor expression": st.column_config.TextColumn(width="medium"),
    "TS Spearman IC": st.column_config.NumberColumn(
        format="%+.3f",
        help="Time-series Spearman correlation: this instrument's factor vs its OWN forward return across dates, at "
             "the declared horizon. Not a cross-sectional Rank IC (that needs a universe of instruments).",
    ),
    "t (n/h)": st.column_config.NumberColumn(
        format="%+.2f",
        help="t of the TS Spearman IC on n/h effective observations -- overlapping h-day forward returns carry only about "
             "n/h independent windows (conservative)",
    ),
    "TS Pearson IC": st.column_config.NumberColumn(
        format="%+.3f", help="Time-series Pearson correlation at the declared horizon"),
    "TS IC mean/std (years)": st.column_config.NumberColumn(
        format="%+.2f",
        help="Mean / std of the yearly TS Spearman ICs -- not a cross-sectional ICIR (no per-date cross-section)",
    ),
    "Coverage": st.column_config.NumberColumn(format="percent"),
    "Break-even cost (bp)": st.column_config.NumberColumn(
        format="%.1f",
        help="One-way cost that would consume the drift-free timing edge of a +/-1 sign position (descriptive)",
    ),
}


def stored_screens(cset: CandidateSignalSet) -> dict[str, CandidateScreen]:
    return SCREEN_STORE.load_many(cset)


def _status(c: CandidateSignal, screens: dict[str, CandidateScreen]) -> ScreenStatus | None:
    s = screens.get(c.candidate_signal_id)
    return s.diagnostics.status if s is not None else None


def _consequences(c: CandidateSignal) -> str:
    seen = dict.fromkeys(f"{o.consequence_label} {_ARROW.get(o.consequence_direction.value, '?')}" for o in c.origins)
    return " · ".join(seen)


def summary_line(cset: CandidateSignalSet, screens: dict[str, CandidateScreen] | None = None) -> str | None:
    """One caption for the collapsed card; ``None`` when nothing is
    measurable."""
    if cset.is_empty:
        return None
    screens = stored_screens(cset) if screens is None else screens
    head = (
        f"Candidate signals: {len(cset.candidates)} testable factor hypotheses on "
        f"{', '.join(cset.instruments())} (e.g. {cset.candidates[0].spec.instrument} "
        f"{cset.candidates[0].expression} → {cset.candidates[0].spec.prediction_horizon.value})"
    )
    if not screens:
        return f"{head} -- none screened yet on 2018-2022 discovery data."
    counts = {s: sum(_status(c, screens) is s for c in cset.candidates) for s in ScreenStatus}
    parts = [f"{n} {_SCREEN_TEXT[s].removeprefix('Screened: ')}" for s, n in counts.items() if n]
    return (f"{head} -- {len(screens)} of {len(cset.candidates)} screened as unconditional factors: "
            f"{' · '.join(parts)}. Not event-conditioned evidence; not validated.")


def expander_label(cset: CandidateSignalSet, screens: dict[str, CandidateScreen] | None = None) -> str:
    screens = stored_screens(cset) if screens is None else screens
    return (
        f"Candidate signals & factor diagnostics · {len(cset.candidates)} candidates · "
        f"{len(screens)} screened"
    )


def candidate_rows(cset: CandidateSignalSet, screens: dict[str, CandidateScreen]) -> list[dict[str, object]]:
    rows = []
    for c in cset.candidates:
        spec = c.spec
        start, end = discovery_window(c)
        rows.append({
            "Signal": c.name,
            "Factor expression": c.expression,
            "Status": _SCREEN_TEXT[_status(c, screens)],
            "Instrument": spec.instrument,
            "Domain": DOMAIN_LABELS[spec.domain],
            "Prediction horizon": spec.prediction_horizon.value,
            "Formation lookback": f"{spec.formation_lookback} bars",
            "Expected relationship": spec.expected_relationship.value.title(),
            "Consequences": _consequences(c),
            "Path types": " · ".join(_PATH_TYPE_TEXT[t] for t in c.path_types),
            "Depth": ", ".join(str(d) for d in c.depths),
            "Fidelity": " · ".join(_FIDELITY_TEXT[f] for f in c.fidelities),
            "Measurement": spec.measurement_label,
            "Transform": spec.transform_label,
            "PIT": "Safe" if spec.data.pit_safe else "Not safe",
            "Data": f"{spec.data.dataset} · {start} → {end} (screen window)",
            "Notes": "; ".join(NOTE_TEXT[n] for n in c.notes) or "--",
        })
    return rows


def lineage_steps(c: CandidateSignal) -> list[tuple[str, tuple[str, ...]]]:
    """"Why this signal?" -- news -> mechanism -> signal paths -> consequence
    -> asset expression -> measurement -> factor expression. Each stage keeps
    every distinct origin (a candidate reached by four paths lists four)."""
    spec = c.spec
    arrow = {"UP": "↑", "DOWN": "↓"}
    anchors = dict.fromkeys(f"starts at {o.route[0]} (graph {o.mechanism_graph_id[:20]}…)" for o in c.origins)
    routes = dict.fromkeys(
        f"{' → '.join(o.route)} · {_PATH_TYPE_TEXT[o.path_type].lower()}, depth {o.transmission_depth}, "
        f"economic lag {o.transmission_lag.value.lower()}, {o.path_status.value.lower()}"
        for o in c.origins
    )
    expressions = dict.fromkeys(
        f"{o.expression_concept} · {DOMAIN_LABELS[spec.domain]} · fidelity {_FIDELITY_TEXT[o.expression_fidelity].lower()}"
        f" · pressure {arrow.get(o.pressure.value, '?') if o.pressure else '?'} (economic lag {o.transmission_lag.value.lower()})"
        for o in c.origins
    )
    data = spec.data
    return [
        ("News / event", tuple(dict.fromkeys(o.event_headline for o in c.origins))),
        ("Mechanism", tuple(anchors)),
        ("Signal paths", tuple(routes)),
        ("Economic consequence", (_consequences(c),)),
        ("Asset expression", tuple(expressions)),
        ("Measurement", (
            f"{spec.measurement_label} of {spec.instrument} -- {data.resolution_status.value.lower().replace('_', ' ')}"
            f", point-in-time safe, {data.dataset}" + (f" (proxy: {data.proxy_note})" if data.proxy_note else ""),
        )),
        ("Factor expression", (c.expression,)),
    ]


def _md(text: str) -> str:
    for ch in ("\\", "*", "_", "$", "`", "[", "]", "#"):
        text = text.replace(ch, "\\" + ch)
    return text


def lineage_markdown(c: CandidateSignal) -> str:
    lines = []
    for i, (stage, items) in enumerate(lineage_steps(c), 1):
        shown = [f"`{x}`" if stage == "Factor expression" else _md(x) for x in items]
        if len(shown) == 1:
            lines.append(f"{i}. **{stage}** -- {shown[0]}")
        else:
            lines.append(f"{i}. **{stage}** ({len(shown)})")
            lines += [f"    - {x}" for x in shown]
    return "\n".join(lines)


def spec_rows(c: CandidateSignal) -> list[dict[str, str]]:
    spec, data = c.spec, c.spec.data
    fs = spec.expression.feature_spec()
    return [
        {"Field": "Input field", "Value": spec.expression.field, "Why": "The measurement reads this bar field."},
        {"Field": "Transform", "Value": spec.transform_label, "Why": spec.rationale.transform},
        {"Field": "Executes", "Value": f"registered feature {fs.kind} {dict(fs.params)} (engine "
                                        f"{spec.expression.feature_metadata().version})",
         "Why": "The formula compiles to exactly this FeatureSpec -- no other transformation runs."},
        {"Field": "Formation lookback", "Value": f"{spec.formation_lookback} bars", "Why": spec.rationale.formation_lookback},
        {"Field": "Normalization", "Value": spec.expression.normalization.value.replace("_", " ").lower(),
         "Why": "Time-series factor on one instrument: a cross-sectional rank would be constant here."},
        {"Field": "Expected relationship", "Value": spec.expected_relationship.value.title(),
         "Why": spec.rationale.sign},
        {"Field": "Prediction horizon", "Value": spec.prediction_horizon.value, "Why": spec.rationale.prediction_horizon},
        {"Field": "Execution lag", "Value": f"{spec.execution_lag_bars} bar",
         "Why": "The factor at bar t's close is acted on no earlier than bar t+1's open."},
        {"Field": "Data", "Value": f"{data.dataset} · {data.data_schema} · {data.field}", "Why": data.coverage_note},
        {"Field": "Point-in-time", "Value": "Safe", "Why": data.pit_rule},
        {"Field": "Identity", "Value": f"{c.candidate_signal_id[:22]}… (factor {c.factor_identity[:22]}…)",
         "Why": "Structural: instrument, data and expression (+ horizon and sign) -- no event, path or timestamp."},
    ]


def diagnostic_rows(cset: CandidateSignalSet, screens: dict[str, CandidateScreen]) -> list[dict[str, object]]:
    rows = []
    for c in cset.candidates:
        s = screens.get(c.candidate_signal_id)
        if s is None:
            continue
        d = s.diagnostics
        dec = d.declared
        evaluable = [p for p in d.subperiods if p.evaluable]
        agree = sum(p.ts_spearman_ic is not None and (p.ts_spearman_ic > 0) == (d.expected_sign > 0) for p in evaluable)
        rows.append({
            "Signal": c.name,
            "Factor expression": s.expression,
            "Screen": _SCREEN_TEXT[d.status],
            "TS Spearman IC": dec.ts_spearman_ic,
            "t (n/h)": dec.ts_spearman_t,
            "TS Pearson IC": dec.ts_pearson_ic,
            "Years with expected sign": f"{agree} of {len(evaluable)}",
            "TS IC mean/std (years)": d.ts_spearman_ir_yearly,
            "Coverage": d.coverage.coverage,
            "Sign flips / yr": d.turnover.sign_flips_per_year,
            "Break-even cost (bp)": d.cost.break_even_cost_bps,
            "Pairs": dec.n_pairs,
        })
    return rows


def refusal_rows(cset: CandidateSignalSet) -> list[dict[str, str]]:
    return [
        {
            "Why not": _REFUSAL_TEXT[r.reason],
            "Expression": r.concept,
            "Domain": DOMAIN_LABELS[r.domain],
            "Instrument": r.instrument or "--",
            "Detail": r.note,
        }
        for r in cset.refusals
    ]


def _run_screens(cset: CandidateSignalSet) -> None:
    with st.status(f"Screening {len(cset.candidates)} candidate signals on 2018-2022 data…", expanded=True) as status:
        def progress(c: CandidateSignal) -> None:
            slow = " (futures bars are rebuilt offline from the raw store, about a minute per root)" \
                if c.spec.domain.value == "FUTURES" else ""
            status.write(f"{c.spec.instrument}: {c.expression} → {c.spec.prediction_horizon.value}{slow}")

        screen_candidates(cset, store=SCREEN_STORE, progress=progress)
        status.update(label="Screening complete", state="complete", expanded=False)


def _render_diagnostic_detail(c: CandidateSignal, s: CandidateScreen, *, key: str) -> None:
    d = s.diagnostics
    st.markdown(f"**{c.name}** -- {_SCREEN_TEXT[d.status]}")
    st.code(s.expression, language=None)
    st.caption(f"Scope: {d.scope.value.replace('_', ' ').lower()} · IC kind: {d.ic_kind.value.replace('_', '-').lower()}"
               f" -- {d.scope_note}")
    st.caption(d.status_note)
    left, right = st.columns(2)
    with left:
        st.caption("TS Spearman IC by horizon (decay)")
        components.plotly_chart(charts.signed_bar_chart(
            [f"{h.horizon_days}D" + (" (declared)" if h.horizon_days == d.declared_horizon_days else "")
             for h in d.horizons],
            [h.ts_spearman_ic for h in d.horizons],
            hover=[f"{h.horizon_days}D: TS Spearman IC {h.ts_spearman_ic:+.3f}, t {h.ts_spearman_t:+.2f}, TS Pearson IC "
                   f"{h.ts_pearson_ic:+.3f}, "
                   f"{h.n_pairs} pairs (~{h.n_independent:.0f} non-overlapping)"
                   if h.ts_spearman_ic is not None and h.ts_spearman_t is not None and h.ts_pearson_ic is not None
                   else f"{h.horizon_days}D: not defined ({h.n_pairs} pairs)" for h in d.horizons],
            yaxis_title="TS Spearman IC",
        ), key=f"{key}-decay")
    with right:
        st.caption(f"TS Spearman IC by year at {d.declared_horizon_days}D (subperiod stability)")
        components.plotly_chart(charts.signed_bar_chart(
            [p.label for p in d.subperiods], [p.ts_spearman_ic for p in d.subperiods],
            muted=[not p.evaluable for p in d.subperiods],
            hover=[f"{p.label}: TS Spearman IC {p.ts_spearman_ic:+.3f}, {p.n_pairs} pairs"
                   + ("" if p.evaluable else " -- too few pairs to evaluate") if p.ts_spearman_ic is not None
                   else f"{p.label}: not defined" for p in d.subperiods],
            yaxis_title="TS Spearman IC",
        ), key=f"{key}-years")
    st.caption(f"Factor values {s.series.window_start} → {s.series.window_end_exclusive} (exclusive)")
    components.plotly_chart(
        charts.factor_series_chart(list(s.series.days), list(s.series.values), name=s.expression),
        key=f"{key}-series",
    )
    t, cost, cov = d.turnover, d.cost, d.coverage
    st.table([
        {"Diagnostic": "Coverage", "Value": f"{cov.n_factor_defined} of {cov.n_days} days ({cov.coverage:.1%}); "
                                            f"first value {cov.first_defined_day}; {cov.undefined_after_warmup} "
                                            "undefined after warm-up (never filled)"},
        {"Diagnostic": "Turnover", "Value": (
            f"autocorrelation {t.autocorrelation_1d:.3f}; {t.sign_flips_per_year:.1f} sign flips per year"
            if t.autocorrelation_1d is not None and t.sign_flips_per_year is not None else "not enough data")},
        {"Diagnostic": "Cost sensitivity", "Value": (
            f"per {cost.horizon_days}D hold: gross edge {cost.gross_edge_bps:+.1f} bp = drift {cost.drift_bps:+.1f} bp "
            f"+ timing {cost.timing_edge_bps:+.1f} bp; break-even one-way cost "
            + (f"{cost.break_even_cost_bps:.1f} bp" if cost.break_even_cost_bps is not None else "none")
            if cost.gross_edge_bps is not None and cost.drift_bps is not None and cost.timing_edge_bps is not None
            else "not enough data") + f". {cost.note}"},
        {"Diagnostic": "Group concentration", "Value": d.concentration_note},
        {"Diagnostic": "Forward return", "Value": d.forward_return_convention},
        {"Diagnostic": "Data", "Value": f"{'; '.join(s.series.input_provenance)} · price domain "
                                        f"{s.series.price_domain.value} · {s.series.data_fingerprint[:18]}…"},
        {"Diagnostic": "Screen rule", "Value": f"{d.rule}: t (n/h) >= {d.policy.t_threshold:g} in the "
                                               f"predeclared direction and >= {d.policy.min_subperiod_sign_share:.0%}"
                                               f" of evaluable years agreeing; >= {d.policy.min_pairs} pairs"},
    ], hide_index=True, border="horizontal")


def render_candidate_signals(cset: CandidateSignalSet, *, key: str) -> None:
    if cset.is_empty:
        st.caption("No candidate signal: no measurement of this event's expressions is real, point-in-time safe "
                   "and executable on history yet.")
        st.caption(cset.not_validated_note)
        return
    screens = stored_screens(cset)
    st.caption(
        "Each candidate is one registered factor expression on one instrument, predicting its forward return over a "
        "declared horizon with a declared sign -- built only from measurements that were real, point-in-time safe "
        "and computable at the time. The formula shown is the one that runs."
    )
    st.dataframe(candidate_rows(cset, screens), width="stretch", hide_index=True, column_config=_CANDIDATE_COLUMNS)
    unscreened = [c for c in cset.candidates if c.candidate_signal_id not in screens]
    if unscreened:
        n_futures = len({c.spec.instrument for c in unscreened if c.spec.domain.value == "FUTURES"})
        wait = f" -- about {n_futures} minute(s) for futures bars" if n_futures else ""
        if st.button(
            f"Run factor diagnostics on 2018-2022 ({len(unscreened)} unscreened){wait}",
            key=f"{key}-run-factor-diagnostics", icon=":material/query_stats:",
            help="Reads already-acquired bars only (no download, no cost), never 2023-2024 validation data or "
                 "the 2025 holdout. Results are cached for every event that reaches the same candidate.",
        ):
            _run_screens(cset)
            st.rerun()

    names = {c.candidate_signal_id: f"{c.name} · {c.expression}" for c in cset.candidates}
    refused = refusal_rows(cset)
    tabs = st.tabs(
        ["Why this signal?", f"Diagnostics ({len(screens)} screened)", f"Not candidates ({len(refused)})"],
        key=f"{key}-candidate-signal-tabs",
    )
    with tabs[0]:
        chosen = st.selectbox("Candidate", list(names), format_func=names.__getitem__,
                              key=f"{key}-candidate-why")
        c = cset.candidate(chosen)
        st.code(c.expression, language=None)
        st.markdown(lineage_markdown(c))
        st.table(spec_rows(c), hide_index=True, border="horizontal")
        cond = c.conditioning
        st.caption(
            f"Event conditioning ({cond.variant.replace('_', ' ').lower()}): event strength "
            f"{cond.event_strength.value.replace('_', ' ').lower()} · exposure "
            f"{cond.exposure.value.replace('_', ' ').lower()} · confirmation "
            f"{cond.confirmation.value.replace('_', ' ').lower()} · transmission weight "
            f"{cond.transmission_weight.value.replace('_', ' ').lower()}. {cond.note}"
        )
        if c.notes:
            st.caption("Notes: " + "; ".join(NOTE_TEXT[n] for n in c.notes) + ".")
        st.caption(f"{len(c.origins)} originating hypotheses share this structural candidate; fidelity is shown as "
                   "provenance and never scores or orders anything.")
    with tabs[1]:
        rows = diagnostic_rows(cset, screens)
        if not rows:
            st.caption("Not screened yet. Run the factor diagnostics above to compute time-series ICs, decay, "
                       "subperiod stability, coverage, turnover and cost sensitivity on 2018-2022 data.")
        else:
            st.dataframe(rows, width="stretch", hide_index=True, column_config=_DIAGNOSTIC_COLUMNS)
            screened = [c for c in cset.candidates if c.candidate_signal_id in screens]
            pick = st.selectbox("Inspect", [c.candidate_signal_id for c in screened],
                                format_func=names.__getitem__, key=f"{key}-candidate-diagnostics")
            _render_diagnostic_detail(cset.candidate(pick), screens[pick], key=f"{key}-diag")
            st.caption(
                f"{len(cset.candidates)} candidates in this set count toward the family validation must correct "
                "across -- screening applies no multiple-testing correction. The diagnostics are unconditional: "
                "they test each factor on every day of 2018-2022 (scope: unconditional factor), not the reaction to this "
                "event -- they are not event-conditioned evidence. How correlated the screened factors are, and which "
                "are one exposure, is in the signal ranking below."
            )
    with tabs[2]:
        if refused:
            st.dataframe(refused, width="stretch", hide_index=True)
        st.caption(
            "Activity measurements are kept as confirmation inputs for a later event-conditioned test; expressions "
            "without a real, point-in-time safe, executable measurement stay out (the Phase D gate)."
        )
    st.caption(cset.not_validated_note)


def run_screens(cset: CandidateSignalSet) -> None:
    """The explicit "run factor diagnostics" action (never run on render)."""
    _run_screens(cset)


def render_diagnostic_detail(c: CandidateSignal, s: CandidateScreen, *, key: str) -> None:
    """Decay by horizon, stability by year, the factor series and the
    coverage / turnover / cost table for one screened candidate."""
    _render_diagnostic_detail(c, s, key=key)


def screen_status_of(c: CandidateSignal, screens: dict[str, CandidateScreen]) -> ScreenStatus | None:
    return _status(c, screens)


def chat_text(cset: CandidateSignalSet) -> str | None:
    if cset.is_empty:
        return None
    screens = stored_screens(cset)
    shown = ", ".join(f"{c.spec.instrument} {c.expression} → {c.spec.prediction_horizon.value}" for c in cset.candidates[:3])
    more = f" (+{len(cset.candidates) - 3} more)" if len(cset.candidates) > 3 else ""
    return (
        f"CANDIDATE SIGNALS: {len(cset.candidates)} testable factor hypotheses -- {shown}{more}; "
        f"{len(screens)} screened on 2018-2022 data. Candidates, not validated factors."
    )

"""Page 6 -- Experiment Log. A read-only browser over the Phase 14/14.1/14.2
SQLite registry. Nothing here can insert, update, or delete a row --
`alpha_agent.registry` forbids in-place mutation by construction, and this
page only ever calls read methods on `ExperimentRegistry`.
"""
from __future__ import annotations

import streamlit as st

from alpha_agent.ui import components, layout, services
from alpha_agent.ui.charts import bar_chart

PAGE_TITLE = "Experiment Log"


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active=None)
    layout.render_header(subtitle="The technical experiment registry -- read-only, append-only.")
    render_body()
    layout.render_disclaimer()


def render_body() -> None:
    """Content only, no page chrome -- see `strategies.render_body`'s
    docstring for why. Used inline by `views/research.py`'s consolidated
    "Experiments" tab."""
    st.markdown("## Experiment Log")
    st.caption(
        "Schema v5: scientific identity vs. execution attempt. Only a VALID attempt supplies an "
        "authoritative result; an INVALID_EXECUTION attempt is engineering/data-pipeline evidence, "
        "never a scientific refutation."
    )

    summary = services.registry_summary()
    violation = services.check_no_holdout_leak(summary, path="$.registry_summary")
    layout.holdout_leak_banner(violation)

    with components.metric_row("el-metrics"):
        m = st.columns(4)
        with m[0]:
            components.metric_card("el-hyp", "Statistical Hypotheses", str(summary["authoritative_statistical_hypotheses"]))
        with m[1]:
            components.metric_card("el-canon", "Canonical", str(summary["canonical"]))
        with m[2]:
            components.metric_card("el-neigh", "Neighbour", str(summary["neighbour"]))
        with m[3]:
            components.metric_card("el-valid", "Valid / Invalid Attempts",
                                    f"{summary['valid_execution_attempts']} / {summary['invalid_execution_attempts']}")

    vc = summary["canonical_verdict_counts"]
    with components.card("el-verdicts"):
        components.plotly_chart(bar_chart(list(vc.keys()), list(vc.values()), title="Canonical-trial headline verdicts"))

    tabs = st.tabs(["Browse Experiments", "Failure Memory", "Failure Records", "Imports / Lineage"])

    with tabs[0]:
        families = sorted({f["family_key"] for f in services.family_catalog()})
        roots = services.approved_universe()
        c1, c2, c3, c4, c5 = st.columns(5)
        family = c1.selectbox(
            "Strategy", ["(any)"] + families,
            format_func=lambda v: v if v == "(any)" else services.strategy_name(v), key="el-family",
        )
        r = c2.selectbox("Market", ["(any)"] + list(roots), key="el-root")
        verdict = c3.selectbox("Verdict", ["(any)", "PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED"], key="el-verdict")
        role = c4.selectbox("Trial role", ["(any)", "CANONICAL", "NEIGHBOUR", "ABLATION", "VARIANT"], key="el-role")
        authority = c5.selectbox("Authority", ["(any)", "AUTHORITATIVE", "SUPERSEDED"], key="el-authority")

        from alpha_agent.registry import RegistryVerdict, TrialRole

        rows = services.list_experiments(
            strategy_family=None if family == "(any)" else family,
            root_symbol=None if r == "(any)" else r,
            verdict=None if verdict == "(any)" else RegistryVerdict(verdict),
            trial_role=None if role == "(any)" else TrialRole(role),
            include_superseded=(authority != "(any)"),
        )
        if authority != "(any)":
            rows = [x for x in rows if x["authority"] == authority]

        st.write(f"{len(rows)} experiment(s)")
        st.dataframe(
            [
                {
                    "Experiment ID": x["experiment_id"], "Phase": x["phase"], "Market": x["root_symbol"],
                    "Strategy": services.strategy_name(x["strategy_family"]), "Role": x["trial_role"],
                    "Authority": x["authority"],
                    "Attempt Status": "VALID" if x["has_valid_authoritative_result"] else "INVALID-ONLY",
                    "Verdict": x["verdict"], "Net PnL": x["net_pnl_usd"], "Sharpe": x["daily_sharpe"],
                    "BH q": x["bh_q"], "DSR": x["dsr_probability"],
                }
                for x in rows
            ],
            width="stretch", hide_index=True, height=420,
        )
        if rows:
            sel = st.selectbox("Inspect experiment", [x["experiment_id"] for x in rows])
            detail = services.get_experiment(sel)
            with components.card("el-detail"):
                exp = detail["experiment"]
                st.markdown("**Identity & Provenance**")
                for label, value in [
                    ("Experiment Identity", exp["experiment_identity"]),
                    ("Strategy Fingerprint", exp["strategy_fingerprint"]),
                    ("Dataset Fingerprint", exp["dataset_fingerprint"]),
                    ("Split Identity", exp["split_identity"]),
                    ("Validation Spec Fingerprint", exp["validation_spec_fingerprint"]),
                    ("Reliability Policy Fingerprint", exp["reliability_policy_fingerprint"]),
                    ("Execution Config Identity", exp["execution_config_identity"]),
                    ("Cost Config Identity", exp["cost_config_identity"]),
                    ("Risk Identity", exp["risk_identity"]),
                    ("Source Artifact", (detail["result"] or {}).get("source_artifact") or "--"),
                ]:
                    components.provenance_row(label, value)
                st.markdown("**Attempt History**")
                st.dataframe(
                    [
                        {"Attempt": a["attempt_ordinal"], "Status": a["attempt_status"],
                         "Invalidation Class": a.get("invalidation_class") or "--",
                         "Engine": a.get("engine") or "--", "Created": a.get("created_at") or "--"}
                        for a in detail["attempts"]
                    ],
                    width="stretch", hide_index=True,
                )
                with st.expander("Full record (JSON)"):
                    st.json(detail)

    with tabs[1]:
        st.caption(
            "The structured answer a Research Agent gets BEFORE proposing or running a hypothesis "
            "(`FailureMemory.lookup`) -- prior evidence and near-duplicates, never a gate override."
        )
        families = sorted({f["family_key"] for f in services.family_catalog()})
        c1, c2 = st.columns(2)
        family = c1.selectbox("Strategy", families, format_func=services.strategy_name, key="fm-family")
        r = c2.selectbox("Market", list(services.approved_universe()), key="fm-root")
        fm = services.failure_memory_lookup(strategy_family=family, root_symbol=r)
        with components.card("el-fm"):
            st.write(
                f"Execution attempts: {fm['execution_attempts_total']} total "
                f"({fm['valid_execution_attempts']} valid / {fm['invalid_execution_attempts']} invalid). "
                f"Valid scientific outcomes: {fm['valid_scientific_outcomes']} "
                f"(refusals: {fm['valid_scientific_refusals']})."
            )
            st.write("Verdict counts:", fm["verdict_counts"])
            for lesson in fm["lessons"]:
                st.info(lesson)
        if fm["prior_experiments"]:
            st.markdown("**Prior Experiments**")
            st.dataframe(fm["prior_experiments"], width="stretch", hide_index=True)
        if fm["related_experiments"]:
            st.markdown("**Related Experiments** (similarity-ranked)")
            st.dataframe(
                [
                    {"Experiment": x["experiment_id"], "Root": x["root_symbol"], "Verdict": x["headline_verdict"],
                     "Similarity": round(x["similarity"]["score"], 3), "Reasons": "; ".join(x["similarity"]["reasons"])}
                    for x in fm["related_experiments"]
                ],
                width="stretch", hide_index=True,
            )

    with tabs[2]:
        split = services.failure_class_split()
        st.caption(
            f"Engineering (never a scientific refutation): {', '.join(split['engineering'])}. "
            f"Scientific (a real outcome): {', '.join(split['scientific'])}."
        )
        c1, c2 = st.columns(2)
        family = c1.selectbox(
            "Strategy", ["(any)"] + sorted({f["family_key"] for f in services.family_catalog()}),
            format_func=lambda v: v if v == "(any)" else services.strategy_name(v), key="fail-family",
        )
        r = c2.selectbox("Market", ["(any)"] + list(services.approved_universe()), key="fail-root")
        failures = services.list_failures(
            strategy_family=None if family == "(any)" else family, root_symbol=None if r == "(any)" else r,
        )
        st.write(f"{len(failures)} failure record(s)")
        st.dataframe(
            [
                {"ID": f["failure_id"], "Class": f["failure_class"],
                 "Kind": "engineering" if f["is_engineering"] else ("scientific" if f["is_scientific"] else "other"),
                 "Code": f["failure_code"], "Summary": f["summary"], "Resolved": f["resolved"]}
                for f in failures
            ],
            width="stretch", hide_index=True, height=360,
        )

    with tabs[3], components.card("el-imports"):
        st.markdown("**Import bundles** (append-only provenance of every batch write)")
        st.json(summary["imports"])
        st.caption(f"Registry content digest: `{summary['content_digest']}`")

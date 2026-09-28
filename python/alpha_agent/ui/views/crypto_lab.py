"""Page 9 -- Crypto Lab. Phase 22 (`prompts/22_CRYPTO_ONCHAIN_OPTIONAL_EXTENSION.md`)
is implemented as a SYNTHETIC SCAFFOLD ONLY (Phase 22 review-gate decision,
2026-09-11; Phase 22.1 hardening, 2026-09-12): zero network calls, zero paid
API use, zero real crypto-vendor connections. This page is READ-ONLY over the
ISOLATED `data/crypto_synthetic/registry.sqlite` store
(`alpha_agent.crypto.synthetic_registry`) -- a different file, a different
schema, from the real Phase 14 experiment registry every other page in this
app reads.

REAL here: the C++ Quant Core, the Phase 10 DSL/compiler, and the Phase 13
validation implementation. SYNTHETIC: the market data, the alternative data,
the contract-economics fixture, the calendar, and therefore the research
evidence itself -- this page never claims to show real CME BTC/ETH historical
performance. Nothing rendered here is, or can become:

* an authoritative scientific verdict (rendered as a "Synthetic Policy
  Outcome (NON-AUTHORITATIVE)", never a plain "Verdict" -- no `Authority` /
  `SUPERSEDES` concept exists in this store at all);
* paper-trading eligible (`alpha_agent.paper.eligibility` never opens this
  store, and the crypto strategy family is deliberately not one of
  `alpha_agent.strategy.candidates_phase_13_5c.BASELINE_FAMILIES`);
* part of the real Phase 13.5C+ BH/FDR statistical family.

Recording a new synthetic result is CLI-only
(`scripts/phase_22_crypto_research.py run`) -- never a button on this page,
exactly like Phase 21's paper-trading page keeps start/step/stop CLI-only.

Phase 22.1b: a legacy Phase 22 (schema v1) database -- from before the Phase
22.1 provenance envelope existed -- is detected and rendered as an honest
"LEGACY SYNTHETIC STORE" state, never crashed on and never displayed as if it
were a v2 validation artifact (`alpha_agent.crypto.synthetic_registry
.SyntheticRegistrySchemaError`).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from alpha_agent.crypto.envelope import SYNTHETIC_POLICY_OUTCOME_LABEL
from alpha_agent.crypto.synthetic_registry import SyntheticRegistrySchemaError
from alpha_agent.ui import components, layout, services

PAGE_TITLE = "Crypto Lab"


def _render_legacy_schema_state(exc: SyntheticRegistrySchemaError) -> None:
    st.warning("LEGACY SYNTHETIC STORE", icon="🗄️")
    st.error(
        "Phase 22 schema v1 detected. These rows are non-authoritative synthetic "
        "fixtures and lack the complete Phase 22.1 provenance envelope "
        "(provenance_by_series, dataset_identity, generator_identity, "
        "content_identity). Archive/reset the synthetic store and rerun the "
        "offline fixture -- see `python scripts/phase_22_crypto_research.py "
        "reset-store --confirm`.",
        icon="⚠️",
    )
    with components.card("cl-legacy-detail"):
        st.caption(str(exc))


def _render_banner() -> None:
    st.error(services.crypto_synthetic_banner(), icon="🧪")


def _render_detail(experiment_id: str) -> None:
    detail = services.get_crypto_synthetic_experiment(experiment_id)
    if detail is None:
        st.error(f"No synthetic experiment {experiment_id!r} in the crypto scaffold store.", icon="⚠️")
        return

    with components.metric_row("cl-metrics"):
        cols = st.columns(4)
        with cols[0]:
            components.metric_card("cl-verdict", SYNTHETIC_POLICY_OUTCOME_LABEL, detail["verdict"])
        with cols[1]:
            components.metric_card("cl-family", "Strategy", services.strategy_name(detail["strategy_family"]))
        with cols[2]:
            components.metric_card("cl-root", "Market (synthetic fixture)", detail["root_symbol"])
        with cols[3]:
            components.metric_card("cl-seed", "Fixture Seed", str(detail["fixture_seed"]))

    st.caption(
        "This outcome is computed by the real, unmodified Phase 13 "
        "ReliabilityPolicy code path -- but the market data and alternative "
        "data behind it are synthetic, so it is a software-pipeline "
        "diagnostic only, never an authoritative scientific result, never a "
        "real ExperimentRegistry entry, and never paper-trading eligible."
    )
    if detail["reason_codes"]:
        st.caption("Reason codes: " + ", ".join(detail["reason_codes"]))

    tabs = st.tabs(["Parameters", "Provenance (per series)", "Dataset / Generator Identity", "Full Report"])
    with tabs[0], components.card(f"cl-params-{experiment_id}"):
        st.json(detail["params"])
    with tabs[1]:
        st.caption(
            "One provenance record per input series -- every field below describes a "
            "deterministic seeded fixture generator run, never a vendor delivery. No "
            "series here is presented as real market or on-chain evidence."
        )
        prov_rows = [
            {
                "Series": name,
                "Role": p["role"],
                "Vendor": p["vendor"],
                "Series Kind": p["series_kind"],
                "Fixture Seed": p["fixture_seed"],
                "Notes": p["notes"],
            }
            for name, p in sorted(detail["provenance_by_series"].items())
        ]
        with components.card(f"cl-provenance-{experiment_id}"):
            st.dataframe(pd.DataFrame(prov_rows), hide_index=True, width="stretch")
    with tabs[2]:
        st.caption(
            "`content_identity` is stable and wall-clock independent -- rerunning "
            "byte-identical synthetic inputs on a different day reproduces the same value."
        )
        with components.card(f"cl-identity-{experiment_id}"):
            st.json(
                {
                    "content_identity": detail["content_identity"],
                    "generator_identity": detail["generator_identity"],
                    "strategy_fingerprint": detail["strategy_fingerprint"],
                    "validation_fingerprint": detail["validation_fingerprint"],
                    "dataset_identity": detail["dataset_identity"],
                }
            )
    with tabs[3], components.card(f"cl-report-{experiment_id}"):
        st.json(detail["report"])


def render() -> None:
    layout.inject_style()
    layout.render_sidebar_nav(active=None)
    layout.render_header(subtitle="Synthetic-fixture crypto research lab -- not part of the certified universe.")

    st.markdown("## Crypto Lab")
    st.caption(
        "Phase 22 -- BTC/ETH derivatives and on-chain feature research. A synthetic "
        "CME-futures-shaped execution fixture (synthetic OHLCV, synthetic alternative "
        "data, a fabricated schema-shape-only contract row, a synthetic 24/7 calendar) is "
        "processed through the REAL, unmodified C++ Quant Core, DSL/compiler, and "
        "validation implementation. No real crypto vendor is connected yet (no network "
        "call, no paid API, no vendor lock-in), so every result below is generated "
        "evidence about the software pipeline, never a claim about real CME BTC/ETH "
        "historical performance."
    )
    _render_banner()

    if not services.crypto_synthetic_db_exists():
        st.info(
            "No synthetic crypto experiment has been recorded yet. Run "
            "`python scripts/phase_22_crypto_research.py run` to generate one -- it is "
            "OFFLINE and free (synthetic data only).",
            icon="ℹ️",
        )
        layout.render_disclaimer()
        return

    try:
        rows = services.list_crypto_synthetic_experiments()
    except SyntheticRegistrySchemaError as exc:
        _render_legacy_schema_state(exc)
        layout.render_disclaimer()
        return
    if not rows:
        st.info("The synthetic scaffold store exists but has no recorded experiments yet.", icon="ℹ️")
        layout.render_disclaimer()
        return

    components.section_header(f"Recorded synthetic experiments ({len(rows)})")
    df = pd.DataFrame(
        [
            {
                "Experiment": r["experiment_id"],
                "Data Role": r["data_role"],
                "Strategy": services.strategy_name(r["strategy_family"]),
                "Market": r["root_symbol"],
                SYNTHETIC_POLICY_OUTCOME_LABEL: r["verdict"],
                "Seed": r["fixture_seed"],
                "Recorded (UTC)": r["created_at_utc"],
            }
            for r in rows
        ]
    )
    with components.card("cl-table"):
        st.dataframe(df, hide_index=True, width="stretch")

    options = {r["experiment_id"]: r["experiment_id"] for r in rows}
    choice = st.selectbox("Select an experiment", list(options.keys()), key="cl-select")
    _render_detail(options[choice])

    st.caption(
        "Record a new run with `python scripts/phase_22_crypto_research.py run` (not from "
        "this page) -- see docs/CRYPTO_ONCHAIN_EXTENSION.md."
    )
    layout.render_disclaimer()

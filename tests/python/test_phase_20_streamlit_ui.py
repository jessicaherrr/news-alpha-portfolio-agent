"""Phase 20 / 20.1 -- Streamlit research interface (prompts 20, 20.1).

`alpha_agent.ui.services` and `alpha_agent.ui.llm_demo` are plain Python (no
`streamlit` import) so the presentation-layer *logic* is tested unconditionally.
The Streamlit rendering smoke tests are gated behind `pytest.importorskip` so
the core suite still runs with no UI extras installed (CLAUDE.md: "keep
optional vendor imports lazy so core tests run without API clients installed").

Phase 20.1 fixes two integrity blockers found in Phase 20 and this file proves
both:

1. A trade/equity artifact is shown ONLY when a committed record verifiably
   binds it to the SELECTED experiment (never by `root_symbol` alone).
2. A validation gate reads PASS ONLY from explicit committed evidence
   (`all_required_gates_satisfied`) -- an absent failure reason code is a
   `NOT_EVALUATED` / `REFUSED_BEFORE_GATE` / `INCONCLUSIVE` state, never an
   inferred PASS.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.registry.holdout_guard import HoldoutAccessError
from alpha_agent.ui import llm_demo, services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)


# ---------------------------------------------------------------------------
# services.py -- read-only registry/report/catalog access
# ---------------------------------------------------------------------------


def test_registry_summary_is_read_only_and_typed():
    summary = services.registry_summary()
    assert summary["schema_version"] == 7
    assert summary["authoritative_statistical_hypotheses"] > 0


def test_services_module_never_calls_a_registry_write_method():
    """Static regression guard: the read-only boundary claim in this module's
    docstring must stay mechanically true, not just true today by accident."""
    src = services.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE",
        r"\.record_signal_path_evidence\(",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, text), f"services.py must never call {pattern!r}"


def test_list_experiments_filters_by_family_and_root():
    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ")
    assert rows
    assert all(r["strategy_family"] == "tsmom" and r["root_symbol"] == "NQ" for r in rows)
    assert all("strategy_fingerprint" in r for r in rows)


def test_get_experiment_round_trips_canonical_result():
    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ", trial_role=None)
    canonical = next(r for r in rows if r["trial_role"] == "CANONICAL")
    detail = services.get_experiment(canonical["experiment_id"])
    assert detail["result"]["headline_verdict"] == canonical["verdict"]


def test_failure_memory_lookup_distinguishes_engineering_from_scientific():
    resp = services.failure_memory_lookup(strategy_family="tsmom", root_symbol="NQ")
    assert "verdict_counts" in resp
    split = services.failure_class_split()
    assert set(split["engineering"]) & set(split["scientific"]) == set()


def test_family_and_feature_catalog_are_nonempty():
    families = {f["family_key"] for f in services.family_catalog()}
    assert {"tsmom", "ma_trend", "breakout", "mean_reversion"} <= families
    assert services.feature_catalog()


def test_catalog_summary_reports_all_five_roots():
    summary = services.catalog_summary()
    assert set(summary["roots"]) == {"CL", "ES", "GC", "NQ", "ZN"}


def test_market_name_and_research_window_are_derived_from_real_data():
    assert services.market_name("NQ") == "Nasdaq 100"
    w = services.research_window()
    assert w["research"] != "unavailable"
    assert w["validation"] != "unavailable"
    assert "2025" in w["holdout"]


def test_load_validation_report_matches_registry_headline_verdict():
    report = services.load_validation_report("NQ", "tsmom")
    assert report is not None
    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ", trial_role=None)
    canonical = next(r for r in rows if r["trial_role"] == "CANONICAL")
    assert report["headline_verdict_global_family"]["verdict"] == canonical["verdict"]


def test_ml_meta_label_report_reflects_the_genuine_phase_15b_refusal():
    report = services.ml_meta_label_report()
    assert report is not None
    assert report["n_pass"] == 0
    assert report["n_refused"] == report["counts"]["statistical_hypotheses_bh_family"]


def test_holdout_guard_actually_fires_on_a_locked_value():
    from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data

    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data({"date": "2025-06-01"}, path="$.test")

    # the UI-facing wrapper catches it and returns a message instead of raising,
    # so a rendered page can fail loud without crashing the whole app.
    violation = services.check_no_holdout_leak({"date": "2025-06-01"}, path="$.test")
    assert violation is not None


def test_holdout_guard_is_silent_on_clean_registry_payloads():
    detail = services.get_experiment(
        services.list_experiments(strategy_family="tsmom", root_symbol="NQ")[0]["experiment_id"]
    )
    assert services.check_no_holdout_leak(detail, path="$.detail") is None


# ---------------------------------------------------------------------------
# Blocker #1 (Phase 20.1) -- experiment-bound trade ledger, never root-matched
# ---------------------------------------------------------------------------


def test_no_experiment_in_the_registry_has_a_bound_trade_ledger_today():
    """Every committed Phase 13.5C / 15B `ResultRecord.source_artifact` is a
    JSON report, never a `trades.csv` -- so this must be None everywhere,
    proving the function does not fall back to a same-root file."""
    rows = services.list_experiments()
    assert rows
    for row in rows:
        detail = services.get_experiment(row["experiment_id"])
        assert services.find_experiment_bound_trade_ledger(detail) is None


def test_trade_ledger_binding_requires_a_source_artifact_naming_it():
    detail_no_source = {"result": {"source_artifact": None}}
    assert services.find_experiment_bound_trade_ledger(detail_no_source) is None

    detail_wrong_kind = {"result": {"source_artifact": "outputs/phase_13_5c/NQ__TSMOM__validation_report.json"}}
    assert services.find_experiment_bound_trade_ledger(detail_wrong_kind) is None

    detail_no_result = {"result": None}
    assert services.find_experiment_bound_trade_ledger(detail_no_result) is None


def test_trade_ledger_binding_refuses_a_hash_mismatch():
    """Even when `source_artifact` DOES name a `trades.csv`, a wrong committed
    hash must refuse to load it -- integrity over convenience."""
    scratch_dir = services.REPO_ROOT / "outputs" / "_test_scratch_phase_20_1_mismatch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    ledger = scratch_dir / "trades.csv"
    ledger.write_text("trade_index,net_pnl_usd\n0,100\n", encoding="utf-8")
    try:
        rel = str(ledger.relative_to(services.REPO_ROOT))
        detail = {"result": {"source_artifact": rel, "source_artifact_sha256": "0" * 64}}
        assert services.find_experiment_bound_trade_ledger(detail) is None
    finally:
        ledger.unlink(missing_ok=True)
        scratch_dir.rmdir()


def test_trade_ledger_binding_succeeds_with_a_verified_hash():
    ledger_dir = services.REPO_ROOT / "outputs" / "_test_scratch_phase_20_1"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "trades.csv"
    ledger_path.write_text("trade_index,net_pnl_usd\n0,100\n1,-40\n", encoding="utf-8")
    try:
        import hashlib

        digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
        rel = str(ledger_path.relative_to(services.REPO_ROOT))
        detail = {"result": {"source_artifact": rel, "source_artifact_sha256": digest}}
        bound = services.find_experiment_bound_trade_ledger(detail)
        assert bound is not None
        assert bound["n_trades"] == 2
        assert bound["source_artifact"] == rel
    finally:
        ledger_path.unlink(missing_ok=True)
        ledger_dir.rmdir()


# ---------------------------------------------------------------------------
# Blocker #2 (Phase 20.1) -- explicit-only gate state, never inferred PASS
# ---------------------------------------------------------------------------


def test_gate_definitions_match_the_real_reason_code_enum():
    from alpha_agent.validation.enums import ReasonCode

    real_values = {c.value for c in ReasonCode}
    for _label, fail_code in services.GATE_DEFINITIONS:
        assert fail_code in real_values, f"{fail_code!r} is not a real ReasonCode value"
    assert "all_required_gates_satisfied" in real_values


def test_gate_state_fail_when_explicit_reason_code_present():
    state = services.gate_state(
        fail_code="fdr_qvalue_above_threshold",
        reason_codes=["null_hypothesis_not_rejected", "fdr_qvalue_above_threshold"],
        verdict="REJECT", trial_role="CANONICAL",
    )
    assert state == "FAIL"


def test_gate_state_never_infers_pass_or_not_evaluated_from_an_absent_reason_code():
    """The Phase 20 bug (inferred PASS), reproduced and pinned, plus its
    Phase 20.1 half-fix (inferred NOT_EVALUATED -- itself an unjustified
    factual claim with no committed proof for this gate): a REJECT verdict
    with this gate's own code absent, and no paired 'not evaluated' code
    either, must read NOT_AVAILABLE -- neither PASS nor NOT_EVALUATED."""
    state = services.gate_state(
        fail_code="performance_concentrated_in_one_root",
        reason_codes=["null_hypothesis_not_rejected", "fdr_qvalue_above_threshold"],
        verdict="REJECT", trial_role="CANONICAL",
    )
    assert state not in ("PASS", "NOT_EVALUATED")
    assert state == "NOT_AVAILABLE"


def test_gate_state_pass_only_with_explicit_all_gates_satisfied():
    state = services.gate_state(
        fail_code="fdr_qvalue_above_threshold",
        reason_codes=["all_required_gates_satisfied"],
        verdict="PASS", trial_role="CANONICAL",
    )
    assert state == "PASS"

    # a PASS verdict without the explicit code is never granted a free PASS
    state2 = services.gate_state(
        fail_code="fdr_qvalue_above_threshold", reason_codes=[], verdict="PASS", trial_role="CANONICAL",
    )
    assert state2 != "PASS"


def test_gate_state_refused_before_gate_on_minimum_sample_inconclusive():
    state = services.gate_state(
        fail_code="fdr_qvalue_above_threshold",
        reason_codes=["insufficient_trades"],
        verdict="INCONCLUSIVE", trial_role="CANONICAL",
    )
    assert state == "REFUSED_BEFORE_GATE"


def test_gate_state_not_evaluated_only_with_this_gates_own_explicit_code():
    """Regime Robustness has a paired committed 'not evaluated' reason code
    (`regime_not_evaluated`) -- the ONE case NOT_EVALUATED is legitimate."""
    state = services.gate_state(
        fail_code="performance_concentrated_in_one_regime",
        reason_codes=["regime_not_evaluated"],
        verdict="INCONCLUSIVE", trial_role="CANONICAL",
    )
    assert state == "NOT_EVALUATED"

    state2 = services.gate_state(
        fail_code="performance_concentrated_in_one_root",
        reason_codes=["cross_market_not_evaluated"],
        verdict="INCONCLUSIVE", trial_role="CANONICAL",
    )
    assert state2 == "NOT_EVALUATED"


def test_gate_state_not_available_when_a_different_gate_has_no_explicit_code():
    """Same INCONCLUSIVE-by-unevaluated-evidence-family scenario, but for a
    gate with NO committed 'not evaluated' code of its own (only Regime /
    Cross-Market have one) -- must be NOT_AVAILABLE, not NOT_EVALUATED and
    not PASS, since nothing commits either claim for THIS gate."""
    state = services.gate_state(
        fail_code="parameter_neighbourhood_unstable",
        reason_codes=["regime_not_evaluated"],
        verdict="INCONCLUSIVE", trial_role="CANONICAL",
    )
    assert state == "NOT_AVAILABLE"


def test_gate_state_not_adjudicated_for_a_neighbour_trial():
    state = services.gate_state(
        fail_code="fdr_qvalue_above_threshold", reason_codes=[], verdict="NOT_ADJUDICATED", trial_role="NEIGHBOUR",
    )
    assert state == "NOT_ADJUDICATED"


def test_gate_state_not_available_with_no_result():
    state = services.gate_state(
        fail_code="fdr_qvalue_above_threshold", reason_codes=[], verdict=None, trial_role="CANONICAL",
    )
    assert state == "NOT_AVAILABLE"


def test_real_registry_never_shows_a_gate_pass_without_the_family_wide_pass():
    """0/107 Phase 13.5C hypotheses PASS (CLAUDE.md / memory) -- so across the
    WHOLE real registry, `gate_table` must never emit PASS for any gate."""
    rows = services.list_experiments()
    seen_any = False
    for row in rows:
        detail = services.get_experiment(row["experiment_id"])
        result = detail["result"]
        table = services.gate_table(
            reason_codes=tuple((result or {}).get("reason_codes") or ()),
            verdict=(result["headline_verdict"] if result else None),
            trial_role=detail["experiment"]["trial_role"],
        )
        seen_any = True
        assert all(g["state"] != "PASS" for g in table), row["experiment_id"]
    assert seen_any


def test_real_registry_only_regime_and_cross_market_gates_can_ever_show_not_evaluated():
    """NOT_EVALUATED requires a gate's own committed 'not evaluated' reason
    code, and only Regime Robustness / Cross-Market Evidence have one -- so no
    other gate may ever show NOT_EVALUATED anywhere in the real registry."""
    allowed = {"Regime Robustness", "Cross-Market Evidence"}
    rows = services.list_experiments()
    for row in rows:
        detail = services.get_experiment(row["experiment_id"])
        result = detail["result"]
        table = services.gate_table(
            reason_codes=tuple((result or {}).get("reason_codes") or ()),
            verdict=(result["headline_verdict"] if result else None),
            trial_role=detail["experiment"]["trial_role"],
        )
        for g in table:
            if g["state"] == "NOT_EVALUATED":
                assert g["label"] in allowed, (row["experiment_id"], g["label"])


def test_gate_evidence_text_never_asserts_a_verdict_word():
    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ", trial_role=None)
    canonical = next(r for r in rows if r["trial_role"] == "CANONICAL")
    detail = services.get_experiment(canonical["experiment_id"])
    stats = services.gate_evidence_text(detail["result"])
    banned = {"PASS", "FAIL", "REJECT"}
    for text in stats.values():
        assert not banned & set(text.upper().split()), text


# ---------------------------------------------------------------------------
# llm_demo.py -- real Phase 16/17 agents, scripted (offline) LLM client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sc", llm_demo.SCENARIOS, ids=lambda s: s.key)
def test_every_demo_scenario_proposes_and_compiles_cleanly(sc):
    universe = services.approved_universe()
    proposal, err, _fm = llm_demo.propose_hypothesis(
        mode=llm_demo.SCRIPTED_MODE, scenario_key=sc.key, universe=universe
    )
    assert err is None, err
    assert proposal.accepted

    compiled, cerr = llm_demo.compile_hypothesis(
        mode=llm_demo.SCRIPTED_MODE,
        scenario_key=sc.key,
        hypothesis=proposal.hypothesis,
        universe=universe,
    )
    assert cerr is None, cerr
    assert compiled.accepted, compiled.rejection_detail
    assert not compiled.feature_coverage.missing
    assert not compiled.feature_coverage.unapproved


def test_live_mode_never_touches_process_environment_for_the_key(monkeypatch):
    """`build_llm_client` must construct `AnthropicClient()` with no key
    argument -- it is the SDK's job to read `ANTHROPIC_API_KEY`, never this
    module's. Simulate the key being absent and confirm the failure is a plain
    typed error, not a crash that could leak partial state."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    universe = services.approved_universe()
    proposal, err, _fm = llm_demo.propose_hypothesis(
        mode=llm_demo.LIVE_MODE, scenario_key="tsmom_nq", universe=universe
    )
    assert proposal is None
    assert err is not None
    assert "ANTHROPIC_API_KEY" not in err  # never echoes the variable's value


# ---------------------------------------------------------------------------
# Streamlit rendering smoke tests (skipped without the `ui` extra installed)
# ---------------------------------------------------------------------------

_VIEW_MODULES = [
    "alpha_agent.ui.views.agent",
    "alpha_agent.ui.views.dashboard",
    "alpha_agent.ui.views.research",
    "alpha_agent.ui.views.strategies",
    "alpha_agent.ui.views.backtests",
    "alpha_agent.ui.views.validation",
    "alpha_agent.ui.views.experiment_log",
    "alpha_agent.ui.views.paper_trading",
    "alpha_agent.ui.views.learn",
    "alpha_agent.ui.views.system",
]


def test_all_ten_pages_render_without_exceptions(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    # Phase 21.1b: PaperLedger.__init__ migrates whatever database it opens.
    # Isolate the Paper Trading page's ledger read so this sweep never
    # touches a real local data/paper_trading/paper_ledger.sqlite.
    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", tmp_path / "does_not_exist.sqlite")

    for mod in _VIEW_MODULES:
        script = f"from {mod} import render\nrender()\n"
        at = AppTest.from_string(script)
        at.run(timeout=90)
        assert not list(at.exception), f"{mod}: {list(at.exception)}"


def test_app_entrypoint_renders_the_default_news_page():
    """The News Alpha Research Thread workspace made News -- "what is worth
    researching?" -- the default landing (`st.Page(..., default=True)` on
    `news.render`); the Agent conversation is the Ask tool. This proves the
    entrypoint actually resolves to News."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(services.REPO_ROOT / "python" / "alpha_agent" / "ui" / "app.py"))
    at.run(timeout=90)
    assert not list(at.exception)
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "What is worth researching?" in markdown_text
    assert "What deserves attention?" not in markdown_text  # Ask's hero, not the landing any more


def test_market_products_own_selector_switches_the_local_selected_root():
    """Sidebar IA pass (task spec section 1): market selection is no longer a
    shared-sidebar control -- it is Market's own local
    `market_selected_root`, changed via the Product Detail area's "Change
    Market" popover (`views/market.py::_render_product_header_and_selector`)."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.market import render\nrender()\n")
    at.run(timeout=90)
    first_root = at.session_state["market_selected_root"]
    assert first_root == services.approved_universe()[0]
    at.button(key="market-selector-NQ").click().run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["market_selected_root"] == "NQ"


def test_agent_to_strategies_handoff_renders_clean():
    """Product refactor: Agent (not Research) is where a hypothesis is
    generated -- Strategies' own compile flow still reads `ra_proposal` /
    `ra_scenario_key` from session state exactly as before, just populated by
    Agent's Generate Hypothesis now instead of the old Research page."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    ra = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    ra.run(timeout=90)
    ra.button(key="agent-send").click().run(timeout=90)
    assert not list(ra.exception)
    assert ra.session_state["ra_proposal"]["accepted"]

    sp = AppTest.from_string("from alpha_agent.ui.views.strategies import render\nrender()\n")
    sp.session_state["ra_proposal"] = ra.session_state["ra_proposal"]
    sp.session_state["ra_scenario_key"] = ra.session_state["ra_scenario_key"]
    sp.run(timeout=90)
    sp.button(key="strategies-compile").click().run(timeout=90)
    assert not list(sp.exception)
    assert sp.session_state["lab_compiled"]["accepted"]


def test_backtests_and_validation_selectboxes_cycle_every_root_and_family():
    """Sweeps every (root, family) combo the pages themselves offer -- catches
    an empty-role / missing-report edge case without a full browser."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    for mod in ("backtests", "validation"):
        at = AppTest.from_string(f"from alpha_agent.ui.views.{mod} import render\nrender()\n")
        at.run(timeout=90)
        for root in services.approved_universe():
            market_sb = next(sb for sb in at.selectbox if sb.label in ("Market", "Root symbol"))
            market_sb.set_value(root).run(timeout=90)
            assert not list(at.exception), f"{mod}: root={root}: {list(at.exception)}"
        for family_key in sorted({f["family_key"] for f in services.family_catalog()}):
            family_sb = next(sb for sb in at.selectbox if sb.label in ("Strategy", "Strategy family"))
            family_sb.set_value(family_key).run(timeout=90)
            assert not list(at.exception), f"{mod}: family={family_key}: {list(at.exception)}"


# Phase 21 landed the real Paper Trading page (target -> order -> simulated
# fill -> portfolio through the hard-risk-gated C++ engine, a persistent
# ledger, drift diagnostics, alerts). Its dedicated, more thorough render
# tests (including a real-run-data smoke test with a seeded ledger fixture)
# live in tests/python/test_phase_21_streamlit_ui.py -- this file no longer
# asserts the old "Phase 21 not built" empty-placeholder shape.


def test_validation_page_never_renders_a_pass_badge_for_a_reject_experiment():
    """End-to-end proof of blocker #2 through the actual page, not just the
    pure function: the real committed NQ/tsmom canonical trial is REJECT, so
    no gate card on that page may render a PASS badge."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ", trial_role=None)
    canonical = next(r for r in rows if r["trial_role"] == "CANONICAL")
    assert canonical["verdict"] == "REJECT"

    at = AppTest.from_string("from alpha_agent.ui.views.validation import render\nrender()\n")
    at.run(timeout=90)
    next(sb for sb in at.selectbox if sb.label in ("Market", "Root symbol")).set_value("NQ").run(timeout=90)
    next(sb for sb in at.selectbox if sb.label in ("Strategy", "Strategy family")).set_value("tsmom").run(timeout=90)
    assert not list(at.exception)
    joined = "\n".join(m.value for m in at.markdown)
    assert "✓ PASS" not in joined

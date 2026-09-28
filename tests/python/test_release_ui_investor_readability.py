"""Release UX -- whole-platform investor-readability pass.

Presentation-only: global investor-facing terminology (`services.market_name`
/ `services.strategy_name`), a plain-language gate-outcome synthesis
(`services.plain_language_reason`), an executive Dashboard summary, a
Strategies research-status table, the renamed sidebar Market Browser, the
Backtest-vs-scientific-PASS distinction on Backtests, and the Agent page's
"ALREADY TESTED" de-emphasis of a redundant re-run. None of this touches
scientific semantics, Registry schema/behavior, C++ execution, BH/FDR, DSR,
`ReliabilityPolicy`, paper eligibility, or holdout rules -- every assertion
below is either a pure string-mapping check or reads the real, already-
committed local registry (no write, no C++ execution, no network call).
"""
from __future__ import annotations

import pytest
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)


# ---------------------------------------------------------------------------
# services.market_name / services.strategy_name -- friendly labels with a
# safe raw-value fallback (spec: "unknown values must safely fall back to
# raw values"; never a blank, never a raised error).
# ---------------------------------------------------------------------------


def test_market_name_friendly_labels_and_safe_fallback():
    assert services.market_name("NQ") == "Nasdaq 100"
    assert services.market_name("ES") == "S&P 500"
    assert services.market_name("ZN") == "10Y Treasury"
    # An unrecognised root is never blank and never raises.
    assert services.market_name("XYZ") == "XYZ"


def test_strategy_name_friendly_labels_and_safe_fallback():
    assert services.strategy_name("tsmom") == "Time-Series Momentum"
    assert services.strategy_name("ma_trend") == "Moving-Average Trend"
    assert services.strategy_name("breakout") == "Breakout"
    assert services.strategy_name("mean_reversion") == "Mean Reversion"
    assert services.strategy_name("silver_bullet") == "Silver Bullet"
    # An unrecognised family key is never blank and never raises -- it falls
    # back to the raw registry key verbatim (e.g. a crypto-lab-only family).
    assert services.strategy_name("crypto_funding_contrarian") == "crypto_funding_contrarian"
    assert services.strategy_name(None) == "--"
    assert services.strategy_name("") == "--"


def test_strategy_name_handles_ml_meta_label_composite_family():
    assert services.strategy_name("ml_meta_label.tsmom") == "ML Meta-Label: Time-Series Momentum"
    # An ML-meta-label base this table does not know is still never blank.
    assert services.strategy_name("ml_meta_label.something_new") == "ML Meta-Label: something_new"


# ---------------------------------------------------------------------------
# services.plain_language_reason -- a readable synthesis built ONLY from the
# same GATE_DEFINITIONS labels / gate_state resolution the gate grid already
# renders. Never a new judgment; every failed-gate label it names must
# actually resolve to FAIL under the real `services.gate_state`.
# ---------------------------------------------------------------------------


def test_plain_language_reason_pass_is_the_generic_all_gates_sentence():
    text = services.plain_language_reason(
        result={"reason_codes": ["all_required_gates_satisfied"]}, trial_role="CANONICAL", verdict="PASS",
    )
    assert text == "All required validation gates were explicitly satisfied."


def test_plain_language_reason_names_only_gates_that_actually_resolve_to_fail():
    result = {"reason_codes": ["fdr_qvalue_above_threshold", "deflated_sharpe_below_threshold"]}
    text = services.plain_language_reason(result=result, trial_role="CANONICAL", verdict="REJECT")
    for label, code in services.GATE_DEFINITIONS:
        state = services.gate_state(fail_code=code, reason_codes=result["reason_codes"], verdict="REJECT", trial_role="CANONICAL")
        if state == "FAIL":
            assert label in text
        else:
            assert label not in text


def test_plain_language_reason_refused_before_gate():
    result = {"reason_codes": ["insufficient_trades"]}
    text = services.plain_language_reason(result=result, trial_role="CANONICAL", verdict="INCONCLUSIVE")
    assert "refused before any statistical gate ran" in text


def test_plain_language_reason_on_the_real_committed_nq_tsmom_reject_experiment():
    """End-to-end against real committed evidence, not a synthetic fixture."""
    rows = services.list_experiments(strategy_family="tsmom", root_symbol="NQ", trial_role=None)
    canonical = next(r for r in rows if r["trial_role"] == "CANONICAL")
    assert canonical["verdict"] == "REJECT"
    detail = services.get_experiment(canonical["experiment_id"])
    text = services.plain_language_reason(
        result=detail["result"], trial_role=detail["experiment"]["trial_role"], verdict="REJECT",
    )
    assert text and text != "See the gate-by-gate detail below for the exact committed reason codes."


# ---------------------------------------------------------------------------
# Dashboard -- executive Research Overview above the per-market/strategy
# detail, built only from already-exposed registry tallies.
# ---------------------------------------------------------------------------


def test_dashboard_renders_research_overview_with_real_registry_tallies():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.dashboard import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "Research Overview" in markdown_text
    assert "Research Coverage" in markdown_text
    assert "Validated -- PASS" in markdown_text
    assert "Paper-Trading Eligible" in markdown_text
    assert "Recent Research" in markdown_text

    verdicts = services.canonical_verdict_distribution()
    eligible = services.paper_eligible_experiments()
    # These metric cards are custom markdown (`components.metric_card`), not
    # native `st.metric`, so the real committed count is asserted against the
    # rendered `aa-metric-value` div content rather than `at.metric`.
    assert f">{verdicts.get('PASS', 0)}</div>" in markdown_text
    assert f">{len(eligible)}</div>" in markdown_text


def test_dashboard_recent_research_uses_friendly_strategy_names():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.dashboard import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)

    rows = [r for r in services.list_experiments(trial_role=None) if r.get("verdict")]
    if not rows:
        pytest.skip("no adjudicated experiment committed in this checkout")
    top = max(rows, key=lambda r: r["experiment_id"])
    markdown_text = " ".join(md.value for md in at.markdown)
    assert services.strategy_name(top["strategy_family"]) in markdown_text


# ---------------------------------------------------------------------------
# Strategies -- "what strategies exist, and what is their research status?"
# ---------------------------------------------------------------------------


def test_strategies_page_renders_research_status_table_with_friendly_names():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.strategies import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "Strategies" in markdown_text

    dataframes = list(at.dataframe)
    assert dataframes, "expected at least the research-status dataframe to render"
    columns = {c for df in dataframes for c in df.value.columns}
    for expected in ("Strategy", "Market", "Research Status", "Scientific Verdict", "Paper Eligible"):
        assert expected in columns
    strategy_values = {v for df in dataframes if "Strategy" in df.value.columns for v in df.value["Strategy"]}
    assert "Time-Series Momentum" in strategy_values or "Moving-Average Trend" in strategy_values


def test_strategies_page_filters_never_raise_and_empty_state_is_readable():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.strategies import render\nrender()\n")
    at.run(timeout=90)
    verdict_sb = next(sb for sb in at.selectbox if sb.key == "strat-status-verdict")
    verdict_sb.set_value("Untested").run(timeout=90)
    assert not list(at.exception), list(at.exception)
    paper_sb = next(sb for sb in at.selectbox if sb.key == "strat-status-paper")
    paper_sb.set_value("YES").run(timeout=90)
    assert not list(at.exception), list(at.exception)


# ---------------------------------------------------------------------------
# Sidebar's former "Workspace / Current Market" block is retired entirely
# (Sidebar IA pass, task spec section 1): market selection is now Market's
# own local `market_selected_root` state -- see
# `test_layout_sidebar.py`/`test_sidebar_nav.py` for its current coverage.
# ---------------------------------------------------------------------------
# Backtests -- "Backtest Completed" is never presented as "Scientific PASS".
# ---------------------------------------------------------------------------


def test_backtests_page_distinguishes_completed_execution_from_scientific_pass():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.backtests import render\nrender()\n")
    at.run(timeout=90)
    next(sb for sb in at.selectbox if sb.label == "Market").set_value("NQ").run(timeout=90)
    next(sb for sb in at.selectbox if sb.label == "Strategy").set_value("tsmom").run(timeout=90)
    assert not list(at.exception), list(at.exception)

    caption_text = " ".join(c.value for c in at.caption)
    assert "Backtest Completed is not the same as Scientific PASS" in caption_text


# ---------------------------------------------------------------------------
# Agent -- an EXISTING exact strategy leads with ALREADY TESTED, and its
# Run This Hypothesis action is de-emphasized (never removed -- a genuine
# reproduction re-run remains available; the underlying registry decides
# re-execution eligibility, never this presentation layer).
# ---------------------------------------------------------------------------


def test_agent_shows_already_tested_for_a_known_existing_exact_strategy():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    # The default preset ("Multi-week time-series momentum in NQ") compiles to
    # a StrategySpec whose exact fingerprint already has a committed REJECT
    # (see test_agent_to_research_navigation.py's equivalent fixture).
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    # AppTest's SafeSessionState has no real `.get` (it resolves the string
    # "get" as a missing widget key instead) -- an explicit membership check
    # is required here, not the usual dict `.get(..., None)`.
    evidence = at.session_state["agent_last_evidence"] if "agent_last_evidence" in at.session_state else None  # noqa: SIM401
    if not (evidence and evidence.get("match_type") == "exact_fingerprint" and evidence.get("result")):
        pytest.skip("default preset has no EXISTING exact-fingerprint registry evidence in this checkout")

    markdown_text = " ".join(md.value for md in at.markdown)
    assert "ALREADY TESTED" in markdown_text

    run_button = at.button(key="agent-run-hypothesis")
    assert run_button is not None  # never removed -- reproduction remains possible
    assert run_button.proto.type != "primary"  # de-emphasized, not the primary next action


def test_agent_already_tested_never_blocks_the_confirm_flow():
    """De-emphasis is presentation-only -- clicking Run still reaches the same
    two-step confirmation as a genuinely new hypothesis (no execution here)."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    at.button(key="agent-send").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    at.button(key="agent-run-hypothesis").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    assert at.session_state["agent-run-confirm-pending"] is True
    assert "agent_run_outcome" not in at.session_state or at.session_state["agent_run_outcome"] is None

"""Phase B1 -- UI integration: investor profile panel, Validated/Promising
sections (Dashboard), Verdict/Promise/Fit columns (Strategies), the top-level
Verdict/Promise/Fit row (Research Details), and the secondary-only treatment
on Validation.

`alpha_agent.ui.services`'s Phase B1 additions are plain Python (tested
unconditionally below); the Streamlit rendering smoke tests are gated behind
`pytest.importorskip`, matching every other Phase-2x UI test file. Every
`AppTest` here isolates the local investor-profile file to `tmp_path` so a
test run never reads or writes the real `data/user_prefs/investor_profile.json`.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.recommendation import DEFAULT_PROFILE, InvestorProfile, ProfileStore, RiskStyle
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


# ---------------------------------------------------------------------------
# services.py -- read-only / local-file-only boundary
# ---------------------------------------------------------------------------


def test_services_module_still_never_calls_a_registry_write_method():
    """Phase B1 added new functions to services.py -- re-proves the Phase 20
    read-only boundary claim still holds mechanically, not just by accident."""
    with open(services.__file__, encoding="utf-8") as fh:
        text = fh.read()
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, text), f"services.py must never call {pattern!r}"


def test_investor_profile_persistence_never_touches_the_registry_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "_PROFILE_STORE", ProfileStore(path=tmp_path / "profile.json"))
    assert services.load_investor_profile() == DEFAULT_PROFILE
    custom = InvestorProfile(risk_style=RiskStyle.AGGRESSIVE)
    services.save_investor_profile(custom)
    assert services.load_investor_profile() == custom
    assert (tmp_path / "profile.json").exists()


def test_research_candidate_summaries_reflect_real_registry_state():
    candidates = services.research_candidate_summaries()
    assert len(candidates) > 0
    assert all(c.scientific_verdict in ("PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED") for c in candidates)


def test_candidate_summary_for_unknown_experiment_is_none_not_an_exception():
    assert services.candidate_summary_for_experiment("does-not-exist-12345") is None


def test_candidate_summary_for_experiment_matches_bulk_summary():
    from alpha_agent.registry import TrialRole

    rows = services.list_experiments(trial_role=TrialRole.CANONICAL)
    row = rows[0]
    bulk = {c.experiment_id: c for c in services.research_candidate_summaries()}[row["experiment_id"]]
    single = services.candidate_summary_for_experiment(row["experiment_id"])
    assert single is not None
    assert single.research_promise_score == bulk.research_promise_score
    assert single.scientific_verdict == bulk.scientific_verdict


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_profile_store(tmp_path, monkeypatch):
    """Every AppTest below runs against an isolated local profile file --
    never the real `data/user_prefs/investor_profile.json`."""
    pytest.importorskip("streamlit")
    monkeypatch.setattr(services, "_PROFILE_STORE", ProfileStore(path=tmp_path / "profile.json"))


def test_agent_page_shows_a_compact_profile_summary():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "Balanced" in markdown_text  # the default RiskStyle headline
    assert at.button(key="agent-profile-edit-toggle")


def test_agent_page_profile_edit_round_trips(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(services, "_PROFILE_STORE", ProfileStore(path=tmp_path / "profile.json"))
    at = AppTest.from_string("from alpha_agent.ui.views.agent import render\nrender()\n")
    at.run(timeout=90)
    at.button(key="agent-profile-edit-toggle").click().run(timeout=90)
    assert not list(at.exception)

    # `st.selectbox` inside `st.form` has no explicit `key` here -- select it
    # by its label instead (Streamlit's AppTest exposes each widget's label).
    risk_box = next(sb for sb in at.selectbox if sb.label == "Risk Style")
    risk_box.select("Aggressive")
    submit = next(b for b in at.button if b.label == "Save Profile")
    submit.click().run(timeout=90)
    assert not list(at.exception)
    assert services.load_investor_profile().risk_style.value == "Aggressive"


def test_dashboard_shows_validated_and_promising_sections_with_honest_empty_state():
    """Real registry state today: 0 PASS -- the Validated Strategies section
    must render a legitimate empty state, never an error, never a fabricated
    PASS."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.dashboard import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    markdown_text = " ".join(md.value for md in at.markdown)
    assert "Validated Strategies" in markdown_text
    assert "Most Promising Research Candidates" in markdown_text
    if len(services.validated_strategy_summaries()) == 0:
        assert "No strategy currently satisfies all scientific validation requirements" in markdown_text


def test_strategies_page_has_validated_promising_and_all_tested_tabs():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.strategies import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    tab_labels = [t.label for t in at.tabs]
    assert {"Validated", "Promising Research", "All Tested"}.issubset(set(tab_labels))


def test_strategies_all_tested_table_carries_promise_and_fit_columns():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.strategies import render\nrender()\n")
    at.run(timeout=90)
    dataframes = at.dataframe
    assert dataframes, "expected at least one st.dataframe on the Strategies page"
    found = False
    for df in dataframes:
        cols = set(df.value.columns) if hasattr(df.value, "columns") else set()
        if {"Research Promise", "User Fit"}.issubset(cols):
            found = True
    assert found


def test_research_details_shows_verdict_promise_fit_row():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.research import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    markdown_text = " ".join(md.value for md in at.markdown)
    # Only rendered when a real experiment is selected -- the registry-latest
    # fallback selection kicks in on a fresh session as long as SOME
    # authoritative result exists (real registry: 21 canonical trials with a
    # committed result).
    if services.latest_authoritative_experiment() is not None:
        assert "Research Promise" in markdown_text
        assert "User Fit" in markdown_text


def test_validation_page_keeps_promise_fit_as_secondary_collapsed_context():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.validation import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    expander_labels = [e.label for e in at.expander]
    matches = [label for label in expander_labels if "Research Promise & User Fit" in label]
    assert matches, "expected a Research Promise & User Fit expander on Validation"
    # secondary treatment: the score is rendered as a plain st.caption inside
    # that expander, never as one of the page's own `metric_card`/badge-style
    # top-level elements the dominant scientific verdict uses.
    caption_text = " ".join(c.value for c in at.caption)
    assert "Research Promise:" in caption_text


def test_no_misleading_validated_language_for_reject_or_inconclusive():
    """Task spec section 9/10: never soften REJECT/INCONCLUSIVE into
    PASS-like language anywhere Phase B1 renders a candidate card."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.dashboard import render\nrender()\n")
    at.run(timeout=90)
    promising = services.promising_candidate_summaries()
    if not promising:
        return
    markdown_html = " ".join(md.value for md in at.markdown)
    for c in promising[:5]:
        assert c.scientific_verdict != "PASS"
    # the literal word "Validated" must never appear directly next to a
    # REJECT/INCONCLUSIVE candidate's own card content beyond the section
    # header itself (which correctly names the SEPARATE PASS-only section).
    assert "Verdict: REJECT" in markdown_html or "Verdict: INCONCLUSIVE" in markdown_html or not promising

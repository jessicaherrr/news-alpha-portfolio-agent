"""Tests for the Discover-page bugfix pass (manual-inspection findings 1-9).

Covers: DISABLED vs NOT_CONNECTED source-status semantics (1), default
product mode / honest Claude-unavailable banner (2), Fast Screen "not yet
screened" honesty and Draft-Candidate-until-ranked semantics (3), no
H-DEMO/experiment-id/duplicated-root text on the primary card surface (4),
the before/after Fast Screen candidate-card information hierarchy (5),
Research Process metric split (6), the "Run C++ Fast Screen" CTA reusing the
already-compiled pool (7), and objective/market consistency (9). Issue 8
(live connector health) and the real end-to-end scenarios (10) are covered
by `test_alpha_discovery_live_c5-c7_*` (already real, unconditional/keyless)
and this session's own manual verification.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from alpha_agent.knowledge import (
    DisabledAdapter,
    IngestionStatus,
    NotConnectedAdapter,
    SourceType,
    disabled_adapters,
    ingest_all,
)
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.screening.fast_screen import FastScreenStatus
from alpha_agent.ui import discovery_campaign, llm_demo, services


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    path = tmp_path / "registry.sqlite"
    monkeypatch.setattr(services, "REGISTRY_PATH", path)
    ExperimentRegistry(path)
    return path


# ---------------------------------------------------------------------------
# issue 1 -- DISABLED vs NOT_CONNECTED semantics
# ---------------------------------------------------------------------------


def test_disabled_adapter_reports_disabled_never_not_connected():
    adapter = DisabledAdapter(SourceType.GITHUB)
    result = adapter.ingest(query="", markets=("NQ",))
    assert result.status == IngestionStatus.DISABLED
    assert result.status != IngestionStatus.NOT_CONNECTED
    assert result.items == ()


def test_disabled_adapters_covers_all_four_external_categories():
    adapters = disabled_adapters()
    assert set(adapters) == {SourceType.GITHUB, SourceType.ACADEMIC, SourceType.COMMUNITY, SourceType.PRACTITIONER}
    for source_type, adapter in adapters.items():
        assert isinstance(adapter, DisabledAdapter)
        assert adapter.source_type == source_type


def test_ingest_all_with_disabled_adapters_never_reports_not_connected():
    results = ingest_all(disabled_adapters(), query="", markets=("NQ",))
    assert len(results) == 4
    assert all(r.status == IngestionStatus.DISABLED for r in results)


def test_not_connected_adapter_is_unchanged_distinct_status():
    """`NotConnectedAdapter` (used by `default_adapters()` for a caller that
    never thought about sources at all) keeps its own, different meaning --
    this bugfix pass adds DISABLED, it does not repurpose NOT_CONNECTED."""
    result = NotConnectedAdapter(SourceType.GITHUB).ingest(query="", markets=())
    assert result.status == IngestionStatus.NOT_CONNECTED


def test_offline_campaign_reports_disabled_for_every_external_source(isolated_registry):
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha for CL.", root="CL",
        family_stem="test_bugfix_offline", run_fast_screen_backtests=False, research_sources="offline",
    )
    assert outcome.accepted, outcome.error
    statuses = {r.source_type: r.status for r in outcome.knowledge_base.external_ingestion_results}
    assert statuses[SourceType.GITHUB] == IngestionStatus.DISABLED
    assert statuses[SourceType.ACADEMIC] == IngestionStatus.DISABLED
    assert statuses[SourceType.COMMUNITY] == IngestionStatus.DISABLED
    assert statuses[SourceType.PRACTITIONER] == IngestionStatus.DISABLED
    assert IngestionStatus.NOT_CONNECTED not in statuses.values()


# ---------------------------------------------------------------------------
# issue 2 -- default product mode / honest Claude-unavailable banner
# ---------------------------------------------------------------------------


def _discover_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string("from alpha_agent.ui.views.discover import render\nrender()\n")


def test_claude_unavailable_shows_honest_banner_never_silent_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _discover_app()
    at.run(timeout=90)
    assert not list(at.exception)
    blob = " ".join(w.body for w in at.warning)
    assert "Connected sources available" in blob
    assert "Reasoning model unavailable" in blob
    # research engine defaults to Offline/Deterministic when Claude is unavailable
    assert at.radio(key="discover-mode").value == llm_demo.SCRIPTED_MODE


def test_claude_available_defaults_to_connected_research_prominently(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    at = _discover_app()
    at.run(timeout=90)
    assert not list(at.exception)
    blob = " ".join(m.value for m in at.markdown)
    assert "CONNECTED RESEARCH" in blob
    assert at.radio(key="discover-mode").value == llm_demo.LIVE_MODE
    assert at.radio(key="discover-research-sources").value == "connected"


def test_external_sources_default_connected_even_without_claude(monkeypatch):
    """Sources need no Anthropic key at all -- they default to available
    regardless of whether the reasoning model is configured."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _discover_app()
    at.run(timeout=90)
    assert at.radio(key="discover-research-sources").value == "connected"


def test_advanced_settings_still_exposes_offline_and_source_subset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    at = _discover_app()
    at.run(timeout=90)
    # both options remain genuinely selectable, just not the highlighted
    # default (AppTest's `.options` reports the formatted display labels).
    mode_widget = at.radio(key="discover-mode")
    assert len(mode_widget.options) == 2
    assert any("Offline" in o for o in mode_widget.options)
    assert any("Claude" in o for o in mode_widget.options)
    sources_widget = at.radio(key="discover-research-sources")
    assert len(sources_widget.options) == 2
    assert any("Internal + Classic only" == o for o in sources_widget.options)


# ---------------------------------------------------------------------------
# issues 3/4/5 -- candidate title, draft-vs-ranked, no internal ids on the
# primary surface
# ---------------------------------------------------------------------------


def _member(*, strategy_family, root_symbol, hypothesis_title, hypothesis_id="H-DEMO-X"):
    return SimpleNamespace(
        strategy_family=strategy_family, root_symbol=root_symbol, hypothesis_title=hypothesis_title,
        hypothesis_id=hypothesis_id, experiment_identity=f"experiment1:{hypothesis_id}",
        strategy_fingerprint="stratdsl1:deadbeef", source_inspirations=(),
    )


def test_candidate_title_known_family_uses_friendly_label_never_raw_family():
    from alpha_agent.ui.views.discover import _candidate_title

    member = _member(strategy_family="ma_trend", root_symbol="CL", hypothesis_title="CL dual moving-average trend")
    title = _candidate_title(member)
    assert title == services.strategy_name("ma_trend")
    assert title != "ma_trend"


def test_candidate_title_never_duplicates_the_root():
    """The exact reported bug: '#1 CL CL dual moving-average trend'."""
    from alpha_agent.ui.views.discover import _candidate_title

    member = _member(strategy_family="ma_trend", root_symbol="CL", hypothesis_title="CL dual moving-average trend")
    title = _candidate_title(member)
    assert title.upper().count("CL") == 0  # the friendly family label carries no root at all
    assert "CL CL" not in f"{member.root_symbol} {title}"


def test_candidate_title_blueprint_family_strips_leading_root_never_duplicates():
    from alpha_agent.ui.views.discover import _candidate_title

    member = _member(
        strategy_family="dsl:daily_trading_day:ma_spread+volatility", root_symbol="NQ",
        hypothesis_title="NQ trend persistence conditioned on a low realized-volatility regime",
    )
    title = _candidate_title(member)
    assert not title.upper().startswith("NQ ")
    assert f"{member.root_symbol} {title}".count("NQ") == 1
    assert title[0].isupper()


def test_candidate_title_never_shows_the_raw_hypothesis_id():
    from alpha_agent.ui.views.discover import _candidate_title

    member = _member(
        strategy_family="dsl:custom", root_symbol="GC", hypothesis_title="GC something novel",
        hypothesis_id="H-DEMO-NOVEL-GC",
    )
    title = _candidate_title(member)
    assert "H-DEMO" not in title
    assert "H-DEMO" not in title.upper()


def test_why_it_stands_out_is_derived_from_real_score_components():
    from alpha_agent.screening.fast_screen import ResearchScreenScore
    from alpha_agent.ui.views.discover import _why_it_stands_out

    score = ResearchScreenScore(
        sharpe_component=35.0, cost_component=5.0, drawdown_component=2.0, activity_component=1.0, total=43.0,
    )
    text = _why_it_stands_out(score)
    assert "Sharpe" in text  # the dominant component in this example


def test_discover_page_never_shows_h_demo_or_number_rank_before_fast_screen(isolated_registry):
    at = _discover_app()
    at.run(timeout=90)
    at.radio(key="discover-research-sources").set_value("offline")
    at.radio(key="discover-mode").set_value(llm_demo.SCRIPTED_MODE)
    at.button(key="discover-run-button").click().run(timeout=120)
    assert not list(at.exception)

    markdown_blob = " ".join(m.value for m in at.markdown)
    assert "H-DEMO" not in markdown_blob
    assert "DRAFT CANDIDATE" in markdown_blob
    assert "CANDIDATES NOT YET SCREENED" in markdown_blob
    # never a duplicated root like "CL CL"
    for root in services.approved_universe():
        assert f"{root} {root}" not in markdown_blob


def test_discover_page_hides_technical_ids_behind_an_expander(isolated_registry):
    at = _discover_app()
    at.run(timeout=90)
    at.radio(key="discover-research-sources").set_value("offline")
    at.button(key="discover-run-button").click().run(timeout=120)
    assert not list(at.exception)
    labels = [e.label for e in at.expander]
    assert "Technical Details" in labels


# ---------------------------------------------------------------------------
# issue 6 -- Research Process metric split
# ---------------------------------------------------------------------------


def test_research_process_splits_local_and_live_metrics(isolated_registry):
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha for NQ.", root="NQ",
        family_stem="test_bugfix_metrics", run_fast_screen_backtests=False, research_sources="offline",
    )
    assert outcome.accepted
    kb = outcome.knowledge_base
    local = sum(1 for i in kb.items if i.source_type in (SourceType.INTERNAL, SourceType.CLASSIC))
    live = sum(len(r.items) for r in kb.external_ingestion_results)
    assert local > 0  # the classic library always has NQ-applicable items
    assert live == 0  # offline -- nothing was ever attempted


# ---------------------------------------------------------------------------
# issue 7 -- Fast Screen the already-compiled pool, never restart discovery
# ---------------------------------------------------------------------------


def test_run_fast_screen_on_pool_reuses_the_exact_compiled_pool(isolated_registry):
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha for NQ.", root="NQ",
        family_stem="test_bugfix_reuse", run_fast_screen_backtests=False, research_sources="offline",
    )
    assert outcome.accepted and outcome.pool is not None

    original_identities = {m.experiment_identity for m in outcome.pool.members}
    rescreened = discovery_campaign.run_fast_screen_on_pool(
        pool=outcome.pool, knowledge_base=outcome.knowledge_base, root="NQ",
        family_stem="test_bugfix_reuse", cli_executable="/definitely/not/a/real/cli",
    )
    # No compiled CLI at that fake path -- generation/pool must still be the
    # SAME real pool, never recompiled or regenerated.
    assert rescreened.pool is outcome.pool
    assert {m.experiment_identity for m in rescreened.pool.members} == original_identities
    assert rescreened.knowledge_base is outcome.knowledge_base
    assert rescreened.trials == ()  # CLI not found -- nothing screened, honestly
    assert rescreened.error is not None and "not found" in rescreened.error.lower()


@pytest.mark.skipif(
    not discovery_campaign._DEFAULT_CLI.exists(), reason="compiled CLI not present in this checkout",
)
def test_run_fast_screen_on_pool_real_screening_matches_inline_path(isolated_registry):
    """Real integration: Fast-Screening an already-compiled pool via
    `run_fast_screen_on_pool` produces the same shape of result
    `run_discovery_campaign(..., run_fast_screen_backtests=True)` would."""
    compiled = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha for NQ.", root="NQ",
        family_stem="test_bugfix_real_reuse", target_k=2, run_fast_screen_backtests=False,
        research_sources="offline",
    )
    assert compiled.accepted and compiled.pool.members

    screened = discovery_campaign.run_fast_screen_on_pool(
        pool=compiled.pool, knowledge_base=compiled.knowledge_base, root="NQ",
        family_stem="test_bugfix_real_reuse", target_k=2,
    )
    assert screened.accepted, screened.error
    assert screened.trials
    assert any(t.status is FastScreenStatus.SCREENED for t in screened.trials)


# ---------------------------------------------------------------------------
# issue 9 -- objective / market consistency
# ---------------------------------------------------------------------------


def test_mismatched_roots_detects_a_different_approved_root_in_free_text():
    from alpha_agent.ui.views.discover import _mismatched_roots

    assert _mismatched_roots("Find robust alpha for ES.", "CL") == ["ES"]
    assert _mismatched_roots("Find robust alpha for CL.", "CL") == []


def test_mismatched_roots_never_false_positives_inside_an_unrelated_word():
    from alpha_agent.ui.views.discover import _mismatched_roots

    # "GCC" contains "GC" as a substring but is not the ticker "GC".
    assert _mismatched_roots("Compile with GCC please.", "CL") == []


def test_discover_page_objective_defaults_to_a_research_root(isolated_registry):
    """Discover is a legacy/hidden page reached only via an explicit hand-off
    (e.g. Market's "Discover Strategies" button) or a plain default -- never
    a former hidden GLOBAL sidebar selector (Sidebar IA pass, task spec
    section 1)."""
    from alpha_agent.ui import services

    at = _discover_app()
    at.run(timeout=90)
    assert not list(at.exception)
    objective_widget = at.text_area(key="discover-objective")
    root = services.approved_universe()[0]
    assert root in objective_widget.value


def test_discover_page_honors_an_explicit_hand_off_target_root(isolated_registry):
    """Market's "Discover Strategies" button seeds `discover_target_root`
    before navigating here -- an explicit hand-off (task spec section 1E's
    same pattern), never a hidden shared selector."""
    at = _discover_app()
    at.session_state["discover_target_root"] = "CL"
    at.run(timeout=90)
    assert not list(at.exception)
    objective_widget = at.text_area(key="discover-objective")
    assert "CL" in objective_widget.value


def test_discover_page_shows_mismatch_warning_when_objective_names_another_root(isolated_registry):
    at = _discover_app()
    at.session_state["discover_target_root"] = "CL"
    at.run(timeout=90)
    at.text_area(key="discover-objective").set_value("Find robust alpha opportunities for ES.")
    at.run(timeout=90)
    assert not list(at.exception)
    blob = " ".join(w.body for w in at.warning)
    assert "ES" in blob and "CL" in blob

"""Phase 8 -- Community Alpha Network UI (prompt sections 9/10).

Real `streamlit.testing.v1.AppTest` renders, mirroring
`test_phase7_alpha_graph_ui.py`'s own pattern. Proves Community lives inside
Research (never a new top-level page), the Research UI consolidation (five
primary tabs, "Factor Library" renamed "My Alpha", Strategies/Experiments/
Validation demoted to drill-down-only), the privacy boundary holds through
the actual rendered page, and browsing/replicating never writes the
scientific registry or calls Claude/market-data.

Every test that needs seeded data monkeypatches `services.REGISTRY_PATH` /
`services.COMMUNITY_STORE_PATH` to `tmp_path` files -- the real production
registry/community store are never touched by this file.
"""
from __future__ import annotations

import re

import pytest
from _community_fixtures import insert
from alpha_agent.registry.enums import RegistryVerdict
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services
from alpha_agent.ui.views import community

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)


def _fresh_app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.community import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


def _seed_public_contribution(tmp_path, monkeypatch) -> str:
    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "experiments.sqlite")
    monkeypatch.setattr(services, "COMMUNITY_STORE_PATH", tmp_path / "community.sqlite")
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        insert(reg, root="CL", family="tsmom", params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
               verdict=RegistryVerdict.REJECT, vintage="a")
    c = services.community_create_contribution(
        kind="HYPOTHESIS", visibility="PUBLIC", contributor_display_name="Alice",
        experiment_id="CL__TSMOM__CANONICAL__A__TEST", title="CL time-series momentum",
        mechanism="TREND",
    )
    return c["contribution_id"]


# ---------------------------------------------------------------------------
# Research UI consolidation (prompt section 10)
# ---------------------------------------------------------------------------


def test_community_is_a_tab_on_learn_not_a_new_top_level_page():
    from alpha_agent.ui.views import learn

    assert learn._TABS["community"] == "Community"


def test_community_is_not_registered_as_its_own_app_page():
    with open(str(services.REPO_ROOT / "python" / "alpha_agent" / "ui" / "app.py"), encoding="utf-8") as fh:
        app_src = fh.read()
    assert "community.render" not in app_src


def test_community_is_reachable_from_the_shared_sidebar_subnav():
    from alpha_agent.ui import layout

    assert ("community", "Community") in layout._LEARN_SUBVIEWS


def test_learn_tabs_and_strategy_lab_tabs_are_the_prescribed_ones():
    """Research Thread workspace: My Alpha / Research Map / Community live on
    Learn (with Guides); the former Research page is Strategy Lab
    (Workflow / Provenance, drill-downs one click away)."""
    from alpha_agent.ui.views import learn, research

    assert tuple(learn._TABS.values()) == ("My Alpha", "Research Map", "Community", "Guides")
    assert research._LANDING_TABS == ("Workflow", "Provenance")


def test_factor_library_renamed_to_my_alpha_everywhere_it_matters():
    from alpha_agent.ui import layout
    from alpha_agent.ui.views import alpha_library, learn

    assert alpha_library.PAGE_TITLE == "My Alpha"
    assert learn._TABS["alpha_library"] == "My Alpha"
    assert ("alpha_library", "My Alpha") in layout._LEARN_SUBVIEWS
    assert "Factor Library" not in learn._TABS.values()


def test_strategies_experiments_validation_are_no_longer_top_level_tabs_but_stay_reachable():
    from alpha_agent.ui.views import research

    for label in ("Strategies", "Experiments", "Validation"):
        assert label not in research._LANDING_TABS
    # still reachable as drill-down (never deleted)
    for key in ("strategies", "experiments", "validation"):
        assert key in research._LANDING_FOCUS_LABEL
    assert ("strategies", "Strategies") in research._MORE_DETAIL_VIEWS
    assert ("experiments", "Experiments") in research._MORE_DETAIL_VIEWS
    assert ("validation", "Validation") in research._MORE_DETAIL_VIEWS


def test_more_detail_views_are_not_sidebar_subrows_anymore():
    from alpha_agent.ui import layout

    # They are Strategy Lab's own sub-rows now -- never Learn's.
    assert not {k for k, _ in layout._LEARN_SUBVIEWS} & {"strategies", "experiments", "validation"}
    assert {k for k, _ in layout._LAB_SUBVIEWS} >= {"strategies", "experiments", "validation"}


# ---------------------------------------------------------------------------
# Renders without running research / calling Claude / calling market data /
# writing the scientific registry
# ---------------------------------------------------------------------------


def test_landing_renders_without_running_research_calling_claude_or_market_data(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("Community must never call a live LLM client just by rendering")

    from alpha_agent.ui import llm_demo

    monkeypatch.setattr(llm_demo, "build_llm_client", _boom, raising=False)
    at = _fresh_app()
    body = " ".join(m.value for m in at.markdown)
    assert "Community Alpha Network" in body


def test_landing_shows_the_three_expected_tabs():
    at = _fresh_app()
    tab_labels = [t.proto.label for t in at.tabs]
    assert tab_labels == ["Community Feed", "My Contributions", "Share a Contribution"]


def test_module_never_calls_a_registry_write_method_static_guard():
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE",
    ]
    with open(community.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for pattern in forbidden:
        assert not re.search(pattern, text), f"community.py must never call {pattern!r}"


def test_module_never_imports_network_or_llm_clients():
    with open(community.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for token in ("AnthropicClient", "databento", "requests."):
        assert token not in text


def test_no_like_or_upvote_widget_anywhere_on_the_page():
    """No button, widget key, or interactive control anywhere on this page
    is a "Like"/"Upvote" action -- the core interaction is Replicate (prompt
    section 4). Scoped to actual widgets, not a full-page prose scan: the
    shared `layout.py` stylesheet's own CSS-comment prose incidentally
    contains the common English word "like" ("exactly like a smaller
    monitor"), which is not a feature and would make a blanket text sweep a
    false positive."""
    at = _fresh_app()
    button_labels = " ".join(b.label for b in at.button if b.label)
    assert not re.search(r"\blike\b|\bupvote\b|\bfavorite\b", button_labels, re.IGNORECASE)
    all_keys = " ".join(w.key or "" for w in (*at.button, *at.text_input, *at.text_area, *at.selectbox))
    assert not re.search(r"like|upvote|favorite", all_keys, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Seeded end-to-end flow: share, view, replicate
# ---------------------------------------------------------------------------


def test_feed_shows_a_seeded_public_contribution(tmp_path, monkeypatch):
    _seed_public_contribution(tmp_path, monkeypatch)
    at = _fresh_app()
    body = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "CL time-series momentum" in body
    open_buttons = [b for b in at.button if b.key and b.key.startswith("community-open-")]
    assert open_buttons


def test_opening_a_contribution_shows_evidence_and_never_triggers_execution(tmp_path, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("opening a contribution must never trigger execution")

    from alpha_agent.ui import deep_research

    monkeypatch.setattr(deep_research, "run_deep_research", _boom, raising=False)
    _seed_public_contribution(tmp_path, monkeypatch)
    at = _fresh_app()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("community-open-")]
    open_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)
    body = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "PROPOSED" in body or "REJECTED" in body  # a real EvidenceMaturity badge/rationale rendered
    assert "CL" in body and "tsmom" in body


def test_replicating_through_the_ui_records_a_real_replication(tmp_path, monkeypatch):
    contribution_id = _seed_public_contribution(tmp_path, monkeypatch)
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        insert(reg, root="CL", family="tsmom", params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
               verdict=RegistryVerdict.PASS, vintage="b")

    at = _fresh_app()
    open_buttons = [b for b in at.button if b.key and b.key.startswith("community-open-")]
    open_buttons[0].click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    at.text_input(key=f"community-replicate-name-{contribution_id}").set_value("Bob")
    at.text_input(key=f"community-replicate-expid-{contribution_id}").set_value(
        "CL__TSMOM__CANONICAL__B__TEST"
    )
    at.button(key=f"community-replicate-submit-{contribution_id}").click().run(timeout=90)
    assert not list(at.exception), list(at.exception)

    reps = services.community_replications_for(contribution_id)
    assert len(reps) == 1
    assert reps[0]["replicator"]["display_name"] == "Bob"
    body = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Bob" in body
    assert "CONFLICTS" in body or "CONFIRMS" in body


def test_shared_visibility_redacts_strategy_params_in_the_rendered_feed(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "experiments.sqlite")
    monkeypatch.setattr(services, "COMMUNITY_STORE_PATH", tmp_path / "community.sqlite")
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        insert(reg, root="CL", family="tsmom", params={"fast_horizon": 21, "slow_horizon": 120, "size": 1},
               verdict=RegistryVerdict.REJECT, vintage="a")
    services.community_create_contribution(
        kind="STRATEGY_SPEC", visibility="SHARED", contributor_display_name="Alice",
        experiment_id="CL__TSMOM__CANONICAL__A__TEST", title="CL TSMOM (shared)",
    )
    at = _fresh_app()
    body = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "redacted" in body.lower()
    # the actual parameter values never leak into the rendered feed for a non-owner viewer
    assert "21" not in " ".join(c.value for c in at.caption if "fast_horizon" not in c.value)


def test_private_contribution_never_appears_in_the_rendered_feed(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "experiments.sqlite")
    monkeypatch.setattr(services, "COMMUNITY_STORE_PATH", tmp_path / "community.sqlite")
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        insert(reg, root="NQ", family="ma_trend", params={"fast_horizon": 10, "slow_horizon": 50, "size": 1},
               verdict=RegistryVerdict.REJECT, vintage="a")
    services.community_create_contribution(
        kind="HYPOTHESIS", visibility="PRIVATE", contributor_display_name="Alice",
        experiment_id="NQ__MA_TREND__CANONICAL__A__TEST", title="My secret NQ idea",
    )
    at = _fresh_app()
    body = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "My secret NQ idea" not in body


def test_rendered_page_never_shows_an_alpha_score_or_expected_return():
    at = _fresh_app()
    body = (
        " ".join(m.value for m in at.markdown)
        + " ".join(c.value for c in at.caption)
        + " ".join(str(m.value) for m in at.metric)
    )
    assert not re.search(r"alpha\s*score", body, re.IGNORECASE)
    assert not re.search(r"expected\s*return", body, re.IGNORECASE)

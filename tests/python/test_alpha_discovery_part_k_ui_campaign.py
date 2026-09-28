"""Alpha Discovery campaign, Part K -- the Discover Strategies UI action
tests (task spec section 78's "Discover Strategies flow readable" +
sections 42/54's "never writes the registry before an explicit strict-
validation step")."""
from __future__ import annotations

from pathlib import Path

import pytest
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import discovery_campaign, llm_demo, services


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    path = tmp_path / "registry.sqlite"
    monkeypatch.setattr(services, "REGISTRY_PATH", path)
    reg = ExperimentRegistry(path)
    reg.close() if hasattr(reg, "close") else None
    return path


def test_default_scenario_mechanism_map_is_mechanism_diverse():
    mechanisms = [m for _, m in discovery_campaign.DEFAULT_SCENARIO_MECHANISM_MAP]
    assert len(mechanisms) == len(set(mechanisms))  # every mapped scenario a distinct mechanism


def test_discovery_campaign_scripted_mode_is_mechanism_diverse_and_writes_nothing(isolated_registry):
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha for NQ.", root="NQ",
        family_stem="test_discovery_ui", target_k=2, run_fast_screen_backtests=False,
    )
    assert outcome.accepted, outcome.error
    assert outcome.pool is not None
    assert len(outcome.pool.members) >= 3  # at least 3 of the 5 scenarios compile for NQ
    families = {m.strategy_family for m in outcome.pool.members}
    assert len(families) >= 2  # mechanism-diverse, not one family repeated
    assert outcome.frozen is None  # fast screen not requested -> nothing frozen
    assert outcome.knowledge_base is not None

    # nothing written to the registry -- plan_family never writes
    with services.open_registry() as reg:
        summary = reg.summary()
    assert summary.canonical == 0
    assert summary.authoritative_statistical_hypotheses == 0


def test_discovery_campaign_reports_knowledge_base_clusters(isolated_registry):
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha for NQ.", root="NQ",
        family_stem="test_discovery_ui_kb", run_fast_screen_backtests=False,
    )
    assert outcome.accepted
    clusters = outcome.knowledge_base.mechanism_clusters(market="NQ")
    assert clusters  # the classic library always has NQ-applicable items


@pytest.mark.skipif(
    not Path(discovery_campaign._DEFAULT_CLI).exists(), reason="compiled CLI not present in this checkout",
)
def test_discovery_campaign_real_fast_screen_and_freeze(isolated_registry):
    """Real integration: candidate generation -> real C++ Fast Screen (2018-
    2022 only) -> Freeze. Never touches strict validation or the registry."""
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha for NQ.", root="NQ",
        family_stem="test_discovery_ui_real", target_k=2, run_fast_screen_backtests=True,
    )
    assert outcome.accepted, outcome.error
    assert outcome.trials
    assert outcome.frozen is not None
    assert outcome.frozen.manifest.planning_complete
    assert outcome.frozen.frozen_count <= 2

    with services.open_registry() as reg:
        summary = reg.summary()
    assert summary.canonical == 0  # still nothing written -- freeze is not execution


def test_discover_page_renders_without_exceptions():
    """A pure render smoke test -- never clicks 'Discover Strategies', so no
    real campaign (LLM or C++) runs; proves the page loads cleanly and the
    empty (no-result-yet) state is legitimate, not an error."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.discover import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)
    assert "Discover Strategies" in " ".join(m.value for m in at.markdown)

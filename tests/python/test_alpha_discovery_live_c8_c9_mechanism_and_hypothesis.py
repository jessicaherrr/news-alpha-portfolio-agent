"""Alpha Discovery live-research campaign, Checkpoints 8-9 -- source
normalization / mechanism extraction / deduplication, and live-source-aware
hypothesis generation.

Checkpoint 8 is largely the EXISTING, unchanged Part C architecture
(`alpha_agent.knowledge.dedup.cluster_by_mechanism`, `StrategyKnowledgeItem`)
now fed by REAL connectors (Checkpoints 5-7) instead of `NotConnectedAdapter`
placeholders -- this module proves that combination end to end with mocked
HTTP (deterministic, no real network).

Checkpoint 9 proves the two concrete wiring fixes this session made:
1. `run_discovery_campaign(research_sources="connected")` actually threads
   `alpha_agent.knowledge.live_adapters()` into the `StrategyKnowledgeBase`
   the candidate pool reads from (previously ALWAYS `NotConnectedAdapter`,
   regardless of any connector's existence);
2. `HypothesisSpec.source_inspirations` -> `FamilyMember.source_inspirations`
   is a real, additive lineage path with NO effect on scientific identity
   (never part of `identity_tuple()` / `_manifest_snapshot()`).
"""
from __future__ import annotations

import json

import pytest
from alpha_agent.agents.orchestrator import FamilyMember
from alpha_agent.knowledge import (
    EconomicMechanism,
    StrategyKnowledgeBase,
    live_adapters,
)
from alpha_agent.registry import ExperimentRegistry
from alpha_agent.ui import discovery_campaign, llm_demo, services


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers: dict = {}
        self.content = b""

    def json(self):
        return self._json_body


class _FakeHttp:
    def __init__(self, routes: dict[str, _FakeResponse]):
        self.routes = routes

    def get(self, url, *, headers=None, params=None, timeout=None):
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                return resp
        # An un-scripted provider (e.g. the ones this test doesn't care
        # about) -- fail closed as ERROR, never silently fabricate data.
        return _FakeResponse(500, {})


def _live_shaped_routes() -> dict[str, _FakeResponse]:
    import base64

    readme = base64.b64encode(
        b"# Trend Bot\n\nA 20/50 moving average crossover trend-following strategy."
    ).decode("ascii")
    return {
        "https://api.github.com/rate_limit": _FakeResponse(200, {"resources": {"core": {"remaining": 5000, "limit": 5000}}}),
        "https://api.github.com/search/repositories": _FakeResponse(200, {"items": [{
            "full_name": "acme/trend-bot", "description": "MA crossover trend following", "html_url": "https://github.com/acme/trend-bot",
            "default_branch": "main", "license": {"spdx_id": "MIT"}, "owner": {"login": "acme"},
            "created_at": "2021-01-01T00:00:00Z", "stargazers_count": 10, "size": 400, "archived": False, "fork": False,
        }]}),
        "https://api.github.com/repos/acme/trend-bot/readme": _FakeResponse(200, {"content": readme, "encoding": "base64"}),
        "https://api.github.com/repos/acme/trend-bot/git/trees/main": _FakeResponse(200, {"tree": []}),
        "https://api.github.com/repos/acme/trend-bot/commits/main": _FakeResponse(200, {"sha": "abc123"}),
        "https://hn.algolia.com/api/v1/search": _FakeResponse(200, {"hits": [
            {"objectID": "1", "title": "Buy Low Sell High: A Trend Following Strategy", "url": "https://example.com/tf", "created_at": "2021-01-01T00:00:00.000Z"}
        ]}),
        "https://api.openalex.org/works": _FakeResponse(200, {"results": []}),
        "https://api.crossref.org/works": _FakeResponse(200, {"message": {"items": []}}),
    }


@pytest.fixture()
def fake_live_sources(monkeypatch):
    fake = _FakeHttp(_live_shaped_routes())
    monkeypatch.setattr("httpx.get", fake.get)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")
    return fake


# ---------------------------------------------------------------------------
# Checkpoint 8: source normalization + mechanism extraction + deduplication
# ---------------------------------------------------------------------------


def test_two_different_live_sources_dedupe_into_one_mechanism_cluster(fake_live_sources):
    kb = StrategyKnowledgeBase(adapters=live_adapters(), query="", markets=("NQ",))
    trend_items = kb.query(mechanism=EconomicMechanism.TREND)
    trend_source_types = {i.source_type.value for i in trend_items}
    assert {"GITHUB", "COMMUNITY"} <= trend_source_types  # both real sources agreed on ONE mechanism

    clusters = kb.mechanism_clusters(market="NQ")
    trend_cluster = next(c for c in clusters if c.economic_mechanism is EconomicMechanism.TREND)
    assert len(trend_cluster.items) >= 2  # one cluster, multiple source references -- not two ideas
    assert len(set(trend_cluster.knowledge_ids)) == len(trend_cluster.items)  # each reference preserved distinctly


def test_every_live_item_keeps_source_claims_separate_from_our_evidence(fake_live_sources):
    kb = StrategyKnowledgeBase(adapters=live_adapters(), query="", markets=("NQ",))
    live_items = [i for i in kb.items if i.source_type.value in ("GITHUB", "ACADEMIC", "COMMUNITY", "PRACTITIONER")]
    assert live_items  # the fixture actually produced live-shaped items
    for item in live_items:
        assert item.reported_results is None  # never our evidence, regardless of source


def test_github_connector_failure_does_not_prevent_other_sources_from_contributing(monkeypatch):
    """Task spec "NETWORK FAILURE HANDLING": one connector failing must not
    abort the whole campaign."""
    routes = _live_shaped_routes()
    routes["https://api.github.com/rate_limit"] = _FakeResponse(500, {})
    fake = _FakeHttp(routes)
    monkeypatch.setattr("httpx.get", fake.get)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")

    kb = StrategyKnowledgeBase(adapters=live_adapters(), query="", markets=("NQ",))
    results_by_source = {r.source_type.value: r.status.value for r in kb.external_ingestion_results}
    assert results_by_source["GITHUB"] == "ERROR"
    assert results_by_source["COMMUNITY"] == "CONNECTED"  # unaffected by GitHub's failure
    assert any(i.source_type.value == "COMMUNITY" for i in kb.items)


# ---------------------------------------------------------------------------
# Checkpoint 9: live-source-aware hypothesis generation
# ---------------------------------------------------------------------------


def test_offline_mode_still_produces_zero_external_items_unchanged(fake_live_sources, tmp_path, monkeypatch):
    """Regression guard: `research_sources="offline"` (the default) must
    behave EXACTLY as it always did -- no live call at all, even though a
    live connector now exists and the fixture would happily answer one.

    Release UX bugfix pass, issue 1: the STATUS reported for "no live call
    attempted" changed from NOT_CONNECTED to DISABLED (NOT_CONNECTED now
    means "attempted and unreachable", never "not requested") -- the
    invariant this test actually guards, zero live items and zero network
    calls, is unchanged."""
    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "reg.sqlite")
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha opportunities for NQ.", root="NQ",
        family_stem="c9_offline_test", scenario_mechanism_map=(("tsmom_nq", EconomicMechanism.MOMENTUM),),
        target_k=1, run_fast_screen_backtests=False, research_sources="offline",
    )
    assert outcome.accepted
    assert all(r.status.value == "DISABLED" for r in outcome.knowledge_base.external_ingestion_results)
    assert all(len(r.items) == 0 for r in outcome.knowledge_base.external_ingestion_results)


def test_connected_mode_wires_real_live_items_into_the_knowledge_base(fake_live_sources, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "reg.sqlite")
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="Find robust alpha opportunities for NQ.", root="NQ",
        family_stem="c9_connected_test", scenario_mechanism_map=(("tsmom_nq", EconomicMechanism.MOMENTUM),),
        target_k=1, run_fast_screen_backtests=False, research_sources="connected",
    )
    assert outcome.accepted, outcome.error
    statuses = {r.source_type.value: r.status.value for r in outcome.knowledge_base.external_ingestion_results}
    assert statuses["GITHUB"] == "CONNECTED"
    assert statuses["COMMUNITY"] == "CONNECTED"
    assert any(i.source_type.value == "GITHUB" for i in outcome.knowledge_base.items)


def test_invalid_research_sources_value_is_a_typed_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "reg.sqlite")
    outcome = discovery_campaign.run_discovery_campaign(
        mode=llm_demo.SCRIPTED_MODE, objective="x", root="NQ", family_stem="c9_bad",
        research_sources="not-a-real-mode",
    )
    assert not outcome.accepted
    assert "research_sources" in (outcome.error or "")


def test_hypothesis_source_inspirations_flows_through_to_family_member_unchanged_identity(tmp_path):
    """`source_inspirations` is pure lineage: it must reach `FamilyMember`
    for display, while `experiment_identity` stays keyed on exactly the same
    inputs as before this field existed (proven by comparing identity across
    two otherwise-identical hypotheses that differ ONLY in
    `source_inspirations`)."""
    from alpha_agent.agents import (
        FamilyPlanSpec,
        IdentityPlanes,
        OrchestratorBudget,
        OrchestratorConfig,
        ResearchAgent,
        ResearchOrchestrator,
        ScriptedExecutionValidationService,
        ScriptedLLMClient,
        StrategyCompilerAgent,
    )
    from alpha_agent.registry.models import MarketWindow
    from alpha_agent.validation.policy import ReliabilityPolicy

    def _hyp(with_inspiration: bool) -> str:
        base = {
            "hypothesis_id": "H-1", "title": "ES multi-week time-series momentum",
            "economic_mechanism": "Gradual diffusion of macro information.",
            "universe": ["ES"], "horizon": "20 trading days", "required_features": ["diff"],
            "signal_description": "Long when both a fast and slow price change are positive.",
            "expected_regime": "trending", "failure_regime": "choppy",
            "falsification_test": "No positive OOS net PnL after costs.",
        }
        if with_inspiration:
            base["source_inspirations"] = ["[GITHUB/TIER_B] acme/trend-bot | mechanism=TREND"]
        return json.dumps(base)

    plan = json.dumps({
        "expressible": True,
        "template": {"family_key": "tsmom", "root_symbol": "ES", "params": {"fast_horizon": 20, "slow_horizon": 120, "size": 1}},
    })
    policy = ReliabilityPolicy()
    planes = IdentityPlanes(
        dataset_fingerprint="valdataset2:x", split_identity="split1:x", validation_spec_fingerprint="validationprotocol1:x",
        reliability_policy_fingerprint=policy.identity(), execution_config_identity="execconfig1:x",
        cost_config_identity="costconfig1:x", risk_identity="riskconfig1:x",
    )
    mw = MarketWindow(label="V", start_date="2023-01-01", end_date="2024-12-31")

    def _plan_family(with_inspiration: bool, reg) -> FamilyMember:
        cfg = OrchestratorConfig(
            objective="test", market_universe=("ES",), market_window=mw, planes=planes,
            family_plan=FamilyPlanSpec(family_stem="c9_ident_test", target_family_size=1),
            budget=OrchestratorBudget(max_planning_attempts_per_family=1),
        )
        orch = ResearchOrchestrator(
            registry=reg, research_agent=ResearchAgent(ScriptedLLMClient([_hyp(with_inspiration)])),
            compiler_agent=StrategyCompilerAgent(ScriptedLLMClient([plan])),
            execution_service=ScriptedExecutionValidationService([]), reliability_policy=policy, config=cfg,
        )
        manifest = orch.plan_family()
        (member,) = manifest.members
        return member

    m1 = _plan_family(False, ExperimentRegistry(tmp_path / "r1.sqlite"))
    m2 = _plan_family(True, ExperimentRegistry(tmp_path / "r2.sqlite"))
    assert m1.source_inspirations == ()
    assert m2.source_inspirations == ("[GITHUB/TIER_B] acme/trend-bot | mechanism=TREND",)
    assert m1.experiment_identity == m2.experiment_identity  # lineage never enters scientific identity
    assert m1.strategy_fingerprint == m2.strategy_fingerprint

"""Alpha Discovery campaign, Part C -- Strategy Knowledge Base tests (task
spec section 71)."""
from __future__ import annotations

import inspect

import pytest
from alpha_agent.knowledge import (
    EconomicMechanism,
    IngestionStatus,
    SourceQualityTier,
    SourceType,
    StrategyKnowledgeBase,
    build_classic_library,
    cluster_by_mechanism,
    deduplicate_exact,
    external_sources,
    internal_items_from_registry_rows,
    registry_source,
    store,
)


def test_classic_library_loads_deterministically():
    a = build_classic_library()
    b = build_classic_library()
    assert a == b
    assert len(a) >= 15


def test_classic_library_covers_required_families():
    titles = {i.title for i in build_classic_library()}
    required_substrings = [
        "Dual Moving Average", "Moving-Average Crossover", "Time-Series Momentum",
        "Donchian", "ATR", "Opening Range", "Bollinger", "RSI", "Prior-Day",
        "Overnight Gap", "Four-Price", "Volatility Contraction",
    ]
    for needle in required_substrings:
        assert any(needle in t for t in titles), f"missing classic item covering {needle!r}"


def test_classic_items_never_claim_a_reported_result():
    for item in build_classic_library():
        assert item.reported_results is None


def test_classic_items_carry_provenance_hash():
    for item in build_classic_library():
        assert item.provenance_hash.startswith("classic1:")


def test_four_price_source_variants_are_preserved_separately_not_merged():
    items = {i.knowledge_id: i for i in build_classic_library()}
    a = items["classic-four-price-source-a"]
    b = items["classic-four-price-source-b"]
    assert a.economic_mechanism != b.economic_mechanism  # continuation vs. fade reading
    assert a.entry_logic_summary != b.entry_logic_summary


def test_source_quality_tiers_never_referenced_by_validation_policy():
    """Task spec section 15: source quality influences priority, never
    thresholds. Static proof: the validation package never imports the
    knowledge package."""
    import alpha_agent.validation.policy as policy_mod

    src = inspect.getsource(policy_mod)
    assert "knowledge" not in src


# ---------------------------------------------------------------------------
# internal source
# ---------------------------------------------------------------------------


def _row(**overrides) -> dict:
    base = {
        "strategy_family": "tsmom", "root_symbol": "NQ", "verdict": "REJECT",
    }
    base.update(overrides)
    return base


def test_internal_source_groups_by_family_and_root():
    rows = [_row(experiment_id="A"), _row(experiment_id="B"), _row(root_symbol="ES")]
    items = internal_items_from_registry_rows(rows)
    assert len(items) == 2  # (tsmom, NQ) and (tsmom, ES)
    ids = {i.knowledge_id for i in items}
    assert ids == {"internal-tsmom-NQ", "internal-tsmom-ES"}


def test_internal_source_skips_unmapped_family():
    rows = [_row(strategy_family="silver_bullet_variant_x")]
    assert internal_items_from_registry_rows(rows) == ()


def test_internal_source_never_fabricates_new_evidence():
    rows = [_row(verdict="REJECT")]
    item = internal_items_from_registry_rows(rows)[0]
    assert "not new evidence" in item.reported_results


# ---------------------------------------------------------------------------
# external sources: safe boundary
# ---------------------------------------------------------------------------


def test_external_adapters_default_to_not_connected():
    adapters = external_sources.default_adapters()
    assert set(adapters) == {
        SourceType.GITHUB, SourceType.ACADEMIC, SourceType.COMMUNITY, SourceType.PRACTITIONER,
    }
    for adapter in adapters.values():
        result = adapter.ingest(query="moving average crossover")
        assert result.status is IngestionStatus.NOT_CONNECTED
        assert result.items == ()


def test_external_ingestion_never_raises_even_on_a_broken_adapter():
    class _Broken:
        source_type = SourceType.GITHUB

        def ingest(self, *, query, markets=()):
            raise RuntimeError("network is definitely not available")

    results = external_sources.ingest_all({SourceType.GITHUB: _Broken()}, query="x")
    assert len(results) == 1
    assert results[0].status is IngestionStatus.NOT_CONNECTED


@pytest.mark.parametrize("module", [external_sources, registry_source, store])
def test_no_forbidden_execution_primitive_anywhere_in_the_knowledge_package(module):
    """Task spec sections 18/19/25/71: GitHub-sourced code is never executed.
    An AST check for an actual CALL to eval/exec/os.system/subprocess.*/
    importlib.import_module -- not a naive text-grep, which would also trip on
    this very module's own docstrings and its `FORBIDDEN_OPERATIONS` data
    literal (both legitimately NAME these operations in prose)."""
    import ast

    tree = ast.parse(inspect.getsource(module))
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("eval", "exec"):
            forbidden_calls.append(func.id)
        elif isinstance(func, ast.Attribute):
            owner = func.value.id if isinstance(func.value, ast.Name) else ""
            if (owner, func.attr) in {
                ("os", "system"), ("subprocess", "Popen"), ("subprocess", "call"),
                ("subprocess", "run"), ("importlib", "import_module"),
            }:
                forbidden_calls.append(f"{owner}.{func.attr}")
    assert forbidden_calls == [], f"forbidden call(s) {forbidden_calls} found in {module.__name__}"


def test_unknown_license_defaults_to_unknown_quality_never_upgraded():
    from alpha_agent.knowledge.models import IngestionStatus as _S
    from alpha_agent.knowledge.models import StrategyKnowledgeItem

    item = StrategyKnowledgeItem(
        knowledge_id="external-x", source_type=SourceType.GITHUB,
        ingestion_status=_S.CONNECTED, title="unlicensed repo idea",
        economic_mechanism=EconomicMechanism.TREND, license=None,
    )
    assert item.source_quality == SourceQualityTier.UNKNOWN


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------


def test_deduplicate_exact_drops_repeated_knowledge_id():
    items = build_classic_library()
    doubled = items + (items[0],)
    assert deduplicate_exact(doubled) == items


def test_cluster_by_mechanism_collapses_many_sources_into_one_cluster():
    from alpha_agent.knowledge.models import StrategyKnowledgeItem

    refs = [
        StrategyKnowledgeItem(
            knowledge_id=f"github-ma-cross-{i}", source_type=SourceType.GITHUB,
            title=f"repo {i} MA crossover", economic_mechanism=EconomicMechanism.TREND,
        )
        for i in range(5)
    ]
    clusters = cluster_by_mechanism(refs)
    assert len(clusters) == 1
    assert len(clusters[0].items) == 5
    assert clusters[0].knowledge_ids == tuple(r.knowledge_id for r in refs)


# ---------------------------------------------------------------------------
# StrategyKnowledgeBase aggregate
# ---------------------------------------------------------------------------


def test_knowledge_base_aggregates_classic_and_internal_and_external():
    kb = StrategyKnowledgeBase(registry_rows=[_row()], query="test", markets=("NQ",))
    sources = {i.source_type for i in kb.items}
    assert SourceType.CLASSIC in sources
    assert SourceType.INTERNAL in sources
    # external adapters ran (NOT_CONNECTED -> zero items, but the ingestion
    # attempt itself is real and reported)
    assert {r.source_type for r in kb.external_ingestion_results} == {
        SourceType.GITHUB, SourceType.ACADEMIC, SourceType.COMMUNITY, SourceType.PRACTITIONER,
    }
    assert all(r.status is IngestionStatus.NOT_CONNECTED for r in kb.external_ingestion_results)


def test_knowledge_base_query_by_mechanism():
    kb = StrategyKnowledgeBase()
    trend_items = kb.query(mechanism=EconomicMechanism.TREND)
    assert trend_items
    assert all(i.economic_mechanism == EconomicMechanism.TREND for i in trend_items)


def test_knowledge_base_query_by_market_never_excludes_a_market_agnostic_item():
    kb = StrategyKnowledgeBase()
    # classic items list all 5 roots -- querying a market must still surface them
    es_items = kb.query(market="ES")
    assert any(i.source_type == SourceType.CLASSIC for i in es_items)


def test_render_for_research_context_is_deterministic_and_bounded():
    kb = StrategyKnowledgeBase()
    items = kb.query(mechanism=EconomicMechanism.TREND)
    snippets_a = kb.render_for_research_context(items, limit=3)
    snippets_b = kb.render_for_research_context(items, limit=3)
    assert snippets_a == snippets_b
    assert len(snippets_a) <= 3
    assert all(isinstance(s, str) for s in snippets_a)


def test_render_for_research_context_survives_holdout_guard():
    """Every snippet must be safe to pass straight into
    `ResearchContext(knowledge_base=...)` -- proves no accidental 2025-shaped
    string sneaks through (classic/internal content has no dates at all)."""
    from alpha_agent.agents.context import build_research_context

    kb = StrategyKnowledgeBase(registry_rows=[_row()])
    snippets = kb.render_for_research_context(kb.items)
    ctx = build_research_context(
        objective="find alpha", market_universe=["NQ"], knowledge_base=snippets,
    )
    assert set(snippets) == set(ctx.knowledge_base)

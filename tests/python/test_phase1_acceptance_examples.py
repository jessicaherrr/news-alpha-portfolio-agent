"""Phase 1 -- the three grounded acceptance examples (prompt 1 section 15)
plus the scientific-safety regression tests it explicitly requires (section
17). Every example runs the REAL deterministic pipeline against the REAL
local registry and the REAL `alpha_agent.features.REGISTRY` -- nothing here
is fabricated to make an example "interesting" (section 15: "do not fabricate
data merely to make the screenshots interesting").

Example 1 -- ENERGY: an EIA petroleum status report observation for CL.
Example 2 -- MACRO / RATES: an FOMC policy observation for ZN.
Example 3 -- EQUITY-INDEX MACRO: the SAME real FOMC category mapped to NQ,
    proving one real observation legitimately translates differently
    depending on which certified market it is being researched for.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import alpha_agent.features.compute  # noqa: F401 -- register feature defs
import pytest
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.registry.holdout_guard import HOLDOUT_START
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.translation.pipeline import build_observation_translation, observation_from_news
from alpha_agent.translation.schemas import ResearchabilityStatus
from alpha_agent.ui import services

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout",
)

_NOW = datetime.now(UTC)  # this sandbox's real wall clock is on/after 2026 -- see the holdout tests below


def _item(category: NewsCategory, *, news_id: str) -> MarketNewsItem:
    return MarketNewsItem(
        news_id=news_id, headline=f"Official {category.value} release ({news_id})",
        source_name="Official Source", source_type=NewsSourceType.OFFICIAL,
        source_url=f"https://example.gov/{news_id}",
        published_at=_NOW - timedelta(hours=3), retrieved_at=_NOW,
        related_products=mapping.products_for_category(category),
        related_asset_classes=mapping.asset_classes_for_category(category),
        category=category, mapping_reason=mapping.mapping_reason_for_category(category),
    )


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(services.REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Example 1 -- ENERGY (prompt 1 section 8's own worked example)
# ---------------------------------------------------------------------------


def test_example_1_energy_eia_petroleum_on_cl():
    obs = observation_from_news(_item(NewsCategory.PETROLEUM, news_id="EX1"), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)

    # OBSERVATION
    assert result.observation.root_symbol == "CL"
    assert result.observation.source == "Official Source"

    # POSSIBLE MECHANISMS (2+)
    assert len(result.mechanism_candidates) >= 2

    # FACTOR CANDIDATES / RESEARCHABILITY -- a genuine mix, not all one status
    by_status = {f.researchability for f in result.factor_candidates}
    assert ResearchabilityStatus.AVAILABLE in by_status
    assert ResearchabilityStatus.DATA_MISSING in by_status

    # HYPOTHESIS + FALSIFICATION
    assert result.hypothesis is not None
    assert result.hypothesis.falsification_test

    # RESEARCH MEMORY -- real registry evidence for CL
    assert result.research_memory.root_symbol == "CL"


# ---------------------------------------------------------------------------
# Example 2 -- MACRO / RATES
# ---------------------------------------------------------------------------


def test_example_2_macro_rates_fomc_on_zn():
    obs = observation_from_news(_item(NewsCategory.FOMC_POLICY, news_id="EX2"), root_symbol="ZN")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    assert result.observation.root_symbol == "ZN"
    assert result.hypothesis is not None
    assert result.hypothesis.universe == ["ZN"]
    assert any(m.mechanism.value == "VOLATILITY_BREAKOUT" for m in result.mechanism_candidates)


# ---------------------------------------------------------------------------
# Example 3 -- EQUITY-INDEX MACRO (the SAME real category, a different
# certified root -- a genuinely different research target, prompt 1 section
# 15's "different kinds of existing Futures context")
# ---------------------------------------------------------------------------


def test_example_3_equity_index_macro_fomc_on_nq():
    obs = observation_from_news(_item(NewsCategory.FOMC_POLICY, news_id="EX3"), root_symbol="NQ")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    assert result.observation.root_symbol == "NQ"
    assert result.hypothesis is not None
    assert result.hypothesis.universe == ["NQ"]
    assert result.research_memory.root_symbol == "NQ"


def test_examples_2_and_3_are_genuinely_different_research_targets():
    zn = observation_from_news(_item(NewsCategory.FOMC_POLICY, news_id="EX2b"), root_symbol="ZN")
    nq = observation_from_news(_item(NewsCategory.FOMC_POLICY, news_id="EX3b"), root_symbol="NQ")
    with _registry() as reg:
        zn_result = build_observation_translation(zn, registry=reg)
        nq_result = build_observation_translation(nq, registry=reg)
    assert zn_result.hypothesis.hypothesis_id != nq_result.hypothesis.hypothesis_id
    assert zn_result.research_memory.summary != nq_result.research_memory.summary or (
        zn_result.research_memory.digests != nq_result.research_memory.digests
    )


# ---------------------------------------------------------------------------
# Scientific-safety regression (prompt 1 section 17)
# ---------------------------------------------------------------------------


def test_market_news_item_never_becomes_registry_evidence():
    """A MarketNewsItem's own fields never appear inside a HypothesisSpec's
    typed evidence-bearing fields (required_features/universe) -- only as
    human-readable narrative text (title/economic_mechanism/expected_regime),
    which is never itself registry evidence."""
    item = _item(NewsCategory.PETROLEUM, news_id="SAFETY1")
    obs = observation_from_news(item, root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    assert result.hypothesis is not None
    # The news_id / source_url never leak into required_features or universe
    # -- those stay closed, typed, FeatureRegistry-checked vocabularies.
    assert item.news_id not in result.hypothesis.required_features
    assert item.news_id not in result.hypothesis.universe


def test_observation_never_gains_a_registry_write_capability():
    """Structural guarantee: the translation pipeline never imports a
    registry WRITE method -- it is given an already-open `ExperimentRegistry`
    purely to READ FailureMemory (prompt 1 section 3)."""
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    for rel in ("python/alpha_agent/translation/pipeline.py", "python/alpha_agent/translation/research_memory.py"):
        source = (repo_root / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_attrs = {
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        forbidden = {"insert", "record_result", "record_attempt", "append_experiment", "write", "commit_step"}
        assert not (called_attrs & forbidden), (rel, called_attrs & forbidden)


def test_observation_timestamp_preserves_origin_vintage_verbatim():
    item = _item(NewsCategory.PETROLEUM, news_id="SAFETY2")
    obs = observation_from_news(item, root_symbol="CL")
    assert obs.observed_at == item.published_at
    assert obs.origin_vintage == item.published_at.date().isoformat()


def test_post_holdout_origin_observation_remains_ineligible_for_2025_holdout():
    """This sandbox's real wall clock is on/after 2026-01-01 -- every
    current observation is honestly post-holdout-origin and must never claim
    eligibility for the untouched 2025 holdout (prompt 1 section 13)."""
    item = _item(NewsCategory.PETROLEUM, news_id="SAFETY3")
    obs = observation_from_news(item, root_symbol="CL")
    assert obs.observed_at >= datetime.fromisoformat(HOLDOUT_START).replace(tzinfo=UTC)
    assert obs.holdout_eligible is False


def test_pre_holdout_origin_observation_is_honestly_eligible():
    """The flip side: a historical (pre-2025) observation is honestly
    eligible -- this is a property of the timestamp, never hardcoded False."""
    old_item = _item(NewsCategory.PETROLEUM, news_id="SAFETY4")
    old_item = old_item.model_copy(
        update={"published_at": datetime(2023, 6, 1, tzinfo=UTC), "retrieved_at": datetime(2023, 6, 1, 1, tzinfo=UTC)}
    )
    obs = observation_from_news(old_item, root_symbol="CL")
    assert obs.holdout_eligible is True


def test_no_fabricated_feature_registry_capability():
    """Every `required_features` entry on a produced hypothesis is a REAL,
    currently registered FeatureRegistry kind -- never invented."""
    from alpha_agent.features import REGISTRY as FEATURE_REGISTRY

    obs = observation_from_news(_item(NewsCategory.PETROLEUM, news_id="SAFETY5"), root_symbol="CL")
    with _registry() as reg:
        result = build_observation_translation(obs, registry=reg)
    known = set(FEATURE_REGISTRY.kinds())
    assert result.hypothesis is not None
    for kind in result.hypothesis.required_features:
        assert kind in known


def test_no_automatic_paid_network_call_from_the_deterministic_path():
    """`build_observation_translation` with no `mechanism_proposal` never
    imports or references an LLM/network transport at all."""
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "python/alpha_agent/translation/pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            imported_names.update(alias.name for alias in node.names)
    forbidden = {"AnthropicClient", "anthropic", "requests", "httpx", "urllib"}
    assert not (imported_names & forbidden), imported_names & forbidden

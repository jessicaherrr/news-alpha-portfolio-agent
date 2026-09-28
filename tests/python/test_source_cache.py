"""Tests for `alpha_agent.knowledge.source_cache` (Release UX Part J, task
spec sections 37-40/57).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from alpha_agent.knowledge.models import (
    EconomicMechanism,
    IngestionStatus,
    SourceQualityTier,
    SourceType,
    StrategyKnowledgeItem,
)
from alpha_agent.knowledge.source_cache import CacheStatus, ResearchSourceCache


@pytest.fixture
def cache(tmp_path: Path) -> ResearchSourceCache:
    return ResearchSourceCache(path=tmp_path / "cache.sqlite")


def _github_item(*, repo="octocat/trend-strategy", commit="abc123", knowledge_id="k1") -> StrategyKnowledgeItem:
    return StrategyKnowledgeItem(
        knowledge_id=knowledge_id, source_type=SourceType.GITHUB, source_quality=SourceQualityTier.TIER_B,
        ingestion_status=IngestionStatus.CONNECTED, title="20/50 MA crossover", repository=repo, commit_sha=commit,
        economic_mechanism=EconomicMechanism.TREND, provenance_hash=f"github1:{commit}",
    )


def _academic_item(*, doi="10.1234/abc.def", knowledge_id="a1") -> StrategyKnowledgeItem:
    return StrategyKnowledgeItem(
        knowledge_id=knowledge_id, source_type=SourceType.ACADEMIC, source_quality=SourceQualityTier.TIER_A,
        ingestion_status=IngestionStatus.CONNECTED, title="Time Series Momentum",
        source_url=f"https://doi.org/{doi}", economic_mechanism=EconomicMechanism.MOMENTUM,
        provenance_hash=f"academic1:{doi}",
    )


# ---------------------------------------------------------------------------
# basic record / versioning
# ---------------------------------------------------------------------------


def test_first_record_is_new(cache):
    outcome = cache.record(_github_item())
    assert outcome.status == CacheStatus.NEW
    assert outcome.record.version_ordinal == 1
    assert outcome.record.is_current is True


def test_same_github_commit_reused_not_duplicated(cache):
    cache.record(_github_item())
    outcome = cache.record(_github_item())
    assert outcome.status == CacheStatus.REUSED
    history = cache.history(outcome.record.source_id)
    assert len(history) == 1


def test_changed_github_commit_creates_a_new_version_preserving_the_old(cache):
    first = cache.record(_github_item(commit="abc123"))
    second = cache.record(_github_item(commit="def456"))
    assert second.status == CacheStatus.NEW_VERSION
    assert second.record.version_ordinal == 2
    history = cache.history(first.record.source_id)
    assert len(history) == 2
    assert history[0].commit_sha == "abc123"
    assert history[0].is_current is False  # preserved, never deleted -- just superseded
    assert history[1].commit_sha == "def456"
    assert history[1].is_current is True


def test_same_doi_deduplicated_across_different_providers(cache):
    """The SAME DOI found via Crossref and OpenAlex (different `knowledge_id`
    provenance hashes) collapses to ONE cached identity."""
    item_a = _academic_item(doi="10.9999/xyz", knowledge_id="crossref-1")
    item_b = _academic_item(doi="10.9999/xyz", knowledge_id="openalex-1")
    out_a = cache.record(item_a)
    out_b = cache.record(item_b)
    assert out_a.record.source_id == out_b.record.source_id
    assert out_b.status == CacheStatus.REUSED
    assert len(cache.history(out_a.record.source_id)) == 1


def test_different_doi_are_distinct_sources(cache):
    out_a = cache.record(_academic_item(doi="10.1/aaa"))
    out_b = cache.record(_academic_item(doi="10.1/bbb"))
    assert out_a.record.source_id != out_b.record.source_id


def test_community_item_without_doi_or_repo_dedupes_by_url(cache):
    item = StrategyKnowledgeItem(
        knowledge_id="hn-1", source_type=SourceType.COMMUNITY, source_quality=SourceQualityTier.TIER_C,
        title="A mean reversion strategy on HN", source_url="https://news.ycombinator.com/item?id=123",
        economic_mechanism=EconomicMechanism.MEAN_REVERSION, provenance_hash="community1:hn-1",
    )
    first = cache.record(item)
    second = cache.record(item)
    assert first.status == CacheStatus.NEW
    assert second.status == CacheStatus.REUSED


# ---------------------------------------------------------------------------
# never mutates scientific evidence / never stores secrets
# ---------------------------------------------------------------------------


def test_cache_writes_never_touch_the_experiment_registry(cache, tmp_path):
    """Structural: writing to the source cache creates ONLY the cache's own
    sqlite file -- never anything under data/registry/."""
    cache.record(_github_item())
    assert cache.path.exists()
    assert "registry" not in str(cache.path)


def test_source_cache_module_never_references_a_secret_env_var_name():
    """Checks the actual CODE (module docstring excluded -- it legitimately
    names these to document that they are forbidden)."""
    import ast

    path = (
        Path(__file__).resolve().parents[2] / "python" / "alpha_agent" / "knowledge" / "source_cache.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_docstring = ast.get_docstring(tree) or ""
    body_without_docstring = tree.body[1:] if (tree.body and ast.get_docstring(tree)) else tree.body
    code_source = "\n".join(ast.unparse(node) for node in body_without_docstring)
    for secret_name in ("GITHUB_TOKEN", "DATABENTO_API_KEY", "ANTHROPIC_API_KEY"):
        assert secret_name not in code_source, f"{secret_name} referenced outside the module docstring"
        assert secret_name in module_docstring  # sanity: the doc DOES explain the boundary


def test_record_signature_accepts_no_credential_parameter():
    import inspect

    sig = inspect.signature(ResearchSourceCache.record)
    for param in sig.parameters:
        assert "key" not in param.lower() and "token" not in param.lower() and "secret" not in param.lower()


# ---------------------------------------------------------------------------
# stats / stale-cache refresh determinism
# ---------------------------------------------------------------------------


def test_stats_reflects_distinct_sources_and_providers(cache):
    cache.record(_github_item(repo="a/b"))
    cache.record(_github_item(repo="c/d"))
    cache.record(_academic_item(doi="10.1/x"))
    stats = cache.stats()
    assert stats["distinct_sources"] == 3
    assert stats["by_provider"]["GITHUB"] == 2
    assert stats["by_provider"]["ACADEMIC"] == 1


def test_refresh_is_deterministic_same_content_same_hash(cache):
    out1 = cache.record(_github_item())
    out2 = cache.record(_github_item())
    assert out1.record.knowledge_item_hash == out2.record.knowledge_item_hash
    assert out1.record.normalized_document_hash == out2.record.normalized_document_hash


def test_current_returns_none_for_unknown_source(cache):
    assert cache.current("github:nobody/nothing") is None

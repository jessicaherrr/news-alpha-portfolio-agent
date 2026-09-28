"""Market Intelligence Data Completion Pass, Section 40 -- static + semantic
proof that `alpha_agent.market_intel` (MarketNewsItem / ScheduledMarketEvent)
and the Checkpoint C/D marketdata additions (FuturesContract /
RelativeMarketSnapshot) can never reach the SCIENTIFIC plane, and that the
scientific plane can never reach INTO `alpha_agent.market_intel` either.
Mirrors `test_release_market_plane_isolation.py`'s own pattern for the
Databento observation provider.
"""
from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SCIENTIFIC_PACKAGE_PREFIXES = (
    "alpha_agent.data",
    "alpha_agent.discovery",
    "alpha_agent.screening",
    "alpha_agent.validation",
    "alpha_agent.registry",
    "alpha_agent.strategy",
    "alpha_agent.holdout",
)

#: Section 15/31: MarketNewsItem and ResearchSource are different concepts,
#: never merged -- `alpha_agent.market_intel` must never import the
#: research-knowledge-base plane either.
FORBIDDEN_IMPORT_PREFIXES = (*SCIENTIFIC_PACKAGE_PREFIXES, "alpha_agent.knowledge")


def _module_path(module_name: str) -> Path:
    return REPO_ROOT / "python" / Path(*module_name.split(".")).with_suffix(".py")


def _module_imports(module_name: str) -> set[str]:
    path = _module_path(module_name)
    source = path.read_text(encoding="utf-8") if path.exists() else ""
    if not source.strip():
        return set()
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _all_python_modules_under(package_prefix: str) -> list[str]:
    pkg_path = REPO_ROOT / "python" / package_prefix.replace(".", "/")
    if not pkg_path.is_dir():
        return []
    out = []
    for path in sorted(pkg_path.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT / "python").with_suffix("")
        out.append(".".join(rel.parts))
    return out


_MARKET_INTEL_MODULES = _all_python_modules_under("alpha_agent/market_intel") + [
    "alpha_agent.ui.market_intel_context",
    "alpha_agent.ui.market_news",
    "alpha_agent.ui.market_events",
]


@pytest.mark.parametrize("module_name", _MARKET_INTEL_MODULES)
def test_market_intel_module_never_imports_the_scientific_or_knowledge_plane(module_name):
    imports = _module_imports(module_name)
    for imported in imports:
        assert not any(imported == p or imported.startswith(p + ".") for p in FORBIDDEN_IMPORT_PREFIXES), (
            f"{module_name} imports {imported!r} -- market_intel (news/events) must never depend on the "
            "scientific plane or the separate ResearchSource knowledge-base plane"
        )


@pytest.mark.parametrize("package_prefix", SCIENTIFIC_PACKAGE_PREFIXES)
def test_scientific_plane_never_imports_market_intel(package_prefix):
    for module_name in _all_python_modules_under(package_prefix):
        imports = _module_imports(module_name)
        for imported in imports:
            assert not imported.startswith(("alpha_agent.market_intel", "alpha_agent.ui.market_intel_context")), (
                f"{module_name} (scientific plane) imports {imported!r} -- MarketNewsItem/ScheduledMarketEvent "
                "must never feed the research pipeline"
            )


def test_market_intel_never_imports_the_registry_module():
    for module_name in _MARKET_INTEL_MODULES:
        imports = _module_imports(module_name)
        assert not any("registry" in i for i in imports), f"{module_name} must never import a registry module"


# ---------------------------------------------------------------------------
# semantic isolation -- a real MarketNewsItem/ScheduledMarketEvent cannot
# alter the registry, verdict, Research Promise, or validation thresholds
# ---------------------------------------------------------------------------


def test_market_news_item_has_no_registry_or_verdict_shaped_field():
    from alpha_agent.market_intel.news_schemas import MarketNewsItem

    forbidden = {"experiment_id", "scientific_verdict", "research_promise", "p_value", "sharpe", "dsr"}
    assert not (set(MarketNewsItem.model_fields) & forbidden)


def test_scheduled_market_event_has_no_registry_or_verdict_shaped_field():
    from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent

    forbidden = {"experiment_id", "scientific_verdict", "research_promise", "p_value", "sharpe", "dsr"}
    assert not (set(ScheduledMarketEvent.model_fields) & forbidden)


def test_futures_contract_has_no_registry_or_verdict_shaped_field():
    from alpha_agent.marketdata.databento_schemas import FuturesContract

    forbidden = {"experiment_id", "scientific_verdict", "research_promise", "p_value", "sharpe", "dsr"}
    assert not (set(FuturesContract.model_fields) & forbidden)


def test_relative_market_snapshot_has_no_registry_or_verdict_shaped_field():
    from alpha_agent.marketdata.relative_markets import RelativeMarketSnapshot

    forbidden = {"experiment_id", "scientific_verdict", "research_promise", "p_value", "sharpe", "dsr"}
    assert not (set(RelativeMarketSnapshot.model_fields) & forbidden)


def test_news_store_and_event_store_write_paths_are_disjoint_from_the_registry_db(tmp_path):
    """Structural: writing real items to `NewsStore`/`EventStore` touches
    ONLY their own sqlite file paths -- never anything under
    `data/registry/`."""
    from alpha_agent.market_intel.event_schemas import EventImportance, ScheduledMarketEvent
    from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
    from alpha_agent.market_intel.store import EventStore, NewsStore

    now = datetime.now(UTC)
    news_store = NewsStore(tmp_path / "news.sqlite")
    news_store.add_item(
        MarketNewsItem(
            news_id="x", headline="h", source_name="s", source_type=NewsSourceType.OFFICIAL,
            source_url="https://x.gov", published_at=now - timedelta(hours=1), retrieved_at=now,
            category=NewsCategory.OTHER, mapping_reason="r",
        )
    )
    event_store = EventStore(tmp_path / "events.sqlite")
    event_store.upsert_event(
        ScheduledMarketEvent(
            event_id="y", name="n", source_name="s", source_url="https://x.gov",
            scheduled_at=now + timedelta(days=1), timezone="America/New_York", category="OTHER",
            importance=EventImportance.LOW, importance_rule="r", mapping_reason="m", retrieved_at=now,
        )
    )
    assert "registry" not in str(news_store._db_path)
    assert "registry" not in str(event_store._db_path)


def test_market_intel_defaults_live_outside_data_registry():
    from alpha_agent.market_intel.store import DEFAULT_DB_PATH, DEFAULT_EVENTS_DB_PATH

    assert "market_intelligence" in str(DEFAULT_DB_PATH)
    assert "registry" not in str(DEFAULT_DB_PATH)
    assert "market_intelligence" in str(DEFAULT_EVENTS_DB_PATH)
    assert "registry" not in str(DEFAULT_EVENTS_DB_PATH)

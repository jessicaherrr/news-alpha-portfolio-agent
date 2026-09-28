"""Market Intelligence Data Completion Pass, Checkpoint E -- MarketNewsItem,
NewsStore, mapping, dedup, and every connector's PARSING logic. Every
connector test injects a fake HTTP layer (monkeypatching
`alpha_agent.market_intel.http_support.http_get_text`/`http_head`/
`http_get_json`) -- NO real network call, ever, mirroring this codebase's
existing "no billable call in unit tests" discipline. Live verification
(real network) is a separate, explicit step (see the session's final
report), never part of this file.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alpha_agent.market_intel import mapping
from alpha_agent.market_intel.connectors import bls as bls_mod
from alpha_agent.market_intel.connectors import eia as eia_mod
from alpha_agent.market_intel.connectors import fed as fed_mod
from alpha_agent.market_intel.connectors import usda as usda_mod
from alpha_agent.market_intel.connectors.commercial import CommercialMarketNewsConnector
from alpha_agent.market_intel.dedup import is_probable_duplicate, normalize_headline, storage_key
from alpha_agent.market_intel.news_schemas import MarketNewsItem, NewsCategory, NewsSourceType
from alpha_agent.market_intel.store import NewsStore
from alpha_agent.marketdata.capability import CapabilityState

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def _item(
    *, news_id="fed:1", headline="Test headline", source_name="Federal Reserve", url="https://example.gov/1",
    published_at=NOW - timedelta(hours=2), category=NewsCategory.FOMC_POLICY,
) -> MarketNewsItem:
    return MarketNewsItem(
        news_id=news_id, headline=headline, source_name=source_name, source_type=NewsSourceType.OFFICIAL,
        source_url=url, published_at=published_at, retrieved_at=NOW, category=category,
        related_products=mapping.products_for_category(category), mapping_reason="test",
    )


# ---------------------------------------------------------------------------
# MarketNewsItem -- Section 15
# ---------------------------------------------------------------------------


def test_published_at_must_never_equal_retrieved_at():
    with pytest.raises(ValueError, match="published_at"):
        MarketNewsItem(
            news_id="x", headline="h", source_name="s", source_type=NewsSourceType.OFFICIAL,
            source_url="https://x.gov", published_at=NOW, retrieved_at=NOW, category=NewsCategory.OTHER,
            mapping_reason="r",
        )


def test_published_at_cannot_be_after_retrieved_at():
    with pytest.raises(ValueError):
        MarketNewsItem(
            news_id="x", headline="h", source_name="s", source_type=NewsSourceType.OFFICIAL,
            source_url="https://x.gov", published_at=NOW + timedelta(hours=1), retrieved_at=NOW,
            category=NewsCategory.OTHER, mapping_reason="r",
        )


def test_retrieved_at_and_published_at_are_distinct_and_preserved():
    item = _item()
    assert item.published_at != item.retrieved_at
    assert item.published_at < item.retrieved_at


# ---------------------------------------------------------------------------
# mapping -- Section 21
# ---------------------------------------------------------------------------


def test_mapping_rule_ids_are_never_blank():
    for category in NewsCategory:
        assert mapping.mapping_rule_id_for_category(category)


def test_fomc_maps_to_equity_rates_fx_gold():
    products = mapping.products_for_category(NewsCategory.FOMC_POLICY)
    assert {"ES", "NQ", "ZN", "GC"}.issubset(set(products))


def test_petroleum_maps_only_to_petroleum_roots():
    products = mapping.products_for_category(NewsCategory.PETROLEUM)
    assert set(products) == {"CL", "MCL", "RB", "HO"}
    assert "NG" not in products  # NG is a distinct category, not petroleum


def test_other_category_maps_to_nothing_never_a_guess():
    assert mapping.products_for_category(NewsCategory.OTHER) == ()


# ---------------------------------------------------------------------------
# dedup -- Section 20
# ---------------------------------------------------------------------------


def test_storage_key_uses_canonical_url():
    a = _item(url="https://example.gov/story-1")
    b = _item(url="https://example.gov/story-1")
    assert storage_key(a) == storage_key(b)


def test_normalize_headline_ignores_case_and_punctuation():
    assert normalize_headline("CPI Rises 0.4%!") == normalize_headline("cpi rises 0 4")


def test_probable_duplicate_requires_same_source_and_close_time():
    a = _item(headline="CPI rises 0.4%", source_name="BLS", url="https://x/1", published_at=NOW - timedelta(hours=1))
    b = _item(headline="CPI Rises 0.4%!", source_name="BLS", url="https://x/2", published_at=NOW - timedelta(hours=1, minutes=10))
    assert is_probable_duplicate(a, b) is True


def test_shared_keyword_alone_is_never_a_duplicate():
    a = _item(headline="CPI rises 0.4% in August", source_name="BLS", url="https://x/1")
    c = _item(headline="CPI expectations survey released", source_name="BLS", url="https://x/3")
    assert is_probable_duplicate(a, c) is False


def test_different_source_is_never_a_duplicate_even_with_identical_headline():
    a = _item(headline="Same headline", source_name="Federal Reserve", url="https://a/1")
    b = _item(headline="Same headline", source_name="EIA", url="https://b/1")
    assert is_probable_duplicate(a, b) is False


# ---------------------------------------------------------------------------
# NewsStore -- Section 16
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> NewsStore:
    return NewsStore(tmp_path / "test_news.sqlite")


def test_store_add_and_list_recent(store: NewsStore):
    assert store.add_item(_item()) is True
    recent = store.list_recent(since=NOW - timedelta(days=1))
    assert len(recent) == 1
    assert recent[0].headline == "Test headline"


def test_store_dedups_by_canonical_url(store: NewsStore):
    a = _item(url="https://example.gov/story")
    b = _item(url="https://example.gov/story", headline="Slightly different headline")
    assert store.add_item(a) is True
    assert store.add_item(b) is False
    assert len(store.list_recent(since=NOW - timedelta(days=1))) == 1


def test_store_dedups_probable_near_duplicates(store: NewsStore):
    a = _item(headline="CPI rises 0.4%", source_name="BLS", url="https://bls.gov/1")
    b = _item(headline="CPI Rises 0.4%!", source_name="BLS", url="https://bls.gov/1?utm=amp")
    assert store.add_item(a) is True
    assert store.add_item(b) is False


def test_store_preserves_materially_different_stories(store: NewsStore):
    a = _item(headline="CPI rises 0.4%", url="https://x/1")
    b = _item(headline="Fed announces new stress test rules", url="https://x/2")
    assert store.add_item(a) is True
    assert store.add_item(b) is True
    assert len(store.list_recent(since=NOW - timedelta(days=1))) == 2


def test_store_count_recent_filters_by_related_product(store: NewsStore):
    petroleum = _item(url="https://x/oil", headline="Crude oil output rises", category=NewsCategory.PETROLEUM)
    macro = _item(
        url="https://x/macro", headline="FOMC holds rates steady", source_name="Federal Reserve",
        category=NewsCategory.FOMC_POLICY,
    )
    store.add_item(petroleum)
    store.add_item(macro)
    assert store.count_recent(since=NOW - timedelta(days=1), related_product="CL") == 1
    assert store.count_recent(since=NOW - timedelta(days=1), related_product="NQ") == 1
    assert store.count_recent(since=NOW - timedelta(days=1), related_product="ZW") == 0


def test_store_never_returns_items_before_since(store: NewsStore):
    old = _item(url="https://x/old", published_at=NOW - timedelta(days=10))
    store.add_item(old)
    assert store.list_recent(since=NOW - timedelta(days=1)) == []


def test_store_is_safe_to_delete_and_rebuild(tmp_path: Path):
    path = tmp_path / "rebuild.sqlite"
    store1 = NewsStore(path)
    store1.add_item(_item())
    path.unlink()
    store2 = NewsStore(path)  # rebuilds schema from scratch, no crash
    assert store2.list_recent(since=NOW - timedelta(days=1)) == []


# ---------------------------------------------------------------------------
# connectors -- fake HTTP only, no network (Section 17-19)
# ---------------------------------------------------------------------------

_FED_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>FOMC statement released</title>
<link><![CDATA[https://www.federalreserve.gov/x.htm]]></link>
<pubDate><![CDATA[Mon, 08 Sep 2026 18:00:00 GMT]]></pubDate></item>
<item><title>Speech on the economy</title>
<link><![CDATA[https://www.federalreserve.gov/y.htm]]></link>
<pubDate><![CDATA[Mon, 01 Sep 2026 18:00:00 GMT]]></pubDate></item>
</channel></rss>"""


def test_fed_news_connector_classifies_fomc_items(monkeypatch):
    monkeypatch.setattr(fed_mod, "http_get_text", lambda url, **kw: _FED_RSS if "monetary" in url else "<rss><channel></channel></rss>")
    connector = fed_mod.FedNewsConnector()
    items = connector.fetch_recent(since=NOW - timedelta(days=30), limit=10)
    assert len(items) == 2  # both real items in the fake monetary feed
    by_url = {i.source_url: i for i in items}
    assert by_url["https://www.federalreserve.gov/x.htm"].category is NewsCategory.FOMC_POLICY
    assert by_url["https://www.federalreserve.gov/y.htm"].category is NewsCategory.OTHER


def test_fed_news_connector_health_reflects_real_connectivity(monkeypatch):
    def _raise(*a, **kw):
        from alpha_agent.market_intel.http_support import MarketIntelHttpError

        raise MarketIntelHttpError("down")

    monkeypatch.setattr(fed_mod, "http_get_text", _raise)
    assert fed_mod.FedNewsConnector().health() is CapabilityState.NOT_CONNECTED


_EIA_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>Crude oil production hits a record</title>
<link>https://www.eia.gov/a</link><pubDate>Mon, 08 Sep 2026 09:00:00 EST</pubDate>
<description>Crude oil output rises.</description></item>
<item><title>Natural gas prices trend near Henry Hub discount</title>
<link>https://www.eia.gov/b</link><pubDate>Tue, 09 Sep 2026 09:00:00 EST</pubDate>
<description>Henry Hub natural gas.</description></item>
<item><title>Electricity demand climbs in ERCOT</title>
<link>https://www.eia.gov/c</link><pubDate>Wed, 10 Sep 2026 09:00:00 EST</pubDate>
<description>ERCOT electricity demand.</description></item>
</channel></rss>"""


def test_eia_news_connector_classifies_petroleum_vs_natgas_vs_other(monkeypatch):
    monkeypatch.setattr(eia_mod, "http_get_text", lambda url, **kw: _EIA_RSS)
    items = eia_mod.EiaNewsConnector().fetch_recent(since=NOW - timedelta(days=30), limit=10)
    by_category = {i.category for i in items}
    assert NewsCategory.PETROLEUM in by_category
    assert NewsCategory.NATURAL_GAS in by_category
    assert NewsCategory.OTHER in by_category  # ERCOT electricity item -- never forced into energy


_BLS_ATOM = """<?xml version='1.0'?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>CPI rises 0.4%</title><link href="https://www.bls.gov/news.release/archives/cpi.htm"/>
<published>2026-09-11T07:50:40-04:00</published><content>CPI content</content></entry>
</feed>"""


def test_bls_news_connector_parses_atom_feed(monkeypatch):
    monkeypatch.setattr(bls_mod, "http_get_text", lambda url, **kw: _BLS_ATOM)
    items = bls_mod.BlsNewsConnector().fetch_recent(since=NOW - timedelta(days=30), limit=10)
    assert len(items) == len(bls_mod._FEEDS)  # same feed content served for every one of the 5 real slugs in this fake
    assert all(i.category is NewsCategory.CPI_PPI_EMPLOYMENT for i in items)
    assert all(i.source_name == "U.S. Bureau of Labor Statistics" for i in items)


def test_usda_publication_connector_uses_real_last_modified_header(monkeypatch):
    monkeypatch.setattr(usda_mod, "http_head", lambda url, **kw: {"last-modified": "Fri, 11 Sep 2026 15:46:22 GMT"})
    items = usda_mod.UsdaPublicationConnector().fetch_recent(since=NOW - timedelta(days=30), limit=10)
    assert len(items) == 1
    assert items[0].published_at == datetime(2026, 9, 11, 15, 46, 22, tzinfo=UTC)
    assert items[0].category is NewsCategory.USDA_GRAIN_OILSEED


def test_usda_mymarketnews_is_not_connected_without_api_key(monkeypatch):
    monkeypatch.delenv("USDA_MMN_API_KEY", raising=False)
    assert usda_mod.UsdaMyMarketNewsConnector().health() is CapabilityState.NOT_CONNECTED
    assert usda_mod.UsdaMyMarketNewsConnector().fetch_recent(since=NOW - timedelta(days=1)) == ()


def test_usda_mymarketnews_key_is_read_from_the_environment_never_hardcoded():
    """The key is always read live via `os.environ.get(...)` -- never a
    hardcoded value literal anywhere in the module's actual code."""
    source = Path(usda_mod.__file__).read_text(encoding="utf-8")
    assert "os.environ.get(_MMN_API_KEY_ENV)" in source


def test_usda_module_never_hardcodes_a_key_value_outside_its_own_docstrings():
    """Every PROSE mention of the env var name is backtick-quoted (never a
    real Python string literal); the single actual QUOTED occurrence is the
    `_MMN_API_KEY_ENV` constant assignment itself."""
    source = Path(usda_mod.__file__).read_text(encoding="utf-8")
    assert source.count('"USDA_MMN_API_KEY"') == 1
    assert "_MMN_API_KEY_ENV = \"USDA_MMN_API_KEY\"" in source


def test_commercial_connector_is_always_disabled():
    connector = CommercialMarketNewsConnector()
    assert connector.health() is CapabilityState.DISABLED
    assert connector.fetch_recent(since=NOW - timedelta(days=1)) == ()


# ---------------------------------------------------------------------------
# secrets never serialized -- Section 18/38
# ---------------------------------------------------------------------------


def test_store_module_never_references_a_secret_env_var_in_code():
    """Mirrors `test_source_cache.py::test_source_cache_module_never_references_a_secret_env_var_name`
    -- the module docstring may (and does) name these to document the
    boundary; the actual CODE never does."""
    import ast

    path = Path(__import__("alpha_agent.market_intel.store", fromlist=["x"]).__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_docstring = ast.get_docstring(tree) or ""
    body_without_docstring = tree.body[1:] if (tree.body and ast.get_docstring(tree)) else tree.body
    code_source = "\n".join(ast.unparse(node) for node in body_without_docstring)
    for secret_name in ("USDA_MMN_API_KEY", "DATABENTO_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN"):
        assert secret_name not in code_source, f"{secret_name} referenced outside the module docstring"
    assert "USDA_MMN_API_KEY" in module_docstring  # sanity: the doc DOES explain the boundary


def test_news_item_schema_has_no_secret_field():
    assert "api_key" not in MarketNewsItem.model_fields
    assert "token" not in MarketNewsItem.model_fields

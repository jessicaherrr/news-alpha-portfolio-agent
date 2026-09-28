"""Alpha Discovery live-research campaign, Checkpoint 7 -- the real community
(Hacker News) and practitioner (institution-allowlisted Crossref/OpenAlex)
connectors.

Mocked-HTTP tests only, except the LIVE smoke tests at the bottom (never
require a credential -- both providers are public and keyless -- and never
fail the suite on a transient provider issue, only on a genuine defect).
"""
from __future__ import annotations

import ast
import inspect

from alpha_agent.knowledge.community_connector import (
    CommunityResearchConnector,
    community_health,
    generate_community_queries,
)
from alpha_agent.knowledge.connector_support import ConnectorBudget
from alpha_agent.knowledge.models import IngestionStatus, SourceType
from alpha_agent.knowledge.practitioner_connector import (
    INSTITUTION_ALLOWLIST,
    PractitionerResearchConnector,
    generate_practitioner_queries,
)


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
        self.calls: list[str] = []

    def get(self, url, *, headers=None, params=None, timeout=None):
        self.calls.append(url)
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                return resp
        raise AssertionError(f"unscripted URL requested: {url}")


# ---------------------------------------------------------------------------
# community (Hacker News)
# ---------------------------------------------------------------------------


_HN_HIT = {
    "objectID": "123", "title": "Simple Mean Reversion Trading Strategy in Python",
    "url": "https://example.com/meanrev", "author": "someone", "created_at": "2020-01-01T00:00:00.000Z",
}


def test_community_query_generation_favors_mechanism_terms():
    queries = generate_community_queries(market="NQ", budget=ConnectorBudget(max_queries=6))
    assert "mean reversion trading strategy" in queries
    assert queries[0] == "mean reversion trading strategy"  # mechanism terms come first


def test_community_ingest_normalizes_a_real_shaped_hit(monkeypatch):
    fake = _FakeHttp({"https://hn.algolia.com/api/v1/search": _FakeResponse(200, {"hits": [_HN_HIT]})})
    monkeypatch.setattr("httpx.get", fake.get)
    conn = CommunityResearchConnector(budget=ConnectorBudget(max_queries=1))
    result = conn.ingest(query="mean reversion trading strategy", markets=("NQ",))
    assert result.status is IngestionStatus.CONNECTED
    (item,) = result.items
    assert item.source_type is SourceType.COMMUNITY
    assert item.economic_mechanism.value == "MEAN_REVERSION"
    assert item.source_quality.value == "TIER_C"  # community gets the lower default weight
    assert item.reported_results is None


def test_community_unclassifiable_hit_is_skipped(monkeypatch):
    fake = _FakeHttp({"https://hn.algolia.com/api/v1/search": _FakeResponse(200, {"hits": [{"objectID": "1", "title": "Show HN: my new budgeting app"}]})})
    monkeypatch.setattr("httpx.get", fake.get)
    conn = CommunityResearchConnector(budget=ConnectorBudget(max_queries=1))
    result = conn.ingest(query="anything", markets=("NQ",))
    assert result.items == ()


def test_community_provider_failure_is_error_not_a_crash(monkeypatch):
    fake = _FakeHttp({"https://hn.algolia.com/api/v1/search": _FakeResponse(500, {})})
    monkeypatch.setattr("httpx.get", fake.get)
    conn = CommunityResearchConnector(budget=ConnectorBudget(max_queries=1))
    result = conn.ingest(query="anything", markets=("NQ",))
    assert result.status is IngestionStatus.ERROR
    assert result.items == ()


def test_community_ingest_url_reads_page_text_only(monkeypatch):
    html = "<html><head><title>My Trend Following Strategy</title></head><body><p>A moving average crossover trend following approach.</p></body></html>"

    class _R:
        status_code = 200
        content = html.encode("utf-8")

    monkeypatch.setattr("httpx.get", lambda *a, **kw: _R())
    conn = CommunityResearchConnector()
    item = conn.ingest_url("https://example.com/my-strategy")
    assert item is not None
    assert item.economic_mechanism.value == "TREND"
    assert item.source_url == "https://example.com/my-strategy"


def test_community_ingest_url_never_treats_html_as_code():
    import alpha_agent.knowledge.community_connector as mod

    src = inspect.getsource(mod)
    assert "eval(" not in src and "exec(" not in src
    assert "BeautifulSoup" not in src  # no heavyweight HTML-parsing dependency -- a plain text strip only


# ---------------------------------------------------------------------------
# practitioner (institution-allowlisted Crossref/OpenAlex)
# ---------------------------------------------------------------------------


_FED_WORK = {
    "title": ["Momentum and Term Structure in Commodity Futures"],
    "publisher": "Federal Reserve Bank of San Francisco",
    "container-title": ["Federal Reserve Bank of San Francisco, Working Paper Series"],
    "DOI": "10.1000/fed-momentum", "author": [{"given": "A.", "family": "Economist"}],
    "issued": {"date-parts": [[2018]]}, "URL": "https://doi.org/10.1000/fed-momentum",
}

_NON_INSTITUTION_WORK = {
    "title": ["Momentum Trading in Retail Brokerage Accounts"],
    "publisher": "Wiley",
    "container-title": ["Journal of Finance"],
    "DOI": "10.1000/wiley-momentum", "author": [], "issued": {"date-parts": [[2018]]}, "URL": "x",
}


def test_institution_allowlist_is_broad_not_a_single_name():
    assert len(INSTITUTION_ALLOWLIST) >= 10
    assert "federal reserve" in INSTITUTION_ALLOWLIST


def test_practitioner_query_generation_pairs_institution_and_mechanism_terms():
    queries = generate_practitioner_queries(market="NQ", budget=ConnectorBudget(max_queries=5))
    assert any("federal reserve" in q.lower() for q in queries)


def test_practitioner_includes_allowlisted_publisher_only(monkeypatch):
    fake = _FakeHttp({
        "https://api.crossref.org/works": _FakeResponse(200, {"message": {"items": [_FED_WORK, _NON_INSTITUTION_WORK]}}),
        "https://api.openalex.org/works": _FakeResponse(200, {"results": []}),
    })
    monkeypatch.setattr("httpx.get", fake.get)
    conn = PractitionerResearchConnector(budget=ConnectorBudget(max_queries=1))
    result = conn.ingest(query="momentum futures", markets=("NQ",))
    assert result.status is IngestionStatus.CONNECTED
    titles = [i.title for i in result.items]
    assert "Momentum and Term Structure in Commodity Futures" in titles
    assert "Momentum Trading in Retail Brokerage Accounts" not in titles  # non-institutional -- excluded


def test_practitioner_both_providers_down_is_error(monkeypatch):
    fake = _FakeHttp({
        "https://api.crossref.org/works": _FakeResponse(500, {}),
        "https://api.openalex.org/works": _FakeResponse(500, {}),
    })
    monkeypatch.setattr("httpx.get", fake.get)
    conn = PractitionerResearchConnector(budget=ConnectorBudget(max_queries=1))
    result = conn.ingest(query="momentum futures", markets=("NQ",))
    assert result.status is IngestionStatus.ERROR


def test_practitioner_never_treats_a_reported_result_as_evidence():
    import alpha_agent.knowledge.practitioner_connector as mod

    src = inspect.getsource(mod)
    assert "reported_results=None" in src


def test_practitioner_module_never_calls_eval_exec():
    import alpha_agent.knowledge.practitioner_connector as mod

    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec")


# ---------------------------------------------------------------------------
# live smoke tests -- no credential required; tolerant of transient
# provider issues.
# ---------------------------------------------------------------------------


def test_live_community_health_check():
    assert community_health() in (IngestionStatus.CONNECTED, IngestionStatus.ERROR)


def test_live_community_ingest_smoke():
    conn = CommunityResearchConnector(budget=ConnectorBudget(max_queries=2, max_documents_total=5))
    result = conn.ingest(query="", markets=("NQ",))
    assert result.status is not IngestionStatus.NOT_CONNECTED
    for item in result.items:
        assert item.source_type is SourceType.COMMUNITY


def test_live_practitioner_ingest_smoke():
    conn = PractitionerResearchConnector(budget=ConnectorBudget(max_queries=2, max_documents_total=5))
    result = conn.ingest(query="", markets=("NQ",))
    for item in result.items:
        assert item.source_type is SourceType.PRACTITIONER
        assert item.provenance_hash.startswith("practitioner1:")

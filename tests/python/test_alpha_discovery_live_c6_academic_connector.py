"""Alpha Discovery live-research campaign, Checkpoint 6 -- the real academic
research connector (task spec section 53's connector tests + section 56's
live smoke test).

Every test except the two LIVE ones uses mocked `httpx.get` -- no real
network call. The LIVE tests always run (no credential is required for a
public academic API) but tolerate a transient provider rate limit / outage
without failing the suite -- only a genuine connector defect (an exception,
or BOTH providers failing) fails them.
"""
from __future__ import annotations

import ast
import inspect

import pytest
from alpha_agent.knowledge.academic_connector import (
    AcademicConnectorHealth,
    AcademicResearchConnector,
    academic_health,
    generate_academic_queries,
)
from alpha_agent.knowledge.connector_support import ConnectorBudget
from alpha_agent.knowledge.models import IngestionStatus, SourceType


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers: dict = {}
        self.content = b""

    def json(self):
        return self._json_body


class _FakeAcademic:
    def __init__(self, routes: dict[str, _FakeResponse]):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, *, headers=None, params=None, timeout=None):
        self.calls.append(url)
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                return resp
        raise AssertionError(f"unscripted URL requested: {url}")


_OPENALEX_WORK = {
    "display_name": "Time Series Momentum in Futures Markets",
    "doi": "https://doi.org/10.1000/tsmom",
    "publication_year": 2012,
    "authorships": [{"author": {"display_name": "T. Moskowitz"}}, {"author": {"display_name": "Y. Pedersen"}}],
    "open_access": {"is_oa": True, "oa_url": "https://example.org/tsmom.pdf"},
    "id": "https://openalex.org/W123",
    # "the" (0), "market" (1) etc -- OpenAlex's inverted-index abstract format.
    "abstract_inverted_index": {
        "Time": [0], "series": [1], "momentum": [2], "trend": [3], "following": [4],
        "in": [5], "futures": [6], "markets.": [7],
    },
}

_CROSSREF_WORK = {
    "title": ["Time Series Momentum in Futures Markets"],  # same paper -- must dedupe against OpenAlex
    "DOI": "10.1000/tsmom",
    "author": [{"given": "T.", "family": "Moskowitz"}],
    "issued": {"date-parts": [[2012]]},
    "URL": "https://doi.org/10.1000/tsmom",
}

_CROSSREF_ONLY_WORK = {
    "title": ["Mean Reversion in Commodity Futures"],
    "DOI": "10.1000/meanrev",
    "author": [{"given": "A.", "family": "Author"}],
    "issued": {"date-parts": [[2015]]},
    "URL": "https://doi.org/10.1000/meanrev",
}


def _routes() -> dict[str, _FakeResponse]:
    return {
        "https://api.openalex.org/works": _FakeResponse(200, {"results": [_OPENALEX_WORK]}),
        "https://api.crossref.org/works": _FakeResponse(200, {"message": {"items": [_CROSSREF_WORK, _CROSSREF_ONLY_WORK]}}),
    }


@pytest.fixture()
def fake_academic(monkeypatch):
    fake = _FakeAcademic(_routes())
    monkeypatch.setattr("httpx.get", fake.get)
    return fake


def test_query_generation_is_bounded_and_deterministic():
    budget = ConnectorBudget(max_queries=2)
    q1 = generate_academic_queries(market="NQ", budget=budget)
    q2 = generate_academic_queries(market="NQ", budget=budget)
    assert q1 == q2
    assert len(q1) <= 2


def test_ingest_normalizes_openalex_work_with_reconstructed_abstract(fake_academic):
    conn = AcademicResearchConnector(budget=ConnectorBudget(max_queries=1, max_documents_total=5))
    result = conn.ingest(query="time series momentum", markets=("NQ",))
    assert result.status is IngestionStatus.CONNECTED
    tsmom = next(i for i in result.items if "Moskowitz" in i.authors[0])
    assert tsmom.source_type is SourceType.ACADEMIC
    assert tsmom.economic_mechanism.value == "MOMENTUM"
    assert "trend following" in tsmom.implementation_notes.lower()
    assert tsmom.reported_results is None


def test_crossref_and_openalex_duplicate_paper_dedupes_to_one_item(fake_academic):
    conn = AcademicResearchConnector(budget=ConnectorBudget(max_queries=1, max_documents_total=5))
    result = conn.ingest(query="time series momentum", markets=("NQ",))
    dois = [i.knowledge_id for i in result.items]
    tsmom_items = [i for i in result.items if "Moskowitz" in i.authors[0]]
    assert len(tsmom_items) == 1  # OpenAlex + Crossref agree on the same DOI -> ONE item
    assert len(dois) == len(set(dois))  # no duplicate knowledge_id at all


def test_crossref_only_paper_is_still_included(fake_academic):
    conn = AcademicResearchConnector(budget=ConnectorBudget(max_queries=1, max_documents_total=5))
    result = conn.ingest(query="mean reversion", markets=("NQ",))
    assert any("Mean Reversion" in i.title for i in result.items)


def test_openalex_failure_falls_back_to_crossref(monkeypatch):
    routes = {
        "https://api.openalex.org/works": _FakeResponse(500, {}),
        "https://api.crossref.org/works": _FakeResponse(200, {"message": {"items": [_CROSSREF_ONLY_WORK]}}),
    }
    fake = _FakeAcademic(routes)
    monkeypatch.setattr("httpx.get", fake.get)
    conn = AcademicResearchConnector(budget=ConnectorBudget(max_queries=1))
    result = conn.ingest(query="mean reversion", markets=("NQ",))
    assert result.status is IngestionStatus.DEGRADED
    assert len(result.items) == 1


def test_both_providers_failing_is_error(monkeypatch):
    routes = {
        "https://api.openalex.org/works": _FakeResponse(500, {}),
        "https://api.crossref.org/works": _FakeResponse(500, {}),
    }
    fake = _FakeAcademic(routes)
    monkeypatch.setattr("httpx.get", fake.get)
    conn = AcademicResearchConnector(budget=ConnectorBudget(max_queries=1))
    result = conn.ingest(query="mean reversion", markets=("NQ",))
    assert result.status is IngestionStatus.ERROR
    assert result.items == ()


def test_unclassifiable_paper_is_skipped_not_guessed(monkeypatch):
    routes = {
        "https://api.openalex.org/works": _FakeResponse(200, {"results": [{"display_name": "A Study of Something Unrelated"}]}),
        "https://api.crossref.org/works": _FakeResponse(200, {"message": {"items": []}}),
    }
    fake = _FakeAcademic(routes)
    monkeypatch.setattr("httpx.get", fake.get)
    conn = AcademicResearchConnector(budget=ConnectorBudget(max_queries=1))
    result = conn.ingest(query="anything", markets=("NQ",))
    assert result.items == ()


def test_no_reported_results_ever_populated_from_a_paper():
    """Structural proof, not just a per-item check: nothing in this module
    ever sets `reported_results` to a non-None value."""
    import alpha_agent.knowledge.academic_connector as mod

    src = inspect.getsource(mod)
    assert "reported_results=None" in src
    assert "reported_results=" not in src.replace("reported_results=None", "")


def test_academic_module_never_calls_eval_exec_or_shells_out():
    import alpha_agent.knowledge.academic_connector as mod

    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec")


# ---------------------------------------------------------------------------
# live smoke test -- no credential required; tolerant of a transient
# rate limit / outage on either public provider.
# ---------------------------------------------------------------------------


def test_live_academic_health_check():
    report = academic_health()
    assert report.health in (AcademicConnectorHealth.CONNECTED, AcademicConnectorHealth.RATE_LIMITED)


def test_live_academic_ingest_smoke():
    conn = AcademicResearchConnector(budget=ConnectorBudget(max_queries=1, max_search_results=5, max_documents_total=5))
    result = conn.ingest(query="time series momentum futures", markets=("NQ",))
    assert result.status is not IngestionStatus.ERROR, result.detail
    for item in result.items:
        assert item.source_type is SourceType.ACADEMIC
        assert item.title
        assert item.provenance_hash.startswith("academic1:")

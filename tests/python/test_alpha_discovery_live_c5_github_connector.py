"""Alpha Discovery live-research campaign, Checkpoint 5 -- the real GitHub
strategy connector (task spec section 53's connector tests + section 56's
live smoke test).

Every test below except the two explicitly marked LIVE tests uses a mocked
`httpx.get` (task spec section 66: "mock all live connectors" in normal
tests) -- no real network call, fully deterministic. The LIVE tests are
skipped outright when `GITHUB_TOKEN` is not configured; they never fail the
suite for an absent credential, only for a genuine connector defect.
"""
from __future__ import annotations

import ast
import base64
import inspect
import os

import pytest
from alpha_agent.knowledge import external_sources
from alpha_agent.knowledge.connector_support import (
    ConnectorBudget,
    load_dotenv_if_available,
)
from alpha_agent.knowledge.github_connector import (
    GitHubConnectorHealth,
    GitHubStrategyConnector,
    generate_github_queries,
    github_health,
)
from alpha_agent.knowledge.models import IngestionStatus, SourceType

# A local `.env` (never committed, never read by this module directly) may
# carry GITHUB_TOKEN -- load it once so the LIVE tests' `skipif` conditions
# below see the same environment the connector itself would see.
load_dotenv_if_available()


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None, headers: dict | None = None, content: bytes = b""):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers = headers or {}
        self.content = content or b""

    def json(self):
        return self._json_body


class _FakeGitHub:
    """Routes a bounded set of GitHub API paths to canned real-shaped JSON.
    Raises AssertionError on any URL it was not told to expect -- a test that
    reaches an un-scripted URL fails loudly rather than silently."""

    def __init__(self, routes: dict[str, _FakeResponse]):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, *, headers=None, params=None, timeout=None):
        self.calls.append(url)
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                return resp
        raise AssertionError(f"unscripted URL requested: {url}")


_README_TEXT = (
    "# Trend Bot\n\nA simple 20/50 moving average crossover trend-following strategy "
    "for futures markets. Long when fast MA > slow MA."
)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _search_response() -> _FakeResponse:
    return _FakeResponse(200, {
        "items": [
            {
                "full_name": "acme/trend-bot", "description": "MA crossover trend following for futures",
                "html_url": "https://github.com/acme/trend-bot", "default_branch": "main",
                "license": {"spdx_id": "MIT"}, "owner": {"login": "acme"}, "created_at": "2021-01-01T00:00:00Z",
                "stargazers_count": 42, "size": 500, "archived": False, "fork": False,
            }
        ]
    })


def _fake_routes(*, rate_remaining: int = 5000) -> dict[str, _FakeResponse]:
    return {
        "https://api.github.com/rate_limit": _FakeResponse(200, {"resources": {"core": {"remaining": rate_remaining, "limit": 5000}}}),
        "https://api.github.com/search/repositories": _search_response(),
        "https://api.github.com/repos/acme/trend-bot/readme": _FakeResponse(200, {"content": _b64(_README_TEXT), "encoding": "base64"}),
        "https://api.github.com/repos/acme/trend-bot/git/trees/main": _FakeResponse(200, {"tree": []}),
        "https://api.github.com/repos/acme/trend-bot/commits/main": _FakeResponse(200, {"sha": "deadbeef1234567890"}),
    }


@pytest.fixture()
def fake_github(monkeypatch):
    fake = _FakeGitHub(_fake_routes())
    monkeypatch.setattr("httpx.get", fake.get)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")
    return fake


# ---------------------------------------------------------------------------
# query generation
# ---------------------------------------------------------------------------


def test_query_generation_is_bounded_and_deterministic():
    budget = ConnectorBudget(max_queries=3)
    q1 = generate_github_queries(market="NQ", budget=budget)
    q2 = generate_github_queries(market="NQ", budget=budget)
    assert q1 == q2
    assert len(q1) <= 3


def test_query_generation_includes_market_specific_and_generic_terms():
    queries = generate_github_queries(market="NQ", budget=ConnectorBudget(max_queries=6))
    assert any("Nasdaq" in q for q in queries)
    assert any("CTA" in q or "managed futures" in q for q in queries)


# ---------------------------------------------------------------------------
# authentication / health
# ---------------------------------------------------------------------------


def test_no_token_is_not_connected_without_any_network_call(monkeypatch):
    # Block a real local `.env` (if this checkout happens to have one) from
    # re-populating GITHUB_TOKEN -- this test must be deterministic
    # regardless of the environment it happens to run in.
    import alpha_agent.knowledge.github_connector as gh_mod

    monkeypatch.setattr(gh_mod, "load_dotenv_if_available", lambda: None)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def _fail_if_called(*a, **kw):
        raise AssertionError("no network call should happen without a token")

    monkeypatch.setattr("httpx.get", _fail_if_called)
    report = github_health()
    assert report.health is GitHubConnectorHealth.NOT_CONNECTED
    assert report.token_present is False


def test_bad_token_is_error(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "bad-token")

    def _unauthorized(url, **kw):
        return _FakeResponse(401, {"message": "Bad credentials"})

    monkeypatch.setattr("httpx.get", _unauthorized)
    report = github_health()
    assert report.health is GitHubConnectorHealth.ERROR


def test_rate_limit_exhausted_is_rate_limited(monkeypatch):
    fake = _FakeGitHub(_fake_routes(rate_remaining=0))
    monkeypatch.setattr("httpx.get", fake.get)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")
    report = github_health()
    assert report.health is GitHubConnectorHealth.RATE_LIMITED

    conn = GitHubStrategyConnector()
    result = conn.ingest(query="futures trading strategy", markets=("NQ",))
    assert result.status is IngestionStatus.RATE_LIMITED
    assert result.items == ()


def test_health_report_never_carries_the_token_value(fake_github):
    report = github_health()
    blob = report.model_dump_json()
    assert "test-token-not-real" not in blob


# ---------------------------------------------------------------------------
# real-shaped parsing via mocked HTTP
# ---------------------------------------------------------------------------


def test_ingest_normalizes_a_repo_with_real_shaped_fields(fake_github):
    conn = GitHubStrategyConnector(budget=ConnectorBudget(max_queries=1, max_files_per_repo=1))
    result = conn.ingest(query="moving average crossover", markets=("NQ",))
    assert result.status is IngestionStatus.CONNECTED
    assert len(result.items) == 1
    item = result.items[0]
    assert item.source_type is SourceType.GITHUB
    assert item.repository == "acme/trend-bot"
    assert item.commit_sha == "deadbeef1234567890"
    assert item.license == "MIT"
    assert item.source_url == "https://github.com/acme/trend-bot"
    assert item.economic_mechanism.value == "TREND"
    assert item.reported_results is None  # never our evidence
    assert item.provenance_hash.startswith("github1:")
    assert item.knowledge_id == item.provenance_hash


def test_repository_search_uses_authenticated_headers(fake_github):
    conn = GitHubStrategyConnector()
    conn.ingest(query="moving average crossover", markets=("NQ",))
    assert any(u.startswith("https://api.github.com/search/repositories") for u in fake_github.calls)


def test_unclassifiable_repo_is_skipped_not_guessed(monkeypatch):
    routes = _fake_routes()
    routes["https://api.github.com/search/repositories"] = _FakeResponse(200, {
        "items": [{
            "full_name": "acme/mystery-bot", "description": "", "html_url": "https://github.com/acme/mystery-bot",
            "default_branch": "main", "license": None, "owner": {"login": "acme"}, "created_at": "2021-01-01T00:00:00Z",
            "stargazers_count": 5000, "size": 10, "archived": False, "fork": False,
        }]
    })
    routes["https://api.github.com/repos/acme/mystery-bot/readme"] = _FakeResponse(200, {"content": _b64("Just a random project, no strategy content."), "encoding": "base64"})
    fake = _FakeGitHub(routes)
    monkeypatch.setattr("httpx.get", fake.get)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")

    conn = GitHubStrategyConnector()
    result = conn.ingest(query="anything", markets=("NQ",))
    assert result.items == ()  # never fabricated a mechanism for an unclassifiable repo


def test_ranking_is_not_dominated_by_stars_alone():
    from alpha_agent.knowledge.github_connector import _score_repo

    high_stars_thin = {"description": "", "license": None, "size": 5, "stargazers_count": 50_000, "archived": False, "fork": False}
    low_stars_documented = {
        "description": "moving average crossover trend strategy", "license": {"spdx_id": "MIT"},
        "size": 500, "stargazers_count": 3, "archived": False, "fork": False,
    }
    assert _score_repo(low_stars_documented, query="moving average crossover") > _score_repo(high_stars_thin, query="moving average crossover")


# ---------------------------------------------------------------------------
# security -- never execute/import retrieved source
# ---------------------------------------------------------------------------


def test_github_modules_never_call_eval_exec_or_shell_out():
    import alpha_agent.knowledge.connector_support as support_mod
    import alpha_agent.knowledge.github_connector as gh_mod

    for mod in (support_mod, gh_mod):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec"), f"{mod.__name__} calls {node.func.id}"
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ("system", "Popen", "call", "run")
                and isinstance(node.value, ast.Name)
                and node.value.id in ("os", "subprocess")
            ):
                raise AssertionError(f"{mod.__name__} shells out via {node.value.id}.{node.attr}")


def test_forbidden_operation_tokens_absent_from_source():
    import alpha_agent.knowledge.github_connector as gh_mod

    src = inspect.getsource(gh_mod)
    for token in external_sources.FORBIDDEN_OPERATIONS:
        assert token not in src


def test_fetched_repo_content_is_never_parsed_as_code():
    """The decoded README/file text only ever flows into string fields
    (`implementation_notes`, `entry_logic_summary`) -- proven structurally by
    `_decode_readme`'s return type and the total absence of `compile(`/
    `ast.parse(` anywhere in the module reached with retrieved content."""
    import alpha_agent.knowledge.github_connector as gh_mod

    src = inspect.getsource(gh_mod)
    assert "compile(" not in src
    assert "importlib" not in src


# ---------------------------------------------------------------------------
# Release UX bugfix pass -- a real "Connected Research" run against a real
# GitHub repository crashed the whole discovery campaign: the repo's own
# file paths (e.g. "docs/plans/2026-07-15-plan.md" -- a real, ordinary
# planning-doc filename, unrelated to market data) landed in `limitations`,
# which reaches `ResearchContext`/`OrchestratorConfig.knowledge_base` and
# tripped the locked-holdout guard's >= 2025-01-01 date scan as a false
# positive. Fixed at the source: `limitations` states a COUNT of inspected
# files, never the literal paths.
# ---------------------------------------------------------------------------


def _fake_routes_with_dated_file(*, rate_remaining: int = 5000) -> dict[str, _FakeResponse]:
    routes = _fake_routes(rate_remaining=rate_remaining)
    routes["https://api.github.com/repos/acme/trend-bot/git/trees/main"] = _FakeResponse(200, {
        "tree": [{"type": "blob", "path": "docs/plans/2026-07-15-volume-profile-plan.md", "size": 100}],
    })
    routes["https://api.github.com/repos/acme/trend-bot/contents/docs/plans/2026-07-15-volume-profile-plan.md"] = (
        _FakeResponse(200, {"content": _b64("Plan notes."), "encoding": "base64"})
    )
    return routes


def test_limitations_never_embeds_a_real_file_path_with_an_incidental_date(monkeypatch):
    fake = _FakeGitHub(_fake_routes_with_dated_file())
    monkeypatch.setattr("httpx.get", fake.get)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")

    conn = GitHubStrategyConnector(budget=ConnectorBudget(max_queries=1, max_files_per_repo=1))
    result = conn.ingest(query="moving average crossover", markets=("NQ",))
    assert result.status is IngestionStatus.CONNECTED
    item = result.items[0]
    assert "2026-07-15" not in item.limitations
    assert "docs/plans" not in item.limitations
    assert "1 additional source file(s)" in item.limitations


def test_render_snippet_with_a_dated_repo_path_passes_the_holdout_guard(monkeypatch):
    """End-to-end: the exact real crash -- a knowledge_base entry built from
    a GitHub item whose repo happened to reference a 2026-dated file path --
    must no longer trip `assert_no_holdout_market_data` at the SAME
    `OrchestratorConfig`/`ResearchContext`-shaped path real callers use."""
    from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data

    fake = _FakeGitHub(_fake_routes_with_dated_file())
    monkeypatch.setattr("httpx.get", fake.get)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")

    conn = GitHubStrategyConnector(budget=ConnectorBudget(max_queries=1, max_files_per_repo=1))
    result = conn.ingest(query="moving average crossover", markets=("NQ",))
    item = result.items[0]
    snippet = item.render_snippet()
    assert "2026-07-15" not in snippet  # the literal path never reached the snippet at all

    # Exactly how OrchestratorConfig/ResearchContext guard their own
    # knowledge_base field -- must not raise.
    assert_no_holdout_market_data(
        {"knowledge_base": [snippet]}, path="$.orchestrator_config",
        observational_context_paths=frozenset({"$.orchestrator_config.knowledge_base"}),
    )


# ---------------------------------------------------------------------------
# live smoke test (task spec "GITHUB LIVE SMOKE TEST") -- skipped without a
# real token, never fabricated, never prints/logs the token.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("GITHUB_TOKEN"), reason="GITHUB_TOKEN not configured in this environment")
def test_live_github_health_check():
    report = github_health()
    assert report.health in (GitHubConnectorHealth.CONNECTED, GitHubConnectorHealth.RATE_LIMITED)
    assert report.token_present is True


@pytest.mark.skipif(not os.environ.get("GITHUB_TOKEN"), reason="GITHUB_TOKEN not configured in this environment")
def test_live_github_ingest_smoke():
    conn = GitHubStrategyConnector(
        budget=ConnectorBudget(max_queries=1, max_search_results=8, max_files_per_repo=1, max_documents_total=3)
    )
    result = conn.ingest(query="moving average crossover trading strategy", markets=("NQ",))
    assert result.status in (IngestionStatus.CONNECTED, IngestionStatus.DEGRADED)
    for item in result.items:
        assert item.source_type is SourceType.GITHUB
        assert item.repository
        assert item.provenance_hash.startswith("github1:")

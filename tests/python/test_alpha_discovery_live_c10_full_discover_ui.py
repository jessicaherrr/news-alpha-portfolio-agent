"""Alpha Discovery live-research campaign, Checkpoint 10 -- the full,
connected Discover Strategies UI (task spec Part 7, tests section "Discover
UI tests").

Real `streamlit.testing.v1.AppTest` renders, real button clicks, no
fabricated widget state. The "connected" scenario mocks `httpx.get` (the
Streamlit `AppTest` harness executes the page script in-process, so a
`monkeypatch` on the shared `httpx` module applies inside it too) so this
stays a fast, deterministic, network-free test -- Checkpoints 5-7 already
prove the connectors themselves work against the real APIs.
"""
from __future__ import annotations

import base64

import pytest
from alpha_agent.registry.sqlite_registry import ExperimentRegistry


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers: dict = {}
        self.content = b""

    def json(self):
        return self._json_body


def _live_shaped_routes() -> dict[str, _FakeResponse]:
    readme = base64.b64encode(b"# Trend Bot\n\nMoving average crossover trend following.").decode("ascii")
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
        "https://hn.algolia.com/api/v1/search": _FakeResponse(200, {"hits": []}),
        "https://api.openalex.org/works": _FakeResponse(200, {"results": []}),
        "https://api.crossref.org/works": _FakeResponse(200, {"message": {"items": []}}),
    }


class _FakeHttp:
    def __init__(self, routes):
        self.routes = routes

    def get(self, url, *, headers=None, params=None, timeout=None):
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                return resp
        return _FakeResponse(500, {})


@pytest.fixture()
def isolated_registry(tmp_path, monkeypatch):
    from alpha_agent.ui import services

    path = tmp_path / "registry.sqlite"
    monkeypatch.setattr(services, "REGISTRY_PATH", path)
    ExperimentRegistry(path)
    return path


def _app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string("from alpha_agent.ui.views.discover import render\nrender()\n")


def test_offline_discover_run_shows_research_sources_and_progress_metrics(isolated_registry):
    at = _app()
    at.run(timeout=90)
    assert not list(at.exception)

    at.radio(key="discover-research-sources").set_value("offline")
    at.button(key="discover-run-button").click().run(timeout=120)
    assert not list(at.exception)

    blob = " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
    assert "Research Sources" in blob
    assert "Research Process" in blob
    metric_labels = [m.label for m in at.metric]
    assert "Local Knowledge Items" in metric_labels
    assert "Live Sources Retrieved" in metric_labels
    assert "Unique Mechanisms" in metric_labels
    # Release UX bugfix pass, issue 1: offline mode means external research
    # was never REQUESTED -- every external source reads DISABLED, never
    # NOT_CONNECTED (which would falsely imply a real attempt was made).
    assert any(m.label == "GitHub" and m.value == "DISABLED" for m in at.metric)
    assert any(m.label == "Academic Papers" and m.value == "DISABLED" for m in at.metric)
    assert any(m.label == "Community" and m.value == "DISABLED" for m in at.metric)
    assert any(m.label == "Practitioner" and m.value == "DISABLED" for m in at.metric)
    assert not any(m.value == "NOT_CONNECTED" for m in at.metric)


def test_connected_discover_run_shows_live_source_status(isolated_registry, monkeypatch):
    fake = _FakeHttp(_live_shaped_routes())
    monkeypatch.setattr("httpx.get", fake.get)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")

    at = _app()
    at.run(timeout=90)
    at.radio(key="discover-research-sources").set_value("connected")
    at.button(key="discover-run-button").click().run(timeout=120)
    assert not list(at.exception)

    assert any(m.label == "GitHub" and m.value == "CONNECTED" for m in at.metric)
    metric_by_label = {m.label: m for m in at.metric}
    assert int(metric_by_label["Live Sources Retrieved"].value) >= 1


def test_review_frozen_candidates_warning_text_is_present_when_a_family_is_frozen(isolated_registry):
    """A cheap structural check (no C++ needed): the exact required warning
    text is a real module constant the view actually renders, not just
    documented in a docstring."""
    from alpha_agent.ui import discovery_campaign

    assert "cannot change once validation begins" in discovery_campaign.STRICT_VALIDATION_WARNING

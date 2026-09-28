"""Tests for `alpha_agent.ui.databento_context` -- the UI's sole boundary
into the Databento observation provider. No real provider/network call: the
module-level `_provider` factory is monkeypatched to a fake in every test.

`real_databento_context`: this file tests `databento_context`'s OWN
delegation/degradation behavior, so it opts OUT of `conftest.py`'s repo-wide
autouse default that replaces the public functions this file needs to
exercise unmodified (see that fixture's docstring).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alpha_agent.marketdata.databento_schemas import (
    ContractResolution,
    DatabentoCapability,
    DatabentoHealth,
    MarketSnapshot,
)
from alpha_agent.registry.holdout_guard import HoldoutAccessError, assert_no_holdout_market_data
from alpha_agent.ui import databento_context

pytestmark = pytest.mark.real_databento_context


class _FakeProvider:
    def __init__(self, snapshot: MarketSnapshot | None = None, health: DatabentoHealth | None = None):
        self._snapshot = snapshot
        self._health = health or DatabentoHealth(
            capability=DatabentoCapability.LATEST_AVAILABLE, dataset="GLBX.MDP3", checked_at=datetime.now(UTC),
        )
        self.clear_cache_calls = 0

    def health(self):
        return self._health

    def get_market_snapshot(self, root):
        return self._snapshot

    def get_recent_ohlcv(self, root, timeframe="1h", lookback_bars=48):
        return None

    def resolve_display_contract(self, root):
        return self._snapshot.contract if self._snapshot else None

    def get_contract_metadata(self, root):
        return None

    def clear_cache(self):
        self.clear_cache_calls += 1


def _snapshot(*, last: float | None = 29123.5, capability=DatabentoCapability.LATEST_AVAILABLE) -> MarketSnapshot:
    return MarketSnapshot(
        root_symbol="NQ",
        contract=ContractResolution(
            root_symbol="NQ", display_symbol="NQ.v.0", resolved_raw_symbol="NQU6", resolved_at=datetime.now(UTC),
        ),
        last=last, as_of=datetime.now(UTC), capability=capability,
    )


@pytest.fixture(autouse=True)
def _reset_provider_cache():
    databento_context._provider.cache_clear()
    yield
    databento_context._provider.cache_clear()


def test_unapproved_root_never_touches_the_provider(monkeypatch):
    calls = []
    monkeypatch.setattr(databento_context, "_provider", lambda: calls.append(1) or _FakeProvider())
    assert databento_context.market_snapshot("ZZ") is None
    assert databento_context.recent_ohlcv("ZZ") is None
    assert databento_context.resolve_display_contract("ZZ") is None
    assert databento_context.contract_metadata("ZZ") is None
    assert calls == []


def test_market_snapshot_delegates_to_provider(monkeypatch):
    fake = _FakeProvider(snapshot=_snapshot())
    monkeypatch.setattr(databento_context, "_provider", lambda: fake)
    snap = databento_context.market_snapshot("nq")
    assert snap is not None
    assert snap.root_symbol == "NQ"


def test_provider_exception_degrades_to_none_never_raises(monkeypatch):
    class Boom:
        def get_market_snapshot(self, root):
            raise RuntimeError("boom")

    monkeypatch.setattr(databento_context, "_provider", lambda: Boom())
    assert databento_context.market_snapshot("NQ") is None


def test_observational_context_note_is_none_without_a_price(monkeypatch):
    fake = _FakeProvider(snapshot=_snapshot(last=None))
    monkeypatch.setattr(databento_context, "_provider", lambda: fake)
    assert databento_context.observational_context_note("NQ") is None


def test_observational_context_note_is_none_when_unavailable_capability(monkeypatch):
    fake = _FakeProvider(snapshot=_snapshot(capability=DatabentoCapability.NOT_CONNECTED))
    monkeypatch.setattr(databento_context, "_provider", lambda: fake)
    assert databento_context.observational_context_note("NQ") is None


def test_observational_context_note_matches_the_closed_grammar_and_passes_the_holdout_guard(monkeypatch):
    fake = _FakeProvider(snapshot=_snapshot())
    monkeypatch.setattr(databento_context, "_provider", lambda: fake)
    note = databento_context.observational_context_note("NQ")
    assert note is not None
    assert note.startswith("OBSERVATIONAL_CONTEXT_ONLY provider=DATABENTO feed=LATEST_AVAILABLE root=NQ")
    # Must pass the SAME guard research-context payloads are checked against,
    # when declared at an observational path -- never at an undeclared one.
    assert_no_holdout_market_data(
        {"knowledge_base": [note]}, path="$.research_context",
        observational_context_paths=frozenset({"$.research_context.knowledge_base"}),
    )
    with pytest.raises(HoldoutAccessError):
        assert_no_holdout_market_data({"scientific_field": note}, path="$.result")


def test_clear_cache_delegates_and_never_raises_on_missing_provider(monkeypatch):
    fake = _FakeProvider()
    monkeypatch.setattr(databento_context, "_provider", lambda: fake)
    databento_context.clear_cache()
    assert fake.clear_cache_calls == 1


def test_health_never_raises_even_on_construction_failure(monkeypatch):
    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(databento_context, "_provider", boom)
    health = databento_context.health()
    assert health.capability == DatabentoCapability.ERROR

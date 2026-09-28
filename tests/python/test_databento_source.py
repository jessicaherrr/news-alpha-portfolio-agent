"""Offline checks for the Databento integration.

None of these hit the network or require an API key. They lock in the
cost-first contract (no estimate -> no download) and verify that the request
we build still matches the installed SDK's expectations.
"""
from __future__ import annotations

import dataclasses

import pytest
from alpha_agent.data import databento_source as ds
from alpha_agent.data.databento_source import HistoricalRequest


def test_historical_request_is_frozen_with_expected_defaults():
    req = HistoricalRequest(
        dataset="GLBX.MDP3",
        symbols=("NQ.v.0",),
        schema="ohlcv-1m",
        start="2024-01-05",
        end="2024-01-06",
        stype_in="continuous",
    )
    assert req.stype_in == "continuous"
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.dataset = "other"  # frozen dataclass

    default_req = HistoricalRequest(
        dataset="GLBX.MDP3", symbols=("ES.v.0",), schema="ohlcv-1m",
        start="2024-01-05", end="2024-01-06",
    )
    assert default_req.stype_in == "raw_symbol"
    assert default_req.stype_out == "instrument_id"


def test_continuous_ohlcv_request_uses_instrument_id_output():
    req = HistoricalRequest(
        dataset="GLBX.MDP3", symbols=("NQ.v.0",), schema="ohlcv-1m",
        start="2024-01-05", end="2024-01-06", stype_in="continuous",
    )
    assert req.stype_out == "instrument_id"


def test_unsupported_symbology_is_rejected_by_the_builder():
    for stype_in in ("continuous", "parent"):
        with pytest.raises(ValueError, match="does not support"):
            HistoricalRequest(
                dataset="GLBX.MDP3", symbols=("NQ.v.0",), schema="ohlcv-1m",
                start="2024-01-05", end="2024-01-06",
                stype_in=stype_in, stype_out="raw_symbol",
            )


def test_definition_request_is_derived():
    req = HistoricalRequest(
        dataset="GLBX.MDP3", symbols=("NQ.v.0", "ES.v.0"), schema="ohlcv-1m",
        start="2024-01-05", end="2024-01-06", stype_in="continuous",
    )
    d = req.definition_request()
    assert d.schema == "definition"
    assert d.stype_in == "parent"
    assert d.stype_out == "instrument_id"          # NOT raw_symbol
    assert set(d.symbols) == {"NQ.FUT", "ES.FUT"}
    assert d.start == req.start and d.end == req.end


def test_estimate_cost_requires_api_key(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    req = HistoricalRequest(
        dataset="GLBX.MDP3", symbols=("NQ.v.0",), schema="ohlcv-1m",
        start="2024-01-05", end="2024-01-06", stype_in="continuous",
    )
    with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
        ds.estimate_cost_usd(req)


def test_estimate_cost_forwards_expected_request(monkeypatch):
    captured: dict = {}

    class _FakeMetadata:
        def get_cost(self, **kwargs):
            captured.update(kwargs)
            return 0.0123

    class _FakeClient:
        metadata = _FakeMetadata()

    monkeypatch.setattr(ds, "_client", lambda api_key=None: _FakeClient())

    req = HistoricalRequest(
        dataset="GLBX.MDP3", symbols=("NQ.v.0",), schema="ohlcv-1m",
        start="2024-01-05", end="2024-01-06", stype_in="continuous",
    )
    cost = ds.estimate_cost_usd(req)

    assert cost == pytest.approx(0.0123)
    assert captured == {
        "dataset": "GLBX.MDP3",
        "symbols": ["NQ.v.0"],
        "schema": "ohlcv-1m",
        "stype_in": "continuous",
        "start": "2024-01-05",
        "end": "2024-01-06",
    }


def test_fetch_aborts_when_estimate_exceeds_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "estimate_cost_usd", lambda request, api_key=None: 5.00)

    def _boom(*_a, **_k):  # must never be reached
        raise AssertionError("download attempted despite cost over cap")

    monkeypatch.setattr(ds, "_client", _boom)

    req = HistoricalRequest(
        dataset="GLBX.MDP3", symbols=("NQ.v.0",), schema="ohlcv-1m",
        start="2024-01-05", end="2024-01-06", stype_in="continuous",
    )
    with pytest.raises(RuntimeError, match="exceeds cap"):
        ds.fetch_and_store_raw(req, max_cost_usd=1.00, root_dir=tmp_path)
    assert not list(tmp_path.rglob("*"))  # nothing written before the cap check


def test_installed_sdk_still_accepts_our_symbology_literals():
    databento = pytest.importorskip("databento")
    assert str(databento.SType.from_str("continuous")) == str(databento.SType.CONTINUOUS)
    assert str(databento.Schema.from_str("ohlcv-1m")) == str(databento.Schema.OHLCV_1M)

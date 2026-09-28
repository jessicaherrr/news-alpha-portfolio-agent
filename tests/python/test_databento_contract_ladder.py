"""Market Intelligence Data Completion Pass, Checkpoint C -- contract ladder
+ term structure. Pure functions (`filter_outright_active_contracts`,
`classify_curve_shape`) are tested directly with no client at all; the
provider-level `get_contract_ladder`/`get_term_structure` wiring is tested
with an injected fake client -- NO real Databento SDK import, NO network
call, ever (mirrors `test_databento_market_data_provider.py`).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from alpha_agent.marketdata.databento_config import DatabentoMarketDataConfig
from alpha_agent.marketdata.databento_provider import (
    DatabentoMarketDataProvider,
    DatabentoUnavailable,
    classify_curve_shape,
    filter_outright_active_contracts,
)
from alpha_agent.marketdata.databento_schemas import CurveShape, DatabentoCapability

AS_OF = datetime(2026, 9, 14, tzinfo=UTC)


# ---------------------------------------------------------------------------
# filter_outright_active_contracts -- pure, no network
# ---------------------------------------------------------------------------


def _def_row(
    raw_symbol: str, *, instrument_class: str = "F", expiration: str, activation: str | None = None,
    instrument_id: int = 1,
) -> dict:
    row = {
        "raw_symbol": raw_symbol, "instrument_class": instrument_class, "instrument_id": instrument_id,
        "expiration": pd.Timestamp(expiration, tz="UTC"),
    }
    if activation is not None:
        row["activation"] = pd.Timestamp(activation, tz="UTC")
    return row


def test_outright_filter_excludes_spreads():
    rows = [
        _def_row("CLV6", instrument_class="F", expiration="2026-09-22"),
        _def_row("CL:BF F7-G7-H7", instrument_class="S", expiration="2027-01-01"),
    ]
    active = filter_outright_active_contracts(rows, as_of=AS_OF, max_contracts=12)
    assert [r["raw_symbol"] for r in active] == ["CLV6"]


def test_expiry_sort_not_lexicographic():
    """CLZ6 sorts before CLA... lexicographically it wouldn't, but Z < nothing
    matters here -- the real regression is symbol vs. expiry ordering: CLF7
    (Jan '27) must sort AFTER CLX6 (Oct '26) even though 'F' < 'X'."""
    rows = [
        _def_row("CLF7", expiration="2026-12-21"),
        _def_row("CLX6", expiration="2026-10-20"),
        _def_row("CLV6", expiration="2026-09-22"),
    ]
    active = filter_outright_active_contracts(rows, as_of=AS_OF, max_contracts=12)
    assert [r["raw_symbol"] for r in active] == ["CLV6", "CLX6", "CLF7"]


def test_expired_contracts_are_excluded():
    rows = [
        _def_row("CLU6", expiration="2026-08-20"),  # already expired relative to AS_OF
        _def_row("CLV6", expiration="2026-09-22"),
    ]
    active = filter_outright_active_contracts(rows, as_of=AS_OF, max_contracts=12)
    assert [r["raw_symbol"] for r in active] == ["CLV6"]


def test_not_yet_activated_contract_is_excluded():
    rows = [
        _def_row("CLZ7", expiration="2027-11-19", activation="2026-10-01"),  # activates in the future
        _def_row("CLV6", expiration="2026-09-22", activation="2018-01-19"),
    ]
    active = filter_outright_active_contracts(rows, as_of=AS_OF, max_contracts=12)
    assert [r["raw_symbol"] for r in active] == ["CLV6"]


def test_missing_activation_is_not_excluded():
    """Section 5: 'activation <= observation time WHEN activation exists' --
    a row with no activation field at all must never be excluded on that
    basis alone."""
    rows = [_def_row("CLV6", expiration="2026-09-22", activation=None)]
    active = filter_outright_active_contracts(rows, as_of=AS_OF, max_contracts=12)
    assert [r["raw_symbol"] for r in active] == ["CLV6"]


def test_max_contracts_is_a_hard_bound():
    rows = [_def_row(f"CL{i}", expiration=f"2027-{((8 + i) % 12) + 1:02d}-01") for i in range(1, 10)]
    active = filter_outright_active_contracts(rows, as_of=AS_OF, max_contracts=3)
    assert len(active) == 3


def test_duplicate_definition_updates_keep_only_the_latest():
    """The same raw symbol can appear more than once in a queried window
    (multiple `security_update_action` rows) -- the LATEST (last in query
    order) must win, never a duplicate row in the ladder."""
    rows = [
        _def_row("CLV6", expiration="2026-09-22", instrument_id=1),
        _def_row("CLV6", expiration="2026-09-22", instrument_id=2),  # a later update, same symbol
    ]
    active = filter_outright_active_contracts(rows, as_of=AS_OF, max_contracts=12)
    assert len(active) == 1
    assert active[0]["instrument_id"] == 2


# ---------------------------------------------------------------------------
# classify_curve_shape -- pure, no network
# ---------------------------------------------------------------------------


def test_curve_shape_contango():
    shape, slope, f2, f3 = classify_curve_shape([100.0, 101.0, 102.5], tolerance_pct=0.05)
    assert shape is CurveShape.CONTANGO
    assert f2 == pytest.approx(1.0)
    assert f3 == pytest.approx(2.5)
    assert slope == pytest.approx(1.25)


def test_curve_shape_backwardation():
    shape, *_ = classify_curve_shape([100.0, 95.0, 90.0], tolerance_pct=0.05)
    assert shape is CurveShape.BACKWARDATION


def test_curve_shape_flat_within_tolerance():
    """A leg move of $0.01 on a $100 front price (0.01%) must not register as
    a real regime under a 0.05% tolerance -- tiny noise, not a signal."""
    shape, *_ = classify_curve_shape([100.0, 100.01, 99.995], tolerance_pct=0.05)
    assert shape is CurveShape.FLAT


def test_curve_shape_mixed_when_legs_disagree_beyond_tolerance():
    shape, *_ = classify_curve_shape([100.0, 105.0, 98.0], tolerance_pct=0.05)
    assert shape is CurveShape.MIXED


def test_curve_shape_insufficient_data_below_two_points():
    shape, slope, f2, f3 = classify_curve_shape([100.0], tolerance_pct=0.05)
    assert shape is CurveShape.INSUFFICIENT_DATA
    assert slope is None and f2 is None and f3 is None

    shape2, *_ = classify_curve_shape([], tolerance_pct=0.05)
    assert shape2 is CurveShape.INSUFFICIENT_DATA


def test_curve_shape_never_regresses_to_a_volume_rank():
    """Regression guard for Section 8's mandatory distinction: two markets
    with the SAME raw closing prices must classify identically regardless of
    which underlying continuous-symbol volume ranking would have picked --
    `classify_curve_shape` never even sees a volume figure, only real,
    expiry-ordered prices."""
    import inspect

    sig = inspect.signature(classify_curve_shape)
    assert "volume" not in sig.parameters


# ---------------------------------------------------------------------------
# provider-level wiring -- fake client, no network
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df


class _FakeMetadata:
    def __init__(self, *, dataset_range: dict, cost: float = 0.0):
        self._dataset_range = dataset_range
        self._cost = cost

    def get_dataset_range(self, *, dataset: str) -> dict:
        return self._dataset_range

    def get_cost(self, **kwargs) -> float:
        return self._cost


class _FakeSymbology:
    def resolve(self, **kwargs) -> dict:
        symbol = kwargs["symbols"][0]
        return {"result": {symbol: [{"d0": "2026-09-08", "d1": "2026-09-13", "s": "999"}]}}


def _resolved_display_definition_df() -> pd.DataFrame:
    """What `resolve_display_contract`'s OWN separate instrument-id lookup
    returns for the continuous proxy's resolved id ("999") -- CLV6, matching
    the front-month contract in `_definition_df()`."""
    return pd.DataFrame(
        [{"raw_symbol": "CLV6", "expiration": pd.Timestamp("2026-09-22", tz="UTC"), "exchange": "XNYM"}]
    )


def _definition_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "raw_symbol": "CLV6", "instrument_class": "F", "instrument_id": 111,
                "expiration": pd.Timestamp("2026-09-22", tz="UTC"),
                "activation": pd.Timestamp("2018-01-19", tz="UTC"),
                "exchange": "XNYM", "min_price_increment": 0.01, "display_factor": 0.01,
            },
            {
                "raw_symbol": "CLX6", "instrument_class": "F", "instrument_id": 222,
                "expiration": pd.Timestamp("2026-10-20", tz="UTC"),
                "activation": pd.Timestamp("2018-01-19", tz="UTC"),
                "exchange": "XNYM", "min_price_increment": 0.01, "display_factor": 0.01,
            },
            {
                "raw_symbol": "CL:BF F7-G7-H7", "instrument_class": "S", "instrument_id": 333,
                "expiration": pd.Timestamp("2027-01-01", tz="UTC"),
                "exchange": "XNYM",
            },
        ]
    )


def _ohlcv1d_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-09-13", tz="UTC"), pd.Timestamp("2026-09-13", tz="UTC")], name="ts_event",
    )
    return pd.DataFrame({"close": [100.0, 95.0], "volume": [1000.0, 500.0], "symbol": ["CLV6", "CLX6"]}, index=idx)


def _statistics_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-09-13", tz="UTC"), pd.Timestamp("2026-09-13", tz="UTC")], name="ts_recv",
    )
    return pd.DataFrame(
        {
            "stat_type": [3, 9],  # settlement price, open interest
            "price": [99.5, float("nan")],
            "quantity": [9_223_372_036_854_775_807, 50000],
            "symbol": ["CLV6", "CLV6"],
        },
        index=idx,
    )


class _FakeTimeseries:
    def __init__(self, *, definition_df, ohlcv_df, statistics_df):
        self._definition_df = definition_df
        self._ohlcv_df = ohlcv_df
        self._statistics_df = statistics_df
        self.calls: list[dict] = []

    def get_range(self, **kwargs) -> _FakeStore:
        self.calls.append(kwargs)
        schema = kwargs["schema"]
        if schema == "definition":
            # The contract-LADDER fetch queries by parent symbology
            # (`{ROOT}.FUT`); `resolve_display_contract`'s own SEPARATE
            # single-instrument lookup queries by instrument_id -- real
            # Databento distinguishes these by `stype_in`.
            if kwargs.get("stype_in") == "instrument_id":
                return _FakeStore(_resolved_display_definition_df())
            return _FakeStore(self._definition_df)
        if schema == "statistics":
            return _FakeStore(self._statistics_df)
        return _FakeStore(self._ohlcv_df)


class _FakeClient:
    def __init__(self, *, cost: float = 0.0, definition_df=None, ohlcv_df=None, statistics_df=None):
        # Baseline hotfix: this MUST be the module's own fixed AS_OF, never
        # the real wall clock. `DatabentoMarketDataProvider.get_contract_ladder`
        # prefers `health.latest_available_ts` (parsed straight from this
        # `dataset_range["end"]`) over `datetime.now(UTC)` as its filtering
        # reference -- a real-clock value here silently re-introduces wall-
        # clock dependence into every fixture contract's expiration check
        # (CLV6 is deliberately fixed at 2026-09-22, one CL expiry cycle
        # after AS_OF) regardless of how the provider itself is written.
        now_iso = AS_OF.isoformat()
        self.metadata = _FakeMetadata(dataset_range={"start": "2010-01-01T00:00:00Z", "end": now_iso}, cost=cost)
        self.symbology = _FakeSymbology()
        self.timeseries = _FakeTimeseries(
            definition_df=definition_df if definition_df is not None else _definition_df(),
            ohlcv_df=ohlcv_df if ohlcv_df is not None else _ohlcv1d_df(),
            statistics_df=statistics_df if statistics_df is not None else _statistics_df(),
        )


def _provider(client: _FakeClient) -> DatabentoMarketDataProvider:
    return DatabentoMarketDataProvider(config=DatabentoMarketDataConfig(), client_factory=lambda api_key: client)


def test_contract_ladder_excludes_spreads_and_marks_front_and_display():
    provider = _provider(_FakeClient())
    ladder = provider.get_contract_ladder("CL", max_contracts=12)
    assert ladder.fetched is True
    symbols = [c.raw_symbol for c in ladder.contracts]
    assert symbols == ["CLV6", "CLX6"]  # the "S" spread row never appears
    assert ladder.contracts[0].is_front_month is True
    assert ladder.contracts[0].is_display_contract is True  # resolved continuous symbol is "999" -> CLV6 here
    assert ladder.contracts[1].is_front_month is False


def test_contract_ladder_prices_and_open_interest_are_real_never_fabricated():
    provider = _provider(_FakeClient())
    ladder = provider.get_contract_ladder("CL", max_contracts=12)
    by_symbol = {c.raw_symbol: c for c in ladder.contracts}
    assert by_symbol["CLV6"].last == 100.0
    assert by_symbol["CLV6"].settlement_price == 99.5
    assert by_symbol["CLV6"].open_interest == 50000.0
    # CLX6 has no statistics row at all in this fixture -- N/A, never 0.
    assert by_symbol["CLX6"].open_interest is None
    assert by_symbol["CLX6"].settlement_price is None


def test_contract_ladder_never_fabricates_open_interest_from_the_int64_sentinel():
    provider = _provider(_FakeClient())
    ladder = provider.get_contract_ladder("CL", max_contracts=12)
    by_symbol = {c.raw_symbol: c for c in ladder.contracts}
    # The settlement row's `quantity` carries the INT64 unset sentinel --
    # must never be read back as an open-interest value.
    assert by_symbol["CLV6"].open_interest != 9_223_372_036_854_775_807


def test_contract_ladder_raw_symbol_provenance_matches_root():
    provider = _provider(_FakeClient())
    ladder = provider.get_contract_ladder("CL", max_contracts=12)
    for c in ladder.contracts:
        assert c.raw_symbol.startswith("CL")
        assert c.root_symbol == "CL"


def test_contract_ladder_disconnected_provider_is_honest_not_fetched():
    provider = DatabentoMarketDataProvider(
        config=DatabentoMarketDataConfig(),
        client_factory=lambda api_key: (_ for _ in ()).throw(DatabentoUnavailable("no key")),
    )
    ladder = provider.get_contract_ladder("CL")
    assert ladder.fetched is False
    assert ladder.contracts == ()


def test_term_structure_reuses_the_ladders_own_expiry_order():
    provider = _provider(_FakeClient())
    ts = provider.get_term_structure("CL", max_contracts=12)
    assert [p.raw_symbol for p in ts.points] == ["CLV6", "CLX6"]
    assert ts.points[0].price == 99.5  # settlement preferred over last
    assert ts.points[0].price_source == "settlement"
    assert ts.points[1].price == 95.0  # CLX6 has no settlement -> falls back to OHLCV close
    assert ts.points[1].price_source == "last"
    assert ts.curve_shape is CurveShape.BACKWARDATION  # 99.5 -> 95.0


def test_term_structure_insufficient_data_when_ladder_fetch_fails():
    provider = DatabentoMarketDataProvider(
        config=DatabentoMarketDataConfig(),
        client_factory=lambda api_key: (_ for _ in ()).throw(DatabentoUnavailable("no key")),
    )
    ts = provider.get_term_structure("CL")
    assert ts.curve_shape is CurveShape.INSUFFICIENT_DATA


def test_contract_ladder_cost_ceiling_is_respected():
    """A definition fetch whose estimated cost exceeds the configured
    ceiling must never proceed -- the ladder stays honestly not-fetched."""
    provider = DatabentoMarketDataProvider(
        config=DatabentoMarketDataConfig(max_auto_fetch_cost_usd=0.0),
        client_factory=lambda api_key: _FakeClient(cost=999.0),
    )
    ladder = provider.get_contract_ladder("CL")
    assert ladder.fetched is False
    assert ladder.capability is DatabentoCapability.ERROR

"""``DatabentoMarketDataProvider`` -- the primary read-only, MARKET-OBSERVATION
market-data provider (mission Part D). Distinct from, and deliberately never
imported by, the frozen scientific research pipeline:

* ``alpha_agent.data.databento_source`` (raw historical acquisition for
  research) is HOLDOUT-GUARDED -- its ``HistoricalRequest`` refuses any window
  reaching 2025-01-01 by construction. That guard is correct for research and
  would make it impossible to ever ask "what does the provider see right now"
  for market observation. This module is therefore a SEPARATE client
  construction and request path, on purpose -- it never imports
  ``alpha_agent.data.databento_source``, and nothing in
  ``alpha_agent.data`` / ``alpha_agent.discovery`` / ``alpha_agent.screening``
  / ``alpha_agent.validation`` / ``alpha_agent.registry`` may import THIS
  module (statically enforced by
  ``tests/python/test_release_market_plane_isolation.py``).
* It never writes to ``data/raw/`` or ``data/processed/``, never computes a
  feature, never runs a backtest, and never touches ``experiment_identity`` or
  ``ExperimentRegistry``.
* The one reuse in the OTHER, safe direction: contract economics (tick size /
  point value) are DERIVED via the same never-hardcoded
  ``alpha_agent.data.contract_economics.derive_contract_economics`` the
  scientific pipeline uses, from a definition record this module fetches
  itself -- a one-way "read a pure, stateless derivation function" import,
  not a coupling to the acquisition pipeline (CLAUDE.md: "never guessed,
  never hardcoded").

CAPABILITY, never assumed (mission section 13): every state
:class:`~alpha_agent.marketdata.databento_schemas.DatabentoCapability` can
report is derived from a real, free ``metadata`` endpoint response --
``LIVE``/``DELAYED`` are never claimed because this module never probes a
Live/streaming gateway (a separate, subscription-gated entitlement). Verified
against the real configured key: ``GLBX.MDP3`` is entitled, historical data
lags "now" by single-digit hours (``LATEST_AVAILABLE``), continuous-front-month
resolution and a bounded recent OHLCV/definition fetch each price at
effectively $0 (see ``scripts/databento_market_capability_probe.py``) -- this
module still calls ``metadata.get_cost()`` before every such fetch and refuses
to proceed past the configured ceiling (CLAUDE.md market-data rules 2-3).

COST SAFETY (mission section 16): every provider call is wrapped in a
process-lifetime TTL cache (:class:`_TTLCache`) so a Streamlit rerun storm
cannot multiply real network/cost activity; every OHLCV/definition fetch is
preceded by a real ``get_cost()`` call and bounded by
``DatabentoMarketDataConfig.max_auto_fetch_cost_usd`` /
``max_lookback_bars`` / ``max_lookback_days``.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from alpha_agent.marketdata.databento_config import (
    DATASET,
    DatabentoMarketDataConfig,
    load_databento_config,
)
from alpha_agent.marketdata.databento_schemas import (
    UNAVAILABLE_CAPABILITIES,
    ContractEconomicsView,
    ContractLadderResult,
    ContractResolution,
    CurveShape,
    DatabentoCapability,
    DatabentoHealth,
    FuturesContract,
    MarketSnapshot,
    OhlcvBar,
    RecentOhlcvResult,
    SessionSummary,
    TermStructurePoint,
    TermStructureResult,
)
from alpha_agent.marketdata.product_catalog import market_universe_roots

#: root -> Databento continuous front-month display proxy (mission section 15
#: -- a research/display convenience, never a tradable symbol). Market
#: Intelligence campaign Checkpoint B: covers the WHOLE catalogued MARKET
#: UNIVERSE (mission Part 4), not just the certified RESEARCH UNIVERSE --
#: this dict only ever states a symbol-naming CONVENTION, never a claim that
#: Databento actually resolves it; `DatabentoMarketDataProvider` still probes
#: and cost-checks every real call exactly as before, so a catalogued-but-
#: unresolvable root fails closed to an honest ERROR/empty result on its own,
#: per-root, the same way it always has (mission Part 4: "Do not blindly
#: hard-code contracts that Databento cannot actually resolve").
DISPLAY_SYMBOL: dict[str, str] = {root: f"{root}.v.0" for root in market_universe_roots()}

#: Requested timeframe -> (Databento base schema, resample rule | None).
#: Databento's OHLCV schemas are 1s/1m/1h/1d only; 5m/15m are honestly
#: labeled as a real-bar resample of 1m data, never a separate vendor grain.
_TIMEFRAME_SCHEMA: dict[str, tuple[str, str | None]] = {
    "1m": ("ohlcv-1m", None),
    "5m": ("ohlcv-1m", "5min"),
    "15m": ("ohlcv-1m", "15min"),
    "1h": ("ohlcv-1h", None),
    "1d": ("ohlcv-1d", None),
}

_BAR_SECONDS: dict[str, int] = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}


class DatabentoUnavailable(RuntimeError):
    """Client construction or a metadata call failed in a way the caller must
    treat as NOT_CONNECTED/ERROR -- never fabricate a value in its place."""


def _default_client_factory(api_key: str | None):
    try:
        import databento as db
    except ImportError as exc:
        raise DatabentoUnavailable("databento SDK not installed") from exc
    import os

    key = api_key or os.getenv("DATABENTO_API_KEY")
    if not key:
        raise DatabentoUnavailable("DATABENTO_API_KEY is not set")
    return db.Historical(key)


def _classify_error(exc: Exception) -> DatabentoCapability:
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return DatabentoCapability.RATE_LIMITED
    return DatabentoCapability.ERROR


def _session_from_bars(root: str, bars: tuple[OhlcvBar, ...]) -> SessionSummary | None:
    """Pure aggregation of an already-fetched real bar window into "today's"
    `SessionSummary` -- no network call. Shared by `get_session_summary` and
    `get_market_snapshot` so the latter never has to fetch a second,
    differently-sized OHLCV window just to compute this."""
    if not bars:
        return None
    latest_date = bars[-1].ts_event.date()
    todays = [b for b in bars if b.ts_event.date() == latest_date]
    if not todays:
        return None
    return SessionSummary(
        root_symbol=root, session_date=latest_date.isoformat(),
        session_open=todays[0].open, session_high=max(b.high for b in todays),
        session_low=min(b.low for b in todays), session_close=todays[-1].close,
        volume=sum(b.volume or 0.0 for b in todays), n_bars=len(todays),
    )


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


#: Databento GLBX.MDP3 `instrument_class` code for a real OUTRIGHT future --
#: verified against a real live `CL.FUT` definition fetch (Checkpoint C
#: research: 125 "F" rows vs. 4132 "S" [spread] rows in one bounded window).
#: Never any other code (calendar spreads, butterflies, ...) is treated as
#: a ladder-eligible contract (Section 5).
OUTRIGHT_FUTURE_INSTRUMENT_CLASS = "F"

#: Real Databento `StatType` enum values (statistics schema) this module
#: reads -- SETTLEMENT_PRICE and OPEN_INTEREST. Verified against a real live
#: CL statistics fetch. Never guessed from a different schema.
_STAT_TYPE_SETTLEMENT_PRICE = 3
_STAT_TYPE_OPEN_INTEREST = 9

#: INT64 "no value" sentinel Databento uses in the `statistics` schema's
#: `quantity` field on rows that carry a `price` instead (e.g. settlement) --
#: never mistaken for a real open-interest count.
_INT64_UNSET_SENTINEL = 9_223_372_036_854_775_807


def filter_outright_active_contracts(
    rows: list[dict[str, Any]], *, as_of: datetime, max_contracts: int,
) -> list[dict[str, Any]]:
    """Checkpoint C Section 5's deterministic contract-ladder filter over raw
    Databento `definition` (``stype_in="parent"``) rows, already de-duplicated
    to the LATEST definition update per raw symbol (a queried window can
    return several ``security_update_action`` rows for the same instrument).

    Kept: ``instrument_class == "F"`` (outright future, never a spread),
    ``expiration >= as_of``, and ``activation <= as_of`` WHEN activation is
    present (Section 5: "when activation exists" -- a row with no activation
    at all is never excluded on that basis alone). Sorted by real expiration
    timestamp (never raw-symbol lexicographic order, never a volume rank),
    ties broken by raw symbol for determinism, truncated to
    ``max_contracts``."""
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("instrument_class") != OUTRIGHT_FUTURE_INSTRUMENT_CLASS:
            continue
        raw_symbol = row.get("raw_symbol")
        if not raw_symbol:
            continue
        latest_by_symbol[raw_symbol] = row  # last occurrence in query order wins -> freshest update

    active: list[dict[str, Any]] = []
    for raw_symbol, row in latest_by_symbol.items():
        expiration = _parse_ts(row.get("expiration"))
        if expiration is None or expiration < as_of:
            continue
        activation = _parse_ts(row.get("activation"))
        if activation is not None and activation > as_of:
            continue
        active.append(row)

    active.sort(key=lambda r: (_parse_ts(r.get("expiration")), r.get("raw_symbol")))
    return active[:max_contracts]


def classify_curve_shape(
    prices: Sequence[float], *, tolerance_pct: float = 0.05,
) -> tuple[CurveShape, float | None, float | None, float | None]:
    """Checkpoint C Section 8 -- deterministic curve-shape classification
    over already expiry-ORDERED real prices (front to back). Returns
    ``(shape, curve_slope, front_second_spread, front_third_spread)``.

    A leg's spread counts as "flat" only within ``tolerance_pct`` percent of
    the front price (Section 8: "Do not classify tiny numerical noise as a
    meaningful curve regime") -- CONTANGO/BACKWARDATION require every
    non-flat leg to agree in sign; any real disagreement is MIXED, and fewer
    than 2 priced legs is the honest ``INSUFFICIENT_DATA`` outcome, never a
    forced guess."""
    if len(prices) < 2:
        return CurveShape.INSUFFICIENT_DATA, None, None, None

    front = prices[0]
    tol_abs = abs(front) * (tolerance_pct / 100.0)
    spreads = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    signs = [0 if abs(s) <= tol_abs else (1 if s > 0 else -1) for s in spreads]
    nonzero_signs = [s for s in signs if s != 0]

    if not nonzero_signs:
        shape = CurveShape.FLAT
    elif all(s > 0 for s in nonzero_signs):
        shape = CurveShape.CONTANGO
    elif all(s < 0 for s in nonzero_signs):
        shape = CurveShape.BACKWARDATION
    else:
        shape = CurveShape.MIXED

    curve_slope = (prices[-1] - prices[0]) / (len(prices) - 1)
    front_second = prices[1] - prices[0] if len(prices) >= 2 else None
    front_third = prices[2] - prices[0] if len(prices) >= 3 else None
    return shape, curve_slope, front_second, front_third


class _TTLCache:
    """Tiny in-process cache -- never persisted, never shared across
    processes. Exists solely to stop a Streamlit rerun (or a chatty UI) from
    multiplying real network calls within one TTL window."""

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any:
        hit = self._store.get(key)
        if hit is None:
            return None
        ts, value = hit
        if time.monotonic() - ts > self._ttl:
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


class DatabentoMarketDataProvider:
    """Read-only, OBSERVATION-plane market data. See module docstring for the
    full isolation boundary this class holds itself to."""

    def __init__(
        self,
        *,
        config: DatabentoMarketDataConfig | None = None,
        api_key: str | None = None,
        client_factory: Any = _default_client_factory,
    ):
        self._config = config or load_databento_config()
        self._api_key = api_key
        self._client_factory = client_factory
        self._client: Any = None
        self._cache = _TTLCache(self._config.ttl_seconds)
        # Section 34: contract DEFINITIONS get their own, much longer-lived
        # cache namespace -- separate from `self._cache` (prices/snapshots),
        # which stays on the short `ttl_seconds` window.
        self._long_cache = _TTLCache(self._config.contract_definitions_ttl_seconds)

    # -- client plumbing ----------------------------------------------------

    def _get_client(self):
        if self._client is None:
            self._client = self._client_factory(self._api_key)
        return self._client

    def clear_cache(self) -> None:
        """The explicit "Refresh Market Data" action (mission section 16) --
        never called automatically on a bare page render. Clears BOTH the
        short-lived price cache and the long-lived contract-definitions
        cache (Section 34) -- an explicit refresh always wins over TTL."""
        self._cache.clear()
        self._long_cache.clear()

    # -- health ---------------------------------------------------------

    def health(self) -> DatabentoHealth:
        cached = self._cache.get("health")
        if cached is not None:
            return cached
        now = datetime.now(UTC)
        if not self._config.enabled:
            health = DatabentoHealth(
                capability=DatabentoCapability.NOT_CONNECTED, dataset=DATASET, checked_at=now,
                detail="Databento market context disabled in configs/databento_market_data.yaml.",
            )
            self._cache.set("health", health)
            return health
        try:
            client = self._get_client()
        except DatabentoUnavailable as exc:
            health = DatabentoHealth(
                capability=DatabentoCapability.NOT_CONNECTED, dataset=DATASET, checked_at=now, detail=str(exc),
            )
            self._cache.set("health", health)
            return health
        try:
            rng = client.metadata.get_dataset_range(dataset=DATASET)
        except Exception as exc:  # noqa: BLE001 -- classify honestly, never crash the caller
            health = DatabentoHealth(
                capability=_classify_error(exc), dataset=DATASET, checked_at=now,
                detail=f"{type(exc).__name__}: {exc}",
            )
            self._cache.set("health", health)
            return health

        end_raw = rng.get("end") if isinstance(rng, dict) else None
        latest = _parse_ts(end_raw)
        if latest is None:
            health = DatabentoHealth(
                capability=DatabentoCapability.HISTORICAL_ONLY, dataset=DATASET, checked_at=now,
                detail="metadata.get_dataset_range returned no parseable end timestamp.",
            )
            self._cache.set("health", health)
            return health
        lag = (now - latest).total_seconds()
        capability = (
            DatabentoCapability.LATEST_AVAILABLE
            if lag <= self._config.latest_available_lag_seconds
            else DatabentoCapability.HISTORICAL_ONLY
        )
        health = DatabentoHealth(
            capability=capability, dataset=DATASET, checked_at=now, latest_available_ts=latest, lag_seconds=lag,
            detail=(
                "metadata.get_dataset_range (free call). No Live/streaming gateway was probed -- "
                "LIVE/DELAYED are never claimed without proving them."
            ),
        )
        self._cache.set("health", health)
        return health

    # -- contract resolution / economics -------------------------------

    def _definition_row(self, root: str) -> dict[str, Any] | None:
        """One definition record for `root`'s current continuous front month,
        cost-checked then fetched. Shared by `resolve_display_contract` and
        `get_contract_metadata` so a root is resolved and its definition
        fetched at most once per TTL window."""
        cache_key = f"definition:{root}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        health = self.health()
        if health.capability in (DatabentoCapability.NOT_CONNECTED, DatabentoCapability.RATE_LIMITED,
                                  DatabentoCapability.ERROR):
            return None
        display = DISPLAY_SYMBOL.get(root.upper())
        if display is None:
            return None
        try:
            client = self._get_client()
            today = health.latest_available_ts.date() if health.latest_available_ts else datetime.now(UTC).date()
            resolved = client.symbology.resolve(
                dataset=DATASET, symbols=[display], stype_in="continuous", stype_out="instrument_id",
                start_date=(today - timedelta(days=5)).isoformat(), end_date=today.isoformat(),
            )
            entries = (resolved.get("result") or {}).get(display) or []
            if not entries:
                self._cache.set(cache_key, None)
                return None
            instrument_id = entries[-1].get("s")
            if not instrument_id:
                self._cache.set(cache_key, None)
                return None

            # Widened past a single day (like the symbology-resolve window
            # above) so a weekend/holiday landing inside a naive 1-day window
            # never starves this of a real definition row -- verified live:
            # a request that falls entirely inside a weekend returns zero
            # rows, not an error, and must not be mistaken for "unresolvable".
            def_start = today - timedelta(days=5)
            cost = client.metadata.get_cost(
                dataset=DATASET, symbols=[str(instrument_id)], schema="definition", stype_in="instrument_id",
                start=def_start.isoformat(), end=today.isoformat(),
            )
            if cost > self._config.max_auto_fetch_cost_usd:
                self._cache.set(cache_key, None)
                return None
            store = client.timeseries.get_range(
                dataset=DATASET, symbols=[str(instrument_id)], schema="definition", stype_in="instrument_id",
                stype_out="instrument_id", start=def_start.isoformat(), end=today.isoformat(),
            )
            df = store.to_df()
            if len(df) == 0:
                self._cache.set(cache_key, None)
                return None
            row = df.iloc[-1].to_dict()
            row["instrument_id"] = instrument_id
        except Exception:  # noqa: BLE001 -- contract-resolution is a display convenience, never fatal
            self._cache.set(cache_key, None)
            return None
        self._cache.set(cache_key, row)
        return row

    def resolve_display_contract(self, root: str) -> ContractResolution | None:
        root = root.upper()
        display = DISPLAY_SYMBOL.get(root)
        if display is None:
            return None
        row = self._definition_row(root)
        now = datetime.now(UTC)
        if row is None:
            return ContractResolution(root_symbol=root, display_symbol=display, resolved_at=now)
        return ContractResolution(
            root_symbol=root, display_symbol=display,
            resolved_raw_symbol=row.get("raw_symbol"),
            resolved_instrument_id=row.get("instrument_id"),
            expiry=_parse_ts(row.get("expiration")),
            exchange=row.get("exchange"),
            resolved_at=now,
        )

    def get_contract_metadata(self, root: str) -> ContractEconomicsView | None:
        """Real, derived (never hardcoded) tick size / point value for
        `root`'s current front-month contract -- see module docstring for
        why importing the derivation function (not the acquisition
        pipeline) from `alpha_agent.data.contract_economics` is the one
        sanctioned reuse in that direction."""
        row = self._definition_row(root.upper())
        if row is None:
            return None
        from alpha_agent.data.contract_economics import EconomicsError, derive_contract_economics

        try:
            econ = derive_contract_economics(
                min_price_increment=row.get("min_price_increment"),
                min_price_increment_amount=row.get("min_price_increment_amount"),
                display_factor=row.get("display_factor"),
                unit_of_measure_qty=row.get("unit_of_measure_qty"),
                unit_of_measure=row.get("unit_of_measure") or "",
                main_fraction=row.get("main_fraction"),
                contract_multiplier=row.get("contract_multiplier"),
                allow_fractional=True,
            )
        except EconomicsError:
            return None
        return ContractEconomicsView(
            quote_tick_size=econ.quote_tick_size, contract_size=econ.contract_size,
            unit_of_measure=econ.unit_of_measure, point_value_usd=econ.point_value_usd,
            tick_value_usd=econ.tick_value_usd, quote_convention=econ.quote_convention.value,
            source=econ.source,
        )

    # -- OHLCV ------------------------------------------------------------

    def get_recent_ohlcv(
        self, root: str, timeframe: str = "1h", lookback_bars: int = 24,
    ) -> RecentOhlcvResult:
        root = root.upper()
        now = datetime.now(UTC)
        display = DISPLAY_SYMBOL.get(root)
        if display is None or timeframe not in _TIMEFRAME_SCHEMA:
            return RecentOhlcvResult(
                root_symbol=root, timeframe=timeframe, as_of=now,
                capability=DatabentoCapability.ERROR, fetched=False,
                detail=f"unsupported root/timeframe: {root}/{timeframe}",
            )
        lookback_bars = min(lookback_bars, self._config.max_lookback_bars)
        cache_key = f"ohlcv:{root}:{timeframe}:{lookback_bars}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        health = self.health()
        if health.capability in (DatabentoCapability.NOT_CONNECTED, DatabentoCapability.RATE_LIMITED,
                                  DatabentoCapability.ERROR):
            result = RecentOhlcvResult(
                root_symbol=root, timeframe=timeframe, as_of=now, capability=health.capability,
                fetched=False, detail=health.detail,
            )
            self._cache.set(cache_key, result)
            return result

        base_schema, resample_rule = _TIMEFRAME_SCHEMA[timeframe]
        wanted_seconds = _BAR_SECONDS[timeframe] * lookback_bars
        # Widen the request window generously (weekends/holidays produce no
        # bars) but never past the hard day ceiling -- defense in depth
        # independent of the cost estimate.
        window_days = min(self._config.max_lookback_days, max(2, (wanted_seconds // 86400) + 5))
        end = health.latest_available_ts or now
        start = end - timedelta(days=window_days)

        try:
            client = self._get_client()
            cost = client.metadata.get_cost(
                dataset=DATASET, symbols=[display], schema=base_schema, stype_in="continuous",
                start=start.isoformat(), end=end.isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            result = RecentOhlcvResult(
                root_symbol=root, timeframe=timeframe, as_of=now, capability=_classify_error(exc),
                fetched=False, detail=f"{type(exc).__name__}: {exc}",
            )
            self._cache.set(cache_key, result)
            return result

        if cost > self._config.max_auto_fetch_cost_usd:
            result = RecentOhlcvResult(
                root_symbol=root, timeframe=timeframe, as_of=now, capability=health.capability,
                estimated_cost_usd=float(cost), fetched=False,
                detail=(
                    f"Estimated cost ${cost:.4f} exceeds the auto-fetch ceiling "
                    f"${self._config.max_auto_fetch_cost_usd:.4f} -- explicit approval required."
                ),
            )
            self._cache.set(cache_key, result)
            return result

        try:
            store = client.timeseries.get_range(
                dataset=DATASET, symbols=[display], schema=base_schema, stype_in="continuous",
                stype_out="instrument_id", start=start.isoformat(), end=end.isoformat(),
            )
            df = store.to_df()
        except Exception as exc:  # noqa: BLE001
            result = RecentOhlcvResult(
                root_symbol=root, timeframe=timeframe, as_of=now, capability=_classify_error(exc),
                estimated_cost_usd=float(cost), fetched=False, detail=f"{type(exc).__name__}: {exc}",
            )
            self._cache.set(cache_key, result)
            return result

        if resample_rule is not None and len(df) > 0:
            df = df.resample(resample_rule).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(subset=["open"])

        df = df.tail(lookback_bars)
        bars = tuple(
            OhlcvBar(
                ts_event=ts if ts.tzinfo else ts.tz_localize(UTC),
                open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
                close=float(row["close"]), volume=float(row["volume"]) if "volume" in row else None,
            )
            for ts, row in df.iterrows()
        )
        result = RecentOhlcvResult(
            root_symbol=root, timeframe=timeframe, bars=bars, as_of=now, capability=health.capability,
            estimated_cost_usd=float(cost), fetched=True,
            detail=f"{len(bars)} real bar(s) from Databento {DATASET} (schema={base_schema}).",
        )
        self._cache.set(cache_key, result)
        return result

    # -- session / snapshot -------------------------------------------------

    def get_session_summary(self, root: str) -> SessionSummary | None:
        root = root.upper()
        ohlcv = self.get_recent_ohlcv(root, timeframe="1h", lookback_bars=48)
        if not ohlcv.fetched or not ohlcv.bars:
            return None
        return _session_from_bars(root, ohlcv.bars)

    def get_market_snapshot(self, root: str) -> MarketSnapshot | None:
        root = root.upper()
        contract = self.resolve_display_contract(root)
        if contract is None:
            return None
        health = self.health()
        now = datetime.now(UTC)
        if health.capability in (DatabentoCapability.NOT_CONNECTED, DatabentoCapability.RATE_LIMITED,
                                  DatabentoCapability.ERROR):
            return MarketSnapshot(root_symbol=root, contract=contract, as_of=now, capability=health.capability)

        ohlcv = self.get_recent_ohlcv(root, timeframe="1h", lookback_bars=72)
        if not ohlcv.fetched or not ohlcv.bars:
            return MarketSnapshot(root_symbol=root, contract=contract, as_of=now, capability=health.capability)

        # MARKET REALITY pass -- derive the session summary from THIS same
        # already-fetched 72-bar window instead of calling
        # `get_session_summary` (which would fetch its own, differently-sized
        # 48-bar window under a different cache key, doubling real network
        # round trips for every snapshot). A 72-bar 1h window is a superset
        # of "today" wherever a 48-bar window would be, so this changes no
        # returned value -- only how many real requests producing it costs.
        session = _session_from_bars(root, ohlcv.bars)
        last = ohlcv.bars[-1].close
        latest_date = ohlcv.bars[-1].ts_event.date()
        prior_bars = [b for b in ohlcv.bars if b.ts_event.date() < latest_date]
        prior_close = prior_bars[-1].close if prior_bars else None
        change = (last - prior_close) if (last is not None and prior_close is not None) else None
        change_pct = (change / prior_close * 100.0) if (change is not None and prior_close) else None
        freshness = (now - ohlcv.bars[-1].ts_event).total_seconds()

        return MarketSnapshot(
            root_symbol=root, contract=contract, last=last, change=change, change_pct=change_pct,
            session=session, prior_close=prior_close, as_of=ohlcv.bars[-1].ts_event,
            capability=health.capability, freshness_seconds=freshness,
        )

    # -- contract ladder / term structure (Checkpoint C) -----------------

    def _outright_definitions(self, root: str) -> list[dict[str, Any]] | None:
        """Real Databento `definition` rows (``stype_in="parent"``,
        ``symbols=[f"{root}.FUT"]``) -- Section 4's preferred workflow (root
        -> parent symbology -> ``{ROOT}.FUT`` -> definition schema). Cached
        under the LONG-TTL namespace (Section 34). Returns ``None`` on any
        failure/unavailability (never partial/fabricated rows); ``[]`` is a
        real, successful "nothing returned" outcome."""
        cache_key = f"definitions:{root}"
        cached = self._long_cache.get(cache_key)
        if cached is not None:
            return cached
        health = self.health()
        if health.capability in UNAVAILABLE_CAPABILITIES:
            return None
        try:
            client = self._get_client()
            end = health.latest_available_ts or datetime.now(UTC)
            start = end - timedelta(days=3)
            cost = client.metadata.get_cost(
                dataset=DATASET, symbols=[f"{root}.FUT"], schema="definition", stype_in="parent",
                start=start.isoformat(), end=end.isoformat(),
            )
            if cost > self._config.max_auto_fetch_cost_usd:
                return None
            store = client.timeseries.get_range(
                dataset=DATASET, symbols=[f"{root}.FUT"], schema="definition", stype_in="parent",
                stype_out="instrument_id", start=start.isoformat(), end=end.isoformat(),
            )
            df = store.to_df()
            rows = df.reset_index().to_dict("records") if len(df) else []
        except Exception:  # noqa: BLE001 -- a contract-ladder fetch is a display convenience, never fatal
            return None
        self._long_cache.set(cache_key, rows)
        return rows

    def _contract_prices(
        self, root: str, raw_symbols: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Real, SHORT-TTL-cached last close + volume per raw symbol, from
        ONE batched ``ohlcv-1d`` request (never one request per contract).
        Returns ``({}, {})`` on any failure -- callers render "N/A", never
        zero (Section 6)."""
        if not raw_symbols:
            return {}, {}
        cache_key = f"contract-prices:{root}:{','.join(raw_symbols)}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        health = self.health()
        empty: tuple[dict[str, float], dict[str, float]] = ({}, {})
        if health.capability in UNAVAILABLE_CAPABILITIES:
            self._cache.set(cache_key, empty)
            return empty
        try:
            client = self._get_client()
            end = health.latest_available_ts or datetime.now(UTC)
            start = end - timedelta(days=5)
            cost = client.metadata.get_cost(
                dataset=DATASET, symbols=list(raw_symbols), schema="ohlcv-1d", stype_in="raw_symbol",
                start=start.isoformat(), end=end.isoformat(),
            )
            if cost > self._config.max_auto_fetch_cost_usd:
                self._cache.set(cache_key, empty)
                return empty
            store = client.timeseries.get_range(
                dataset=DATASET, symbols=list(raw_symbols), schema="ohlcv-1d", stype_in="raw_symbol",
                stype_out="instrument_id", start=start.isoformat(), end=end.isoformat(),
            )
            df = store.to_df()
        except Exception:  # noqa: BLE001
            self._cache.set(cache_key, empty)
            return empty

        last_by: dict[str, float] = {}
        volume_by: dict[str, float] = {}
        if len(df) and "symbol" in df.columns:
            for sym, group in df.groupby("symbol"):
                latest = group.sort_index().iloc[-1]
                last_by[str(sym)] = float(latest["close"])
                if "volume" in latest and pd.notna(latest["volume"]):
                    volume_by[str(sym)] = float(latest["volume"])
        result = (last_by, volume_by)
        self._cache.set(cache_key, result)
        return result

    def _contract_statistics(self, root: str, raw_symbols: tuple[str, ...]) -> dict[str, dict[str, float]]:
        """Real, SHORT-TTL-cached settlement price + open interest per raw
        symbol, from ONE batched ``statistics`` request -- only where the
        entitlement actually supports it (Section 6). A symbol absent from
        the returned dict means "no such statistic was returned", rendered
        as N/A by the caller, never fabricated as ``0``."""
        if not raw_symbols:
            return {}
        cache_key = f"contract-stats:{root}:{','.join(raw_symbols)}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        health = self.health()
        if health.capability in UNAVAILABLE_CAPABILITIES:
            self._cache.set(cache_key, {})
            return {}
        try:
            client = self._get_client()
            end = health.latest_available_ts or datetime.now(UTC)
            start = end - timedelta(days=5)
            cost = client.metadata.get_cost(
                dataset=DATASET, symbols=list(raw_symbols), schema="statistics", stype_in="raw_symbol",
                start=start.isoformat(), end=end.isoformat(),
            )
            if cost > self._config.max_auto_fetch_cost_usd:
                self._cache.set(cache_key, {})
                return {}
            store = client.timeseries.get_range(
                dataset=DATASET, symbols=list(raw_symbols), schema="statistics", stype_in="raw_symbol",
                stype_out="instrument_id", start=start.isoformat(), end=end.isoformat(),
            )
            df = store.to_df()
        except Exception:  # noqa: BLE001
            self._cache.set(cache_key, {})
            return {}

        out: dict[str, dict[str, float]] = {}
        if len(df) and "symbol" in df.columns and "stat_type" in df.columns:
            for sym, group in df.groupby("symbol"):
                group = group.sort_index()
                entry: dict[str, float] = {}
                settlement_rows = group[group["stat_type"] == _STAT_TYPE_SETTLEMENT_PRICE]
                if len(settlement_rows):
                    price = settlement_rows.iloc[-1]["price"]
                    if pd.notna(price):
                        entry["settlement"] = float(price)
                oi_rows = group[group["stat_type"] == _STAT_TYPE_OPEN_INTEREST]
                if len(oi_rows):
                    qty = oi_rows.iloc[-1]["quantity"]
                    if pd.notna(qty) and 0 <= qty < _INT64_UNSET_SENTINEL:  # not NaN, not the sentinel
                        entry["open_interest"] = float(qty)
                if entry:
                    out[str(sym)] = entry
        self._cache.set(cache_key, out)
        return out

    def get_contract_ladder(self, root: str, *, max_contracts: int = 12) -> ContractLadderResult:
        """Checkpoint C -- the real, expiry-ordered outright-contract ladder
        for ``root`` (Sections 4-7). Never exposes a raw Databento
        DataFrame; everything here is a typed `FuturesContract`."""
        root = root.upper()
        now = datetime.now(UTC)
        if root not in DISPLAY_SYMBOL:
            return ContractLadderResult(
                root_symbol=root, as_of=now, capability=DatabentoCapability.ERROR, fetched=False,
                detail=f"{root!r} is not a catalogued futures root.",
            )
        health = self.health()
        if health.capability in UNAVAILABLE_CAPABILITIES:
            return ContractLadderResult(
                root_symbol=root, as_of=now, capability=health.capability, fetched=False, detail=health.detail,
            )

        raw_rows = self._outright_definitions(root)
        if raw_rows is None:
            return ContractLadderResult(
                root_symbol=root, as_of=now, capability=DatabentoCapability.ERROR, fetched=False,
                detail="Definition fetch failed or exceeded the auto-fetch cost ceiling.",
            )

        as_of_ref = health.latest_available_ts or now
        active_rows = filter_outright_active_contracts(raw_rows, as_of=as_of_ref, max_contracts=max_contracts)
        if not active_rows:
            return ContractLadderResult(
                root_symbol=root, as_of=now, capability=health.capability, fetched=True,
                detail=f"No active outright contracts resolved for {root} in the queried window.",
            )

        raw_symbols = tuple(str(row["raw_symbol"]) for row in active_rows)
        last_by, volume_by = self._contract_prices(root, raw_symbols)
        stats_by = self._contract_statistics(root, raw_symbols)
        display = self.resolve_display_contract(root)
        display_raw_symbol = display.resolved_raw_symbol if display else None

        from alpha_agent.data.contract_economics import usable_number

        contracts: list[FuturesContract] = []
        for i, row in enumerate(active_rows):
            raw_symbol = str(row["raw_symbol"])
            expiration = _parse_ts(row.get("expiration"))
            stats = stats_by.get(raw_symbol, {})
            contracts.append(
                FuturesContract(
                    root_symbol=root, raw_symbol=raw_symbol, instrument_id=int(row["instrument_id"]),
                    activation=_parse_ts(row.get("activation")), expiration=expiration,
                    min_price_increment=usable_number(row.get("min_price_increment")),
                    display_factor=usable_number(row.get("display_factor")),
                    venue=row.get("exchange"),
                    is_front_month=(i == 0), is_display_contract=(raw_symbol == display_raw_symbol),
                    last=last_by.get(raw_symbol), volume=volume_by.get(raw_symbol),
                    settlement_price=stats.get("settlement"), open_interest=stats.get("open_interest"),
                    days_to_expiry=(expiration.date() - as_of_ref.date()).days if expiration else None,
                )
            )

        return ContractLadderResult(
            root_symbol=root, contracts=tuple(contracts), as_of=now, capability=health.capability, fetched=True,
            detail=(
                f"{len(contracts)} real outright contract(s) from Databento definitions "
                f"(parent symbology {root}.FUT)."
            ),
        )

    def get_term_structure(
        self, root: str, *, max_contracts: int = 12, tolerance_pct: float = 0.05,
    ) -> TermStructureResult:
        """Checkpoint C Section 8 -- built from `get_contract_ladder`'s OWN
        real, expiry-ordered outright contracts, never from a continuous
        volume-ranked ``ROOT.v.N`` symbol."""
        root = root.upper()
        now = datetime.now(UTC)
        ladder = self.get_contract_ladder(root, max_contracts=max_contracts)
        if not ladder.fetched:
            return TermStructureResult(
                root_symbol=root, curve_shape=CurveShape.INSUFFICIENT_DATA, tolerance_pct=tolerance_pct,
                as_of=now, capability=ladder.capability, detail=ladder.detail,
            )

        points = []
        for c in ladder.contracts:
            if c.settlement_price is not None:
                price, source = c.settlement_price, "settlement"
            elif c.last is not None:
                price, source = c.last, "last"
            else:
                price, source = None, "unavailable"
            points.append(
                TermStructurePoint(
                    raw_symbol=c.raw_symbol, expiration=c.expiration, price=price, price_source=source,
                    volume=c.volume, open_interest=c.open_interest,
                )
            )

        priced_prices = [p.price for p in points if p.price is not None]
        shape, slope, front_second, front_third = classify_curve_shape(priced_prices, tolerance_pct=tolerance_pct)
        return TermStructureResult(
            root_symbol=root, points=tuple(points), front_second_spread=front_second,
            front_third_spread=front_third, curve_slope=slope, curve_shape=shape, tolerance_pct=tolerance_pct,
            as_of=now, capability=ladder.capability,
            detail=f"{len(priced_prices)} of {len(points)} contract(s) priced.",
        )


__all__ = [
    "DISPLAY_SYMBOL",
    "OUTRIGHT_FUTURE_INSTRUMENT_CLASS",
    "DatabentoMarketDataProvider",
    "DatabentoUnavailable",
    "classify_curve_shape",
    "filter_outright_active_contracts",
]

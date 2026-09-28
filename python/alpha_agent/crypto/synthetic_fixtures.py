"""Deterministic, network-free synthetic fixtures for the Phase 22 crypto scaffold.

Mirrors the discipline of ``tests/python/validation_fixtures.py`` (Phase 13):
every generator is seeded, reproducible, and explicitly labelled as a software
fixture, never a market claim. Nothing here calls a network, reads an API key,
or spends money -- CLAUDE.md "MONEY/NETWORK" and the Phase 22 review-gate
decision (synthetic-scaffold-only) both forbid it at this phase.

Covers every Phase 22 alternative-data surface (prompt 22): CME BTC/ETH
futures basis, perpetual funding + open interest, liquidations, exchange
flows, and on-chain MVRV/SOPR/active addresses -- each as a typed row list
(:mod:`alpha_agent.crypto.schemas`) carrying an explicit
``available_ts_ns`` (Phase 22.1: point-in-time availability, see that
module's docstring), plus one merged research-ready daily frame
(:func:`synthetic_crypto_bars`) that runs every surface through the causal
as-of join in :mod:`alpha_agent.crypto.dataset` before exposing it as a
feature column. A vendor-neutral typed adapter Protocol implementation exists
for each surface (:mod:`alpha_agent.crypto.vendors`), tagged
``provenance_role = SYNTHETIC``.

:func:`synthetic_contract_row` is a fabricated, clearly-labelled placeholder
CME contract spec (schema-shape only; never derived from a real Databento
definition record, never used for real accounting -- CLAUDE.md's
contract-economics rules govern the REAL derivation path in
:mod:`alpha_agent.data.contract_economics`, untouched by this module). It lets
the real, unmodified C++ Quant Core be exercised end-to-end over a
SYNTHETIC CME-futures-shaped execution fixture -- it is never claimed to be a
real CME BTC/ETH contract specification (Phase 22.1 presentation-semantics
fix; see docs/CRYPTO_ONCHAIN_EXTENSION.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_agent.crypto.dataset import (
    aggregate_daily_liquidations,
    build_causal_feature_columns,
    daily_series_from_exchange_flow_bars,
    series_from_alt_data_rows,
)
from alpha_agent.crypto.provenance import (
    CryptoDataProvenance,
    DataProvenanceRole,
    synthetic_provenance,
)
from alpha_agent.crypto.schemas import (
    CmeCryptoBasisBar,
    ExchangeFlowBar,
    LiquidationEvent,
    OnChainMetricBar,
    PerpFundingBar,
)
from alpha_agent.schemas.market_data import CONTRACT_COLUMNS
from alpha_agent.validation.fingerprint import fingerprint

DAY_NS = 86_400_000_000_000
_HOUR_NS = 3_600_000_000_000
_MINUTE_NS = 60_000_000_000

#: 2016-01-04 UTC, a Monday -- well before any real acquisition window this
#: repository has ever touched (2018-2024) and nowhere near the 2025 locked
#: holdout, so a synthetic-fixture timestamp can never be mistaken for one.
SYNTHETIC_BASE_NS = pd.Timestamp("2016-01-04T00:00:00Z").value

#: Fabricated, schema-shape-only placeholder. NOT a real CME contract
#: specification (CLAUDE.md's derived-point-value rule governs the real path;
#: this value is never run through it and never used for real accounting).
_PLACEHOLDER_TICK_SIZE = 5.0
_PLACEHOLDER_MULTIPLIER = 5.0

# ---------------------------------------------------------------------------
# Phase 22.1 -- publication delays. Each documents WHY that surface is (or
# isn't) immediately available; a delay of 0 means "known the instant it
# happens" (funding settlement, a live price), never a default/omission.
# ---------------------------------------------------------------------------
FUNDING_PUBLICATION_DELAY_NS = 0            # settles and is known immediately
BASIS_PUBLICATION_DELAY_NS = 0              # built from two live prices
LIQUIDATION_EVENT_PUBLICATION_DELAY_NS = 0  # a single print is near-real-time
#: A day's TOTAL liquidation notional cannot be known until the day is over --
#: the aggregate's own delay is applied on top of that in `dataset.py`
#: (`day_end + this`), never at the day's own start.
LIQUIDATION_AGGREGATE_PUBLICATION_DELAY_NS = 15 * _MINUTE_NS
#: On-chain analytics need block confirmations + provider processing.
ONCHAIN_PUBLICATION_DELAY_NS = 6 * _HOUR_NS
#: Exchange-flow attribution needs confirmation + address clustering.
EXCHANGE_FLOW_PUBLICATION_DELAY_NS = 4 * _HOUR_NS

GENERATOR_SCHEMA = "phase22-synthetic-crypto-generator/1"


def synthetic_crypto_generator_identity(
    *, root_symbol: str, n_days: int, start_ns: int, seed: int, start_price_usd: float,
) -> str:
    """A stable, content-based fingerprint over every parameter that
    deterministically controls :func:`synthetic_crypto_bars`'s output --
    generator schema version, seed, and every shape/delay parameter. There is
    no wall-clock input here to exclude: two calls with the same arguments
    always produce the same identity, on any machine, on any day (Phase 22.1
    "stable scientific/content identity")."""
    return fingerprint(
        "cryptogenerator1",
        {
            "schema": GENERATOR_SCHEMA,
            "root_symbol": root_symbol,
            "n_days": n_days,
            "start_ns": start_ns,
            "seed": seed,
            "start_price_usd": start_price_usd,
            "publication_delays_ns": {
                "funding": FUNDING_PUBLICATION_DELAY_NS,
                "basis": BASIS_PUBLICATION_DELAY_NS,
                "liquidation_event": LIQUIDATION_EVENT_PUBLICATION_DELAY_NS,
                "liquidation_aggregate": LIQUIDATION_AGGREGATE_PUBLICATION_DELAY_NS,
                "onchain": ONCHAIN_PUBLICATION_DELAY_NS,
                "exchange_flow": EXCHANGE_FLOW_PUBLICATION_DELAY_NS,
            },
        },
    )


def synthetic_contract_row(
    *, instrument_id: int, raw_symbol: str, root_symbol: str, expiration_ns: int
) -> pd.DataFrame:
    """A fabricated, single-row contract spec matching ``CONTRACT_COLUMNS`` --
    schema-shape only, so the real C++ engine can be exercised end-to-end over
    a SYNTHETIC CME-futures-shaped execution fixture. Never a real CME BTC/ETH
    contract specification; never routed through
    :mod:`alpha_agent.data.contract_economics`."""
    return pd.DataFrame(
        [
            {
                "instrument_id": instrument_id,
                "raw_symbol": raw_symbol,
                "root_symbol": root_symbol,
                "exchange": "XCME",
                "tick_size": _PLACEHOLDER_TICK_SIZE,
                "multiplier": _PLACEHOLDER_MULTIPLIER,
                "activation_ns": 1,
                "expiration_ns": expiration_ns,
                "first_notice_ns": "",
                "last_trade_ns": "",
            }
        ]
    )[list(CONTRACT_COLUMNS)]


@dataclass(frozen=True)
class SyntheticCryptoBundle:
    """The merged fixture :func:`synthetic_crypto_bars` returns."""

    bars: pd.DataFrame
    #: One :class:`CryptoDataProvenance` per column surface -- "ohlcv" plus
    #: every causally-joined feature column. Phase 22.1: never collapsed to a
    #: single record (see :mod:`alpha_agent.crypto.envelope`).
    provenance_by_series: dict[str, CryptoDataProvenance]
    generator_identity: str


#: Every causal feature column `synthetic_crypto_bars` can produce, covering
#: every Phase 22 alternative-data surface named in prompts/22. Not every
#: strategy uses every column -- the requirement is that the full typed
#: surface CAN feed the same FeatureSpec/StrategySpec pipeline, not that one
#: demonstration strategy does.
CRYPTO_FEATURE_COLUMNS: tuple[str, ...] = (
    "funding_rate", "open_interest_usd",
    "mvrv_ratio", "sopr_ratio", "active_addresses",
    "basis_pct",
    "long_liquidation_usd", "short_liquidation_usd", "liquidation_imbalance",
    "exchange_inflow_usd", "exchange_outflow_usd", "exchange_netflow_usd",
)


def synthetic_crypto_bars(
    *,
    root_symbol: str = "BTC",
    instrument_id: int,
    n_days: int,
    start_ns: int = SYNTHETIC_BASE_NS,
    seed: int = 22,
    start_price_usd: float = 40_000.0,
) -> SyntheticCryptoBundle:
    """One deterministic daily frame: synthetic CME-futures-shaped OHLCV plus
    every synthetic crypto/on-chain feature column, each produced by its own
    typed row generator and then CAUSALLY AS-OF JOINED
    (:mod:`alpha_agent.crypto.dataset`) onto the OHLCV timeline -- a column's
    value at bar ``T`` only ever reflects an observation whose
    ``available_ts_ns <= T``. No forward-fill of not-yet-available data, and
    every row is freshly generated with no market-data gap to paper over."""
    rng = np.random.default_rng(seed)
    ts = start_ns + np.arange(n_days, dtype="int64") * DAY_NS

    # -- OHLCV: a bounded random walk, always positive. This is the base
    # execution timeline the C++ boundary reads (BOUNDARY_BAR_COLUMNS) and is
    # never itself subject to a publication delay -- a bar's own close is
    # definitionally available at its own ts_event_ns.
    log_ret = rng.normal(0.0002, 0.02, size=n_days)
    close = start_price_usd * np.exp(np.cumsum(log_ret))
    openp = np.empty(n_days)
    openp[0] = start_price_usd
    openp[1:] = close[:-1]
    spread = close * 0.003
    high = np.maximum(openp, close) + spread
    low = np.minimum(openp, close) - spread
    volume = rng.integers(500, 5000, size=n_days)

    funding_rows = synthetic_perp_funding_bars(
        symbol=f"{root_symbol}-PERP", n=n_days, start_ns=start_ns, seed=seed,
    )
    basis_rows = synthetic_cme_crypto_basis_bars(
        root_symbol=root_symbol, n=n_days, start_ns=start_ns, seed=seed,
        start_price_usd=start_price_usd,
    )
    onchain_rows = synthetic_onchain_metric_bars(
        asset=root_symbol, n=n_days, start_ns=start_ns, seed=seed,
    )
    liquidation_events = synthetic_liquidation_events(
        symbol=f"{root_symbol}-PERP", n_days=n_days, start_ns=start_ns, seed=seed,
    )
    flow_rows = synthetic_exchange_flow_bars(
        asset=root_symbol, n=n_days, start_ns=start_ns, seed=seed,
    )

    series_by_column = {
        "funding_rate": series_from_alt_data_rows(funding_rows, value_fn=lambda r: r.funding_rate),
        "open_interest_usd": series_from_alt_data_rows(
            funding_rows, value_fn=lambda r: r.open_interest_usd
        ),
        "basis_pct": series_from_alt_data_rows(basis_rows, value_fn=lambda r: r.basis_pct),
        "mvrv_ratio": series_from_alt_data_rows(onchain_rows, value_fn=lambda r: r.mvrv_ratio),
        "sopr_ratio": series_from_alt_data_rows(onchain_rows, value_fn=lambda r: r.sopr_ratio),
        "active_addresses": series_from_alt_data_rows(
            onchain_rows, value_fn=lambda r: r.active_addresses
        ),
        **aggregate_daily_liquidations(
            liquidation_events, start_ns=start_ns, day_ns=DAY_NS, n_days=n_days,
            publication_delay_ns=LIQUIDATION_AGGREGATE_PUBLICATION_DELAY_NS,
        ),
        **daily_series_from_exchange_flow_bars(flow_rows),
    }
    feature_cols = build_causal_feature_columns(ts, series_by_column)

    bars = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts_event_ns": ts, "instrument_id": instrument_id,
                    "open": openp, "high": high, "low": low, "close": close, "volume": volume,
                }
            ),
            feature_cols,
        ],
        axis=1,
    )

    provenance_by_series: dict[str, CryptoDataProvenance] = {
        "ohlcv": synthetic_provenance("ohlcv", seed=seed, notes=f"root={root_symbol}"),
        "funding_rate": funding_rows[0].provenance,
        "open_interest_usd": funding_rows[0].provenance,
        "basis_pct": basis_rows[0].provenance,
        "mvrv_ratio": onchain_rows[0].provenance,
        "sopr_ratio": onchain_rows[0].provenance,
        "active_addresses": onchain_rows[0].provenance,
        "long_liquidation_usd": synthetic_provenance(
            "liquidation_aggregate", seed=seed, notes="derived: sum(notional_usd where side=LONG) per day"
        ),
        "short_liquidation_usd": synthetic_provenance(
            "liquidation_aggregate", seed=seed, notes="derived: sum(notional_usd where side=SHORT) per day"
        ),
        "liquidation_imbalance": synthetic_provenance(
            "liquidation_aggregate", seed=seed, notes="derived: long_liquidation_usd - short_liquidation_usd"
        ),
        "exchange_inflow_usd": flow_rows[0].provenance,
        "exchange_outflow_usd": flow_rows[0].provenance,
        "exchange_netflow_usd": synthetic_provenance(
            "exchange_flows", seed=seed, notes="derived: exchange_inflow_usd - exchange_outflow_usd"
        ),
    }
    gen_id = synthetic_crypto_generator_identity(
        root_symbol=root_symbol, n_days=n_days, start_ns=start_ns, seed=seed,
        start_price_usd=start_price_usd,
    )
    return SyntheticCryptoBundle(
        bars=bars, provenance_by_series=provenance_by_series, generator_identity=gen_id,
    )


# ---------------------------------------------------------------------------
# typed single-series builders (back the Protocol adapters in .vendors)
# ---------------------------------------------------------------------------
def synthetic_perp_funding_bars(
    *, symbol: str, venue: str = "SYNTH-PERP", n: int, start_ns: int = SYNTHETIC_BASE_NS,
    seed: int = 22,
) -> list[PerpFundingBar]:
    rng = np.random.default_rng(seed)
    prov = synthetic_provenance("perp_funding", seed=seed, notes=f"symbol={symbol}")
    rate = np.zeros(n)
    for i in range(1, n):
        rate[i] = 0.85 * rate[i - 1] + rng.normal(0.0, 0.0006)
    oi = 5.0e8 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    mark = 40_000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, size=n)))
    return [
        PerpFundingBar(
            ts_event_ns=int(start_ns + i * DAY_NS),
            available_ts_ns=int(start_ns + i * DAY_NS) + FUNDING_PUBLICATION_DELAY_NS,
            venue=venue, symbol=symbol,
            funding_rate=float(rate[i]), open_interest_usd=float(oi[i]),
            mark_price_usd=float(mark[i]), provenance=prov,
        )
        for i in range(n)
    ]


def synthetic_liquidation_events(
    *, symbol: str, venue: str = "SYNTH-PERP", n_days: int, start_ns: int = SYNTHETIC_BASE_NS,
    seed: int = 22, max_events_per_day: int = 6,
) -> list[LiquidationEvent]:
    """Zero or more individual liquidation prints per calendar day, each at a
    random intraday timestamp -- NOT one homogenized event per day. Each
    individual print is near-real-time
    (``available_ts_ns == ts_event_ns + LIQUIDATION_EVENT_PUBLICATION_DELAY_NS``);
    the DAY'S TOTAL is a separate, later-available aggregate built by
    :func:`alpha_agent.crypto.dataset.aggregate_daily_liquidations`."""
    rng = np.random.default_rng(seed)
    prov = synthetic_provenance("liquidations", seed=seed, notes=f"symbol={symbol}")
    rows: list[LiquidationEvent] = []
    for day in range(n_days):
        day_start = start_ns + day * DAY_NS
        n_events = int(rng.integers(0, max_events_per_day + 1))
        if n_events == 0:
            continue
        offsets = np.sort(rng.integers(0, DAY_NS, size=n_events))
        sides = rng.choice(["LONG", "SHORT"], size=n_events)
        notional = rng.exponential(2.0e5, size=n_events)
        for j in range(n_events):
            ts = int(day_start + offsets[j])
            rows.append(
                LiquidationEvent(
                    ts_event_ns=ts,
                    available_ts_ns=ts + LIQUIDATION_EVENT_PUBLICATION_DELAY_NS,
                    venue=venue, symbol=symbol,
                    side_liquidated=str(sides[j]), notional_usd=float(notional[j]), provenance=prov,
                )
            )
    return rows


def synthetic_onchain_metric_bars(
    *, asset: str, n: int, start_ns: int = SYNTHETIC_BASE_NS, seed: int = 22,
) -> list[OnChainMetricBar]:
    """One daily on-chain observation per day. Modelled as a full-day summary
    (MVRV/SOPR/active-address providers typically report on a >= 1-day lag) --
    NOT available until the day it describes has fully elapsed PLUS
    ``ONCHAIN_PUBLICATION_DELAY_NS``, never at that day's own start (prompt
    22: "an on-chain value cannot be treated as available at the beginning of
    that same day")."""
    rng = np.random.default_rng(seed)
    prov = synthetic_provenance("onchain_metrics", seed=seed, notes=f"asset={asset}")
    x = np.arange(n)
    mvrv = 1.8 + 0.6 * np.sin(x / 45.0) + rng.normal(0.0, 0.05, size=n)
    sopr = 1.0 + 0.05 * np.sin(x / 30.0 + 1.0) + rng.normal(0.0, 0.01, size=n)
    active_addr = np.clip(9.0e5 * (1.0 + 0.1 * np.sin(x / 60.0)) + rng.normal(0.0, 1.5e4, size=n), 1.0, None)
    rows = []
    for i in range(n):
        day_start = int(start_ns + i * DAY_NS)
        day_end = day_start + DAY_NS
        rows.append(
            OnChainMetricBar(
                ts_event_ns=day_start,
                available_ts_ns=day_end + ONCHAIN_PUBLICATION_DELAY_NS,
                asset=asset, mvrv_ratio=float(mvrv[i]),
                sopr_ratio=float(sopr[i]), active_addresses=float(active_addr[i]), provenance=prov,
            )
        )
    return rows


def synthetic_exchange_flow_bars(
    *, asset: str, n: int, start_ns: int = SYNTHETIC_BASE_NS, seed: int = 22,
) -> list[ExchangeFlowBar]:
    """One daily net exchange-flow observation per day, NOT available until
    ``EXCHANGE_FLOW_PUBLICATION_DELAY_NS`` after the day it describes."""
    rng = np.random.default_rng(seed)
    prov = synthetic_provenance("exchange_flows", seed=seed, notes=f"asset={asset}")
    inflow = np.abs(rng.normal(5.0e6, 2.0e6, size=n))
    outflow = np.abs(rng.normal(5.0e6, 2.0e6, size=n))
    rows = []
    for i in range(n):
        day_start = int(start_ns + i * DAY_NS)
        day_end = day_start + DAY_NS
        rows.append(
            ExchangeFlowBar(
                ts_event_ns=day_start,
                available_ts_ns=day_end + EXCHANGE_FLOW_PUBLICATION_DELAY_NS,
                asset=asset, inflow_usd=float(inflow[i]), outflow_usd=float(outflow[i]),
                provenance=prov,
            )
        )
    return rows


def synthetic_cme_crypto_basis_bars(
    *, root_symbol: str, n: int, start_ns: int = SYNTHETIC_BASE_NS, seed: int = 22,
    start_price_usd: float = 40_000.0,
) -> list[CmeCryptoBasisBar]:
    rng = np.random.default_rng(seed)
    prov = synthetic_provenance("cme_crypto_basis", seed=seed, notes=f"root={root_symbol}")
    fut = start_price_usd * np.exp(np.cumsum(rng.normal(0.0002, 0.02, size=n)))
    basis_pct = np.zeros(n)
    for i in range(1, n):
        basis_pct[i] = 0.9 * basis_pct[i - 1] + rng.normal(0.0, 0.15)
    spot = fut / (1.0 + basis_pct / 100.0)
    return [
        CmeCryptoBasisBar(
            ts_event_ns=int(start_ns + i * DAY_NS),
            available_ts_ns=int(start_ns + i * DAY_NS) + BASIS_PUBLICATION_DELAY_NS,
            root_symbol=root_symbol,
            futures_price_usd=float(fut[i]), spot_index_price_usd=float(spot[i]),
            basis_pct=float(basis_pct[i]), provenance=prov,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# synthetic Protocol implementations (alpha_agent.crypto.vendors)
# ---------------------------------------------------------------------------
class SyntheticPerpFundingAdapter:
    provenance_role = DataProvenanceRole.SYNTHETIC

    def __init__(self, *, seed: int = 22) -> None:
        self._seed = seed

    def fetch_funding(self, *, symbol: str, start_ts_ns: int, end_ts_ns: int) -> list[PerpFundingBar]:
        n = max(1, int((end_ts_ns - start_ts_ns) // DAY_NS))
        return synthetic_perp_funding_bars(symbol=symbol, n=n, start_ns=start_ts_ns, seed=self._seed)


class SyntheticLiquidationAdapter:
    provenance_role = DataProvenanceRole.SYNTHETIC

    def __init__(self, *, seed: int = 22) -> None:
        self._seed = seed

    def fetch_liquidations(
        self, *, symbol: str, start_ts_ns: int, end_ts_ns: int
    ) -> list[LiquidationEvent]:
        n_days = max(1, int((end_ts_ns - start_ts_ns) // DAY_NS))
        return synthetic_liquidation_events(
            symbol=symbol, n_days=n_days, start_ns=start_ts_ns, seed=self._seed
        )


class SyntheticOnChainAdapter:
    provenance_role = DataProvenanceRole.SYNTHETIC

    def __init__(self, *, seed: int = 22) -> None:
        self._seed = seed

    def fetch_metrics(self, *, asset: str, start_ts_ns: int, end_ts_ns: int) -> list[OnChainMetricBar]:
        n = max(1, int((end_ts_ns - start_ts_ns) // DAY_NS))
        return synthetic_onchain_metric_bars(asset=asset, n=n, start_ns=start_ts_ns, seed=self._seed)


class SyntheticExchangeFlowAdapter:
    provenance_role = DataProvenanceRole.SYNTHETIC

    def __init__(self, *, seed: int = 22) -> None:
        self._seed = seed

    def fetch_flows(self, *, asset: str, start_ts_ns: int, end_ts_ns: int) -> list[ExchangeFlowBar]:
        n = max(1, int((end_ts_ns - start_ts_ns) // DAY_NS))
        return synthetic_exchange_flow_bars(asset=asset, n=n, start_ns=start_ts_ns, seed=self._seed)


class SyntheticCmeCryptoBasisAdapter:
    provenance_role = DataProvenanceRole.SYNTHETIC

    def __init__(self, *, seed: int = 22) -> None:
        self._seed = seed

    def fetch_basis(
        self, *, root_symbol: str, start_ts_ns: int, end_ts_ns: int
    ) -> list[CmeCryptoBasisBar]:
        n = max(1, int((end_ts_ns - start_ts_ns) // DAY_NS))
        return synthetic_cme_crypto_basis_bars(
            root_symbol=root_symbol, n=n, start_ns=start_ts_ns, seed=self._seed
        )

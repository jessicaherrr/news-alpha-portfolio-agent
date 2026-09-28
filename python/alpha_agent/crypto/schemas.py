"""Typed row schemas for the Phase 22 crypto/on-chain data surface.

These are the payload types a real vendor adapter would eventually populate
(prompts/22: CME BTC/ETH futures basis, perpetual funding/open-interest/
liquidations, exchange flows, on-chain MVRV/SOPR/active-address style
features). Only the deterministic synthetic generator in
:mod:`alpha_agent.crypto.synthetic_fixtures` populates them today -- every row
carries a :class:`~alpha_agent.crypto.provenance.CryptoDataProvenance` that is
checked ``SYNTHETIC`` at construction time.

Kept structurally parallel to (but separate from) the Phase 02.5 market-data
schemas: CLAUDE.md's "prefer explicit typed schemas over free-form text" and
"do not mix crypto venue assumptions with CME futures execution semantics" --
these types never enter the C++ boundary directly, only pre-merged, causally
as-of-joined numeric feature columns do (:mod:`alpha_agent.crypto.dataset`).

Phase 22.1 -- point-in-time availability (every record, not just an
afterthought): :class:`_AltDataRecord` is the shared base every alternative-
data row subclasses. It carries BOTH:

* ``ts_event_ns`` -- when the thing happened / was observed;
* ``available_ts_ns`` -- when the value became knowable to a strategy.

with the invariant ``available_ts_ns >= ts_event_ns`` (equality is allowed for
a genuinely immediate synthetic event -- e.g. a funding-rate settlement is
known the instant it settles). A vendor that only reports one timestamp is not
yet a valid Phase 22 data surface: collapsing the two onto each other is
exactly the "backdated full-day aggregate available at the start of its own
day" mistake this field exists to make impossible to express.
:mod:`alpha_agent.crypto.dataset` is the ONLY place that turns a list of these
rows into a feature column, and it enforces
``available_ts_ns <= decision_ts_ns`` on every value it emits.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from alpha_agent.crypto.provenance import CryptoDataProvenance


class _AltDataRecord(BaseModel):
    """Shared shape + causal invariant for every Phase 22 alternative-data row."""

    model_config = {"frozen": True, "extra": "forbid"}

    ts_event_ns: int = Field(gt=0)
    #: When this value became knowable to a strategy -- NOT when it happened.
    #: Must be >= ts_event_ns (equality allowed for a genuinely immediate
    #: synthetic event). See module docstring.
    available_ts_ns: int = Field(gt=0)
    provenance: CryptoDataProvenance

    @model_validator(mode="after")
    def _availability_not_before_event(self) -> _AltDataRecord:
        if self.available_ts_ns < self.ts_event_ns:
            raise ValueError(
                f"available_ts_ns ({self.available_ts_ns}) is before ts_event_ns "
                f"({self.ts_event_ns}) -- a value cannot become available before "
                "it is observed"
            )
        return self


class PerpFundingBar(_AltDataRecord):
    """One funding-interval observation for a perpetual future. Funding
    settles and is known immediately (``available_ts_ns == ts_event_ns`` in
    the synthetic generator)."""

    venue: str
    symbol: str                          # e.g. "BTC-PERP"
    funding_rate: float                  # per-interval rate, e.g. 0.0001 = 1bp
    open_interest_usd: float = Field(ge=0)
    mark_price_usd: float = Field(gt=0)


class LiquidationEvent(_AltDataRecord):
    """One individual liquidation print for a perpetual future. A single event
    is near-real-time (``available_ts_ns == ts_event_ns`` in the synthetic
    generator); the DAILY AGGREGATE built from many events
    (:func:`alpha_agent.crypto.dataset.aggregate_daily_liquidations`) is not
    available until well after the day closes -- see that function's
    docstring."""

    venue: str
    symbol: str
    side_liquidated: str                 # "LONG" | "SHORT"
    notional_usd: float = Field(ge=0)


class OnChainMetricBar(_AltDataRecord):
    """One daily on-chain metric observation for a base asset. On-chain
    analytics require block confirmation + provider processing, so the
    synthetic generator gives this a same-day-or-later
    ``available_ts_ns`` -- never available at the start of the day it
    describes."""

    asset: str                           # "BTC" | "ETH"
    mvrv_ratio: float
    sopr_ratio: float
    active_addresses: float = Field(ge=0)


class ExchangeFlowBar(_AltDataRecord):
    """One daily net exchange-flow observation for a base asset. ``inflow_usd``
    is value moved ONTO tracked exchange addresses (often read as latent
    selling pressure); ``outflow_usd`` is value moved OFF exchange addresses
    (often read as accumulation / custody). Like on-chain metrics, attributing
    a day's flows requires confirmation + address-clustering processing, so
    this is never available at the start of the day it describes."""

    asset: str                           # "BTC" | "ETH"
    inflow_usd: float = Field(ge=0)
    outflow_usd: float = Field(ge=0)


class CmeCryptoBasisBar(_AltDataRecord):
    """CME-listed crypto future vs. spot/index basis for one bar. Built from
    two live prices, so it is immediate
    (``available_ts_ns == ts_event_ns`` in the synthetic generator).

    ``futures_price_usd`` would come from the EXISTING Databento GLBX.MDP3
    adapter unchanged (CME BTC/ETH futures already trade on ``GLBX.MDP3`` --
    no new vendor); ``spot_index_price_usd`` is the new, still-unconnected
    input this bar type exists to carry.
    """

    root_symbol: str                     # CME BTC/ETH-shaped root, e.g. "BTC" / "ETH"
    futures_price_usd: float = Field(gt=0)
    spot_index_price_usd: float = Field(gt=0)
    basis_pct: float


def _rows_to_frame(rows: list[BaseModel]) -> pd.DataFrame:
    if not rows:
        raise ValueError("at least one row is required")
    df = pd.DataFrame([r.model_dump(mode="json", exclude={"provenance"}) for r in rows])
    return df.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)


def funding_frame(rows: list[PerpFundingBar]) -> pd.DataFrame:
    return _rows_to_frame(rows)


def liquidation_frame(rows: list[LiquidationEvent]) -> pd.DataFrame:
    return _rows_to_frame(rows)


def onchain_frame(rows: list[OnChainMetricBar]) -> pd.DataFrame:
    return _rows_to_frame(rows)


def exchange_flow_frame(rows: list[ExchangeFlowBar]) -> pd.DataFrame:
    return _rows_to_frame(rows)


def basis_frame(rows: list[CmeCryptoBasisBar]) -> pd.DataFrame:
    return _rows_to_frame(rows)

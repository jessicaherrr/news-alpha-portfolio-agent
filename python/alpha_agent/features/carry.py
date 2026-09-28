"""Carry / term-structure interface (section 11).

Carry needs **individual raw contracts / curve observations**, never a
back-adjusted continuous series (back-adjustment removes exactly the contract
basis that carry measures). This module defines the typed inputs and the
point-in-time carry math; it does **not** fabricate production carry values when
the dataset lacks overlapping curve observations -- ``build_curve_observations``
returns what it is given and flags an insufficient curve.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from alpha_agent.schemas.market_data import (
    PriceDomain,
    is_normalized_price,
    is_tradable_contract_symbol,
)


class CarryError(ValueError):
    """A carry computation was fed the wrong price domain or an invalid curve."""


class ContractQuote(BaseModel):
    """One real raw contract's price observation at a single timestamp."""

    model_config = {"frozen": True}

    instrument_id: int = Field(gt=0)
    raw_symbol: str = Field(min_length=1)
    expiration_ns: int = Field(gt=0)
    price: float
    price_field: str = "close"

    @model_validator(mode="after")
    def _check(self) -> ContractQuote:
        if not is_tradable_contract_symbol(self.raw_symbol):
            raise CarryError(f"carry quote needs a real contract, got {self.raw_symbol!r}")
        if not is_normalized_price(self.price):
            raise CarryError(f"carry quote price {self.price!r} is not a normalized value")
        return self


class CurveObservation(BaseModel):
    """The futures curve for one root at one timestamp: >= 2 contracts ordered by
    expiration. Prices are raw-contract prices (signed-safe)."""

    model_config = {"frozen": True}

    ts_event_ns: int = Field(gt=0)
    root_symbol: str = Field(min_length=1)
    quotes: tuple[ContractQuote, ...]

    @model_validator(mode="after")
    def _check(self) -> CurveObservation:
        if len(self.quotes) < 2:
            raise CarryError("a curve observation needs at least a front and a next contract")
        exps = [q.expiration_ns for q in self.quotes]
        if exps != sorted(exps):
            raise CarryError("curve quotes must be ordered by ascending expiration")
        if len(set(exps)) != len(exps):
            raise CarryError("curve quotes must have distinct expirations")
        return self

    @property
    def front(self) -> ContractQuote:
        return self.quotes[0]

    @property
    def nxt(self) -> ContractQuote:
        return self.quotes[1]

    @property
    def maturity_spacing_days(self) -> float:
        return (self.nxt.expiration_ns - self.front.expiration_ns) / 86_400_000_000_000


class CarrySnapshot(BaseModel):
    ts_event_ns: int
    root_symbol: str
    front_raw_symbol: str
    next_raw_symbol: str
    front_price: float
    next_price: float
    maturity_spacing_days: float
    calendar_spread: float                    # next - front (signed-safe)
    annualized_carry: float | None            # None when undefined (see notes)
    normalized_carry: float | None


def assert_carry_domain(domain: PriceDomain | str) -> None:
    d = PriceDomain(domain) if not isinstance(domain, PriceDomain) else domain
    if d is not PriceDomain.RAW_CONTRACT:
        raise CarryError(
            f"carry must be derived from raw individual contracts, not price_domain={d.value}"
        )


def calendar_spread(front_price: float, next_price: float) -> float:
    """next - front. Well-defined for zero / negative prices."""
    return float(next_price) - float(front_price)


def annualized_carry(
    front_price: float, next_price: float, maturity_spacing_days: float,
    *, day_count_year: float = 365.0,
) -> float | None:
    """Annualized roll yield of being long the front contract:

        ((front - next) / next) * (year / spacing)

    Positive => backwardation (front above next) => positive carry for a long.
    Returns ``None`` (explicit missing) when ``next`` <= 0 (the percentage is not
    meaningful) or the spacing is non-positive.
    """
    if next_price <= 0.0 or maturity_spacing_days <= 0.0:
        return None
    raw = (front_price - next_price) / next_price
    ann = raw * (day_count_year / maturity_spacing_days)
    return float(ann) if math.isfinite(ann) else None


def normalized_carry(spread: float, reference: float | None) -> float | None:
    """calendar spread divided by a positive reference scale (e.g. rolling vol
    or |front price|). ``None`` when the reference is missing or ~0."""
    if reference is None or abs(reference) <= 1e-12 or not math.isfinite(reference):
        return None
    return float(spread / reference)


def build_curve_observations(
    quotes_by_ts: dict[int, Iterable[ContractQuote]],
    *,
    root_symbol: str,
    price_domain: PriceDomain | str = PriceDomain.RAW_CONTRACT,
) -> tuple[list[CurveObservation], list[str]]:
    """Assemble per-timestamp curve observations from raw-contract quotes.

    Returns ``(observations, warnings)``. A timestamp with fewer than two
    contracts is skipped with a warning -- no synthetic contract is invented.
    """
    assert_carry_domain(price_domain)
    obs: list[CurveObservation] = []
    warnings: list[str] = []
    for ts in sorted(quotes_by_ts):
        qs = sorted(quotes_by_ts[ts], key=lambda q: q.expiration_ns)
        if len(qs) < 2:
            warnings.append(f"ts {ts}: only {len(qs)} contract(s) on the curve -- skipped")
            continue
        obs.append(CurveObservation(ts_event_ns=ts, root_symbol=root_symbol, quotes=tuple(qs)))
    if not obs:
        warnings.append("no timestamp had >= 2 overlapping contracts -- carry curve is empty")
    return obs, warnings


def carry_snapshot(obs: CurveObservation, *, reference: float | None = None) -> CarrySnapshot:
    spread = calendar_spread(obs.front.price, obs.nxt.price)
    ann = annualized_carry(obs.front.price, obs.nxt.price, obs.maturity_spacing_days)
    return CarrySnapshot(
        ts_event_ns=obs.ts_event_ns,
        root_symbol=obs.root_symbol,
        front_raw_symbol=obs.front.raw_symbol,
        next_raw_symbol=obs.nxt.raw_symbol,
        front_price=obs.front.price,
        next_price=obs.nxt.price,
        maturity_spacing_days=obs.maturity_spacing_days,
        calendar_spread=spread,
        annualized_carry=ann,
        normalized_carry=normalized_carry(spread, reference if reference is not None
                                          else abs(obs.nxt.price) or None),
    )


def carry_frame(observations: list[CurveObservation]) -> pd.DataFrame:
    """Point-in-time carry table: each row uses only that timestamp's curve."""
    rows = [carry_snapshot(o).model_dump() for o in observations]
    cols = list(CarrySnapshot.model_fields)
    return pd.DataFrame(rows, columns=cols)


# --- synthetic test helper -------------------------------------------------

def synthetic_curve(
    ts_event_ns: int,
    *,
    root_symbol: str = "CL",
    front_price: float,
    next_price: float,
    spacing_days: int = 30,
    front_expiration_ns: int | None = None,
) -> CurveObservation:
    """Two hand-specified contracts on one root -- for deterministic carry tests
    only. Not a production data path."""
    fe = front_expiration_ns or (ts_event_ns + 5 * 86_400_000_000_000)
    ne = fe + spacing_days * 86_400_000_000_000
    return CurveObservation(
        ts_event_ns=ts_event_ns,
        root_symbol=root_symbol,
        quotes=(
            ContractQuote(instrument_id=1, raw_symbol=f"{root_symbol}F6",
                          expiration_ns=fe, price=front_price),
            ContractQuote(instrument_id=2, raw_symbol=f"{root_symbol}G6",
                          expiration_ns=ne, price=next_price),
        ),
    )

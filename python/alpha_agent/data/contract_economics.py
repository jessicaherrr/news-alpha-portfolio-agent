"""Contract economics -- Databento instrument-definition fields -> USD quantities.

Concepts are kept explicit and separate (prompt 03.5 rule 1):

    quote_tick_size   min_price_increment, in quoted price units          (NQ: 0.25)
    contract_size     unit_of_measure_qty                                  (NQ: 20.0, IPNT)
    price_scale       USD per (1 quoted unit * 1 contract_size unit)       (NQ: 1.0)
    point_value_usd   USD PnL per 1.0 move in the quoted price             (NQ: 20.0)
    tick_value_usd    USD per tick                                         (NQ: 5.0)

The word "multiplier" is avoided here. The frozen ``ContractSpec.multiplier`` /
``ContractSpecModel.multiplier`` field carries ``point_value_usd`` -- see
docs/CONTRACT_ECONOMICS.md.

No NQ-specific constants: everything is derived from the definition record via
Databento's documented (non-fractional) derivation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from alpha_agent.data.fractional_price import QuoteConvention

# Integer "unset" sentinels seen in GLBX.MDP3 definition records
# (INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64 maxima).
_INT_SENTINELS: frozenset[int] = frozenset(
    {2**7 - 1, 2**8 - 1, 2**15 - 1, 2**16 - 1, 2**31 - 1, 2**32 - 1, 2**63 - 1, 2**64 - 1}
)


# Databento contract-notional convention for FRACTIONALLY-QUOTED products
# (CBOT Treasuries): the quoted price is a PERCENT OF PAR, so one full price
# point is 1% of unit_of_measure_qty (the face value).
FRACTIONAL_PRICE_SCALE = 0.01


class EconomicsError(RuntimeError):
    """Contract economics cannot be derived without guessing -- caller must stop."""


def usable_number(value) -> float | None:
    """Return ``value`` as a positive finite float, or ``None`` when it is
    missing / non-positive / an integer sentinel (e.g. 2147483647, 255)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0.0:
        return None
    if f.is_integer() and int(f) in _INT_SENTINELS:
        return None
    return f


@dataclass(frozen=True)
class ContractEconomics:
    quote_tick_size: float      # min_price_increment, in NORMALIZED (decimal) price units
    contract_size: float        # unit_of_measure_qty
    unit_of_measure: str        # e.g. "IPNT"
    price_scale: float
    point_value_usd: float      # -> ContractSpec.multiplier
    tick_value_usd: float
    source: str                 # provenance of the derivation
    quote_convention: QuoteConvention = QuoteConvention.DECIMAL

    def cross_check(self, tol: float = 1e-6) -> bool:
        """quote_tick_size * point_value_usd == tick_value_usd."""
        return abs(self.quote_tick_size * self.point_value_usd - self.tick_value_usd) <= tol


def derive_contract_economics(
    *,
    min_price_increment,
    min_price_increment_amount,
    display_factor,
    unit_of_measure_qty,
    unit_of_measure: str,
    main_fraction=None,
    contract_multiplier=None,   # raw vendor value -- NEVER used as the point value (often a sentinel)
    allow_fractional: bool = False,
) -> ContractEconomics:
    """Databento documented derivation.

    tick_value_from_definition = min_price_increment_amount / display_factor
    price_scale                = tick_value_from_definition / (min_price_increment * unit_of_measure_qty)
    point_value_usd            = unit_of_measure_qty * price_scale
    tick_value_usd             = min_price_increment * point_value_usd

    That decimal derivation is correct only for DECIMAL-quoted products, where
    ``min_price_increment_amount / display_factor`` really is the USD tick value
    and ``unit_of_measure_qty`` really is the point value.

    FRACTIONALLY-QUOTED products (CBOT Treasuries -- a non-sentinel
    ``main_fraction``) are quoted as a PERCENT OF PAR, and ``unit_of_measure_qty``
    is the contract's FACE value, not its point value. Databento's documented
    contract-notional logic fixes the scale for this case:

        price_scale     = 0.01                       (percent of par)
        point_value_usd = unit_of_measure_qty * 0.01
        tick_value_usd  = min_price_increment * point_value_usd

    e.g. ZN: 100000 * 0.01 = $1,000 / point, 0.015625 * 1000 = $15.625 / tick --
    matching the published CME contract specification. Their
    ``min_price_increment_amount`` does NOT carry the USD tick value (ZN reports
    0.001 with ``display_factor`` 1.0) and is therefore never used on this path.

    With ``allow_fractional=False`` a fractional product still raises (Phase 03.5
    default -- refuse rather than mis-derive). ``contract_multiplier`` is never
    trusted (it is routinely the INT32 sentinel). Raises ``EconomicsError``
    rather than guessing.
    """
    mpi = usable_number(min_price_increment)
    mpia = usable_number(min_price_increment_amount)
    df = usable_number(display_factor)
    uomq = usable_number(unit_of_measure_qty)

    if mpi is None:
        raise EconomicsError("min_price_increment is missing or a sentinel")
    if uomq is None:
        raise EconomicsError(
            "unit_of_measure_qty is missing or a sentinel -- cannot derive the "
            "contract size without guessing (contract_multiplier is not trusted)"
        )

    is_fractional = usable_number(main_fraction) is not None
    if is_fractional and not allow_fractional:
        raise EconomicsError(
            f"main_fraction={main_fraction} indicates a fractionally-quoted product; "
            "pass allow_fractional=True to use the percent-of-par derivation "
            "-- real ZN field interpretation is verified in Phase 04.5"
        )

    if is_fractional:
        # PERCENT OF PAR. unit_of_measure_qty is the FACE value; the quoted price
        # is a percentage of it, so one full price point is 1% of face.
        # min_price_increment_amount / display_factor is NOT the USD tick value
        # for these products and is deliberately not consulted.
        convention = QuoteConvention.FRACTIONAL_32
        price_scale = FRACTIONAL_PRICE_SCALE
        source = (
            "unit_of_measure_qty * 0.01 (fractionally-quoted percent-of-par "
            "contract-notional convention; min_price_increment_amount not used)"
        )
    else:
        if mpia is None or df is None:
            raise EconomicsError(
                "min_price_increment_amount / display_factor missing or a sentinel -- "
                "cannot derive tick value"
            )
        convention = QuoteConvention.DECIMAL
        tick_value_from_definition = mpia / df
        price_scale = round(tick_value_from_definition / (mpi * uomq), 12)
        source = "unit_of_measure_qty * price_scale (min_price_increment_amount / display_factor)"

    point_value_usd = round(uomq * price_scale, 6)
    tick_value_usd = round(mpi * point_value_usd, 6)

    return ContractEconomics(
        quote_tick_size=mpi,
        contract_size=uomq,
        unit_of_measure=str(unit_of_measure),
        price_scale=price_scale,
        point_value_usd=point_value_usd,
        tick_value_usd=tick_value_usd,
        source=source,
        quote_convention=convention,
    )

"""The single price-normalization primitive.

Databento DBN prices are int64 fixed-point (1e-9). Everything downstream --
canonical bars, the raw-contract overlap, RollEvent prices, continuous series,
back-adjustment -- must work in the SAME normalized (decimal points) domain.

``descale_fixed_point`` is the only place ``/ PRICE_SCALE`` happens.
``assert_same_price_domain`` catches a fixed-point vs normalized mixup between
two frames without a product-specific absolute threshold.
"""
from __future__ import annotations

import math

import pandas as pd

from alpha_agent.schemas.market_data import PRICE_SCALE, UNDEF_PRICE

PRICE_COLS_DEFAULT: tuple[str, ...] = ("open", "high", "low", "close")


class PriceScaleError(RuntimeError):
    """A price entered a layer in the wrong scale domain."""


def descale_fixed_point(
    df: pd.DataFrame,
    *,
    price_cols: tuple[str, ...] = PRICE_COLS_DEFAULT,
    drop_undef: bool = True,
) -> pd.DataFrame:
    """DBN int64 fixed-point OHLC -> normalized float64 points.

    Raises ``PriceScaleError`` if a price column is already float dtype (that
    means the frame is already normalized and dividing again would be wrong).
    ``UNDEF_PRICE`` sentinels become NaN (when ``drop_undef``).
    """
    out = df.copy()
    for c in price_cols:
        if c not in out.columns:
            continue
        s = out[c]
        if s.dtype.kind == "f":
            raise PriceScaleError(
                f"column {c!r} is float dtype -- frame looks already normalized; "
                f"refusing to divide by {PRICE_SCALE} again"
            )
        s = s.astype("int64")
        scaled = s.astype("float64") / PRICE_SCALE
        out[c] = scaled.where(s != UNDEF_PRICE) if drop_undef else scaled
    return out


def median_abs_price(df: pd.DataFrame, price_col: str = "close") -> float:
    m = float(pd.to_numeric(df[price_col], errors="coerce").abs().median())
    return 0.0 if math.isnan(m) else m


def assert_same_price_domain(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    price_col: str = "close",
    max_ratio: float = 50.0,
    label: str = "",
) -> None:
    """Raise ``PriceScaleError`` if the two frames' median |price| differ by more
    than ``max_ratio`` (a fixed-point vs normalized mixup is ~1e9x)."""
    ma, mb = median_abs_price(a, price_col), median_abs_price(b, price_col)
    if ma <= 0 or mb <= 0:
        return
    ratio = max(ma, mb) / min(ma, mb)
    if ratio > max_ratio:
        raise PriceScaleError(
            f"price-domain mismatch{(' ' + label) if label else ''}: "
            f"median |{price_col}| {ma:.6g} vs {mb:.6g} (ratio {ratio:.3g}x > {max_ratio}x). "
            f"One frame is likely raw DBN fixed-point -- run descale_fixed_point first."
        )

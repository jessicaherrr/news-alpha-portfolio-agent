"""Cross-market feature architecture (section 17). MVP is deliberately small.

Alignment is strictly **backward**: a base-market row at ``T`` is joined to the
most recent other-market row at or before ``T`` (``merge_asof`` direction
``backward``). A future other-market row is never used. All windows are
backward-looking.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class PairSpec(BaseModel):
    model_config = {"frozen": True}

    base_root: str = Field(min_length=1)
    other_root: str = Field(min_length=1)
    kind: str = "spread"                  # "spread" | "ratio" | "return_corr"
    window: int = 20
    tolerance_ns: int | None = None       # max age of the joined other-market row

    @property
    def name(self) -> str:
        base = f"xm_{self.kind}_{self.base_root}_{self.other_root}"
        if self.kind == "return_corr":
            return f"{base}_{self.window}"
        return base


def align_backward(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    on: str = "ts_event_ns",
    value_cols: tuple[str, ...] = ("close",),
    suffix: str = "_other",
    tolerance_ns: int | None = None,
) -> pd.DataFrame:
    """Left-join ``other`` onto ``base`` by nearest earlier ``on`` value."""
    b = base.sort_values(on, kind="stable").reset_index(drop=True)
    o = other.loc[:, [on, *value_cols]].sort_values(on, kind="stable").reset_index(drop=True)
    o = o.rename(columns={c: f"{c}{suffix}" for c in value_cols})
    merged = pd.merge_asof(
        b, o, on=on, direction="backward",
        tolerance=int(tolerance_ns) if tolerance_ns is not None else None,
    )
    return merged


def build_cross_market_feature(
    frames_by_root: dict[str, pd.DataFrame],
    spec: PairSpec,
    *,
    price_col: str = "close",
) -> pd.DataFrame:
    """Return a frame ``[ts_event_ns, <spec.name>]`` (NaN where the other market
    has no earlier observation or a window is not yet full)."""
    if spec.base_root not in frames_by_root or spec.other_root not in frames_by_root:
        raise ValueError(f"missing frame(s) for pair {spec.base_root}/{spec.other_root}")
    base = frames_by_root[spec.base_root]
    other = frames_by_root[spec.other_root]
    merged = align_backward(
        base, other, value_cols=(price_col,), suffix="_other", tolerance_ns=spec.tolerance_ns,
    )
    b = pd.to_numeric(merged[price_col], errors="coerce").astype("float64")
    o = pd.to_numeric(merged[f"{price_col}_other"], errors="coerce").astype("float64")

    if spec.kind == "spread":
        out = b - o
    elif spec.kind == "ratio":
        out = (b / o).where(o != 0.0)
    elif spec.kind == "return_corr":
        rb = b.pct_change().where(b.shift() > 0)
        ro = o.pct_change().where(o.shift() > 0)
        out = rb.rolling(spec.window, min_periods=spec.window).corr(ro)
    else:
        raise ValueError(f"unknown cross-market kind {spec.kind!r}")

    return pd.DataFrame({"ts_event_ns": merged["ts_event_ns"].to_numpy(),
                         spec.name: out.to_numpy()})

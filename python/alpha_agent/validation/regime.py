"""Regime-stability evidence (section 17).

Regimes are PREDEFINED and CAUSAL. Any threshold learned from data is derived
only from allowed past/train data -- never from the partition that makes the
strategy look best, and never from future strategy PnL. If no regime data is
available the evidence is marked ``NOT_EVALUATED`` rather than guessed.

The framework ships one causal labeller (trailing realised-volatility tertiles
with train-derived cut points); callers may supply their own causal label array
for trend or session regimes.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.validation.enums import EvidenceStatus, RegimeKind
from alpha_agent.validation.fingerprint import fingerprint


class RegimeLabelling(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: RegimeKind
    # per OOS trading-day label; length == len(daily returns)
    labels: tuple[str, ...]
    # thresholds / definition, recorded for audit -- derived from train data only
    definition: dict = Field(default_factory=dict)
    derived_from: str = "train_only"

    def identity(self) -> str:
        return fingerprint(
            "valregime1",
            {"kind": self.kind.value, "definition": self.definition, "derived_from": self.derived_from},
        )


def causal_volatility_regime_labels(
    daily_abs_move: np.ndarray,
    *,
    train_n_days: int,
    lookback: int = 20,
) -> RegimeLabelling:
    """Label each OOS day low/mid/high volatility using trailing mean absolute
    daily move over ``lookback`` days, bucketed by tertile cut points estimated
    ONLY from the first ``train_n_days`` observations (train data)."""
    x = np.asarray(daily_abs_move, dtype=float)
    n = x.size
    trail = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - lookback)
        if i - lo >= 3:
            trail[i] = np.mean(x[lo:i]) if i > lo else np.nan
    train = trail[:train_n_days]
    train = train[np.isfinite(train)]
    if train.size < 6:
        q1 = q2 = float("nan")
    else:
        q1, q2 = np.quantile(train, [1 / 3, 2 / 3])
    labels = []
    for v in trail:
        if not np.isfinite(v) or not np.isfinite(q1):
            labels.append("unknown")
        elif v <= q1:
            labels.append("low_vol")
        elif v <= q2:
            labels.append("mid_vol")
        else:
            labels.append("high_vol")
    return RegimeLabelling(
        kind=RegimeKind.VOLATILITY,
        labels=tuple(labels),
        definition={"lookback": lookback, "train_q1": float(q1), "train_q2": float(q2)},
    )


class RegimeBucket(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    n_days: int
    mean_daily_return: float
    daily_sharpe: float
    net_pnl_usd: float
    pnl_share: float                           # net_pnl_usd / total abs pnl


class RegimeStabilityResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    status: EvidenceStatus
    kind: RegimeKind | None = None
    labelling_fingerprint: str | None = None
    buckets: tuple[RegimeBucket, ...] = ()
    max_regime_pnl_share: float = 0.0
    is_concentrated: bool = False              # one regime holds > concentration_threshold of PnL
    concentration_threshold: float = 0.9


def evaluate_regime_stability(
    returns: np.ndarray,
    daily_pnl_usd: np.ndarray,
    labelling: RegimeLabelling | None,
    *,
    concentration_threshold: float = 0.9,
) -> RegimeStabilityResult:
    if labelling is None:
        return RegimeStabilityResult(status=EvidenceStatus.NOT_EVALUATED)
    r = np.asarray(returns, dtype=float)
    pnl = np.asarray(daily_pnl_usd, dtype=float)
    labs = np.asarray(labelling.labels)
    if labs.size != r.size:
        return RegimeStabilityResult(status=EvidenceStatus.NOT_EVALUATED)

    total_abs = float(np.sum(np.abs(pnl))) or 1.0
    buckets: list[RegimeBucket] = []
    shares: list[float] = []
    for lab in sorted(set(labs.tolist())):
        m = labs == lab
        rr = r[m]
        pp = pnl[m]
        sd = rr.std(ddof=1) if rr.size >= 2 else 0.0
        share = float(np.sum(pp) / total_abs)
        if lab != "unknown":
            shares.append(share)
        buckets.append(
            RegimeBucket(
                label=lab,
                n_days=int(m.sum()),
                mean_daily_return=float(rr.mean()) if rr.size else 0.0,
                daily_sharpe=float(rr.mean() / sd) if sd else 0.0,
                net_pnl_usd=float(np.sum(pp)),
                pnl_share=share,
            )
        )
    max_share = max((abs(s) for s in shares), default=0.0)
    return RegimeStabilityResult(
        status=EvidenceStatus.EVALUATED,
        kind=labelling.kind,
        labelling_fingerprint=labelling.identity(),
        buckets=tuple(buckets),
        max_regime_pnl_share=float(max_share),
        is_concentrated=bool(max_share > concentration_threshold and len(shares) >= 2),
        concentration_threshold=concentration_threshold,
    )

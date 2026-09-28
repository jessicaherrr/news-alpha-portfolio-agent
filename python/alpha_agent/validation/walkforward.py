"""Causal chronological walk-forward evaluation (sections 6 & 7).

    train_1 -> test_1 ,  train_2 -> test_2 ,  ... ,  train_k -> test_k

No random train/test split. For Phase 13 a PREDECLARED fixed ``StrategySpec`` is
evaluated across folds -- there is no parameter optimisation inside a fold. Each
out-of-sample test fold:

* sees only past feature warm-up (``warmup_bars`` bars before the test start,
  used to compute features, never to open an economic position);
* starts flat -- no position or PnL is inherited from the training segment
  (``PortfolioStatePolicy.FLAT_START``, the documented default, section 6);
* is a fully independent C++ backtest over the test-window bars (+ the read-only
  warm-up bars) so no future bar is ever used to close a training trade and then
  counted as test performance (section 7);
* is separated from its training window by an ``embargo_days`` gap.

This module builds the fold boundaries and summarises the per-fold C++ results.
The reruns are orchestrated by :mod:`.runner`.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator

from alpha_agent.validation.enums import PortfolioStatePolicy
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.metrics import annualized_sharpe, daily_sharpe
from alpha_agent.validation.returns import DailyReturnSeries, concat_oos_series
from alpha_agent.validation.splits import NS_PER_DAY


class WalkForwardConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "walk-forward/1"
    n_folds: int = Field(default=4, ge=1, le=100)
    scheme: str = "expanding"                  # "expanding" | "rolling"
    min_train_days: int = Field(default=60, ge=1)
    train_days: int = Field(default=120, ge=1)  # rolling scheme only
    test_days: int = Field(default=30, ge=1)
    embargo_days: int = Field(default=5, ge=0)
    portfolio_state_policy: PortfolioStatePolicy = PortfolioStatePolicy.FLAT_START
    warmup_bars: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> WalkForwardConfig:
        if self.scheme not in ("expanding", "rolling"):
            raise ValueError("scheme must be 'expanding' or 'rolling'")
        return self

    def identity(self) -> str:
        return fingerprint("valwalkfwd1", self.model_dump(mode="json"))


class WalkForwardFold(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    fold_index: int
    train_start_ts_ns: int
    train_end_ts_ns: int                       # exclusive
    warmup_start_ts_ns: int                    # feature history only, no position
    test_start_ts_ns: int
    test_end_ts_ns: int                        # exclusive

    @model_validator(mode="after")
    def _causal(self) -> WalkForwardFold:
        if not (self.train_start_ts_ns < self.train_end_ts_ns <= self.test_start_ts_ns
                < self.test_end_ts_ns):
            raise ValueError("fold windows must be causally ordered train < test")
        if self.warmup_start_ts_ns > self.test_start_ts_ns:
            raise ValueError("warmup window cannot start after the test window")
        if self.warmup_start_ts_ns < self.train_start_ts_ns:
            raise ValueError("warmup cannot reach before the training data begins")
        return self


def build_folds(
    research_span_ns: tuple[int, int],
    config: WalkForwardConfig,
    *,
    bar_span_ns: int | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Build up to ``config.n_folds`` causal folds inside the research span
    (never the holdout). Returns as many folds as fit; the caller's minimum-fold
    gate decides whether that is enough."""
    start, end = research_span_ns
    warm_ns = (config.warmup_bars * bar_span_ns) if bar_span_ns else 0
    test_ns = config.test_days * NS_PER_DAY
    emb_ns = config.embargo_days * NS_PER_DAY
    min_train_ns = config.min_train_days * NS_PER_DAY
    train_ns = config.train_days * NS_PER_DAY

    folds: list[WalkForwardFold] = []
    test_start = start + min_train_ns + emb_ns
    for i in range(config.n_folds):
        test_end = test_start + test_ns
        if test_end > end:
            break
        train_end = test_start - emb_ns
        train_start = start if config.scheme == "expanding" else max(start, train_end - train_ns)
        if train_end - train_start < min_train_ns:
            break
        warm_start = max(train_start, test_start - warm_ns)
        folds.append(
            WalkForwardFold(
                fold_index=i,
                train_start_ts_ns=train_start,
                train_end_ts_ns=train_end,
                warmup_start_ts_ns=warm_start,
                test_start_ts_ns=test_start,
                test_end_ts_ns=test_end,
            )
        )
        test_start = test_end
    return tuple(folds)


class WalkForwardFoldResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    fold_index: int
    test_start_ts_ns: int
    test_end_ts_ns: int
    n_trading_days: int
    n_fills: int
    n_trades: int
    oos_net_pnl_usd: float
    daily_sharpe: float
    is_positive: bool
    daily: DailyReturnSeries


class WalkForwardResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    config_fingerprint: str
    n_folds_planned: int
    n_folds_evaluated: int
    positive_fold_count: int
    fold_consistency: float                    # positive_fold_count / n_folds_evaluated
    combined_oos_net_pnl_usd: float
    combined_oos_daily_sharpe: float
    combined_oos_annualized_sharpe: float
    folds: tuple[WalkForwardFoldResult, ...]
    combined_oos_daily: DailyReturnSeries | None = None


def summarize_walk_forward(
    config: WalkForwardConfig,
    n_folds_planned: int,
    fold_results: list[WalkForwardFoldResult],
) -> WalkForwardResult:
    n = len(fold_results)
    pos = sum(1 for f in fold_results if f.is_positive)
    combined = concat_oos_series([f.daily for f in fold_results]) if fold_results else None
    r = combined.returns_array() if combined is not None else np.empty(0)
    return WalkForwardResult(
        config_fingerprint=config.identity(),
        n_folds_planned=n_folds_planned,
        n_folds_evaluated=n,
        positive_fold_count=pos,
        fold_consistency=(pos / n) if n else 0.0,
        combined_oos_net_pnl_usd=float(sum(f.oos_net_pnl_usd for f in fold_results)),
        combined_oos_daily_sharpe=daily_sharpe(r) if r.size else float("nan"),
        combined_oos_annualized_sharpe=annualized_sharpe(r) if r.size else float("nan"),
        folds=tuple(fold_results),
        combined_oos_daily=combined,
    )

"""Deterministic time-series-aware bootstrap (section 10).

Naive IID resampling destroys the short-range serial dependence of a daily PnL
series, so it is **not** offered as a main-path method. Two dependent-data
resamplers are implemented, both circular (wrap-around) so every observation is
equally likely to appear:

* **moving block** -- fixed block length ``L``; ``ceil(n / L)`` blocks with
  uniform random starts, concatenated and truncated to ``n``.
* **stationary** (Politis & Romano 1994) -- geometric(1/L) random block lengths
  with uniform random starts; the expected block length is ``L``.

Determinism: ``numpy.random.default_rng(seed)`` only; same ``(returns, config)``
=> byte-identical resample indices and confidence intervals.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.validation.enums import BootstrapMethod
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.metrics import annualized_sharpe, daily_sharpe


class BootstrapConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    method: BootstrapMethod = BootstrapMethod.STATIONARY
    n_resamples: int = Field(default=2000, ge=100, le=200_000)
    # moving_block: the fixed block length. stationary: the EXPECTED block length.
    mean_block_length: int = Field(default=20, ge=1, le=10_000)
    seed: int = Field(default=0, ge=0)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    def identity(self) -> str:
        return fingerprint("valbootstrap1", self.model_dump(mode="json"))


class BootstrapCI(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    statistic: str
    point_estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    std_error: float
    n_resamples: int
    method: str
    block_length: int
    seed: int


class BootstrapResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    config_fingerprint: str
    n_days: int
    mean_daily_return: BootstrapCI
    annualized_sharpe: BootstrapCI


def resample_indices(
    rng: np.random.Generator, n: int, method: BootstrapMethod, block_length: int
) -> np.ndarray:
    """One circular block-resample of ``range(n)`` -> an index array of length n.
    Exposed for tests (checklist H/I): block structure must be preserved."""
    if n <= 0:
        return np.empty(0, dtype=int)
    L = max(1, min(block_length, n))
    out = np.empty(n, dtype=int)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        if method == BootstrapMethod.MOVING_BLOCK:
            blk = L
        else:  # stationary: geometric block length, expectation L
            blk = int(rng.geometric(1.0 / L)) if L > 1 else 1
        blk = max(1, blk)
        take = min(blk, n - filled)
        idx = (start + np.arange(take)) % n
        out[filled:filled + take] = idx
        filled += take
    return out


def _bootstrap_samples(
    returns: np.ndarray, config: BootstrapConfig, statfn
) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    n = x.size
    rng = np.random.default_rng(config.seed)
    samples = np.empty(config.n_resamples, dtype=float)
    for b in range(config.n_resamples):
        idx = resample_indices(rng, n, config.method, config.mean_block_length)
        samples[b] = statfn(x[idx])
    return samples


def _percentile_ci(
    samples: np.ndarray, point: float, statistic: str, config: BootstrapConfig
) -> BootstrapCI:
    finite = samples[np.isfinite(samples)]
    alpha = (1.0 - config.ci_level) / 2.0
    if finite.size == 0:
        lo = hi = se = float("nan")
    else:
        lo = float(np.quantile(finite, alpha))
        hi = float(np.quantile(finite, 1.0 - alpha))
        se = float(finite.std(ddof=1)) if finite.size >= 2 else float("nan")
    return BootstrapCI(
        statistic=statistic,
        point_estimate=float(point),
        ci_low=lo,
        ci_high=hi,
        ci_level=config.ci_level,
        std_error=se,
        n_resamples=config.n_resamples,
        method=config.method.value,
        block_length=config.mean_block_length,
        seed=config.seed,
    )


def bootstrap_daily_returns(returns: np.ndarray, config: BootstrapConfig) -> BootstrapResult:
    """Confidence intervals for the mean daily return and the annualised Sharpe
    (section 10 minimum requirement)."""
    x = np.asarray(returns, dtype=float)
    if x.size < 3:
        raise ValueError("bootstrap needs at least 3 daily observations")

    mean_samples = _bootstrap_samples(x, config, np.mean)
    # bootstrap the DAILY sharpe then annualise the CI endpoints -- the series is
    # daily, annualisation is a fixed sqrt(252) scale (section 9).
    sharpe_samples = _bootstrap_samples(x, config, daily_sharpe)
    sharpe_ann_samples = sharpe_samples * np.sqrt(252.0)

    return BootstrapResult(
        config_fingerprint=config.identity(),
        n_days=int(x.size),
        mean_daily_return=_percentile_ci(mean_samples, float(np.mean(x)), "mean_daily_return", config),
        annualized_sharpe=_percentile_ci(
            sharpe_ann_samples, annualized_sharpe(x), "annualized_sharpe", config
        ),
    )

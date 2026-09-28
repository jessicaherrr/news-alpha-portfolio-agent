"""Deflated Sharpe Ratio -- Bailey & Lopez de Prado (2014), "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
*Journal of Portfolio Management* 40(5) (section 13).

FREQUENCY CONVENTION (Phase 13.2 sections 7-8). Every quantity that enters the
PSR / DSR inference is on the **same per-observation frequency -- DAILY**, the
framework's only return series:

  * ``SR_hat``  = mean(daily_returns) / std(daily_returns)   -- the DAILY Sharpe
  * ``N``       = number of daily observations
  * ``g3, g4``  = sample skewness / (non-excess, Pearson) kurtosis of those SAME
                  daily returns
  * ``SR*_0``   = expected-maximum benchmark on the DAILY scale, built from the
                  variance of the trials' DAILY Sharpe estimates

The annualised Sharpe ``SR_hat * sqrt(252)`` is a REPORTING metric only
(``observed_annualized_sharpe``); it never enters any formula below. So rescaling
the reporting Sharpe to annualised form does not change the DSR probability.

Exact equations used
--------------------
Probabilistic Sharpe Ratio against a benchmark daily Sharpe ``SR*`` (B&LdP 2014,
eq. for PSR; Bailey & Lopez de Prado 2012):

    PSR(SR*) = Phi(  (SR_hat - SR*) * sqrt(N - 1)
                     / sqrt( 1 - g3*SR_hat + (g4 - 1)/4 * SR_hat^2 )  )

(a normal sample has ``g4 == 3`` so the correction term ``(g4-1)/4 -> 0.5``.)

Deflated Sharpe Ratio (B&LdP 2014, eq. 9):

    DSR   = PSR(SR*_0)

    SR*_0 = sqrt(Var[{SR_n}]) * ( (1 - gamma) * Z^-1( 1 - 1/T )
                                  +    gamma   * Z^-1( 1 - 1/(T*e) ) )

where ``Z^-1`` is the standard-normal inverse CDF, ``T`` the effective number of
trials, ``Var[{SR_n}]`` the cross-trial variance of the trials' DAILY Sharpe
estimates, ``gamma`` the Euler-Mascheroni constant (~0.5772) and ``e`` Euler's
number. ``SR*_0`` is the DAILY Sharpe the best of ``T`` independent zero-skill
trials is expected to show; DSR is the probability the observed DAILY Sharpe
beats that selection-inflated DAILY benchmark -- both on the same scale
(section 8).

Assumptions / edge cases:
* ``N < 2`` or zero return variance -> DSR undefined (``is_valid = False``).
* Non-normality denominator ``1 - g3*SR + (g4-1)/4*SR^2 <= 0`` -> undefined.
* ``T <= 1`` or ``Var[{SR_n}] <= 0`` -> no deflation, ``SR*_0 = 0``, DSR = PSR(0).
* DSR is a probability in ``[0, 1]``. It does **not**, on its own, prove alpha.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.validation.metrics import (
    daily_sharpe,
    sample_kurtosis,
    sample_skewness,
)

_N = NormalDist()
EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    observed_daily_sharpe: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    benchmark_daily_sharpe: float = 0.0,
) -> float:
    if n_obs < 2 or not np.isfinite(observed_daily_sharpe):
        return float("nan")
    sr = float(observed_daily_sharpe)
    denom = 1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    z = (sr - benchmark_daily_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(_N.cdf(z))


def deflated_benchmark_sharpe(sharpe_variance_across_trials: float, n_trials: int) -> float:
    """SR*_0 -- the selection-inflated zero-skill benchmark daily Sharpe."""
    if n_trials <= 1 or sharpe_variance_across_trials <= 0.0:
        return 0.0
    t = float(n_trials)
    a = _N.inv_cdf(1.0 - 1.0 / t)
    b = _N.inv_cdf(1.0 - 1.0 / (t * math.e))
    return float(
        math.sqrt(sharpe_variance_across_trials)
        * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)
    )


class DeflatedSharpeResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    observed_daily_sharpe: float
    observed_annualized_sharpe: float
    n_obs: int = Field(ge=0)
    return_skewness: float
    return_kurtosis: float

    n_trials: int = Field(ge=1)
    sharpe_variance_across_trials: float      # Var[{SR_n}] on the DAILY Sharpe scale
    benchmark_daily_sharpe: float             # SR*_0, DAILY scale (same as observed)

    psr_vs_zero: float                       # PSR(0): significance ignoring selection
    deflated_sharpe_ratio: float             # DSR == PSR(SR*_0), a probability in [0,1]
    is_valid: bool
    # every quantity in the inference is on this per-observation frequency; the
    # annualised Sharpe is reporting-only and never enters the formulas.
    inference_frequency: str = "daily"
    note: str = ""


def deflated_sharpe_ratio(
    daily_returns: np.ndarray,
    *,
    n_trials: int,
    trial_daily_sharpes: list[float] | None = None,
    sharpe_variance_across_trials: float | None = None,
) -> DeflatedSharpeResult:
    """Compute the DSR for one strategy's daily return series within a family of
    ``n_trials`` inspected variants.

    Provide the cross-trial Sharpe dispersion as either ``trial_daily_sharpes``
    (the daily Sharpe of every inspected variant, including this one and every
    failed one) or an explicit ``sharpe_variance_across_trials``.
    """
    x = np.asarray(daily_returns, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    sr_d = daily_sharpe(x)
    sr_a = sr_d * math.sqrt(252.0) if np.isfinite(sr_d) else float("nan")
    skew = sample_skewness(x)
    kurt = sample_kurtosis(x)

    if sharpe_variance_across_trials is not None:
        var_tr = float(sharpe_variance_across_trials)
    elif trial_daily_sharpes:
        arr = np.asarray([s for s in trial_daily_sharpes if np.isfinite(s)], dtype=float)
        var_tr = float(arr.var(ddof=1)) if arr.size >= 2 else 0.0
    else:
        var_tr = 0.0

    benchmark = deflated_benchmark_sharpe(var_tr, max(1, int(n_trials)))

    valid = n >= 2 and np.isfinite(sr_d) and np.isfinite(skew) and np.isfinite(kurt)
    if not valid:
        return DeflatedSharpeResult(
            observed_daily_sharpe=float(sr_d) if np.isfinite(sr_d) else 0.0,
            observed_annualized_sharpe=float(sr_a) if np.isfinite(sr_a) else 0.0,
            n_obs=n,
            return_skewness=float(skew) if np.isfinite(skew) else 0.0,
            return_kurtosis=float(kurt) if np.isfinite(kurt) else 0.0,
            n_trials=max(1, int(n_trials)),
            sharpe_variance_across_trials=var_tr,
            benchmark_daily_sharpe=benchmark,
            psr_vs_zero=0.0,
            deflated_sharpe_ratio=0.0,
            is_valid=False,
            note="DSR undefined: fewer than 2 observations or zero return variance.",
        )

    psr0 = probabilistic_sharpe_ratio(sr_d, n, skew, kurt, 0.0)
    dsr = probabilistic_sharpe_ratio(sr_d, n, skew, kurt, benchmark)
    denom_ok = np.isfinite(dsr)
    return DeflatedSharpeResult(
        observed_daily_sharpe=float(sr_d),
        observed_annualized_sharpe=float(sr_a),
        n_obs=n,
        return_skewness=float(skew),
        return_kurtosis=float(kurt),
        n_trials=max(1, int(n_trials)),
        sharpe_variance_across_trials=var_tr,
        benchmark_daily_sharpe=benchmark,
        psr_vs_zero=float(psr0) if np.isfinite(psr0) else 0.0,
        deflated_sharpe_ratio=float(dsr) if denom_ok else 0.0,
        is_valid=bool(denom_ok),
        note="" if denom_ok else "DSR undefined: non-normality denominator <= 0.",
    )

"""Typed multiple-testing / trial-family accounting (section 12).

Every strategy variant inspected for performance is a trial: the canonical
strategy, every predeclared parameter neighbour, every ablation, every related
hypothesis variant, every cross-market run. All of them stay in the denominator
-- the winner is never counted alone and failed trials are never dropped.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.validation.fdr import FdrResult, benjamini_hochberg_decisions
from alpha_agent.validation.fingerprint import fingerprint


class TrialRecord(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    label: str                                # human label -- NOT in the family fingerprint
    role: str                                 # canonical | neighbour | ablation | cross_market | variant
    strategy_fingerprint: str
    schedule_hash: str | None = None
    # One-sided empirical p-value for positive performance (section 11). A trial
    # that was inspected but never null-tested gets 1.0 -- it cannot be
    # "significant", and it still counts toward the family size.
    p_value: float = Field(default=1.0, ge=0.0, le=1.0)
    null_tested: bool = False
    observed_daily_sharpe: float | None = None
    observed_net_pnl_usd: float | None = None


class MultipleTestingFamily(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "trial-family/1"
    family_id: str
    trials: tuple[TrialRecord, ...]

    def n_trials(self) -> int:
        return len(self.trials)

    def tested_fingerprints(self) -> tuple[str, ...]:
        return tuple(sorted({t.strategy_fingerprint for t in self.trials}))

    def p_values(self) -> list[float]:
        return [t.p_value for t in self.trials]

    def observed_daily_sharpes(self) -> list[float]:
        return [
            t.observed_daily_sharpe
            for t in self.trials
            if t.observed_daily_sharpe is not None and np.isfinite(t.observed_daily_sharpe)
        ]

    def sharpe_variance_across_trials(self) -> float:
        arr = np.asarray(self.observed_daily_sharpes(), dtype=float)
        return float(arr.var(ddof=1)) if arr.size >= 2 else 0.0

    def family_fingerprint(self) -> str:
        payload = {
            "family_id": self.family_id,
            "schema_version": self.schema_version,
            # identity only -- roles + strategy/schedule hashes, sorted; no labels
            "trials": sorted(
                [t.role, t.strategy_fingerprint, t.schedule_hash or ""] for t in self.trials
            ),
        }
        return fingerprint("trialfamily1", payload)

    def benjamini_hochberg(self, q_threshold: float = 0.10) -> FdrResult:
        return benjamini_hochberg_decisions(
            self.p_values(), q_threshold, labels=[t.label for t in self.trials]
        )

    def canonical_trial_index(self) -> int:
        for i, t in enumerate(self.trials):
            if t.role == "canonical":
                return i
        raise ValueError(f"family {self.family_id!r} has no 'canonical' trial")

"""Chronological train / validation / locked-holdout split model (sections 5, 7).

No random train/test split for market time series -- windows are contiguous
calendar spans in strict chronological order. There is exactly one
``LOCKED_HOLDOUT`` window and it is always last, separated from all research data
by an explicit embargo so fold-edge state (feature lookback, strategy state,
pending latency, maximum holding) cannot leak across the boundary.

The split object itself only *describes* windows. The structural guarantee that
the holdout is never touched during research lives in :mod:`.holdout`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from alpha_agent.validation.enums import SplitRole
from alpha_agent.validation.fingerprint import fingerprint

NS_PER_DAY = 86_400_000_000_000


class SplitWindow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    role: SplitRole
    start_ts_ns: int = Field(ge=0)
    end_ts_ns: int = Field(gt=0)          # exclusive

    @model_validator(mode="after")
    def _order(self) -> SplitWindow:
        if self.end_ts_ns <= self.start_ts_ns:
            raise ValueError(f"split window end {self.end_ts_ns} <= start {self.start_ts_ns}")
        return self

    def contains(self, ts_ns: int) -> bool:
        return self.start_ts_ns <= ts_ns < self.end_ts_ns


class SplitPlan(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "split-plan/1"
    windows: tuple[SplitWindow, ...]
    # Bars/days of separation required around every fold and split boundary
    # (section 7). Documented, part of the fingerprint.
    embargo_days: int = Field(default=5, ge=0, le=1000)

    @property
    def embargo_ns(self) -> int:
        return self.embargo_days * NS_PER_DAY

    @model_validator(mode="after")
    def _chronological(self) -> SplitPlan:
        if len(self.windows) < 2:
            raise ValueError("a SplitPlan needs at least a research window and a locked holdout")
        for a, b in zip(self.windows, self.windows[1:]):
            if b.start_ts_ns < a.end_ts_ns:
                raise ValueError("split windows must not overlap and must be chronological")
        holdouts = [w for w in self.windows if w.role == SplitRole.LOCKED_HOLDOUT]
        if len(holdouts) != 1:
            raise ValueError("exactly one LOCKED_HOLDOUT window is required")
        if self.windows[-1].role != SplitRole.LOCKED_HOLDOUT:
            raise ValueError("the LOCKED_HOLDOUT window must be chronologically last")
        research_end = max(
            w.end_ts_ns for w in self.windows if w.role != SplitRole.LOCKED_HOLDOUT
        )
        if self.windows[-1].start_ts_ns - research_end < self.embargo_ns:
            raise ValueError(
                f"locked holdout starts {self.windows[-1].start_ts_ns - research_end} ns after "
                f"research data; the embargo requires >= {self.embargo_ns} ns"
            )
        return self

    # -- accessors ----------------------------------------------------------
    def window(self, role: SplitRole) -> SplitWindow:
        for w in self.windows:
            if w.role == role:
                return w
        raise KeyError(role)

    def has_role(self, role: SplitRole) -> bool:
        return any(w.role == role for w in self.windows)

    def research_span_ns(self) -> tuple[int, int]:
        """(start, end) covering every non-holdout window."""
        rs = [w for w in self.windows if w.role != SplitRole.LOCKED_HOLDOUT]
        return (min(w.start_ts_ns for w in rs), max(w.end_ts_ns for w in rs))

    def holdout_window(self) -> SplitWindow:
        return self.window(SplitRole.LOCKED_HOLDOUT)

    def split_fingerprint(self) -> str:
        return fingerprint(
            "splitplan1",
            {
                "schema_version": self.schema_version,
                "embargo_days": self.embargo_days,
                "windows": sorted(
                    [w.role.value, w.start_ts_ns, w.end_ts_ns] for w in self.windows
                ),
            },
        )

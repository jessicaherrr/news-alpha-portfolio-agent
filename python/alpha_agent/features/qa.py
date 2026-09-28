"""Feature-engine QA diagnostics.

Parallel to ``alpha_agent.data.diagnostics`` but scoped to feature computation.
Nothing here silently "fixes" data: a repair (``inf`` -> missing) is always
recorded as an issue so the caller can see what happened.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field


class FeatureIssueKind(str, Enum):
    NAN_IN_OUTPUT = "nan_in_output"                     # NaN after the warmup region
    INF_IN_OUTPUT = "inf_in_output"                     # +/-inf produced -> coerced to missing
    INSUFFICIENT_LOOKBACK = "insufficient_lookback"     # fewer than min_observations available
    ZERO_DENOMINATOR = "zero_denominator"               # division by ~0 -> missing
    NON_POSITIVE_LOG_INPUT = "non_positive_log_input"   # log() of a <= 0 price -> missing
    NEGATIVE_RETURN_BASE = "negative_return_base"       # percentage return with a negative base
    NON_POSITIVE_RETURN_BASE = "non_positive_return_base"  # base price <= 0 -> return missing
    STALE_INPUT = "stale_input"                         # window spans an unexpectedly old bar
    MISSING_BAR_GAP = "missing_bar_gap"                 # a gap larger than one bar interval
    SESSION_BOUNDARY_RESET = "session_boundary_reset"   # a window was reset at a boundary
    EXTREME_VALUE = "extreme_value"                     # |value| beyond a sanity magnitude (kept)
    PRICE_DOMAIN_MISMATCH = "price_domain_mismatch"     # source domain not allowed by the spec
    LOOKAHEAD_UNSAFE_REQUEST = "lookahead_unsafe_request"  # retrospective feature, PIT contract


# Sanity magnitude for a feature value. Values beyond this are kept verbatim
# (never clipped) but flagged EXTREME_VALUE.
EXTREME_VALUE_ABS_LIMIT = 1.0e12


class FeatureIssue(BaseModel):
    kind: FeatureIssueKind
    feature: str
    message: str
    count: int = 1
    context: dict = Field(default_factory=dict)


class FeatureQAReport(BaseModel):
    issues: list[FeatureIssue] = Field(default_factory=list)

    def add(
        self,
        kind: FeatureIssueKind,
        feature: str,
        message: str,
        *,
        count: int = 1,
        **context,
    ) -> None:
        self.issues.append(
            FeatureIssue(kind=kind, feature=feature, message=message, count=count, context=context)
        )

    def for_feature(self, feature: str) -> list[FeatureIssue]:
        return [i for i in self.issues if i.feature == feature]

    def has_kind(self, kind: FeatureIssueKind, *, feature: str | None = None) -> bool:
        return any(
            i.kind is kind and (feature is None or i.feature == feature) for i in self.issues
        )

    @property
    def by_kind(self) -> dict[str, int]:
        c: Counter[str] = Counter()
        for i in self.issues:
            c[i.kind.value] += i.count
        return dict(c)

    def summary(self) -> dict:
        return {"n_issues": len(self.issues), "by_kind": self.by_kind}

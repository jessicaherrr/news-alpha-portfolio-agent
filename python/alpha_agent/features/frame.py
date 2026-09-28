"""The point-in-time :class:`FeatureFrame` and its lineage / cache key.

A FeatureFrame keeps identifiers (``ts_event_ns`` + instrument / root / continuous
symbol + session labels) strictly separate from the numeric feature columns.
Rows are never silently dropped: a value that cannot be computed because history
is insufficient is an explicit ``NaN`` with ``mask == False``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
from pydantic import BaseModel, Field

from alpha_agent.data.lineage import code_commit
from alpha_agent.features.qa import FeatureQAReport
from alpha_agent.features.registry import FEATURE_ENGINE_VERSION
from alpha_agent.features.safety import (
    FEATURE_NEVER_EXECUTION_PRICE,
    FeatureSafety,
    FeatureSafetyError,
    SourceSafety,
)
from alpha_agent.features.spec import FeatureMetadata, FeatureSpec


class FeatureLineage(BaseModel):
    engine_version: str = FEATURE_ENGINE_VERSION
    source_price_domain: str
    # Phase 04 back-adjustment mode when source_price_domain == "back_adjusted".
    source_adjustment_mode: str | None = None
    source_paths: list[str] = Field(default_factory=list)
    source_fingerprint: str
    n_source_rows: int
    as_of_ts_ns: int | None = None
    # Safety the SOURCE confers, before features (see alpha_agent.features.safety).
    source_point_in_time_safe: bool = True
    source_signal_safe: bool = True
    source_execution_price_safe: bool = False
    code_commit: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    feature_names: list[str] = Field(default_factory=list)
    feature_specs: list[dict] = Field(default_factory=list)
    identity: dict[str, str] = Field(default_factory=dict)


def feature_cache_key(
    source_fingerprint: str, spec: FeatureSpec, *, engine_version: str = FEATURE_ENGINE_VERSION,
    as_of_ts_ns: int | None = None,
) -> str:
    """Reproducibility key: same source lineage + same spec + same engine version
    => same key. Changing any of them changes the key."""
    payload = json.dumps(
        {
            "source_fingerprint": source_fingerprint,
            "spec": spec.canonical_json(),
            "engine_version": engine_version,
            "as_of_ts_ns": as_of_ts_ns,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class FeatureFrame:
    identifiers: pd.DataFrame
    features: pd.DataFrame
    mask: pd.DataFrame                       # True where the feature value is present
    availability: pd.DataFrame               # per feature: first ts_event_ns it is present
    metadata: dict[str, FeatureMetadata]
    specs: dict[str, FeatureSpec]
    lineage: FeatureLineage
    safety: dict[str, FeatureSafety] = field(default_factory=dict)
    qa: FeatureQAReport = field(default_factory=FeatureQAReport)

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    @property
    def source_safety(self) -> SourceSafety:
        """Safety the SOURCE price series conferred (before features). This is
        where ``execution_price_safe`` is a real property -- a RawContract feed
        is source-execution-price-safe; a continuous / back-adjusted feed is
        not. It does NOT propagate to the feature columns (see ``frame_safety``)."""
        lin = self.lineage
        return SourceSafety(
            point_in_time_safe=lin.source_point_in_time_safe,
            signal_safe=lin.source_signal_safe,
            execution_price_safe=lin.source_execution_price_safe,
        )

    @property
    def frame_safety(self) -> FeatureSafety:
        """Aggregate DERIVED-FEATURE safety: a use is safe for the frame iff it is
        safe for every feature column. ``execution_price_safe`` is always
        ``False`` -- a computed feature value is never an execution price."""
        names = self.feature_names
        reasons: dict[str, str] = {}
        for n in names:
            for k, v in self.safety[n].reasons.items():
                reasons[f"{n}:{k}"] = v
        return FeatureSafety(
            feature="<frame>",
            point_in_time_safe=bool(names) and all(self.safety[n].point_in_time_safe for n in names),
            signal_safe=bool(names) and all(self.safety[n].signal_safe for n in names),
            execution_price_safe=False,
            reasons=reasons or {"execution_price": FEATURE_NEVER_EXECUTION_PRICE},
        )

    def frame(self) -> pd.DataFrame:
        """Identifiers + feature columns as one wide DataFrame (identifiers first)."""
        return pd.concat([self.identifiers, self.features], axis=1)

    def missing_report(self) -> pd.DataFrame:
        rows = []
        for name in self.feature_names:
            m = self.mask[name]
            rows.append(
                {
                    "feature": name,
                    "n_present": int(m.sum()),
                    "n_missing": int((~m).sum()),
                    "first_available_ts_ns": self.availability.loc[name, "first_available_ts_ns"],
                }
            )
        return pd.DataFrame(rows)

    def assert_point_in_time_safe(self) -> None:
        bad = {n: self.safety[n].reasons for n in self.feature_names
               if not self.safety[n].point_in_time_safe}
        if bad:
            raise FeatureSafetyError(
                f"FeatureFrame is not point-in-time safe: {sorted(bad)}; reasons={bad}"
            )

    def assert_signal_safe(self) -> None:
        """Raise unless every feature may drive a strategy ``Signal`` (Phase 10
        DSL contract). Point-in-time back-adjusted trend features pass; a
        retrospective back-adjusted source or a retrospective roll feature does
        not."""
        bad = {n: self.safety[n].reasons for n in self.feature_names
               if not self.safety[n].signal_safe}
        if bad:
            raise FeatureSafetyError(
                f"FeatureFrame is not signal-safe: {sorted(bad)}; reasons={bad}"
            )

    def assert_execution_price_safe(self) -> None:
        """ALWAYS raises. No ``FeatureFrame`` is execution-price-safe: a computed
        feature value is a transform, never an execution reference / Fill /
        slippage / contract execution price -- whatever its source. Execution
        prices are selected only by the deterministic C++ execution path from
        ``RawContract`` market data (BOUNDARY_CONTRACT E; ``make_fill`` is the
        final guard). Use the ``RawContract`` MarketEvent for the executable
        price; ``self.source_safety.execution_price_safe`` only reports whether
        the *source feed* was a real contract."""
        raise FeatureSafetyError(
            "no FeatureFrame is execution-price-safe: "
            + FEATURE_NEVER_EXECUTION_PRICE
            + f" (source_execution_price_safe={self.lineage.source_execution_price_safe}; "
            "features drive Signals, not Fills)"
        )

    def cache_keys(self) -> dict[str, str]:
        return {
            name: feature_cache_key(
                self.lineage.source_fingerprint, spec, as_of_ts_ns=self.lineage.as_of_ts_ns
            )
            for name, spec in self.specs.items()
        }


def build_lineage(
    *, source, specs: dict[str, FeatureSpec], as_of_ts_ns: int | None,
) -> FeatureLineage:
    ss = source.safety()
    return FeatureLineage(
        source_price_domain=source.price_domain.value,
        source_adjustment_mode=source.adjustment_mode,
        source_paths=list(source.source_paths),
        source_fingerprint=source.fingerprint(),
        n_source_rows=len(source.frame),
        as_of_ts_ns=as_of_ts_ns,
        source_point_in_time_safe=ss.point_in_time_safe,
        source_signal_safe=ss.signal_safe,
        source_execution_price_safe=ss.execution_price_safe,
        code_commit=code_commit(),
        feature_names=sorted(specs),
        feature_specs=[json.loads(s.canonical_json()) for s in specs.values()],
        identity=dict(source.identity),
    )

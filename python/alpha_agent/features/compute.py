"""FeatureComputer: turn a SourceSeries + typed FeatureSpecs into a FeatureFrame.

Guarantees
----------
* **Point-in-time.** The source is truncated to ``as_of_ts_ns`` before anything
  is computed; every window primitive is backward-looking; a value at ``T`` is
  bit-identical whether the series ends at ``T`` or later (deterministic replay).
* **No silent NaN/inf.** ``+/-inf`` is coerced to missing and recorded. Missing
  values are explicit (``mask == False``); rows are never dropped.
* **Price-domain enforcement.** A spec that does not permit the source domain is
  a hard error. A back-adjusted source is signal-safe (point-in-time mode) or
  look-ahead unsafe (retrospective mode); never execution-price safe.
* **Deterministic names.** The same spec always produces the same column name; a
  collision between two differing specs is a hard error.
"""
from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np
import pandas as pd

from alpha_agent.features.enums import SessionPolicy
from alpha_agent.features.frame import FeatureFrame, build_lineage, feature_cache_key
from alpha_agent.features.qa import EXTREME_VALUE_ABS_LIMIT, FeatureIssueKind, FeatureQAReport
from alpha_agent.features.registry import REGISTRY, FeatureComputeContext, FeatureRegistry
from alpha_agent.features.safety import combine
from alpha_agent.features.source import (
    SourceSeries,
    assert_no_execution_identity_leak,
    coerce_source,
)
from alpha_agent.features.spec import FeatureSpec


class FeaturePriceDomainError(ValueError):
    """The source price domain is not permitted by a requested feature."""


class LookaheadUnsafeError(ValueError):
    """A look-ahead-unsafe feature was requested under a point-in-time contract --
    either the feature itself is retrospective, or the source is a
    retrospective_research back-adjusted series (future roll gaps applied)."""


class FeatureNameCollision(ValueError):
    """Two different specs resolved to the same canonical feature name."""


class FeatureComputer:
    def __init__(self, registry: FeatureRegistry = REGISTRY) -> None:
        self.registry = registry

    def compute(
        self,
        source: SourceSeries | pd.DataFrame,
        specs: list[FeatureSpec],
        *,
        as_of_ts_ns: int | None = None,
        require_point_in_time: bool = True,
        cache: MutableMapping[str, pd.Series] | None = None,
        price_domain=None,
        adjustment_mode: str | None = None,
    ) -> FeatureFrame:
        src: SourceSeries = coerce_source(
            source, price_domain=price_domain, adjustment_mode=adjustment_mode
        )
        src = src.truncate_as_of(as_of_ts_ns)
        frame = src.frame
        idx = frame.index

        resolved: dict[str, FeatureSpec] = {}
        for spec in specs:
            fdef = self.registry.get(spec.kind)
            fdef.resolve_params(spec.params)  # validate early / raise on unknown params
            name = fdef.canonical_name(spec)
            if name in resolved and resolved[name].canonical_json() != spec.canonical_json():
                raise FeatureNameCollision(
                    f"specs {resolved[name].canonical_json()} and {spec.canonical_json()} "
                    f"both resolve to feature name {name!r}"
                )
            resolved[name] = spec

        qa = FeatureQAReport()
        lineage = build_lineage(source=src, specs=resolved, as_of_ts_ns=as_of_ts_ns)
        fingerprint = lineage.source_fingerprint
        src_safety = src.safety()

        seg_cache: dict = {}
        feat_cols: dict[str, pd.Series] = {}
        mask_cols: dict[str, pd.Series] = {}
        avail_rows: dict[str, object] = {}
        metadata: dict = {}
        safety: dict = {}

        for name, spec in resolved.items():
            fdef = self.registry.get(spec.kind)
            params = fdef.resolve_params(spec.params)
            metadata[name] = fdef.metadata(spec, source_safety=src_safety)
            feat_pit = fdef.is_point_in_time_safe(spec)
            safety[name] = combine(
                src_safety, feature=name, feature_point_in_time_safe=feat_pit,
                feature_reason=fdef.safety_reason(spec),
            )

            # -- price-domain gate
            allowed = fdef.effective_domains(spec)
            if src.price_domain not in allowed:
                qa.add(
                    FeatureIssueKind.PRICE_DOMAIN_MISMATCH, name,
                    f"source domain {src.price_domain.value} not in {[d.value for d in allowed]}",
                )
                raise FeaturePriceDomainError(
                    f"feature {name!r} does not permit source price domain "
                    f"{src.price_domain.value}"
                )

            # -- look-ahead gate (feature-inherent OR retrospective back-adjusted source)
            if require_point_in_time and not safety[name].point_in_time_safe:
                reasons = safety[name].reasons
                qa.add(
                    FeatureIssueKind.LOOKAHEAD_UNSAFE_REQUEST, name,
                    f"look-ahead-unsafe feature request; reasons={reasons}",
                )
                raise LookaheadUnsafeError(
                    f"feature {name!r} is not point-in-time safe under a point-in-time "
                    f"contract (reasons={reasons}); pass require_point_in_time=False for "
                    f"explicit offline research"
                )

            # -- session / missing-data QA (recorded, never auto-fixed)
            policy = fdef.effective_session_policy(spec)
            if policy not in seg_cache:
                seg_cache[policy] = src.segment_ids(policy)
            seg = seg_cache[policy]
            _record_window_context(src, seg, policy, name, qa)

            # -- cache
            key = feature_cache_key(fingerprint, spec, as_of_ts_ns=as_of_ts_ns)
            if cache is not None and key in cache:
                series = cache[key].reindex(idx).astype("float64")
            else:
                ctx = FeatureComputeContext(
                    source=src, frame=frame, segment_id=seg, params=params,
                    price_field=spec.price_field, spec=spec, name=name, qa=qa,
                    interval_ns=src.interval_ns, min_obs=fdef.effective_min_obs(spec),
                )
                series = pd.Series(fdef.compute(ctx), index=idx, dtype="float64")
                if cache is not None:
                    cache[key] = series.copy()

            series, present = _finalize(series, name, qa, fdef.effective_min_obs(spec))
            feat_cols[name] = series
            mask_cols[name] = present
            first_ts = frame.loc[present, "ts_event_ns"]
            avail_rows[name] = int(first_ts.iloc[0]) if len(first_ts) else None

        feature_names = sorted(feat_cols)
        assert_no_execution_identity_leak(feature_names)

        features = pd.DataFrame({n: feat_cols[n] for n in feature_names}, index=idx)
        mask = pd.DataFrame({n: mask_cols[n] for n in feature_names}, index=idx)
        availability = pd.DataFrame(
            {"first_available_ts_ns": {n: avail_rows[n] for n in feature_names}}
        )
        lineage.feature_names = feature_names

        return FeatureFrame(
            identifiers=src.identifiers(),
            features=features,
            mask=mask,
            availability=availability,
            metadata=metadata,
            specs={n: resolved[n] for n in feature_names},
            lineage=lineage,
            safety={n: safety[n] for n in feature_names},
            qa=qa,
        )


def _record_window_context(src, seg, policy, name: str, qa: FeatureQAReport) -> None:
    gaps = src.gap_breaks()
    n_gaps = int(gaps.to_numpy().sum())
    n_resets = int(seg.iloc[-1]) if len(seg) else 0
    if n_gaps:
        if policy in (SessionPolicy.RESET_ON_GAP, SessionPolicy.RESET_ON_SESSION_AND_GAP):
            qa.add(FeatureIssueKind.MISSING_BAR_GAP, name,
                   f"{n_gaps} missing-bar gap(s); the window is reset at each", count=n_gaps)
        else:
            qa.add(FeatureIssueKind.STALE_INPUT, name,
                   f"{n_gaps} missing-bar gap(s) and policy={policy.value}: a window may span "
                   f"stale data", count=n_gaps)
    if policy is not SessionPolicy.CONTINUOUS and n_resets:
        qa.add(FeatureIssueKind.SESSION_BOUNDARY_RESET, name,
               f"{n_resets} window reset(s) under policy={policy.value}", count=n_resets)


def _finalize(
    series: pd.Series, name: str, qa: FeatureQAReport, min_obs: int
) -> tuple[pd.Series, pd.Series]:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    inf_mask = np.isinf(s)
    if inf_mask.any():
        qa.add(FeatureIssueKind.INF_IN_OUTPUT, name,
               f"{int(inf_mask.sum())} inf value(s) coerced to missing", count=int(inf_mask.sum()))
        s = s.mask(inf_mask)

    present = s.notna()
    if present.any():
        first = present.to_numpy().argmax()
        tail_missing = int((~present.to_numpy()[first:]).sum())
        if tail_missing:
            qa.add(FeatureIssueKind.NAN_IN_OUTPUT, name,
                   f"{tail_missing} missing value(s) after the feature first became available",
                   count=tail_missing)
    else:
        qa.add(FeatureIssueKind.INSUFFICIENT_LOOKBACK, name,
               f"feature never had {min_obs} observations in any segment")

    extreme = present & (s.abs() > EXTREME_VALUE_ABS_LIMIT)
    if extreme.any():
        qa.add(FeatureIssueKind.EXTREME_VALUE, name,
               f"{int(extreme.sum())} value(s) exceed |{EXTREME_VALUE_ABS_LIMIT:g}| "
               f"(kept, not clipped)",
               count=int(extreme.sum()))
    return s, present


def compute_features(
    source: SourceSeries | pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    as_of_ts_ns: int | None = None,
    require_point_in_time: bool = True,
    cache: MutableMapping | None = None,
    registry: FeatureRegistry = REGISTRY,
    price_domain=None,
    adjustment_mode: str | None = None,
) -> FeatureFrame:
    return FeatureComputer(registry).compute(
        source, specs, as_of_ts_ns=as_of_ts_ns,
        require_point_in_time=require_point_in_time, cache=cache, price_domain=price_domain,
        adjustment_mode=adjustment_mode,
    )

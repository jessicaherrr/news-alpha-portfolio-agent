"""The pooled, causal model matrix for one Phase 15 primary family.

The five per-root :class:`~alpha_agent.ml.corpus.CorpusResult`s of one primary
family are pooled into ONE chronologically ordered dataset with ``root_symbol``
as a categorical feature (prompt 15A s.6, s.10). Every numeric column is a frozen
Phase 15 :class:`FeatureSpec` read at the decision bar, so
``feature_timestamp <= decision_timestamp`` holds by construction and is
re-asserted per event.

Missing feature -> the event is EXCLUDED with the typed
``FEATURES_MISSING_AT_DECISION`` reason at corpus-build time; nothing is imputed
here (imputation is a fitted transform and lives strictly inside a training
fold).

Providers:

* :class:`SyntheticFeatureProvider` -- deterministic feature vectors keyed on the
  event id, weakly predictive of the label by a tunable ``signal_strength`` so
  the integration harness can exercise both the PASS and non-PASS adjudication
  paths. ``--synthetic-integration`` only.
* :class:`Phase135cFeatureProvider` -- real features through the Feature Engine on
  the frozen Phase 13.5C daily signal series. ``--run-real`` only.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from alpha_agent.ml.features import MLFeatureSetSpec
from alpha_agent.ml.labels import MetaLabelEvent
from alpha_agent.ml.manifest import ML_FEATURE_SET, ROOTS
from alpha_agent.ml.splits import EventTimeline

#: The two frozen regime-baseline inputs, by feature-set alias (prompt 15A s.7).
REGIME_INPUT_ALIASES: tuple[str, ...] = ("rvol_20", "trend_strength_50_200")


class PooledMatrix(BaseModel):
    """One primary family's pooled, chronologically ordered model matrix."""

    model_config = {"frozen": True, "extra": "forbid"}

    primary_family: str
    feature_set_fingerprint: str
    ordered_aliases: tuple[str, ...]
    fitted_columns: tuple[str, ...]          # numeric + one-hot categoricals
    allowlist_columns: tuple[str, ...]       # the conceptual inputs (11 + 2 cats)
    event_ids: tuple[str, ...]
    root_symbols: tuple[str, ...]
    n_events: int
    n_positive: int
    per_root_counts: dict[str, int]
    regime_input_column_indices: tuple[int, ...]
    #: opaque handles; the arrays live on the companion object
    matrix_fingerprint: str

    @property
    def positive_rate(self) -> float:
        return self.n_positive / self.n_events if self.n_events else 0.0


class PooledDataset:
    """The pooled matrix plus its dense arrays (not a pydantic model)."""

    def __init__(
        self,
        meta: PooledMatrix,
        *,
        X: np.ndarray,
        y: np.ndarray,
        timeline: EventTimeline,
        primary_strategy_fingerprints: tuple[str, ...],
    ):
        self.meta = meta
        self.X = X
        self.y = y
        self.timeline = timeline
        self.primary_strategy_fingerprints = primary_strategy_fingerprints


class FeatureProvider(Protocol):
    provider_tag: str

    def features_for(
        self, event: MetaLabelEvent, feature_set: MLFeatureSetSpec
    ) -> np.ndarray | None: ...


# --------------------------------------------------------------------------
# synthetic provider
# --------------------------------------------------------------------------
class SyntheticFeatureProvider:
    """Deterministic, weakly-predictive feature vectors. TESTS ONLY."""

    provider_tag = "synthetic_feature_provider__software_fixture_only"

    def __init__(self, *, signal_strength: float = 0.55, noise: float = 1.0):
        self._signal = float(signal_strength)
        self._noise = float(noise)

    def features_for(
        self, event: MetaLabelEvent, feature_set: MLFeatureSetSpec
    ) -> np.ndarray | None:
        seed = int.from_bytes(
            hashlib.sha256(event.event_id.encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed)
        k = len(feature_set.ordered_aliases)
        x = self._noise * rng.standard_normal(k)
        # make the first two features weakly informative of the label
        x[0] += self._signal * (2 * event.label - 1)
        x[1] += 0.5 * self._signal * (2 * event.label - 1)
        return x


# --------------------------------------------------------------------------
# real provider (gated)
# --------------------------------------------------------------------------
class Phase135cFeatureProvider:
    """Real Phase 15 features via the Feature Engine on the daily signal series.

    Built for ``--run-real`` only. Given the per-root daily signal frame it
    computes the frozen ordered feature set once and serves the row whose
    ``ts_event_ns`` matches (or is the last on or before) the decision bar. A
    decision with no covered feature row returns ``None`` -> typed exclusion.
    """

    provider_tag = "phase_13_5c_feature_provider"

    def __init__(self, per_root_feature_frames: dict[str, object]):
        # {root: pandas.DataFrame indexed by ts_event_ns with the ordered aliases}
        self._frames = per_root_feature_frames

    def features_for(
        self, event: MetaLabelEvent, feature_set: MLFeatureSetSpec
    ) -> np.ndarray | None:
        frame = self._frames.get(event.root_symbol)
        if frame is None:
            return None
        import pandas as pd  # local: real path only

        ts = int(event.feature_timestamp)
        col_ts = frame["ts_event_ns"].to_numpy("int64")
        pos = int(np.searchsorted(col_ts, ts, side="right")) - 1
        if pos < 0:
            return None
        if int(col_ts[pos]) > ts:  # never a future row
            return None
        row = frame.iloc[pos]
        try:
            vec = np.asarray(
                [float(row[a]) for a in feature_set.ordered_aliases], dtype=float
            )
        except KeyError:
            return None
        if not np.all(np.isfinite(vec)):
            return None
        del pd
        return vec


# --------------------------------------------------------------------------
# the pooled builder
# --------------------------------------------------------------------------
def build_pooled_matrix(
    primary_family: str,
    events_by_root: dict[str, tuple[MetaLabelEvent, ...]],
    *,
    provider: FeatureProvider,
    feature_set: MLFeatureSetSpec = ML_FEATURE_SET,
    training_roots: tuple[str, ...] = ROOTS,
) -> tuple[PooledDataset, tuple[MetaLabelEvent, ...]]:
    """Pool the five roots' events into one chronological causal matrix.

    Returns ``(dataset, feature_missing_events)`` -- any event whose provider
    returned ``None`` is dropped from the matrix and returned so the caller can
    record the typed ``FEATURES_MISSING_AT_DECISION`` exclusion.
    """
    aliases = feature_set.ordered_aliases
    root_index = {r: i for i, r in enumerate(training_roots)}
    regime_idx = tuple(aliases.index(a) for a in REGIME_INPUT_ALIASES)

    rows: list[tuple[int, str, str, int, int, int, np.ndarray]] = []
    missing: list[MetaLabelEvent] = []
    for root in training_roots:
        for ev in events_by_root.get(root, ()):
            if ev.feature_timestamp > ev.decision_timestamp:
                raise ValueError(f"event {ev.event_id}: feature ts after decision ts")
            vec = provider.features_for(ev, feature_set)
            if vec is None:
                missing.append(ev)
                continue
            vec = np.asarray(vec, dtype=float)
            if vec.shape != (len(aliases),) or not np.all(np.isfinite(vec)):
                missing.append(ev)
                continue
            rows.append(
                (
                    ev.decision_timestamp,
                    ev.event_id,
                    root,
                    1 if ev.primary_side > 0 else 0,
                    ev.label,
                    ev.label_end_timestamp,
                    vec,
                )
            )

    rows.sort(key=lambda r: (r[0], r[1]))
    n = len(rows)
    n_num = len(aliases)
    n_root = len(training_roots)
    X = np.zeros((n, n_num + n_root + 1), dtype=float)
    y = np.zeros(n, dtype=int)
    event_ids: list[str] = []
    roots_out: list[str] = []
    decision_ts: list[int] = []
    label_start: list[int] = []
    label_end: list[int] = []
    per_root: dict[str, int] = {r: 0 for r in training_roots}

    for i, (d_ts, eid, root, side_code, label, le_ts, vec) in enumerate(rows):
        X[i, :n_num] = vec
        X[i, n_num + root_index[root]] = 1.0
        X[i, n_num + n_root] = float(side_code)
        y[i] = label
        event_ids.append(eid)
        roots_out.append(root)
        decision_ts.append(d_ts)
        label_start.append(d_ts)
        label_end.append(le_ts)
        per_root[root] += 1

    timeline = EventTimeline(
        decision_ts_ns=tuple(decision_ts),
        label_start_ts_ns=tuple(label_start),
        label_end_ts_ns=tuple(label_end),
    )
    fitted_columns = (
        *aliases,
        *(f"root_symbol__{r}" for r in training_roots),
        "primary_side__long",
    )
    matrix_fp = hashlib.sha256(
        np.ascontiguousarray(np.round(X, 10)).tobytes()
        + np.ascontiguousarray(y).tobytes()
        + "|".join(event_ids).encode()
    ).hexdigest()
    meta = PooledMatrix(
        primary_family=primary_family,
        feature_set_fingerprint=feature_set.identity(),
        ordered_aliases=aliases,
        fitted_columns=fitted_columns,
        allowlist_columns=(*aliases, *feature_set.categoricals),
        event_ids=tuple(event_ids),
        root_symbols=tuple(roots_out),
        n_events=n,
        n_positive=int(y.sum()),
        per_root_counts=per_root,
        regime_input_column_indices=regime_idx,
        matrix_fingerprint=f"mlmatrix1:{matrix_fp}",
    )
    primaries = tuple(
        sorted({ev.strategy_fingerprint for evs in events_by_root.values() for ev in evs})
    )
    dataset = PooledDataset(
        meta, X=X, y=y, timeline=timeline, primary_strategy_fingerprints=primaries
    )
    return dataset, tuple(missing)

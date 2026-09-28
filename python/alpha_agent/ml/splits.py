"""Nested chronological validation with purge + embargo (prompt 15A s.10).

::

    OUTER fold k   train  [ ................. ] |embargo| test [ ..... ]
                          purged of any event whose LABEL WINDOW
                          overlaps the test block
    INNER fold j   the same construction, built INSIDE outer-train only

There is no random split anywhere. :func:`refuse_random_split` exists so that a
shuffled or K-fold request fails loudly rather than being silently reinterpreted.

Purge vs embargo -- they are different and Phase 15 needs both
---------------------------------------------------------------
* **Purge** removes a training event whose *label window*
  ``[label_start, label_end]`` overlaps the test block. A meta-label resolves
  when the primary episode's last fill closes, which can be months after the
  decision; without purging, a training row's outcome would be partly determined
  by the very period being scored.
* **Embargo** additionally removes training events whose label window ends
  inside a buffer around the test block, covering serial dependence that the
  label window itself does not express (overlapping volatility state, a shared
  roll, a straddled macro shock).

Both operate on the label window, never on the decision timestamp alone -- using
the decision timestamp would leave exactly the long-horizon leakage that matters
most here.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator

from alpha_agent.ml.enums import FoldKind
from alpha_agent.ml.errors import FoldIsolationError, RandomSplitForbidden
from alpha_agent.ml.guards import (
    DEVELOPMENT_CORPUS_END_NS,
    DEVELOPMENT_CORPUS_START_NS,
    assert_within_development_corpus,
)
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.splits import NS_PER_DAY

NESTED_CV_SCHEMA = "ml-nested-cv/1"


def refuse_random_split(*_args: object, **_kwargs: object) -> None:
    """Market time series are never shuffled. This is the tripwire."""
    raise RandomSplitForbidden(
        "a random / shuffled / K-fold split is forbidden for market time series: it "
        "trains on the future to predict the past and destroys the serial structure "
        "the label horizon depends on. Phase 15 uses chronological folds only."
    )


class NestedCVSpec(BaseModel):
    """The frozen nested chronological design."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = NESTED_CV_SCHEMA
    corpus_start_ts_ns: int = DEVELOPMENT_CORPUS_START_NS
    corpus_end_ts_ns: int = DEVELOPMENT_CORPUS_END_NS          # exclusive
    #: Outer test blocks, chronological, non-overlapping, given as explicit
    #: [start, end) epoch-ns spans so the design is auditable by eye.
    outer_test_blocks: tuple[tuple[int, int], ...]
    n_inner_folds: int = Field(default=3, ge=2, le=10)
    #: Embargo either side of every test block, in calendar days.
    embargo_days: int = Field(default=10, ge=0, le=365)
    #: An outer training window shorter than this is not trained.
    min_train_days: int = Field(default=365, ge=1)
    #: Refuse to train a fold with fewer labelled training events than this.
    min_train_events: int = Field(default=100, ge=1)
    #: Refuse to score a fold with fewer labelled test events than this.
    min_test_events: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def _check(self) -> NestedCVSpec:
        if not self.outer_test_blocks:
            raise ValueError("at least one outer test block is required")
        for a, b in self.outer_test_blocks:
            if b <= a:
                raise ValueError(f"outer test block [{a}, {b}) is empty or reversed")
            if a < self.corpus_start_ts_ns or b > self.corpus_end_ts_ns:
                raise ValueError(
                    f"outer test block [{a}, {b}) escapes the development corpus "
                    f"[{self.corpus_start_ts_ns}, {self.corpus_end_ts_ns})"
                )
        for (a1, b1), (a2, _b2) in zip(self.outer_test_blocks, self.outer_test_blocks[1:]):
            if a2 < b1:
                raise ValueError("outer test blocks must be chronological and non-overlapping")
            del a1
        if self.corpus_end_ts_ns > DEVELOPMENT_CORPUS_END_NS:
            raise ValueError(
                "the Phase 15 development corpus ends at 2025-01-01 exclusive; 2025 is the "
                "LOCKED_FINAL_HOLDOUT"
            )
        return self

    @property
    def embargo_ns(self) -> int:
        return self.embargo_days * NS_PER_DAY

    @property
    def n_outer_folds(self) -> int:
        return len(self.outer_test_blocks)

    def identity(self) -> str:
        return fingerprint(
            "mlnestedcv1",
            {
                "schema_version": self.schema_version,
                "corpus": [self.corpus_start_ts_ns, self.corpus_end_ts_ns],
                "outer_test_blocks": [list(b) for b in self.outer_test_blocks],
                "n_inner_folds": self.n_inner_folds,
                "embargo_days": self.embargo_days,
                "min_train_days": self.min_train_days,
                "min_train_events": self.min_train_events,
                "min_test_events": self.min_test_events,
            },
        )


class Fold(BaseModel):
    """One resolved fold: explicit row index sets, already purged and embargoed."""

    model_config = {"frozen": True, "extra": "forbid"}

    kind: FoldKind
    fold_index: int = Field(ge=0)
    parent_outer_index: int | None = None
    test_start_ts_ns: int
    test_end_ts_ns: int                          # exclusive
    train_row_indices: tuple[int, ...]
    test_row_indices: tuple[int, ...]
    purged_row_indices: tuple[int, ...] = ()
    embargoed_row_indices: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _disjoint(self) -> Fold:
        overlap = set(self.train_row_indices) & set(self.test_row_indices)
        if overlap:
            raise FoldIsolationError(
                f"fold {self.kind.value}[{self.fold_index}] has {len(overlap)} row(s) in both "
                f"train and test (first {sorted(overlap)[:5]})"
            )
        return self

    def identity(self) -> str:
        return fingerprint(
            "mlfold1",
            {
                "kind": self.kind.value,
                "fold_index": self.fold_index,
                "parent_outer_index": self.parent_outer_index,
                "test_span": [self.test_start_ts_ns, self.test_end_ts_ns],
                "train_row_indices": list(self.train_row_indices),
                "test_row_indices": list(self.test_row_indices),
            },
        )


class EventTimeline(BaseModel):
    """The decision and label timestamps the split planner reasons over.

    Row ``i`` is the ``i``-th labelled event in chronological decision order.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    decision_ts_ns: tuple[int, ...]
    label_start_ts_ns: tuple[int, ...]
    label_end_ts_ns: tuple[int, ...]

    @model_validator(mode="after")
    def _check(self) -> EventTimeline:
        n = len(self.decision_ts_ns)
        if not (len(self.label_start_ts_ns) == len(self.label_end_ts_ns) == n):
            raise ValueError("timeline arrays must be the same length")
        for i in range(n):
            if self.label_end_ts_ns[i] < self.label_start_ts_ns[i]:
                raise ValueError(f"event {i}: label window ends before it starts")
            if self.label_end_ts_ns[i] <= self.decision_ts_ns[i]:
                raise ValueError(f"event {i}: label does not resolve after the decision")
        if any(a > b for a, b in zip(self.decision_ts_ns, self.decision_ts_ns[1:])):
            raise ValueError("events must be in non-decreasing decision-time order")
        assert_within_development_corpus(self.decision_ts_ns, what="EventTimeline.decision_ts_ns")
        assert_within_development_corpus(self.label_end_ts_ns, what="EventTimeline.label_end_ts_ns")
        return self

    @property
    def n_events(self) -> int:
        return len(self.decision_ts_ns)


def _select_train_rows(
    timeline: EventTimeline,
    *,
    train_span: tuple[int, int],
    test_start: int,
    test_end: int,
    embargo_ns: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return ``(train, purged, embargoed)`` row indices for one test block.

    A candidate training row must (a) lie in the training span by DECISION time,
    (b) survive purging -- its label window must not overlap the test block, and
    (c) survive the embargo buffer around the test block.
    """
    lo, hi = train_span
    train: list[int] = []
    purged: list[int] = []
    embargoed: list[int] = []
    emb_lo = test_start - embargo_ns
    emb_hi = test_end + embargo_ns

    for i in range(timeline.n_events):
        d = timeline.decision_ts_ns[i]
        ls = timeline.label_start_ts_ns[i]
        le = timeline.label_end_ts_ns[i]
        if not (lo <= d < hi):
            continue
        if ls < test_end and le > test_start:      # label window overlaps the test block
            purged.append(i)
            continue
        if le > emb_lo and ls < emb_hi:            # inside the embargo buffer
            embargoed.append(i)
            continue
        train.append(i)
    return train, purged, embargoed


def build_outer_folds(spec: NestedCVSpec, timeline: EventTimeline) -> tuple[Fold, ...]:
    """Expanding-origin chronological outer folds over the development corpus.

    Every outer training window starts at the corpus start; only the end moves.
    Training is always strictly before testing, then purged, then embargoed.
    """
    folds: list[Fold] = []
    for k, (test_start, test_end) in enumerate(spec.outer_test_blocks):
        train, purged, embargoed = _select_train_rows(
            timeline,
            train_span=(spec.corpus_start_ts_ns, test_start),
            test_start=test_start,
            test_end=test_end,
            embargo_ns=spec.embargo_ns,
        )
        test = [
            i for i in range(timeline.n_events)
            if test_start <= timeline.decision_ts_ns[i] < test_end
        ]
        folds.append(
            Fold(
                kind=FoldKind.OUTER,
                fold_index=k,
                test_start_ts_ns=test_start,
                test_end_ts_ns=test_end,
                train_row_indices=tuple(train),
                test_row_indices=tuple(test),
                purged_row_indices=tuple(purged),
                embargoed_row_indices=tuple(embargoed),
            )
        )
    return tuple(folds)


def build_inner_folds(
    spec: NestedCVSpec, timeline: EventTimeline, outer: Fold
) -> tuple[Fold, ...]:
    """Chronological inner folds built INSIDE one outer training window.

    The inner window is split into ``n_inner_folds + 1`` equal calendar blocks;
    blocks 1..n are inner test blocks with expanding-origin training before each,
    purged and embargoed by the same rule. The outer test block is never visible
    here -- that is what makes the outer estimate honest.
    """
    inner_lo = spec.corpus_start_ts_ns
    inner_hi = outer.test_start_ts_ns - spec.embargo_ns
    if inner_hi <= inner_lo:
        return ()
    n_blocks = spec.n_inner_folds + 1
    width = (inner_hi - inner_lo) // n_blocks
    if width <= 0:
        return ()

    folds: list[Fold] = []
    for j in range(spec.n_inner_folds):
        test_start = inner_lo + width * (j + 1)
        test_end = inner_lo + width * (j + 2) if j + 2 < n_blocks else inner_hi
        train, purged, embargoed = _select_train_rows(
            timeline,
            train_span=(inner_lo, test_start),
            test_start=test_start,
            test_end=test_end,
            embargo_ns=spec.embargo_ns,
        )
        test = [
            i for i in range(timeline.n_events)
            if test_start <= timeline.decision_ts_ns[i] < test_end
        ]
        folds.append(
            Fold(
                kind=FoldKind.INNER,
                fold_index=j,
                parent_outer_index=outer.fold_index,
                test_start_ts_ns=test_start,
                test_end_ts_ns=test_end,
                train_row_indices=tuple(train),
                test_row_indices=tuple(test),
                purged_row_indices=tuple(purged),
                embargoed_row_indices=tuple(embargoed),
            )
        )
    return tuple(folds)


def assert_fold_is_causal(fold: Fold, timeline: EventTimeline) -> None:
    """Every training row decides before the test block and resolves before it.

    This is the property the whole design exists to guarantee, asserted directly
    rather than inferred from how the fold was built.
    """
    for i in fold.train_row_indices:
        if timeline.decision_ts_ns[i] >= fold.test_start_ts_ns:
            raise FoldIsolationError(
                f"training row {i} decides at {timeline.decision_ts_ns[i]}, at or after the "
                f"test block start {fold.test_start_ts_ns}"
            )
        if timeline.label_end_ts_ns[i] > fold.test_start_ts_ns:
            raise FoldIsolationError(
                f"training row {i} has a label horizon {timeline.label_end_ts_ns[i]} reaching "
                f"into the test block starting {fold.test_start_ts_ns}; purge/embargo failed"
            )
    for i in fold.test_row_indices:
        if not (fold.test_start_ts_ns <= timeline.decision_ts_ns[i] < fold.test_end_ts_ns):
            raise FoldIsolationError(f"test row {i} decides outside its own test block")


def assert_global_temporal_isolation(
    fold: Fold, timeline: EventTimeline, root_symbols: tuple[str, ...] | list[str]
) -> dict[str, dict[str, int]]:
    """Chronological isolation holds GLOBALLY across a POOLED root universe.

    Phase 15 trains one model over ES/NQ/CL/GC/ZN with ``root_symbol`` as a
    categorical feature. Pooling is a modelling choice; it must not become a
    temporal one. If 2023 is an outer TEST block, then 2023 events of *every*
    root are excluded from training -- not merely those of the root the
    economics are later measured on.

    The boundary is a timestamp, so the property is true by construction. It is
    still asserted directly and PER ROOT, because "true by construction" is the
    claim a pooled design most easily breaks later (a per-root split, a
    root-specific warm-up, a resampled join), and a silent breach here would make
    every out-of-sample number in the phase meaningless.

    Returns a per-root census -- ``{root: {n_train, n_test, first/last ts}}`` --
    so the isolation can be reported as evidence rather than asserted in prose.
    """
    if len(root_symbols) != timeline.n_events:
        raise ValueError(
            f"root_symbols has {len(root_symbols)} entries for {timeline.n_events} events"
        )

    census: dict[str, dict[str, int]] = {}
    for i in fold.train_row_indices:
        root = root_symbols[i]
        if timeline.decision_ts_ns[i] >= fold.test_start_ts_ns:
            raise FoldIsolationError(
                f"pooled temporal isolation broken: root {root} contributes training row {i} "
                f"deciding at {timeline.decision_ts_ns[i]}, at or after the test block start "
                f"{fold.test_start_ts_ns}. Pooling across roots never weakens chronological "
                "isolation: at a given test block, NO root may contribute events from that "
                "test/future period to training."
            )
        if timeline.label_end_ts_ns[i] > fold.test_start_ts_ns:
            raise FoldIsolationError(
                f"pooled temporal isolation broken: root {root} contributes training row {i} "
                f"whose label horizon {timeline.label_end_ts_ns[i]} reaches into the test block "
                f"starting {fold.test_start_ts_ns}"
            )
        c = census.setdefault(root, {"n_train": 0, "n_test": 0})
        c["n_train"] += 1
    for i in fold.test_row_indices:
        root = root_symbols[i]
        if not (fold.test_start_ts_ns <= timeline.decision_ts_ns[i] < fold.test_end_ts_ns):
            raise FoldIsolationError(
                f"root {root} test row {i} decides outside its own test block"
            )
        c = census.setdefault(root, {"n_train": 0, "n_test": 0})
        c["n_test"] += 1
    return census


def pooled_isolation_report(
    folds: tuple[Fold, ...], timeline: EventTimeline, root_symbols: tuple[str, ...] | list[str]
) -> tuple[dict[str, object], ...]:
    """Per-fold, per-root isolation evidence for the audit artifacts."""
    return tuple(
        {
            "fold_kind": f.kind.value,
            "fold_index": f.fold_index,
            "test_start_ts_ns": f.test_start_ts_ns,
            "test_end_ts_ns": f.test_end_ts_ns,
            "per_root": assert_global_temporal_isolation(f, timeline, root_symbols),
        }
        for f in folds
    )


def assert_no_train_test_overlap_across_folds(folds: tuple[Fold, ...]) -> None:
    """Outer test blocks partition the evaluation support: no row is scored twice."""
    seen: dict[int, int] = {}
    for f in folds:
        for i in f.test_row_indices:
            if i in seen:
                raise FoldIsolationError(
                    f"row {i} is a test row in both outer fold {seen[i]} and {f.fold_index}; "
                    "an OOF prediction must be produced exactly once"
                )
            seen[i] = f.fold_index


def evaluation_support_row_indices(folds: tuple[Fold, ...]) -> np.ndarray:
    """The rows that receive an OOF prediction, in chronological order."""
    rows = sorted({i for f in folds for i in f.test_row_indices})
    return np.asarray(rows, dtype=int)

"""Out-of-fold predictions, with per-row provenance.

An OOF prediction is only meaningful if you can say *which* model produced it and
prove that model never saw the row. So every prediction carries its producing
fold and the fingerprint of the model artifact that emitted it, and
:func:`assert_oof_provenance` checks each row against that fold's own training
set. That turns "these are out-of-sample" from an assertion in a document into a
property the test suite verifies.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator

from alpha_agent.ml.enums import MetaLabelAction
from alpha_agent.ml.errors import FoldIsolationError
from alpha_agent.ml.splits import Fold
from alpha_agent.validation.fingerprint import fingerprint

ML_PREDICTION_FRAME_SCHEMA = "ml-prediction-frame/1"


class MLPrediction(BaseModel):
    """One out-of-fold decision about one primary signal."""

    model_config = {"frozen": True, "extra": "forbid"}

    event_id: str
    row_index: int = Field(ge=0)
    root_symbol: str
    decision_timestamp: int = Field(gt=0)
    #: calibration / audit only -- never converted into position size
    probability_take: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    action: MetaLabelAction
    label: int | None = None                    # the realised label, for scoring
    outer_fold_index: int = Field(ge=0)
    producing_model_fingerprint: str

    @model_validator(mode="after")
    def _action_matches(self) -> MLPrediction:
        expected = (
            MetaLabelAction.TAKE if self.probability_take >= self.threshold
            else MetaLabelAction.SKIP
        )
        if self.action is not expected:
            raise ValueError(
                f"event {self.event_id}: action {self.action.value} disagrees with "
                f"p={self.probability_take} vs threshold={self.threshold}. The action is a "
                "deterministic function of the probability and the frozen threshold; it is "
                "never overridden."
            )
        return self


class MLPredictionFrame(BaseModel):
    """Every OOF prediction of one ML experiment, chronologically ordered."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = ML_PREDICTION_FRAME_SCHEMA
    experiment_label: str = ""                  # cosmetic
    predictions: tuple[MLPrediction, ...]

    @model_validator(mode="after")
    def _ordered_and_unique(self) -> MLPredictionFrame:
        seen: set[str] = set()
        prev = 0
        for p in self.predictions:
            if p.event_id in seen:
                raise FoldIsolationError(
                    f"event {p.event_id} received more than one OOF prediction; each row is "
                    "scored exactly once, by exactly one outer fold"
                )
            seen.add(p.event_id)
            if p.decision_timestamp < prev:
                raise ValueError("OOF predictions must be in chronological decision order")
            prev = p.decision_timestamp
        return self

    @property
    def n(self) -> int:
        return len(self.predictions)

    @property
    def take_rate(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.action is MetaLabelAction.TAKE for p in self.predictions) / len(self.predictions)

    def probabilities(self) -> np.ndarray:
        return np.asarray([p.probability_take for p in self.predictions], dtype=float)

    def labels(self) -> np.ndarray:
        return np.asarray(
            [(-1 if p.label is None else p.label) for p in self.predictions], dtype=int
        )

    def actions_by_event(self) -> dict[str, MetaLabelAction]:
        return {p.event_id: p.action for p in self.predictions}

    def prediction_hash(self) -> str:
        """Reproducibility anchor: same inputs + same seed => same hash."""
        return fingerprint(
            "mloofpred1",
            {
                "schema_version": self.schema_version,
                "rows": [
                    [
                        p.event_id,
                        p.decision_timestamp,
                        round(p.probability_take, 12),
                        p.threshold,
                        p.action.value,
                        p.outer_fold_index,
                        p.producing_model_fingerprint,
                    ]
                    for p in self.predictions
                ],
            },
        )


def assert_oof_provenance(
    frame: MLPredictionFrame, folds: tuple[Fold, ...]
) -> None:
    """No OOF prediction may come from a model trained on that very row.

    This is the single most important test in the whole protocol: it is what
    makes an "out-of-sample" number actually out-of-sample.
    """
    by_index = {f.fold_index: f for f in folds}
    for p in frame.predictions:
        fold = by_index.get(p.outer_fold_index)
        if fold is None:
            raise FoldIsolationError(
                f"prediction for {p.event_id} names outer fold {p.outer_fold_index}, which does "
                "not exist"
            )
        if p.row_index in set(fold.train_row_indices):
            raise FoldIsolationError(
                f"OOF prediction for {p.event_id} (row {p.row_index}) came from the model of "
                f"outer fold {p.outer_fold_index}, which was TRAINED on that row"
            )
        if p.row_index not in set(fold.test_row_indices):
            raise FoldIsolationError(
                f"OOF prediction for {p.event_id} (row {p.row_index}) is not a test row of the "
                f"fold {p.outer_fold_index} that produced it"
            )

"""Fold-isolated preprocessing: the ONE way a fitted object is produced.

Every fitted quantity in Phase 15 -- an imputation median, a standardisation
mean/scale, a regime quantile cut point, a cluster centroid, a one-hot category
list -- goes through :class:`FoldFittedTransform`. That is deliberate: a single
choke point is what makes "preprocessing sees the training fold only" a testable
property rather than a claim in a document.

The transform records the exact row index set it was fitted on. Transforming a
row that was in a *later* fold is fine (that is what OOS scoring is). Fitting on
one is a :class:`FoldIsolationError`, and so is transforming with a transform
whose fit set intersects the rows being evaluated.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.ml.enums import PreprocessingKind
from alpha_agent.ml.errors import FoldIsolationError, LeakageError
from alpha_agent.validation.fingerprint import fingerprint

PREPROCESSING_SPEC_SCHEMA = "ml-preprocessing-spec/1"


class PreprocessingSpec(BaseModel):
    """What to fit. Frozen; never chosen after seeing performance."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = PREPROCESSING_SPEC_SCHEMA
    kind: PreprocessingKind = PreprocessingKind.MEDIAN_IMPUTE_STANDARDIZE
    #: Winsorise to +/- this many train-fold standard deviations before scaling.
    #: 0 disables. Cut points are train-fold quantities like everything else.
    winsorize_sigma: float = Field(default=0.0, ge=0.0, le=20.0)

    def identity(self) -> str:
        return fingerprint("mlpreprocspec1", self.model_dump(mode="json"))


class FoldFittedTransform:
    """A transform fitted on one explicit, recorded set of training row indices."""

    def __init__(
        self,
        spec: PreprocessingSpec,
        *,
        fit_row_indices: np.ndarray,
        feature_order: tuple[str, ...],
    ):
        self.spec = spec
        self.feature_order = tuple(feature_order)
        self._fit_rows = frozenset(int(i) for i in np.asarray(fit_row_indices).ravel())
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._impute_median: np.ndarray | None = None
        self._fitted = False

    # -- introspection used by the fold-isolation tests -----------------------
    @property
    def fit_row_indices(self) -> frozenset[int]:
        return self._fit_rows

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def statistics(self) -> dict[str, list[float]]:
        if not self._fitted:
            raise FoldIsolationError("transform has not been fitted")
        return {
            "center": [float(v) for v in (self._center if self._center is not None else [])],
            "scale": [float(v) for v in (self._scale if self._scale is not None else [])],
        }

    def fingerprint(self) -> str:
        return fingerprint(
            "mlpreprocfitted1",
            {
                "spec": self.spec.identity(),
                "feature_order": list(self.feature_order),
                "statistics": self.statistics(),
            },
            allow_non_finite=True,
        )

    # -- fit / transform ------------------------------------------------------
    def fit(self, X: np.ndarray, *, row_indices: np.ndarray) -> FoldFittedTransform:
        """Fit on ``X``, whose rows must be EXACTLY the declared training rows."""
        rows = frozenset(int(i) for i in np.asarray(row_indices).ravel())
        if rows != self._fit_rows:
            extra = sorted(rows - self._fit_rows)[:5]
            raise FoldIsolationError(
                "preprocessing fit received rows outside its declared training fold "
                f"(first offenders {extra}); a scaler, imputer, quantile boundary or "
                "clusterer is never fitted on validation or future rows"
            )
        if X.shape[1] != len(self.feature_order):
            raise ValueError(
                f"X has {X.shape[1]} columns but the feature order declares "
                f"{len(self.feature_order)}"
            )
        if self.spec.kind is PreprocessingKind.NONE:
            self._center = np.zeros(X.shape[1])
            self._scale = np.ones(X.shape[1])
            self._fitted = True
            return self

        with np.errstate(invalid="ignore"):
            median = np.nanmedian(X, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(X), X, median)

        if self.spec.kind is PreprocessingKind.MEDIAN_IMPUTE:
            self._center = median
            self._scale = np.ones(X.shape[1])
        else:
            mean = filled.mean(axis=0)
            std = filled.std(axis=0, ddof=0)
            # A constant train-fold column scales by 1.0 rather than exploding.
            std = np.where(std > 0.0, std, 1.0)
            self._center = mean
            self._scale = std
        self._impute_median = median
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, *, row_indices: np.ndarray | None = None) -> np.ndarray:
        """Apply the fitted transform. Rows may be anything -- that is scoring.

        ``row_indices`` is optional and, when given, is checked for the *inverse*
        error: evaluating rows that the transform was itself fitted on, which
        would make an "out-of-sample" score in-sample.
        """
        if not self._fitted:
            raise FoldIsolationError("transform used before it was fitted")
        if row_indices is not None:
            rows = frozenset(int(i) for i in np.asarray(row_indices).ravel())
            overlap = rows & self._fit_rows
            if overlap:
                raise FoldIsolationError(
                    f"{len(overlap)} evaluation row(s) were part of this transform's own fit "
                    f"set (first offenders {sorted(overlap)[:5]}); that is not out-of-sample"
                )
        if self.spec.kind is PreprocessingKind.NONE:
            return np.asarray(X, dtype=float)
        filled = np.where(np.isfinite(X), X, self._impute_median)
        if self.spec.kind is PreprocessingKind.MEDIAN_IMPUTE:
            return filled
        out = (filled - self._center) / self._scale
        if self.spec.winsorize_sigma > 0.0:
            lim = self.spec.winsorize_sigma
            out = np.clip(out, -lim, lim)
        return out


def assert_no_whole_dataset_fit(
    transform: FoldFittedTransform, *, all_row_indices: np.ndarray
) -> None:
    """Refuse a transform fitted on the entire dataset (prompt 15A s.7).

    A fit set equal to every row means no fold isolation happened at all.
    """
    everything = frozenset(int(i) for i in np.asarray(all_row_indices).ravel())
    if transform.fit_row_indices == everything and len(everything) > 0:
        raise LeakageError(
            "preprocessing was fitted on the whole dataset; standardisation, imputation, "
            "quantile boundaries and clustering are fitted on a TRAIN FOLD only"
        )

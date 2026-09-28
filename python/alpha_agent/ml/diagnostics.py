"""Predeclared ML classification diagnostics (prompt 15A s.9).

ROC-AUC, PR-AUC, log loss, Brier, calibration, precision, recall, class balance,
prediction coverage, TAKE rate. **Every one of these is DIAGNOSTIC ONLY.** The
scientific question is economic; improvement is never claimed from a
classification metric, and none of these gates a verdict
(``MLValidationSpec.ml_diagnostics_gate_verdict`` is ``False`` and refused
otherwise). Calibration is looked at first.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from alpha_agent.ml.predictions import MLPredictionFrame

_EPS = 1e-12


class CalibrationBin(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    lo: float
    hi: float
    n: int
    mean_predicted: float
    empirical_rate: float


class MLDiagnostics(BaseModel):
    """Diagnostic classification metrics for one experiment's OOF predictions."""

    model_config = {"frozen": True, "extra": "forbid"}

    n_predictions: int = Field(ge=0)
    n_labelled: int = Field(ge=0)
    class_balance_positive: float
    prediction_coverage: float
    take_rate: float

    roc_auc: float
    pr_auc: float
    log_loss: float
    brier_score: float
    precision: float
    recall: float

    calibration_bins: tuple[CalibrationBin, ...]
    calibration_mae: float

    is_diagnostic_only: bool = True

    def diagnostic_fingerprint(self) -> str:
        from alpha_agent.validation.fingerprint import fingerprint

        return fingerprint(
            "mldiagnostics1", self.model_dump(mode="json"), allow_non_finite=True
        )


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic with tie-corrected mid-ranks."""
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=float)
    sorted_p = p[order]
    i = 0
    while i < p.size:
        j = i
        while j < p.size and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0     # mid-rank for the tie block
        i = j
    rank_sum_pos = float(ranks[y == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def compute_ml_diagnostics(frame: MLPredictionFrame, *, n_calibration_bins: int = 10) -> MLDiagnostics:
    preds = frame.predictions
    p = np.asarray([x.probability_take for x in preds], dtype=float)
    y = np.asarray([(-1 if x.label is None else x.label) for x in preds], dtype=int)
    labelled = y >= 0
    n = len(preds)
    n_lab = int(labelled.sum())
    yl = y[labelled].astype(int)
    pl = np.clip(p[labelled], _EPS, 1 - _EPS)

    if n_lab == 0:
        bins: tuple[CalibrationBin, ...] = ()
        roc = pr = ll = brier = prec = rec = cal_mae = float("nan")
    else:
        ll = float(-np.mean(yl * np.log(pl) + (1 - yl) * np.log(1 - pl)))
        brier = float(np.mean((pl - yl) ** 2))
        roc = _roc_auc(yl, pl)
        pr = _pr_auc(yl, pl)
        take = pl >= 0.5
        tp = int(np.sum(take & (yl == 1)))
        fp = int(np.sum(take & (yl == 0)))
        fn = int(np.sum(~take & (yl == 1)))
        prec = float(tp / (tp + fp)) if (tp + fp) else float("nan")
        rec = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        edges = np.linspace(0.0, 1.0, n_calibration_bins + 1)
        blist: list[CalibrationBin] = []
        errs: list[float] = []
        for b in range(n_calibration_bins):
            lo, hi = edges[b], edges[b + 1]
            m = (pl >= lo) & (pl < hi if b < n_calibration_bins - 1 else pl <= hi)
            if not m.any():
                continue
            mp = float(pl[m].mean())
            er = float(yl[m].mean())
            blist.append(CalibrationBin(lo=float(lo), hi=float(hi), n=int(m.sum()),
                                        mean_predicted=mp, empirical_rate=er))
            errs.append(abs(mp - er))
        bins = tuple(blist)
        cal_mae = float(np.mean(errs)) if errs else float("nan")

    return MLDiagnostics(
        n_predictions=n,
        n_labelled=n_lab,
        class_balance_positive=float(yl.mean()) if n_lab else float("nan"),
        prediction_coverage=float(n_lab / n) if n else 0.0,
        take_rate=frame.take_rate,
        roc_auc=roc,
        pr_auc=pr,
        log_loss=ll,
        brier_score=brier,
        precision=prec,
        recall=rec,
        calibration_bins=bins,
        calibration_mae=cal_mae,
    )

"""Predeclared model families and their FROZEN finite hyperparameter grid.

Prompt 15A sections 8 and 9: a deliberately small predeclared set; no AutoML, no
Bayesian optimisation, no Optuna, no adaptive grid expansion, and above all no
"try another parameter because this result looks bad". The grid below is frozen
into ``configs/phase_15_ml.yaml`` and into the committed candidate manifest
BEFORE the first real ML performance number exists.

Dependency posture (Phase 15A.1)
--------------------------------
``scikit-learn`` and ``scipy`` are installed under a one-time explicitly approved
install. Nothing else is: ``lightgbm``, ``xgboost``, ``catboost``, ``optuna``,
``tensorflow`` and ``torch`` are all listed in :data:`FORBIDDEN_ML_PACKAGES` and
:func:`assert_no_forbidden_ml_packages` fails loudly if one appears. Keeping the
research set at two families -- one linear probabilistic model and one controlled
nonlinear tree -- is deliberate statistical control.

Every backend still imports its package lazily and raises a typed
:class:`~alpha_agent.ml.errors.MLDependencyMissing` when it is absent; nothing
here ever installs anything. :class:`SyntheticDeterministicBackend` is a
pure-numpy stand-in used by the leakage / fold-isolation / determinism tests so
they stay independent of any vendor library. The manifest refuses it for a
research candidate.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field, model_validator

from alpha_agent.ml.enums import (
    MLModelFamily,
    ModelSelectionObjective,
    ModelSelectionTieBreak,
)
from alpha_agent.ml.errors import ForbiddenDependency, MLDependencyMissing
from alpha_agent.validation.fingerprint import fingerprint

ML_MODEL_SPEC_SCHEMA = "ml-model-spec/1"
MODEL_SEARCH_SPEC_SCHEMA = "ml-model-search-spec/1"

#: The single predeclared random seed for the whole Phase 15 model search.
#: It is a SEMANTIC input, not provenance: a different seed is a different
#: (if usually similar) search, so it enters pre-run identity.
DEFAULT_ML_RANDOM_SEED: int = 20250101

#: FROZEN finite grid. Values only; the keys are the exact constructor arguments.
FROZEN_HYPERPARAMETER_GRID: dict[MLModelFamily, tuple[dict[str, Any], ...]] = {
    MLModelFamily.LOGISTIC_L2: (
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ),
    MLModelFamily.HIST_GRADIENT_BOOSTING: (
        {"max_depth": 2, "learning_rate": 0.05, "max_iter": 100},
        {"max_depth": 2, "learning_rate": 0.05, "max_iter": 300},
        {"max_depth": 2, "learning_rate": 0.10, "max_iter": 100},
        {"max_depth": 2, "learning_rate": 0.10, "max_iter": 300},
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 100},
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300},
        {"max_depth": 3, "learning_rate": 0.10, "max_iter": 100},
        {"max_depth": 3, "learning_rate": 0.10, "max_iter": 300},
    ),
    MLModelFamily.SYNTHETIC_DETERMINISTIC: ({"ridge": 1.0},),
}

#: The FROZEN TAKE threshold. A single predeclared constant -- NOT a search
#: dimension (prompt 15A.1 s.3).
#:
#: Phase 15A swept ``{0.45, 0.50, 0.55, 0.60}`` inside the inner loop while
#: scoring every candidate with the SAME objective, mean inner-fold log loss.
#: Log loss is a property of the predicted PROBABILITY and is completely
#: independent of the decision threshold, so all four thresholds scored
#: identically and the "selection" resolved through a deterministic tie-break --
#: a search dimension that was never actually searched, while still inflating
#: the DSR search-breadth term fourfold. The sweep is removed.
#:
#: ``probability_take >= 0.50 -> TAKE``, otherwise ``SKIP``. A probability never
#: becomes a continuous position size. Thresholds such as 0.55 / 0.60 are a
#: DECISION-POLICY hypothesis and may only return as an explicitly predeclared
#: new search family -- never as a free parameter of this one, and never chosen
#: on PnL.
FROZEN_TAKE_THRESHOLD: float = 0.50


def assert_frozen_take_threshold(value: float, *, what: str = "take_threshold") -> float:
    """Refuse any TAKE threshold other than the frozen Phase 15 constant."""
    if float(value) != FROZEN_TAKE_THRESHOLD:
        raise ValueError(
            f"{what} {value!r} is not the FROZEN Phase 15 TAKE threshold "
            f"{FROZEN_TAKE_THRESHOLD}. The threshold is predeclared, not selected: the inner "
            "objective (log loss) is threshold-independent, so sweeping thresholds searched "
            "nothing while inflating the search-breadth term. A different threshold is a new "
            "decision-policy hypothesis and needs its own predeclared search family."
        )
    return float(value)

#: Model families that may appear in a RESEARCH candidate.
RESEARCH_MODEL_FAMILIES: tuple[MLModelFamily, ...] = (
    MLModelFamily.LOGISTIC_L2,
    MLModelFamily.HIST_GRADIENT_BOOSTING,
)

#: Packages that are explicitly NOT approved for Phase 15. Keeping the model
#: family at two -- one linear probabilistic model, one controlled nonlinear
#: tree -- is deliberate statistical control, not a claim that these are bad
#: models. ``HistGradientBoostingClassifier`` is the frozen tree baseline and
#: ships with scikit-learn, so no gradient-boosting package is needed.
FORBIDDEN_ML_PACKAGES: tuple[str, ...] = (
    "lightgbm", "xgboost", "catboost", "optuna", "tensorflow", "torch",
)

#: The approved Phase 15 numerical stack, recorded in environment provenance.
APPROVED_ML_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "scipy", "sklearn")

_REQUIRED_PACKAGE: dict[MLModelFamily, str] = {
    MLModelFamily.LOGISTIC_L2: "sklearn",
    MLModelFamily.HIST_GRADIENT_BOOSTING: "sklearn",
}


class MLModelSpec(BaseModel):
    """One model family at ONE frozen hyperparameter point."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = ML_MODEL_SPEC_SCHEMA
    family: MLModelFamily
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    random_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _in_frozen_grid(self) -> MLModelSpec:
        grid = FROZEN_HYPERPARAMETER_GRID[self.family]
        if self.hyperparameters not in [dict(g) for g in grid]:
            raise ValueError(
                f"hyperparameters {self.hyperparameters} are not a point of the FROZEN grid for "
                f"{self.family.value}. The grid is predeclared before any performance is seen; "
                "it is never expanded because a result looked bad."
            )
        return self

    def identity(self) -> str:
        return fingerprint(
            "mlmodelspec1",
            {
                "schema_version": self.schema_version,
                "family": self.family.value,
                "hyperparameters": {k: self.hyperparameters[k] for k in sorted(self.hyperparameters)},
                "random_seed": self.random_seed,
            },
        )

    def required_package(self) -> str | None:
        return _REQUIRED_PACKAGE.get(self.family)


def frozen_model_specs(family: MLModelFamily, *, random_seed: int = 0) -> tuple[MLModelSpec, ...]:
    """Every frozen grid point of one family, in declaration order."""
    return tuple(
        MLModelSpec(family=family, hyperparameters=dict(h), random_seed=random_seed)
        for h in FROZEN_HYPERPARAMETER_GRID[family]
    )


def _declared(value: object) -> str:
    """Normalise a declared closed-enum value to the string that is hashed."""
    return value.value if isinstance(value, Enum) else str(value)


class ModelSearchSpec(BaseModel):
    """The whole PREDECLARED model-search PROCEDURE for one experiment.

    This -- not one selected hyperparameter point -- is the scientific unit of a
    nested-CV study, and it is what enters pre-run experiment identity.

    Why (Phase 15A.2)
    -----------------
    Under nested CV a hyperparameter point is *chosen inside each outer training
    fold*, so different outer folds may legitimately select different points and
    no single point exists before the run. An identity built on one point is
    therefore not a pre-run quantity at all: it could not be computed before
    fitting, it would split one economic hypothesis into ``H`` registry rows
    (60 trials would become 380), and duplicate detection would require already
    knowing the answer the run is supposed to produce.

    The scientific claim a Phase 15 trial makes is *"this primary, filtered by a
    model of THIS family selected from THIS frozen grid by THIS objective under
    THIS tie-break and THIS seed, improves risk-adjusted net economics"*. Every
    one of those is knowable from the proposal, so all of them are identity, and
    ``experiment_identity`` stays computable before a single fit.

    What is NOT here
    ----------------
    The hyperparameters an outer fold actually selected. Those are POST-FIT
    PROVENANCE, recorded per fold on
    :class:`~alpha_agent.ml.training.OuterFoldRecord` and in the training report,
    and they never change which experiment this is.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = MODEL_SEARCH_SPEC_SCHEMA
    family: MLModelFamily
    #: The FULL frozen grid the inner loop chooses among, in declaration order.
    #: Order is semantic: it IS the tie-break rule. Defaulted from
    #: :data:`FROZEN_HYPERPARAMETER_GRID` and refused if it disagrees with it.
    search_grid: tuple[dict[str, Any], ...] = ()
    selection_objective: ModelSelectionObjective = (
        ModelSelectionObjective.MEAN_INNER_FOLD_LOG_LOSS_MINIMISED
    )
    selection_tie_break: ModelSelectionTieBreak = (
        ModelSelectionTieBreak.FROZEN_GRID_DECLARATION_ORDER_FIRST
    )
    random_seed: int = Field(default=DEFAULT_ML_RANDOM_SEED, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _default_to_the_frozen_grid(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("search_grid"):
            family = data.get("family")
            if family is not None:
                grid = FROZEN_HYPERPARAMETER_GRID[MLModelFamily(family)]
                data = {**data, "search_grid": tuple(dict(h) for h in grid)}
        return data

    @model_validator(mode="after")
    def _grid_is_the_frozen_one(self) -> ModelSearchSpec:
        frozen = [dict(h) for h in FROZEN_HYPERPARAMETER_GRID[self.family]]
        if [dict(h) for h in self.search_grid] != frozen:
            raise ValueError(
                f"search_grid for {self.family.value} is not the FROZEN grid (in declaration "
                "order). The grid is predeclared before any performance is seen: it is never "
                "expanded, reordered or trimmed because a result looked bad. Changing the "
                "frozen grid is a change to the search PROCEDURE and therefore a different "
                "pre-run experiment identity, which must be predeclared, not slipped in."
            )
        return self

    @property
    def objective_value(self) -> str:
        """The declared objective as a plain string.

        Read through a property rather than ``.value`` so identity stays total
        over whatever is declared: the enums are closed today, but a fingerprint
        that raises on an unrecognised declaration would fail *silently late*,
        after a caller had already treated the object as a valid procedure.
        """
        return _declared(self.selection_objective)

    @property
    def tie_break_value(self) -> str:
        return _declared(self.selection_tie_break)

    @property
    def n_configurations(self) -> int:
        """The trial-level DSR search breadth: 3 for logistic, 8 for the tree."""
        return len(self.search_grid)

    def grid_specs(self) -> tuple[MLModelSpec, ...]:
        """The ordered candidate points the inner loop will actually fit."""
        return tuple(
            MLModelSpec(family=self.family, hyperparameters=dict(h), random_seed=self.random_seed)
            for h in self.search_grid
        )

    def identity(self) -> str:
        return fingerprint(
            "mlmodelsearch1",
            {
                "schema_version": self.schema_version,
                "family": self.family.value,
                # ordered: the declaration order IS the tie-break rule
                "search_grid": [
                    {k: h[k] for k in sorted(h)} for h in self.search_grid
                ],
                "n_configurations": self.n_configurations,
                "selection_objective": self.objective_value,
                "selection_tie_break": self.tie_break_value,
                "random_seed": self.random_seed,
            },
        )

    def contains(self, model: MLModelSpec) -> bool:
        """Is ``model`` one of the points this procedure could have selected?"""
        return (
            model.family is self.family
            and model.random_seed == self.random_seed
            and dict(model.hyperparameters) in [dict(h) for h in self.search_grid]
        )


def frozen_model_search(
    family: MLModelFamily, *, random_seed: int = DEFAULT_ML_RANDOM_SEED
) -> ModelSearchSpec:
    """The frozen search procedure for one model family."""
    return ModelSearchSpec(family=family, random_seed=random_seed)


def assert_selection_is_within_search(
    search: ModelSearchSpec, selected: MLModelSpec, *, what: str = "outer-fold selection"
) -> MLModelSpec:
    """A selected point must be one the PREDECLARED procedure could reach.

    Post-fit provenance is still auditable provenance: a fold that reports a
    configuration outside the frozen grid means the executed search was not the
    declared one, which is an integrity failure, not a new experiment.
    """
    if not search.contains(selected):
        raise ValueError(
            f"{what} reported {selected.family.value} {selected.hyperparameters} "
            f"(seed {selected.random_seed}), which is not a point of the predeclared search "
            f"{search.identity()}. The executed search must be the declared search."
        )
    return selected


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class ModelBackend(Protocol):
    """The only interface the training loop knows about."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelBackend: ...

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray: ...

    def artifact_metadata(self) -> dict: ...


def _require(package: str, family: MLModelFamily) -> Any:
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise MLDependencyMissing(
            f"model family {family.value} needs {package!r}, which is not installed in this "
            "environment. Installing it requires network access and explicit human approval "
            "(CLAUDE.md autonomous-execution STOP condition 1). Phase 15A never installs a "
            "package and never silently substitutes a different model."
        ) from exc


class SklearnBackend:
    """Lazy scikit-learn backend for both research families."""

    def __init__(self, spec: MLModelSpec, *, feature_order: tuple[str, ...]):
        self.spec = spec
        self.feature_order = tuple(feature_order)
        self._model: Any = None

    def _construct(self) -> Any:
        _require("sklearn", self.spec.family)
        if self.spec.family is MLModelFamily.LOGISTIC_L2:
            from sklearn.linear_model import LogisticRegression

            # L2 is expressed by leaving the penalty at its default. scikit-learn
            # 1.8 deprecated the `penalty=` argument in favour of `l1_ratio`,
            # whose default 0.0 IS pure L2 -- and was pure L2 before the rename
            # too. Passing neither therefore gives the identical estimator on
            # every supported version, so the model semantics are unchanged and
            # only the deprecated spelling is dropped.
            return LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=self.spec.random_seed,
                **self.spec.hyperparameters,
            )
        if self.spec.family is MLModelFamily.HIST_GRADIENT_BOOSTING:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                random_state=self.spec.random_seed,
                early_stopping=False,     # a validation cut inside fit would be a hidden split
                **self.spec.hyperparameters,
            )
        raise MLDependencyMissing(f"{self.spec.family.value} has no scikit-learn backend")

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnBackend:
        self._model = self._construct()
        self._model.fit(X, y)
        return self

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise MLDependencyMissing("backend used before fit")
        return np.asarray(self._model.predict_proba(X))[:, 1]

    def artifact_metadata(self) -> dict:
        import sklearn

        return {
            "backend": "sklearn",
            "sklearn_version": sklearn.__version__,
            "model_spec": self.spec.identity(),
            "family": self.spec.family.value,
            "hyperparameters": dict(self.spec.hyperparameters),
            "random_seed": self.spec.random_seed,
            "feature_order": list(self.feature_order),
        }


class SyntheticDeterministicBackend:
    """Pure-numpy ridge-regularised least squares on {0,1} targets. TESTS ONLY.

    It exists so the leakage, fold-isolation, determinism and artifact-metadata
    guarantees can be proven today, with scikit-learn absent and with no real
    market data. It is NOT a research model: :func:`assert_research_model_family`
    refuses it, and the frozen candidate manifest cannot contain it.
    """

    def __init__(self, spec: MLModelSpec, *, feature_order: tuple[str, ...]):
        if spec.family is not MLModelFamily.SYNTHETIC_DETERMINISTIC:
            raise ValueError("SyntheticDeterministicBackend only serves SYNTHETIC_DETERMINISTIC")
        self.spec = spec
        self.feature_order = tuple(feature_order)
        self._coef: np.ndarray | None = None
        self._n_train: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> SyntheticDeterministicBackend:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.hstack([np.ones((X.shape[0], 1)), X])
        ridge = float(self.spec.hyperparameters["ridge"])
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        self._coef = np.linalg.solve(gram, design.T @ y)
        self._n_train = int(X.shape[0])
        return self

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise ValueError("backend used before fit")
        X = np.asarray(X, dtype=float)
        design = np.hstack([np.ones((X.shape[0], 1)), X])
        return 1.0 / (1.0 + np.exp(-(design @ self._coef)))

    def artifact_metadata(self) -> dict:
        return {
            "backend": "synthetic_deterministic",
            "model_spec": self.spec.identity(),
            "family": self.spec.family.value,
            "hyperparameters": dict(self.spec.hyperparameters),
            "random_seed": self.spec.random_seed,
            "feature_order": list(self.feature_order),
            "n_train_rows": self._n_train,
            "coefficients": [float(c) for c in (self._coef if self._coef is not None else [])],
            "research_use": "FORBIDDEN -- software test fixture only",
        }


def make_backend(spec: MLModelSpec, *, feature_order: tuple[str, ...]) -> ModelBackend:
    if spec.family is MLModelFamily.SYNTHETIC_DETERMINISTIC:
        return SyntheticDeterministicBackend(spec, feature_order=feature_order)
    return SklearnBackend(spec, feature_order=feature_order)


def assert_research_model_family(family: MLModelFamily) -> None:
    if family not in RESEARCH_MODEL_FAMILIES:
        raise ValueError(
            f"{family.value} is not a research model family; the predeclared Phase 15 set is "
            f"{[f.value for f in RESEARCH_MODEL_FAMILIES]}"
        )


def ml_dependency_status() -> dict[str, object]:
    """Offline report of which predeclared families are runnable RIGHT NOW."""
    import importlib.util

    status: dict[str, object] = {}
    for pkg in ("sklearn", "scipy", *FORBIDDEN_ML_PACKAGES):
        status[pkg] = importlib.util.find_spec(pkg) is not None
    status["runnable_research_families"] = [
        f.value for f in RESEARCH_MODEL_FAMILIES
        if status.get(_REQUIRED_PACKAGE.get(f, ""), False)
    ]
    status["forbidden_packages_present"] = [
        p for p in FORBIDDEN_ML_PACKAGES if status.get(p, False)
    ]
    return status


def assert_no_forbidden_ml_packages() -> None:
    """Refuse to proceed if an unapproved model/search package is importable.

    Approval covered scikit-learn and scipy only. A gradient-boosting library or
    a hyperparameter-search library appearing in the environment is a change to
    the predeclared search, so it fails loudly rather than being silently
    available to a later phase.
    """
    import importlib.util

    present = [p for p in FORBIDDEN_ML_PACKAGES if importlib.util.find_spec(p) is not None]
    if present:
        raise ForbiddenDependency(
            f"unapproved ML/search package(s) importable: {present}. The Phase 15 model family "
            f"is exactly {[f.value for f in RESEARCH_MODEL_FAMILIES]} -- one linear "
            "probabilistic model and one controlled nonlinear tree -- and the model zoo is not "
            "expanded before any Phase 15 result exists."
        )


def ml_environment_provenance() -> dict[str, object]:
    """Exact versions of the approved Phase 15 numerical stack.

    Recorded in every Phase 15 provenance artifact. Versions are provenance, not
    semantics: they never enter a fingerprint, but a result produced under a
    different stack must be traceable to it.
    """
    import importlib
    import importlib.util
    import platform
    import sys

    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for pkg in APPROVED_ML_PACKAGES:
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except ImportError:
            versions[pkg] = None
    versions["forbidden_packages_absent"] = [
        p for p in FORBIDDEN_ML_PACKAGES if importlib.util.find_spec(p) is None
    ]
    versions["forbidden_packages_present"] = [
        p for p in FORBIDDEN_ML_PACKAGES if importlib.util.find_spec(p) is not None
    ]
    return versions

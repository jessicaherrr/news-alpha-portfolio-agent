"""Regime representations for Phase 15 (prompt 15A s.7).

Two strictly separate concepts, and the boundary between them is the whole point:

A. **Deterministic causal regime features.** Computed from typed
   :class:`FeatureSpec`s with train-fold-derived cut points. This is the
   REFERENCE BASELINE. Phase 15B.1b freezes its exact transformation semantics:

   * two axes -- input 0 ``realized_vol(20)``, input 1 ``trend_strength(50,200)``;
   * for EACH axis independently, the ``1/3`` and ``2/3`` empirical quantile cut
     points, computed from the CURRENT TRAIN FOLD only (linear interpolation),
     giving three ordered buckets ``lo <= mid <= hi`` per axis
     (``np.digitize`` with ``right=False`` -- a value on a cut goes up);
   * the regime state is the CARTESIAN PRODUCT of the two axes -> ``3 x 3 = 9``
     states, indexed ``state = vol_bucket * 3 + trend_bucket`` (axis-0-major);
   * the 9 states are exposed as ``regime_state_0 .. regime_state_8`` one-hot
     columns.

   The two inputs are NEVER standardised-and-averaged; there is no PCA, no
   clustering, no learned discovery and no performance-based tuning. The signed
   trend axis reads low = strongly negative ``trend_strength`` (fast MA well
   below slow), high = strongly positive; the column names stay
   interpretation-free (``regime_state_<i>``) and the semantic mapping lives in
   :meth:`RegimeSpec.identity`.

B. **Optional learned regime representation.** A clusterer / PCA / quantiser.
   Anything fitted is fitted on a TRAIN FOLD ONLY -- see
   :class:`~alpha_agent.ml.preprocessing.FoldFittedTransform`.

A learned cluster is NEVER given an economic name (:func:`assert_regime_label_is_not_economic`).
"""
from __future__ import annotations

from itertools import product

from pydantic import BaseModel, Field, model_validator

from alpha_agent.features.spec import FeatureSpec
from alpha_agent.ml.enums import RegimeSpecKind
from alpha_agent.ml.errors import LeakageError
from alpha_agent.validation.fingerprint import fingerprint

REGIME_SPEC_SCHEMA = "ml-regime-spec/1"

#: Neutral cluster labels. Deliberately not "bull"/"bear"/"risk_on"/"risk_off".
NEUTRAL_CLUSTER_PREFIX = "regime_cluster_"
#: Neutral deterministic state column prefix.
REGIME_STATE_PREFIX = "regime_state_"

#: The frozen Phase 15B.1b deterministic regime algorithm (bound into identity).
DETERMINISTIC_REGIME_ALGORITHM = "independent_axis_tertiles_cartesian_product/1"
DETERMINISTIC_QUANTILE_PROBS: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
DETERMINISTIC_QUANTILE_INTERPOLATION = "linear"
DETERMINISTIC_DIGITIZE_RIGHT = False


class RegimeSpec(BaseModel):
    """A causal regime representation, deterministic or learned."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = REGIME_SPEC_SCHEMA
    kind: RegimeSpecKind = RegimeSpecKind.DETERMINISTIC_CAUSAL_VOL_TREND
    #: inputs to the regime, as typed causal FeatureSpecs (axis order is semantic)
    inputs: tuple[FeatureSpec, ...] = ()
    #: legacy field, kept so a NONE ablation spec fingerprints exactly as it did
    #: before Phase 15B.1b. Not used by the deterministic algorithm.
    n_buckets: int = Field(default=3, ge=2, le=10)
    #: learned kinds only
    n_clusters: int = Field(default=3, ge=2, le=10)
    random_seed: int = Field(default=0, ge=0)

    # -- deterministic 2D vol x signed-trend regime (Phase 15B.1b, frozen) ----
    algorithm: str = DETERMINISTIC_REGIME_ALGORITHM
    #: tertiles per axis
    axis_bucket_count: int = Field(default=3, ge=2, le=6)
    #: the per-axis quantile probabilities, ascending in (0, 1)
    quantile_probs: tuple[float, ...] = DETERMINISTIC_QUANTILE_PROBS
    quantile_interpolation: str = DETERMINISTIC_QUANTILE_INTERPOLATION
    #: np.digitize(right=): False => a value exactly on a cut goes to the HIGHER
    #: bucket (deterministic tie rule)
    digitize_right: bool = DETERMINISTIC_DIGITIZE_RIGHT
    combination: str = "cartesian_product_axis0_major"
    #: one prose note per axis, in input order -- what its lo/mid/hi buckets MEAN.
    #: Recorded for provenance (and hashed); never a column name.
    axis_semantics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> RegimeSpec:
        if self.kind is RegimeSpecKind.NONE:
            if self.inputs:
                raise ValueError("RegimeSpecKind.NONE must declare no inputs")
            return self
        if not self.inputs:
            raise ValueError(f"{self.kind.value} needs at least one causal input FeatureSpec")
        if self.kind is RegimeSpecKind.DETERMINISTIC_CAUSAL_VOL_TREND:
            if list(self.quantile_probs) != sorted(self.quantile_probs) or not all(
                0.0 < p < 1.0 for p in self.quantile_probs
            ):
                raise ValueError("quantile_probs must be ascending values strictly in (0, 1)")
            if len(self.quantile_probs) != self.axis_bucket_count - 1:
                raise ValueError(
                    f"{len(self.quantile_probs)} quantile cut points cannot form "
                    f"{self.axis_bucket_count} buckets per axis"
                )
            if self.axis_semantics and len(self.axis_semantics) != len(self.inputs):
                raise ValueError("axis_semantics, when given, needs one note per input axis")
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether this regime requires fitting -- and therefore fold isolation."""
        return self.kind in (
            RegimeSpecKind.DETERMINISTIC_CAUSAL_VOL_TREND,   # quantile cut points are fitted
            RegimeSpecKind.LEARNED_TRAIN_FOLD_KMEANS,
        )

    @property
    def is_learned(self) -> bool:
        return self.kind is RegimeSpecKind.LEARNED_TRAIN_FOLD_KMEANS

    @property
    def n_states(self) -> int:
        """Deterministic regime: the Cartesian product size over the axes."""
        if self.kind is not RegimeSpecKind.DETERMINISTIC_CAUSAL_VOL_TREND:
            raise ValueError("n_states is only defined for the deterministic vol x trend regime")
        return self.axis_bucket_count ** len(self.inputs)

    def state_map(self) -> tuple[tuple[int, ...], ...]:
        """State index -> the per-axis bucket index tuple. Axis-0-major, so
        state 0 is (0, ..., 0), state 1 increments the LAST axis, and the first
        axis is the slowest-varying. Deterministic and total."""
        rng = range(self.axis_bucket_count)
        return tuple(product(*([rng] * len(self.inputs))))

    def output_columns(self) -> tuple[str, ...]:
        if self.kind is RegimeSpecKind.NONE:
            return ()
        if self.kind is RegimeSpecKind.LEARNED_TRAIN_FOLD_KMEANS:
            return tuple(f"{NEUTRAL_CLUSTER_PREFIX}{i}" for i in range(self.n_clusters))
        return tuple(f"{REGIME_STATE_PREFIX}{i}" for i in range(self.n_states))

    def identity(self) -> str:
        return fingerprint("mlregimespec1", self.identity_payload())

    def identity_payload(self) -> dict:
        # A NONE ablation spec MUST fingerprint exactly as it did before Phase
        # 15B.1b -- the 20 NO_REGIME experiment identities are frozen.
        base = {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "inputs": [s.canonical_json() for s in self.inputs] if self.inputs else [],
            "n_buckets": self.n_buckets,
            "n_clusters": self.n_clusters,
            "random_seed": self.random_seed,
        }
        if self.kind is RegimeSpecKind.NONE:
            return base
        if self.kind is RegimeSpecKind.DETERMINISTIC_CAUSAL_VOL_TREND:
            base["deterministic_transformation"] = {
                "algorithm": self.algorithm,
                "axis_input_specs_in_order": [s.canonical_json() for s in self.inputs],
                "axis_bucket_count": self.axis_bucket_count,
                "quantile_probs": [float(p) for p in self.quantile_probs],
                "quantile_interpolation": self.quantile_interpolation,
                "quantile_scope": "current_train_fold_only",
                "digitize_right": self.digitize_right,
                "combination": self.combination,
                "state_order": "axis0_major",
                "state_map": [list(s) for s in self.state_map()],
                "output_columns": list(self.output_columns()),
                "axis_semantics": list(self.axis_semantics),
                "missing_input_policy": "typed_feature_exclusion_never_zero_imputed",
                "no_standardize_no_pca_no_clustering_no_perf_tuning": True,
            }
        return base


def assert_regime_label_is_not_economic(label: str) -> None:
    """A learned cluster may not be handed an economic interpretation.

    Prompt 15A s.7: an unsupervised cluster does not automatically mean bull /
    bear / risk-on / risk-off unless the definition supports it. In Phase 15 no
    definition does, so the vocabulary is refused outright.
    """
    banned = {"bull", "bear", "risk_on", "risk-off", "risk_off", "risk-on",
              "crisis", "calm", "expansion", "recession"}
    if label.strip().lower() in banned:
        raise LeakageError(
            f"regime label {label!r} asserts an economic interpretation that an unsupervised "
            f"partition does not support; use a neutral {NEUTRAL_CLUSTER_PREFIX}<i> label"
        )

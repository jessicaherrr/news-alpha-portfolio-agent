"""The ML feature universe -- typed :class:`FeatureSpec`s only (prompt 15A s.6).

There is deliberately no way to hand this layer an ad-hoc pandas column. Every
ML feature is a Phase 09 :class:`~alpha_agent.features.spec.FeatureSpec`, so it
inherits, for free and provably:

* a deterministic canonical name and fingerprint,
* an availability timestamp (point-in-time truncation),
* a declared :class:`PriceDomain`,
* signal-safety / look-ahead-safety provenance.

:func:`assert_feature_set_signal_safe` re-checks the last two against the
registry rather than trusting the caller, because "I only used causal features"
is exactly the claim a research framework must not take on faith.

Excluded from the MVP on purpose
--------------------------------
* **Carry / term structure.** ``alpha_agent.features.carry`` is a separate
  curve-construction API, not a registered ``FeatureSpec`` kind, and the acquired
  Plan-B dataset does not carry a point-in-time deferred-contract curve for every
  root. Prompt 15A s.6 permits carry "ONLY where current PIT-safe curve data
  supports it"; here it does not, so it is out.
* **Any execution-plane field.** A fill price, spread or commission is never a
  feature, however convenient.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from alpha_agent.features.registry import FEATURE_ENGINE_VERSION, REGISTRY
from alpha_agent.features.safety import SourceSafety
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.ml.errors import LeakageError
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.validation.fingerprint import fingerprint

ML_FEATURE_SET_SCHEMA = "ml-feature-set/1"

#: Feature-engine column names that are categorical identity rather than a
#: computed series. They are declared separately so the ordered numeric matrix
#: stays unambiguous.
CATEGORICAL_FEATURES: tuple[str, ...] = ("root_symbol", "primary_side")


class MLFeature(BaseModel):
    """One ordered column of the model matrix."""

    model_config = {"frozen": True, "extra": "forbid"}

    alias: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    spec: FeatureSpec
    rationale: str = ""              # prose; never in the fingerprint

    @field_validator("spec")
    @classmethod
    def _registered(cls, v: FeatureSpec) -> FeatureSpec:
        if v.kind not in REGISTRY:
            raise ValueError(
                f"feature kind {v.kind!r} is not registered in the Phase 09 Feature Engine; "
                "an ML feature may never be an ad-hoc pandas column"
            )
        return v

    def feature_fingerprint(self) -> str:
        """The frozen Phase 13.5C feature-fingerprint rule, reused verbatim.

        Same rule, same prefix, same canonical JSON -- so a feature shared with a
        Phase 13.5C strategy fingerprints identically in both places.
        """
        import hashlib

        return "feat1:" + hashlib.sha256(self.spec.canonical_json().encode()).hexdigest()


class MLFeatureSetSpec(BaseModel):
    """The ORDERED feature universe of one ML candidate.

    Order is semantic: it is the column order of every matrix, of every persisted
    model artifact and of the fingerprint. A reordering is a different feature
    set, because a tree model fitted under one order cannot be scored under
    another.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = ML_FEATURE_SET_SCHEMA
    feature_engine_version: str = FEATURE_ENGINE_VERSION
    label: str = ""                                  # cosmetic
    features: tuple[MLFeature, ...] = Field(min_length=1)
    #: categorical identity columns, one-hot encoded with a train-fold-derived
    #: category list. Never a learned embedding in the MVP.
    categoricals: tuple[str, ...] = ()
    #: The signal price domain the features are computed on. Back-adjusted
    #: continuous is legitimate for SIGNALS (CLAUDE.md market-data rule 7); it is
    #: never an execution price and never reaches a fill.
    price_domain: PriceDomain = PriceDomain.BACK_ADJUSTED

    @model_validator(mode="after")
    def _unique(self) -> MLFeatureSetSpec:
        aliases = [f.alias for f in self.features]
        if len(set(aliases)) != len(aliases):
            raise ValueError("duplicate ML feature alias")
        for c in self.categoricals:
            if c not in CATEGORICAL_FEATURES:
                raise ValueError(
                    f"categorical {c!r} is not a declared identity column {CATEGORICAL_FEATURES}"
                )
        return self

    @property
    def ordered_aliases(self) -> tuple[str, ...]:
        return tuple(f.alias for f in self.features)

    def canonical_feature_names(self) -> tuple[str, ...]:
        """The Feature-Engine canonical column name of each feature, in alias
        order. ``compute_features`` returns columns keyed by this name (and
        alphabetically ordered), so a caller mapping the real feature frame to
        the ordered alias matrix MUST select by this, never positionally."""
        return tuple(REGISTRY.name_for(f.spec) for f in self.features)

    def feature_fingerprints(self) -> tuple[str, ...]:
        return tuple(f.feature_fingerprint() for f in self.features)

    def identity(self) -> str:
        return fingerprint(
            "mlfeatureset1",
            {
                "schema_version": self.schema_version,
                "feature_engine_version": self.feature_engine_version,
                # ordered, NOT sorted -- column order is semantic
                "features": [
                    {"alias": f.alias, "fingerprint": f.feature_fingerprint()}
                    for f in self.features
                ],
                "categoricals": list(self.categoricals),
                "price_domain": self.price_domain.value,
            },
        )


#: Source safety of each signal price domain, assuming a CAUSAL lineage
#: (point-in-time back-adjustment, never retrospective). Mirrors the matrix in
#: :mod:`alpha_agent.features.safety`; a retrospective back-adjusted source is
#: simply not offered here.
_CAUSAL_SOURCE_SAFETY: dict[PriceDomain, SourceSafety] = {
    PriceDomain.RAW_CONTRACT: SourceSafety(
        point_in_time_safe=True, signal_safe=True, execution_price_safe=True
    ),
    PriceDomain.RAW_CONTINUOUS: SourceSafety(
        point_in_time_safe=True, signal_safe=True, execution_price_safe=False
    ),
    PriceDomain.BACK_ADJUSTED: SourceSafety(
        point_in_time_safe=True, signal_safe=True, execution_price_safe=False
    ),
}


def feature_metadata(spec: MLFeatureSetSpec, feature: MLFeature):
    """Resolved Phase 09 metadata for one ML feature under the declared domain."""
    return REGISTRY.metadata_for(
        feature.spec, source_safety=_CAUSAL_SOURCE_SAFETY[spec.price_domain]
    )


def assert_feature_set_signal_safe(spec: MLFeatureSetSpec) -> None:
    """Re-derive every feature's safety from the registry and refuse the unsafe.

    A feature must be signal-safe and point-in-time safe under the declared price
    domain. ``execution_price_safe`` is hard-wired ``False`` for every computed
    feature and is not checked here: a feature is never an execution price by
    construction -- selecting one is the C++ core's job.
    """
    for f in spec.features:
        meta = feature_metadata(spec, f)
        if not meta.signal_safe:
            raise LeakageError(
                f"ML feature {f.alias!r} ({meta.feature_name}) is not signal-safe under "
                f"price domain {spec.price_domain.value}"
            )
        if not meta.point_in_time_safe:
            raise LeakageError(
                f"ML feature {f.alias!r} ({meta.feature_name}) is not point-in-time safe under "
                f"price domain {spec.price_domain.value}; a retrospectively back-adjusted or "
                "future-roll-aware feature would leak"
            )
        if meta.execution_price_safe:
            raise LeakageError(
                f"ML feature {f.alias!r} reports execution-price safety; a computed feature "
                "is never an execution reference"
            )


def max_lookback_bars(spec: MLFeatureSetSpec) -> int:
    """Warm-up the event builder must respect before the first labelled event."""
    return max(feature_metadata(spec, f).lookback for f in spec.features)

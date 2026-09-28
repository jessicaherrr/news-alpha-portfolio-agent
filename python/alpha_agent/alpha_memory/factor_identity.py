"""Deterministic Factor identity (prompt 2 section 7; hardened first by the
identity-hardening patch, section 1 ("Factor must not equal Mechanism"),
then by the final Phase 2 semantic fix, section 1 ("equal input-data
requirements only prove shared data requirements, not equal quantitative
factor representation").

Mechanism is a PARENT semantic dimension, never the Factor identity itself.
Two strategy families that both implement the same ``EconomicMechanism`` are
NOT automatically the same Factor.

Conservative V1 rule (final fix): ``BaselineFamilyDoc.required_data`` (the
declared market-data SHAPE a family needs) is real, typed, and worth
exposing for transparency, but it is NOT sufficient proof that two families
compute the same quantitative factor -- tsmom (price differencing over two
horizons) and ma_trend (moving-average crossover) both only need "one
continuous or raw OHLC price series per root", yet they are mathematically
different transforms. Merging them on that basis alone would be exactly the
kind of false merging section 7 warns against ("false duplication is
preferable to false merging").

So V1 NEVER automatically merges two different strategy families into one
Factor: :func:`group_families_for_factor_identity` returns one singleton
group per mapped family. Merging remains architecturally possible --
:func:`compute_factor_identity` still accepts an arbitrary
``strategy_families`` tuple -- but only a caller holding independent,
genuinely typed proof of equivalence (e.g. two strategies sharing an
identical, KIND-level registered feature set -- not merely a matching
``required_data`` string) may legitimately pass more than one family. No
such proof exists for any pair of families in this repository today, so
every real call in this codebase passes exactly one family. This is
deliberately NOT a universal free-text factor ontology: nothing here hashes
prose from ``formula_and_timing`` or ``economic_mechanism``, and no new
hand-curated equivalence table is introduced.
"""
from __future__ import annotations

from alpha_agent.alpha_memory.schemas import FACTOR_IDENTITY_SCHEMA, FactorIdentity
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.strategy.baselines.families import BASELINE_FAMILIES, BaselineFamilyDoc
from alpha_agent.strategy.baselines.silver_bullet import SILVER_BULLET_FAMILY
from alpha_agent.translation.research_memory import MECHANISM_TO_KNOWN_FAMILIES
from alpha_agent.validation.fingerprint import fingerprint

__all__ = [
    "compute_factor_identity",
    "family_structural_signature",
    "group_families_for_factor_identity",
]

_FAMILY_DOCS: dict[str, BaselineFamilyDoc] = {d.key: d for d in (*BASELINE_FAMILIES, SILVER_BULLET_FAMILY)}

_PROVENANCE_NOTE = (
    "factor_identity is a fingerprint over (mechanism, the exact set of strategy families this "
    "Factor bundles) -- V1 bundles exactly one family per Factor by default (see "
    "group_families_for_factor_identity). structural_signature (BaselineFamilyDoc.required_data) is "
    "informational only: it documents what market-data SHAPE this family needs, but two families "
    "needing the same shape are NOT proven to compute the same quantitative factor, so it is never "
    "used to decide whether families merge. related_strategy_families is the exact family set this "
    "Factor was built from -- never the mechanism's whole mapped family list unconditionally."
)


def family_structural_signature(strategy_family: str) -> tuple[str, ...]:
    """The real, typed, but INFORMATIONAL-ONLY data-shape signature for one
    strategy family: its declared ``required_data`` (sorted). Never used to
    decide Factor equivalence (see module docstring) -- exposed purely so a
    reader can see WHAT DATA a Factor's family needs, not to justify merging
    it with another family that happens to need the same shape. An unknown
    family (no catalog entry) gets an empty tuple, never a fabricated one."""
    doc = _FAMILY_DOCS.get(strategy_family)
    if doc is None:
        return ()
    return tuple(sorted(doc.required_data))


def compute_factor_identity(mechanism: EconomicMechanism, strategy_families: tuple[str, ...]) -> FactorIdentity:
    """One Factor's identity: a fingerprint over ``mechanism`` PLUS the
    EXACT set of ``strategy_families`` it bundles.

    ``strategy_families`` is normally a singleton -- see
    :func:`group_families_for_factor_identity`, V1's conservative default.
    Passing more than one family is only legitimate when the CALLER
    independently holds genuinely typed proof those families compute the
    same factor (not merely a matching ``required_data`` shape); nothing in
    this repository supplies that proof today. ``structural_signature`` on
    the returned object is informational context about the (single, in
    every real V1 case) family's own data-shape requirement -- it plays no
    role in this function's identity computation.
    """
    families = tuple(sorted(strategy_families))
    ident = fingerprint(
        "factoridentity3",
        {"schema": FACTOR_IDENTITY_SCHEMA, "mechanism": mechanism.value, "strategy_families": list(families)},
    )
    signature = family_structural_signature(families[0]) if len(families) == 1 else ()
    return FactorIdentity(
        mechanism=mechanism,
        factor_identity=ident,
        structural_signature=signature,
        related_strategy_families=families,
        provenance_note=_PROVENANCE_NOTE,
    )


def group_families_for_factor_identity(mechanism: EconomicMechanism) -> tuple[tuple[str, ...], ...]:
    """V1's conservative default grouping: every strategy family Phase 1's
    frozen bridge maps to ``mechanism`` gets its OWN singleton Factor group
    -- NEVER merged with a sibling family merely because both are mapped
    under the same mechanism, and NEVER merged merely because both declare
    the same ``required_data`` (final Phase 2 semantic fix, section 1:
    equal input-data requirements prove only shared data requirements, not
    equal quantitative factor representation). "False duplication is
    preferable to false merging" -- so today, e.g., TREND -> (("ma_trend",),
    ("tsmom",)), two separate singleton groups, even though both need OHLC
    data and both are frozen-mapped to the same mechanism.

    This function is the ONE place V1's "never merge automatically" policy
    is enforced; :func:`compute_factor_identity` stays general so a future
    phase with genuine typed transform/feature-kind equivalence evidence can
    extend grouping without touching identity computation itself.
    """
    families = MECHANISM_TO_KNOWN_FAMILIES.get(mechanism, ())
    return tuple((f,) for f in families)

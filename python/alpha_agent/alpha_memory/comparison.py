""""What changed this time?" (prompt 2 section 15; hardened by the
identity-hardening patch, section 4).

Compares a proposed candidate (mechanism / root / strategy family / params)
against an existing :class:`AlphaResearchObject`, separating SAME from
CHANGED using only dimensions the actual structured evidence can compare.
Never infers novelty from free text (section 15), never scores similarity
(that belongs to :mod:`alpha_agent.registry.similarity`'s own near-duplicate
retrieval, reused here only for its declared-range parameter distance).

Factor sameness is NEVER inferred from mechanism sameness (the bug this
patch fixes): it is computed by rebuilding the candidate's own
``FactorIdentity`` from ``(mechanism, strategy_family)`` via
:func:`alpha_agent.alpha_memory.factor_identity.compute_factor_identity` and
comparing the resulting ``factor_identity`` fingerprint to ``alpha.factor``'s.
Two candidates can share a mechanism and still land on genuinely different
Factors (e.g. different declared ``required_data``); this module must be
able to say exactly that. When no ``strategy_family`` is given, the
candidate's own Factor cannot be computed at all -- "factor" is then simply
omitted from both ``same`` and ``changed`` (an honest "not yet comparable",
never a guess).
"""
from __future__ import annotations

from alpha_agent.alpha_memory.factor_identity import compute_factor_identity
from alpha_agent.alpha_memory.schemas import AlphaResearchObject, WhatChanged
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.similarity import parameter_distance

__all__ = ["what_changed"]


def what_changed(
    alpha: AlphaResearchObject,
    *,
    mechanism: EconomicMechanism,
    root_symbol: str,
    strategy_family: str | None = None,
    params: dict | None = None,
) -> WhatChanged:
    same: list[str] = []
    changed: list[str] = []
    notes: list[str] = []

    if root_symbol == alpha.root_symbol:
        same.append("root")
    else:
        changed.append("root")

    if mechanism == alpha.mechanism:
        same.append("mechanism")
    else:
        changed.append("mechanism")

    if strategy_family is not None:
        candidate_factor = compute_factor_identity(mechanism, (strategy_family,))
        if candidate_factor.factor_identity == alpha.factor.factor_identity:
            same.append("factor")
        else:
            changed.append("factor")
    # strategy_family is None -> the candidate's own Factor cannot be
    # computed yet, so "factor" is deliberately absent from both lists
    # rather than inherited from the mechanism comparison above.

    variant = None
    if strategy_family is not None:
        variant = next((v for v in alpha.strategy_variants if v.strategy_family == strategy_family), None)
        if variant is not None:
            same.append("strategy_family")
        else:
            changed.append("strategy_family")
            notes.append(
                f"'{strategy_family}' has not been implemented under this factor before on {alpha.root_symbol} "
                "-- a genuinely new strategy variant, not just a new parameter set."
            )

    param_diffs: dict[str, tuple[float, float]] = {}
    if variant is not None and variant.canonical_params and params:
        _dist, per = parameter_distance(strategy_family, params, variant.canonical_params)
        for key in per:
            old, new = variant.canonical_params.get(key), params.get(key)
            if isinstance(old, (int, float)) and isinstance(new, (int, float)) and float(old) != float(new):
                param_diffs[key] = (float(old), float(new))
        if param_diffs:
            changed.append("parameterization")
        elif per:
            same.append("parameterization")

    if not same and not changed:
        notes.append("Insufficient structured evidence to compare this candidate to prior research.")

    return WhatChanged(
        same=tuple(same),
        changed=tuple(changed),
        parameter_differences=param_diffs,
        notes=" ".join(notes),
    )

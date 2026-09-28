"""Research recycling (Phase 6 closure, instruction 4).

"Failure Memory is evidence, not a blacklist." This module answers, for a
PROPOSED Factor/Strategy/context -- before it is compiled or executed --
whether it is essentially the same thing already tested, and if so, why, and
what would count as a meaningful change to justify researching it again.

Deliberately NOT a parallel implementation. It composes two existing, real,
frozen tools unchanged:

* :meth:`alpha_agent.registry.sqlite_registry.ExperimentRegistry.find_related`
  -- the SAME deterministic near-duplicate retrieval the runtime orchestrator
  (:mod:`alpha_agent.agents.orchestrator`) already uses inside
  ``plan_family``'s ``NEAR_DUPLICATE_NO_NOVELTY`` disposition, including its
  ``asset_domain`` scoping (ETF and Futures never share near-duplicate
  evidence) and its frozen similarity threshold
  (:attr:`alpha_agent.agents.orchestrator.FamilyPlanSpec.near_duplicate_block_threshold`,
  reused verbatim below, never re-declared).
* :func:`alpha_agent.alpha_memory.comparison.what_changed` -- Phase 2's real
  SAME-vs-CHANGED comparison (root / mechanism / factor / strategy_family /
  parameterization), reused to explain a proposal's decision in the exact
  vocabulary Alpha Memory already uses, when an existing
  :class:`~alpha_agent.alpha_memory.schemas.AlphaResearchObject` is available
  for the proposal's (root, mechanism).

This module never blocks anything, never writes the registry, never calls an
LLM, and never computes an Alpha Score / expected-return / success
probability. ``DEPRIORITIZE_NO_NOVELTY`` is a recommendation with an
explicit, inspectable reason; ``RECONSIDER`` covers everything else,
including "no prior evidence at all" and "prior evidence exists but is not a
blocking near-duplicate, or something meaningful changed, or novelty was
stated." A rejected experiment never permanently removes a Factor from
consideration -- it only raises the bar for restating why this attempt is
different.

Micro-fix (on top of the above, never touching ``similarity.py``): a family
with no declared parameter-grid range in
:mod:`alpha_agent.strategy.baselines.families` (e.g. ETF's ``etf_tsmom``,
which is ETF-specific and never part of that Futures-only table) always
scores its generic ``parameter_proximity`` similarity component at 0,
regardless of whether the proposed params are identical to or wildly
different from a prior experiment's -- so a truly identical repeat can score
BELOW ``near_duplicate_block_threshold`` and slip through as ``RECONSIDER``.
:func:`_find_exact_structured_repeat` checks for an EXACT match on
``asset_domain``/``root_symbol``/``strategy_family``/normalized ``params``
(plus cadence, only when both sides specify one) straight off existing
registry evidence -- never a similarity score -- and runs BEFORE the generic
near-duplicate logic below.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from alpha_agent.agents.orchestrator import FamilyPlanSpec
from alpha_agent.alpha_memory.comparison import what_changed
from alpha_agent.alpha_memory.schemas import AlphaResearchObject, WhatChanged
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.registry.enums import AssetDomain, RegistryVerdict
from alpha_agent.registry.similarity import SimilarityBreakdown
from alpha_agent.registry.sqlite_registry import ExperimentRegistry, RelatedExperiment

__all__ = [
    "RESEARCH_RECYCLING_SCHEMA",
    "RecyclingDecision",
    "ResearchRecyclingAssessment",
    "assess_research_recycling",
]

RESEARCH_RECYCLING_SCHEMA = "research-recycling/1"

#: Reused verbatim from the runtime orchestrator's own frozen family-plan
#: config default -- never re-declared, so the two can never silently drift
#: apart (see ``test_research_recycling.py``'s explicit cross-check).
DEFAULT_NEAR_DUPLICATE_THRESHOLD: float = FamilyPlanSpec.model_fields[
    "near_duplicate_block_threshold"
].default


class RecyclingDecision(str, Enum):
    """Advisory only -- see module docstring. Never a hard block."""

    RECONSIDER = "RECONSIDER"
    DEPRIORITIZE_NO_NOVELTY = "DEPRIORITIZE_NO_NOVELTY"


class ResearchRecyclingAssessment(BaseModel):
    """One proposal's research-recycling read, with its evidence attached so
    a caller (Agent or UI) never has to trust the decision label alone."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = RESEARCH_RECYCLING_SCHEMA
    decision: RecyclingDecision
    explanation: str
    nearest_related: RelatedExperiment | None = None
    what_changed: WhatChanged | None = None


def _normalize_for_exact_match(params: dict) -> dict:
    """Canonicalise numeric params for exact comparison -- an integer-valued
    float (``21.0``) must equal its ``int`` form (``21``), the same
    normalization the registry importer itself applies before hashing."""
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, float) and v.is_integer():
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _cadence_matches(query: str, stored: str | None) -> bool:
    """Compare cadence only when BOTH sides actually specify one -- an
    absent stored or query cadence must never block an otherwise-exact
    structural match (the real XLE row, for instance, was persisted with no
    cadence recorded on either field)."""
    if not query or not stored:
        return True
    return query == stored


def _find_exact_structured_repeat(
    registry: ExperimentRegistry,
    *,
    strategy_family: str,
    root_symbol: str,
    asset_domain: AssetDomain,
    params: dict,
    signal_cadence: str,
    execution_cadence: str,
) -> RelatedExperiment | None:
    """An EXACT structured repeat, straight off existing registry evidence:
    same ``asset_domain`` (the query itself is already domain-scoped), same
    ``root_symbol``, same ``strategy_family``, same normalized ``params``,
    and matching cadence whenever both sides declare one. See the module
    docstring's "Micro-fix" note for why this is needed alongside the
    generic near-duplicate score rather than instead of it."""
    normalized_query = _normalize_for_exact_match(params)
    for view in registry.experiments(
        strategy_family=strategy_family, root_symbol=root_symbol,
        asset_domain=asset_domain, authoritative_only=True,
    ):
        spec = view.experiment.strategy_spec_json
        stored_params = _normalize_for_exact_match(dict(spec.get("params", {})))
        if stored_params != normalized_query:
            continue
        if not _cadence_matches(signal_cadence, spec.get("signal_cadence")):
            continue
        if not _cadence_matches(execution_cadence, spec.get("execution_cadence")):
            continue
        return RelatedExperiment(
            experiment_id=view.experiment_id,
            experiment_identity=view.experiment_identity,
            display_name=view.experiment.display_name,
            strategy_family=view.experiment.strategy_family,
            root_symbol=view.experiment.root_symbol,
            trial_role=view.experiment.trial_role,
            params=stored_params,
            headline_verdict=view.verdict,
            reason_codes=view.result.reason_codes if view.result else (),
            authority=view.authority,
            similarity=SimilarityBreakdown(score=1.0, reasons=("exact_structured_repeat",)),
        )
    return None


def assess_research_recycling(
    registry: ExperimentRegistry,
    *,
    mechanism: EconomicMechanism,
    strategy_family: str,
    root_symbol: str,
    asset_domain: AssetDomain,
    params: dict,
    novelty_notes: str = "",
    signal_cadence: str = "",
    execution_cadence: str = "",
    feature_fingerprints: tuple[str, ...] = (),
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    alpha: AlphaResearchObject | None = None,
) -> ResearchRecyclingAssessment:
    """Assess whether a proposed (mechanism, strategy_family, root_symbol,
    params) is worth researching again, given existing registry evidence.

    ``alpha``, when the caller already has the relevant
    :class:`AlphaResearchObject` on hand (e.g. from
    :mod:`alpha_agent.alpha_memory.builder`), sharpens the explanation with a
    real ``what_changed`` breakdown; it is optional because ``find_related``
    alone is already sufficient to make the decision.
    """
    changed: WhatChanged | None = None
    if alpha is not None:
        changed = what_changed(
            alpha,
            mechanism=mechanism,
            root_symbol=root_symbol,
            strategy_family=strategy_family,
            params=params,
        )
    meaningfully_changed = bool(changed and changed.changed)

    # Exact structured repeat, checked BEFORE generic similarity logic (see
    # the module docstring's "Micro-fix" note): a family with no declared
    # parameter-grid range can score an identical repeat below the generic
    # near-duplicate threshold, so this check never relies on that score.
    exact = _find_exact_structured_repeat(
        registry, strategy_family=strategy_family, root_symbol=root_symbol,
        asset_domain=asset_domain, params=params,
        signal_cadence=signal_cadence, execution_cadence=execution_cadence,
    )
    if (
        exact is not None
        and exact.headline_verdict in (RegistryVerdict.REJECT, RegistryVerdict.INCONCLUSIVE)
        and not novelty_notes.strip()
        and not meaningfully_changed
    ):
        prior_reasons = ", ".join(exact.reason_codes) or "no reason codes recorded"
        return ResearchRecyclingAssessment(
            decision=RecyclingDecision.DEPRIORITIZE_NO_NOVELTY,
            explanation=(
                f"Exact structured repeat of {exact.experiment_id} (same "
                f"asset_domain/root_symbol/strategy_family/params, prior "
                f"{exact.headline_verdict.value}: {prior_reasons}) with no stated "
                "novelty and no structurally meaningful change. This is evidence "
                "to weigh, not a permanent block -- state what is meaningfully "
                "different (market context, measurable transform, Strategy "
                "implementation, instrument/asset domain, new legitimate data, "
                "or hypothesis) to reconsider."
            ),
            nearest_related=exact,
            what_changed=changed,
        )

    related = registry.find_related(
        strategy_family=strategy_family,
        root_symbol=root_symbol,
        asset_domain=asset_domain,
        params=params,
        signal_cadence=signal_cadence,
        execution_cadence=execution_cadence,
        feature_fingerprints=feature_fingerprints,
        top_k=3,
    )

    if not related:
        return ResearchRecyclingAssessment(
            decision=RecyclingDecision.RECONSIDER,
            explanation=(
                "No related prior experiment found for this Factor/Strategy/"
                "context in this asset domain -- unresearched territory."
            ),
            what_changed=changed,
        )

    top = related[0]
    is_near_duplicate = (
        top.similarity.score >= near_duplicate_threshold
        and top.headline_verdict in (RegistryVerdict.REJECT, RegistryVerdict.INCONCLUSIVE)
    )

    if is_near_duplicate and not novelty_notes.strip() and not meaningfully_changed:
        prior_reasons = ", ".join(top.reason_codes) or "no reason codes recorded"
        return ResearchRecyclingAssessment(
            decision=RecyclingDecision.DEPRIORITIZE_NO_NOVELTY,
            explanation=(
                f"Near-duplicate of {top.experiment_id} "
                f"(similarity {top.similarity.score:.3f}, prior "
                f"{top.headline_verdict.value if top.headline_verdict else '?'}: {prior_reasons}) "
                "with no stated novelty and no structurally meaningful change. This is "
                "evidence to weigh, not a permanent block -- state what is meaningfully "
                "different (market context, measurable transform, Strategy implementation, "
                "instrument/asset domain, new legitimate data, or hypothesis) to reconsider."
            ),
            nearest_related=top,
            what_changed=changed,
        )

    reasons: list[str] = []
    if novelty_notes.strip():
        reasons.append(f"stated novelty: {novelty_notes.strip()}")
    if meaningfully_changed:
        reasons.append(f"meaningfully changed: {', '.join(changed.changed)}")
    if not is_near_duplicate:
        verdict_label = top.headline_verdict.value if top.headline_verdict else "NOT_ADJUDICATED"
        reasons.append(
            f"nearest prior evidence ({top.experiment_id}) is not a blocking "
            f"near-duplicate (similarity {top.similarity.score:.3f}, verdict {verdict_label})"
        )
    return ResearchRecyclingAssessment(
        decision=RecyclingDecision.RECONSIDER,
        explanation="; ".join(reasons) or "related evidence exists but does not block reconsideration.",
        nearest_related=top,
        what_changed=changed,
    )

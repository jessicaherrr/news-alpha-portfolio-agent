"""Phase B1 -- the candidate presentation/research model and ranking helpers.

`ResearchCandidateSummary` is the one typed object the UI consumes (task spec
section 13: "UI should consume typed deterministic outputs. Do not put
scoring logic directly inside Streamlit views."). It is built from a single
registry row (as returned by `alpha_agent.ui.services.list_experiments`, one
row per CANONICAL trial) plus an `InvestorProfile` -- nothing here reads the
registry directly, so this module is fully unit-testable against synthetic
rows with no database.

Three dimensions stay architecturally separate all the way through this
module (task spec sections 2/9/12):

* `scientific_verdict` -- copied VERBATIM from the row's own `verdict`. Never
  recomputed, never influenced by promise/fit.
* `research_promise_score` -- `alpha_agent.recommendation.promise`, profile-
  independent.
* `user_fit_score` -- `alpha_agent.recommendation.fit`, verdict-independent
  and promise-independent.

`validated_strategies` / `promising_candidates` enforce task spec section 18:
only a PASS verdict may ever appear in the validated list; a REJECT or
INCONCLUSIVE candidate can rank arbitrarily high in the promising list but can
never be relabeled or promoted into "validated".
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel

from alpha_agent.recommendation.explanations import (
    next_research_direction,
    validation_blockers,
    why_promising,
)
from alpha_agent.recommendation.fit import PersonalizationState, score_user_fit
from alpha_agent.recommendation.profile import DEFAULT_PROFILE, InvestorProfile
from alpha_agent.recommendation.promise import ResearchPromiseBreakdown, score_research_promise


class ResearchCandidateSummary(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    experiment_id: str
    experiment_identity: str
    root_symbol: str
    strategy_family: str
    friendly_strategy_name: str

    #: Existing authoritative evidence, copied verbatim -- see module
    #: docstring. One of PASS / REJECT / INCONCLUSIVE / NOT_ADJUDICATED.
    scientific_verdict: str

    research_promise_score: float
    research_promise_label: str
    research_promise_breakdown: ResearchPromiseBreakdown

    user_fit_score: float
    user_fit_label: str
    user_fit_evidence_coverage: dict[str, str]
    #: Section 6 -- the UI must consult this before ever rendering
    #: `user_fit_score`/`user_fit_label` as a percentage: `NOT_PERSONALIZED`
    #: means there is nothing measured yet (see `alpha_agent.ui.
    #: recommendation_views.format_user_fit`).
    user_fit_personalization_state: PersonalizationState
    #: Section 7 -- "Based on N measurable dimensions".
    user_fit_measurable_dimension_count: int

    annualized_sharpe: float | None
    net_pnl_usd: float | None
    trade_count: int | None
    fold_consistency: float | None

    why_promising: str
    validation_blockers: tuple[str, ...]
    next_research_direction: str


def build_candidate_summary(
    row: dict[str, Any],
    *,
    profile: InvestorProfile = DEFAULT_PROFILE,
    strategy_name_fn: Callable[[str | None], str] | None = None,
) -> ResearchCandidateSummary:
    """Build one `ResearchCandidateSummary` from one registry row. `row` is
    expected to carry at least the keys `alpha_agent.ui.services.
    list_experiments` returns (experiment_id/root_symbol/strategy_family/
    verdict/annualized_sharpe/... plus the extended result fields:
    fold_consistency/parameter_stability/cost_stress/regime_evidence/
    gating_null_p/evidence_completeness/n_oos_days)."""
    verdict = row.get("verdict") or "NOT_ADJUDICATED"
    promise = score_research_promise(row)
    fit = score_user_fit(strategy_family=row.get("strategy_family"), result=row, profile=profile)
    family = row.get("strategy_family") or "--"
    friendly_name = strategy_name_fn(family) if strategy_name_fn else family

    return ResearchCandidateSummary(
        experiment_id=row["experiment_id"],
        experiment_identity=row.get("experiment_identity", ""),
        root_symbol=row["root_symbol"],
        strategy_family=family,
        friendly_strategy_name=friendly_name,
        scientific_verdict=verdict,
        research_promise_score=promise.total,
        research_promise_label=promise.label,
        research_promise_breakdown=promise,
        user_fit_score=fit.total,
        user_fit_label=fit.label,
        user_fit_evidence_coverage=fit.evidence_coverage,
        user_fit_personalization_state=fit.personalization_state,
        user_fit_measurable_dimension_count=fit.measurable_dimension_count,
        annualized_sharpe=row.get("annualized_sharpe"),
        net_pnl_usd=row.get("net_pnl_usd"),
        trade_count=row.get("n_trades"),
        fold_consistency=row.get("fold_consistency"),
        why_promising=why_promising(row, promise),
        validation_blockers=validation_blockers(row, verdict),
        next_research_direction=next_research_direction(row),
    )


def build_candidate_summaries(
    rows: Iterable[dict[str, Any]],
    *,
    profile: InvestorProfile = DEFAULT_PROFILE,
    strategy_name_fn: Callable[[str | None], str] | None = None,
) -> list[ResearchCandidateSummary]:
    return [
        build_candidate_summary(row, profile=profile, strategy_name_fn=strategy_name_fn) for row in rows
    ]


def validated_strategies(
    candidates: Iterable[ResearchCandidateSummary],
) -> list[ResearchCandidateSummary]:
    """PASS only (task spec section 18). Deterministic tie-break by
    `experiment_id` so repeated calls over the same input always return the
    same order."""
    return sorted(
        (c for c in candidates if c.scientific_verdict == "PASS"),
        key=lambda c: c.experiment_id,
    )


SortKey = Literal["promise", "fit"]


def _has_no_committed_evidence(candidate: ResearchCandidateSummary) -> bool:
    """True for a NOT_ADJUDICATED trial that carries literally zero committed
    evidence (a typed pre-adjudication refusal, e.g. Phase 15B's
    `PHASE_15_TRIAL_REFUSED` / `INSUFFICIENT_TRAIN_EVENTS`) -- as opposed to a
    NOT_ADJUDICATED trial that DOES carry partial evidence. Used only to keep
    such a trial out of the ranked promising list (task spec section 22: "a
    NOT_ADJUDICATED partial evidence should not dominate fully evaluated
    candidates" -- generalised here to BOTH sort modes, since a `sort_by="fit"`
    view would otherwise let a zero-evidence trial's trivial mechanism-match
    fit score rank it above every genuinely evaluated REJECT/INCONCLUSIVE
    candidate). It remains visible everywhere else (e.g. the Strategies page's
    "All Tested" view) -- this never hides or deletes a failure record."""
    return candidate.scientific_verdict == "NOT_ADJUDICATED" and candidate.research_promise_score == 0.0


def promising_candidates(
    candidates: Iterable[ResearchCandidateSummary], *, sort_by: SortKey = "promise"
) -> list[ResearchCandidateSummary]:
    """Every non-PASS candidate that carries at least some committed
    evidence (REJECT / INCONCLUSIVE, or a NOT_ADJUDICATED trial with partial
    evidence), ranked by Research Promise by default (task spec section 17:
    "Do not default to User Fit alone"). `sort_by="fit"` offers the optional
    personalized ordering -- it never changes `scientific_verdict` and never
    moves a candidate into/out of `validated_strategies`.

    Deterministic tie-break: the secondary score, then `experiment_id`, so
    ties never depend on input/dict iteration order.
    """
    pool = [
        c for c in candidates
        if c.scientific_verdict != "PASS" and not _has_no_committed_evidence(c)
    ]
    if sort_by == "fit":
        return sorted(pool, key=lambda c: (-c.user_fit_score, -c.research_promise_score, c.experiment_id))
    return sorted(pool, key=lambda c: (-c.research_promise_score, -c.user_fit_score, c.experiment_id))

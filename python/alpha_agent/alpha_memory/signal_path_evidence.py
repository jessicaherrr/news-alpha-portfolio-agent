"""News Alpha Phase H -- one news-to-portfolio research run as typed SIGNAL-PATH
EVIDENCE (the registry's schema-v7 `signal_path_evidence`).

Every hypothesis the run produced is accounted for at the furthest stage it
reached, with a typed reason:

    a signal path no expression takes up          -> SIGNAL_PATH
    an expression refused before a candidate      -> ASSET_EXPRESSION / MEASUREMENT
    a candidate signal (one record per origin path) -> its fate through
        screening -> ranking -> the portfolio gate -> validation of the portfolio

Outcome and scope are kept apart on purpose:

* FAILURE / UNRESOLVED / SUCCESS says how the hypothesis fared;
* HYPOTHESIS / SCREENING / PORTFOLIO says what the evidence is ABOUT. A data
  gap is a HYPOTHESIS-scope failure (retry when the data exists), an
  in-sample screen is SCREENING-scope, and a validation result is
  PORTFOLIO-scope -- a fact about the tested portfolio, never about the
  member signal, its path or its mechanism. SUCCESS exists only at PORTFOLIO
  scope (the record model refuses anything else).

Deterministic: the same run produces the same records (``evidence_id`` is a
content fingerprint), so recording it twice writes nothing new.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from alpha_agent.news_alpha.asset_expression import (
    AssetExpression,
    AssetExpressionPlan,
    ExpressionStatus,
)
from alpha_agent.news_alpha.candidate_signals import (
    CandidateRefusal,
    CandidateSignalSet,
    RefusalReason,
)
from alpha_agent.news_alpha.measurement import ResolutionStatus
from alpha_agent.news_alpha.signal_paths import PathStatus, SignalPath, SignalPathDiscovery
from alpha_agent.portfolio.plan import PortfolioPlan
from alpha_agent.portfolio.validation import PortfolioValidationReport, PortfolioValidationStatus
from alpha_agent.recommendation.signal_ranking import (
    NotRankableReason,
    RankedSignalSet,
    RankingTier,
)
from alpha_agent.registry.enums import (
    EvidenceScope,
    HypothesisStage,
    PathEvidenceOutcome,
    PathEvidenceReason,
)
from alpha_agent.registry.models import SignalPathEvidenceRecord
from alpha_agent.screening.candidate_signal_screen import CandidateScreen
from alpha_agent.screening.factor_diagnostics import ScreenStatus
from alpha_agent.validation.enums import ReasonCode
from alpha_agent.validation.fingerprint import fingerprint

__all__ = [
    "EVIDENCE_RULE",
    "EventResearch",
    "build_signal_path_evidence",
    "research_run_id",
]

EVIDENCE_RULE = "signal-path-evidence/1"

#: status precedence (Phase D): the primary blocker is the first one present
_BLOCKER_ORDER = (ResolutionStatus.DOMAIN_UNAVAILABLE, ResolutionStatus.NOT_PIT_SAFE, ResolutionStatus.MISSING,
                  ResolutionStatus.NOT_EXECUTABLE)


@dataclass(frozen=True)
class EventResearch:
    """One event's hypothesis-plane objects, as the pipeline built them."""

    discovery: SignalPathDiscovery
    expressions: AssetExpressionPlan
    candidates: CandidateSignalSet

    def __post_init__(self) -> None:
        if self.expressions.signal_paths_id != self.discovery.fingerprint():
            raise ValueError("the expression plan was built from different signal paths")
        if self.candidates.asset_expression_id != self.expressions.fingerprint():
            raise ValueError("the candidate set was built from a different expression plan")


def research_run_id(events: Sequence[EventResearch], *, ranked: RankedSignalSet | None, plan: PortfolioPlan | None,
                    validation: PortfolioValidationReport | None) -> str:
    """Identity of one research run: its inputs and the outcomes it reached."""
    return fingerprint("newsalpharun1", {
        "rule": EVIDENCE_RULE,
        "events": sorted([e.candidates.event_id, e.candidates.fingerprint()] for e in events),
        "ranked": ranked.fingerprint() if ranked else None,
        "plan": plan.fingerprint() if plan else None,
        "validation": None if validation is None else [
            validation.status.value, validation.strategy_fingerprint,
            validation.report.report_fingerprint() if validation.report else None],
    })


@dataclass(frozen=True)
class _Fate:
    stage: HypothesisStage
    outcome: PathEvidenceOutcome
    scope: EvidenceScope
    reason: PathEvidenceReason
    detail: str
    evidence: dict
    strategy_fingerprint: str | None = None
    experiment_identity: str | None = None


def _path_fields(p: SignalPath) -> dict:
    return {"mechanism_graph_id": p.mechanism_graph_id, "path_id": p.path_id, "path_signature": p.signature,
            "path_type": p.path_type.value if p.path_type else None, "transmission_depth": p.transmission_depth,
            "consequence_state": p.consequence_state}


def _expression_fields(e: AssetExpression) -> dict:
    return {"expression_id": e.expression_id, "expression_concept": e.concept, "expression_domain": e.domain.value,
            "expression_fidelity": e.fidelity.value}


def _refusal_fate(r: CandidateRefusal, e: AssetExpression, plan: AssetExpressionPlan) -> tuple[_Fate, str | None]:
    """The typed cause behind a Phase E refusal, and the measurement status it rests on."""
    if r.reason is RefusalReason.NO_SIGNAL_RULE:
        return _Fate(HypothesisStage.MEASUREMENT, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.HYPOTHESIS,
                     PathEvidenceReason.NO_SIGNAL_RULE, r.note, {}), None
    if r.reason is RefusalReason.CONFIRMATION_ONLY_MEASUREMENT:
        return _Fate(HypothesisStage.MEASUREMENT, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.HYPOTHESIS,
                     PathEvidenceReason.CONFIRMATION_ONLY_MEASUREMENT, r.note, {}), None
    by_status = {
        ExpressionStatus.EXCLUDED_BY_MANDATE: (PathEvidenceOutcome.UNRESOLVED,
                                               PathEvidenceReason.EXPRESSION_EXCLUDED_BY_MANDATE),
        ExpressionStatus.REMOVED_BY_CONSTRAINTS: (PathEvidenceOutcome.UNRESOLVED,
                                                  PathEvidenceReason.REMOVED_BY_CONSTRAINTS),
        ExpressionStatus.DOMAIN_UNAVAILABLE: (PathEvidenceOutcome.FAILURE, PathEvidenceReason.DOMAIN_UNAVAILABLE),
        ExpressionStatus.NO_INSTRUMENT: (PathEvidenceOutcome.FAILURE, PathEvidenceReason.NO_INSTRUMENT),
    }
    if e.status in by_status:
        outcome, reason = by_status[e.status]
        return _Fate(HypothesisStage.ASSET_EXPRESSION, outcome, EvidenceScope.HYPOTHESIS, reason, r.note,
                     {"expression_status": e.status.value}), None
    statuses = [m.resolution.status for m in plan.measurements_for(e)]
    blocker = next((s for s in _BLOCKER_ORDER if s in statuses), None)
    return _Fate(HypothesisStage.MEASUREMENT, PathEvidenceOutcome.FAILURE, EvidenceScope.HYPOTHESIS,
                 PathEvidenceReason.MEASUREMENT_UNAVAILABLE, r.note,
                 {"measurement_statuses": sorted({s.value for s in statuses})}), blocker.value if blocker else None


def _validation_fate(cid: str, validation: PortfolioValidationReport | None, plan: PortfolioPlan | None,
                     experiment_identity: str | None) -> _Fate:
    """A screen-supported signal's fate at the portfolio gate and in validation."""
    if validation is None:
        return _Fate(HypothesisStage.RANKING, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.SCREENING,
                     PathEvidenceReason.NOT_ELIGIBLE_FOR_VALIDATION, "Not taken to validation in this run.", {})
    status = validation.status
    if status is PortfolioValidationStatus.NO_ELIGIBLE_PORTFOLIO:
        return _Fate(HypothesisStage.PORTFOLIO, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.HYPOTHESIS,
                     PathEvidenceReason.NOT_ELIGIBLE_FOR_VALIDATION, validation.status_detail,
                     {"gate": [f.reason.value for f in validation.eligibility.findings]})
    if status is PortfolioValidationStatus.EXECUTION_UNSUPPORTED:
        return _Fate(HypothesisStage.PORTFOLIO, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.HYPOTHESIS,
                     PathEvidenceReason.EXECUTION_UNSUPPORTED, validation.status_detail, {})
    if status is PortfolioValidationStatus.COST_MODEL_INCOMPLETE:
        return _Fate(HypothesisStage.PORTFOLIO, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.HYPOTHESIS,
                     PathEvidenceReason.COST_MODEL_INCOMPLETE, validation.status_detail, {})
    strategy = validation.strategy
    if strategy is None or cid not in {m.candidate_signal_id for m in strategy.members}:
        return _Fate(HypothesisStage.RANKING, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.SCREENING,
                     PathEvidenceReason.NOT_ELIGIBLE_FOR_VALIDATION,
                     "Screen-supported, but not a member of the validated portfolio.", {})
    fp = validation.strategy_fingerprint
    if status is PortfolioValidationStatus.PRIOR_RESULT_CITED:
        outcome = {"PASS": PathEvidenceOutcome.SUCCESS, "REJECT": PathEvidenceOutcome.FAILURE}.get(
            validation.cited_verdict or "", PathEvidenceOutcome.UNRESOLVED)
        return _Fate(HypothesisStage.VALIDATION, outcome, EvidenceScope.PORTFOLIO,
                     PathEvidenceReason.PORTFOLIO_PRIOR_RESULT_CITED, validation.status_detail,
                     {"cited_verdict": validation.cited_verdict}, fp, experiment_identity)
    codes = [c.value for c in validation.report.reason_codes] if validation.report else []
    evidence = {"verdict": validation.report.verdict.value if validation.report else None, "reason_codes": codes}
    if status is PortfolioValidationStatus.VALIDATED:
        return _Fate(HypothesisStage.VALIDATION, PathEvidenceOutcome.SUCCESS, EvidenceScope.PORTFOLIO,
                     PathEvidenceReason.PORTFOLIO_VALIDATED, validation.scope_note, evidence, fp, experiment_identity)
    if status is PortfolioValidationStatus.REJECTED:
        reason = (PathEvidenceReason.PORTFOLIO_COST_FRAGILE if ReasonCode.COST_STRESS_FAILED.value in codes
                  else PathEvidenceReason.PORTFOLIO_REJECTED)
        return _Fate(HypothesisStage.VALIDATION, PathEvidenceOutcome.FAILURE, EvidenceScope.PORTFOLIO, reason,
                     validation.status_detail, evidence, fp, experiment_identity)
    return _Fate(HypothesisStage.VALIDATION, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.PORTFOLIO,
                 PathEvidenceReason.PORTFOLIO_INCONCLUSIVE, validation.status_detail, evidence, fp,
                 experiment_identity)


def _signal_fate(cid: str, ranked: RankedSignalSet | None, screens: Mapping[str, CandidateScreen],
                 plan: PortfolioPlan | None, validation: PortfolioValidationReport | None,
                 experiment_identity: str | None) -> _Fate:
    screen = screens.get(cid)
    signal = next((s for s in ranked.signals if s.candidate_signal_id == cid), None) if ranked else None
    if signal is None or signal.tier is RankingTier.NOT_RANKABLE and signal.not_rankable_reason in (
            NotRankableReason.NOT_SCREENED, NotRankableReason.NO_DIAGNOSTIC_METHOD, None):
        reason = (PathEvidenceReason.NO_DIAGNOSTIC_METHOD
                  if signal is not None and signal.not_rankable_reason is NotRankableReason.NO_DIAGNOSTIC_METHOD
                  else PathEvidenceReason.NOT_SCREENED)
        return _Fate(HypothesisStage.CANDIDATE_SIGNAL, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.HYPOTHESIS,
                     reason, "No discovery screen of this candidate yet.", {})
    evidence = {"tier": signal.tier.value}
    if screen is not None:
        d = screen.diagnostics
        evidence |= {"screen_status": d.status.value, "diagnostic_scope": d.scope.value,
                     "prediction_horizon_days": signal.prediction_horizon_days}
    if signal.tier is RankingTier.EXCLUDED_BY_MANDATE:
        return _Fate(HypothesisStage.RANKING, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.HYPOTHESIS,
                     PathEvidenceReason.SIGNAL_EXCLUDED_BY_MANDATE, signal.summaries.user_fit, evidence)
    status = screen.diagnostics.status if screen is not None else None
    if signal.tier is RankingTier.NOT_RANKABLE:
        reason = (PathEvidenceReason.SCREEN_INSUFFICIENT_DATA
                  if signal.not_rankable_reason is NotRankableReason.INSUFFICIENT_EVIDENCE
                  else PathEvidenceReason.NOT_RANKABLE)
        return _Fate(HypothesisStage.SCREEN, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.SCREENING, reason,
                     signal.summaries.scientific_quality, evidence)
    if status is ScreenStatus.CONTRADICTS_EXPECTED_SIGN:
        return _Fate(HypothesisStage.SCREEN, PathEvidenceOutcome.FAILURE, EvidenceScope.SCREENING,
                     PathEvidenceReason.SCREEN_CONTRADICTS_SIGN, screen.diagnostics.status_note, evidence)
    if status is ScreenStatus.NO_SCREEN_SUPPORT:
        return _Fate(HypothesisStage.SCREEN, PathEvidenceOutcome.FAILURE, EvidenceScope.SCREENING,
                     PathEvidenceReason.SCREEN_NO_SUPPORT, screen.diagnostics.status_note, evidence)
    if status is not ScreenStatus.SCREEN_CONTINUE:
        return _Fate(HypothesisStage.SCREEN, PathEvidenceOutcome.UNRESOLVED, EvidenceScope.SCREENING,
                     PathEvidenceReason.SCREEN_INSUFFICIENT_DATA, screen.diagnostics.status_note if screen else "",
                     evidence)
    fate = _validation_fate(cid, validation, plan, experiment_identity)
    return _Fate(fate.stage, fate.outcome, fate.scope, fate.reason, fate.detail, evidence | fate.evidence,
                 fate.strategy_fingerprint, fate.experiment_identity)


def build_signal_path_evidence(
    events: Sequence[EventResearch],
    *,
    ranked: RankedSignalSet | None,
    screens: Mapping[str, CandidateScreen],
    plan: PortfolioPlan | None,
    validation: PortfolioValidationReport | None,
    recorded_at: str,
    experiment_identity: str | None = None,
) -> tuple[SignalPathEvidenceRecord, ...]:
    """Every hypothesis of the run, once per (event, path, expression,
    measurement, candidate) it touched. ``experiment_identity`` links the
    records of a validated portfolio's members to its registry experiment
    (pass it only after that experiment is recorded)."""
    run_id = research_run_id(events, ranked=ranked, plan=plan, validation=validation)
    common = {"research_run_id": run_id, "recorded_at": recorded_at,
              "portfolio_plan_fingerprint": plan.fingerprint() if plan else None}
    out: dict[str, SignalPathEvidenceRecord] = {}

    def add(**fields) -> None:
        record = SignalPathEvidenceRecord.create(**common, **fields)
        out.setdefault(record.evidence_id, record)

    for er in events:
        rules = (*er.discovery.rules, *er.expressions.rules, *er.candidates.rules)
        head = {"event_id": er.candidates.event_id, "event_headline": er.candidates.event_headline,
                "rule_versions": (*rules, EVIDENCE_RULE)}
        paths = {p.path_id: p for p in er.discovery.paths}
        expressions = {e.expression_id: e for e in er.expressions.expressions}
        expressed_paths = {pid for e in er.expressions.expressions for pid in e.path_ids}

        for p in er.discovery.paths:
            if p.path_id in expressed_paths:
                continue
            unresolved = p.status in (PathStatus.UNRESOLVED, PathStatus.REJECTED)
            add(**head, **_path_fields(p), stage_reached=HypothesisStage.SIGNAL_PATH,
                outcome=PathEvidenceOutcome.UNRESOLVED if unresolved else PathEvidenceOutcome.FAILURE,
                evidence_scope=EvidenceScope.HYPOTHESIS,
                reason_code=PathEvidenceReason.PATH_UNRESOLVED if unresolved else PathEvidenceReason.PATH_NOT_EXPRESSED,
                detail=(f"Path status {p.status.value}." if unresolved
                        else f"No reviewed asset expression takes up {p.consequence_label}."),
                evidence={"path_status": p.status.value})

        for r in er.candidates.refusals:
            e = expressions[r.expression_id]
            fate, blocker = _refusal_fate(r, e, er.expressions)
            for pid in e.path_ids:
                add(**head, **_path_fields(paths[pid]), **_expression_fields(e), instrument=r.instrument,
                    measurement_id=r.measurement_id, measurement_status=blocker, stage_reached=fate.stage,
                    outcome=fate.outcome, evidence_scope=fate.scope, reason_code=fate.reason, detail=fate.detail,
                    evidence=fate.evidence)

        for c in er.candidates.candidates:
            fate = _signal_fate(c.candidate_signal_id, ranked, screens, plan, validation, experiment_identity)
            for o in c.origins:
                add(**head, **_path_fields(paths[o.path_id]), **_expression_fields(expressions[o.expression_id]),
                    instrument=c.spec.instrument, measurement_id=o.measurement_id,
                    measurement_status=c.spec.data.resolution_status.value, candidate_signal_id=c.candidate_signal_id,
                    factor_identity=c.factor_identity, portfolio_strategy_fingerprint=fate.strategy_fingerprint,
                    experiment_identity=fate.experiment_identity, stage_reached=fate.stage, outcome=fate.outcome,
                    evidence_scope=fate.scope, reason_code=fate.reason, detail=fate.detail, evidence=fate.evidence)
    return tuple(sorted(out.values(), key=lambda r: (r.event_id, r.path_signature or "", r.expression_id or "",
                                                     r.candidate_signal_id or "", r.evidence_id)))

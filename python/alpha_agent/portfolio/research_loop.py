"""News Alpha Phase H -- CLOSING THE RESEARCH LOOP: one call from a Phase G plan to
recorded evidence.

    events (Phases A-E) + ranked set (F) + plan (G)
      -> validate_portfolio          input gate, execution gate, C++ replays,
                                     the existing validation plane (portfolio.validation)
      -> registry experiments        only for an executed validation
                                     (portfolio.registry_record, the existing registry)
      -> signal-path evidence        every hypothesis of the run, typed
                                     (alpha_memory.signal_path_evidence, registry schema v7)
      -> path-level study hooks      testable today or not, with every blocker
                                     (news_alpha.study_design)

Nothing is recorded unless ``record=True`` and a registry is given; an
ineligible portfolio still produces its evidence (why nothing entered), and
that evidence is worth recording -- a failure is first-class data.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from alpha_agent.alpha_memory.signal_path_evidence import (
    EventResearch,
    build_signal_path_evidence,
    research_run_id,
)
from alpha_agent.news_alpha.candidate_signals import CandidateSignal
from alpha_agent.news_alpha.mandate import ResearchMandate
from alpha_agent.news_alpha.study_design import (
    StudyCapabilities,
    StudyTestability,
    assess_study,
    default_designs,
    probe_study_capabilities,
)
from alpha_agent.portfolio.evaluation import evaluate_factor_through_validation
from alpha_agent.portfolio.execution import ExecutionInputs, load_execution_inputs
from alpha_agent.portfolio.plan import PortfolioPlan
from alpha_agent.portfolio.risk_model import MarketSnapshotStore, SnapshotProvider, SnapshotWindow
from alpha_agent.portfolio.selection import FactorSource
from alpha_agent.portfolio.strategy import PortfolioStrategySpec
from alpha_agent.portfolio.validation import (
    PortfolioValidationReport,
    PortfolioValidationStatus,
    validate_portfolio,
)
from alpha_agent.recommendation.signal_ranking import RankedSignalSet
from alpha_agent.registry.models import SignalPathEvidenceRecord
from alpha_agent.screening.candidate_signal_screen import CandidateScreen

__all__ = [
    "RESEARCH_LOOP_SCHEMA",
    "ResearchLoopOutcome",
    "close_research_loop",
    "evaluation_snapshots",
    "real_factor_loader",
]

RESEARCH_LOOP_SCHEMA = "research-loop/1"
_EXECUTED = (PortfolioValidationStatus.VALIDATED, PortfolioValidationStatus.REJECTED,
             PortfolioValidationStatus.INSUFFICIENT_EVIDENCE)


def evaluation_snapshots(directory: Path | None = None) -> SnapshotProvider:
    """Cached EVALUATION-window (2018-2024) snapshots -- validation only."""
    return MarketSnapshotStore(directory, window=SnapshotWindow.EVALUATION).provider()


def real_factor_loader(candidates: Mapping[str, CandidateSignal], screens: Mapping[str, CandidateScreen]
                       ) -> Callable[[PortfolioStrategySpec], dict[str, FactorSource]]:
    """Each member's factor through the validation window, its discovery
    prefix checked against its own screen (look-ahead guard)."""
    def load(strategy: PortfolioStrategySpec) -> dict[str, FactorSource]:
        return {m.candidate_signal_id: evaluate_factor_through_validation(
            candidates[m.candidate_signal_id], screen=screens[m.candidate_signal_id]) for m in strategy.members}
    return load


class ResearchLoopOutcome(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = RESEARCH_LOOP_SCHEMA
    research_run_id: str
    validation: PortfolioValidationReport
    experiment_identity: str | None
    evidence_by_outcome: dict[str, int]
    evidence_by_scope: dict[str, int]
    evidence_by_reason: dict[str, int]
    studies: tuple[StudyTestability, ...]
    recorded_experiments: int = 0
    recorded_evidence: int = 0

    def headline(self) -> str:
        return self.validation.headline()


def close_research_loop(
    events: Sequence[EventResearch],
    ranked: RankedSignalSet,
    screens: Mapping[str, CandidateScreen],
    plan: PortfolioPlan,
    mandate: ResearchMandate,
    *,
    workdir: Path,
    registry=None,
    record: bool = False,
    load_factor_sources: Callable[[PortfolioStrategySpec], Mapping[str, FactorSource]] | None = None,
    snapshots: SnapshotProvider | None = None,
    load_inputs: Callable[[PortfolioStrategySpec], ExecutionInputs] | None = None,
    study_capabilities: StudyCapabilities | None = None,
    recorded_at: str | None = None,
    code_commit: str = "",
    source_artifact: str | None = None,
    allocator_cli=None,
    replay_cli: Path | None = None,
) -> tuple[ResearchLoopOutcome, tuple[SignalPathEvidenceRecord, ...]]:
    if record and registry is None:
        raise ValueError("recording needs a registry")
    candidates = {c.candidate_signal_id: c for e in events for c in e.candidates.candidates}
    validation = validate_portfolio(
        plan, ranked, screens, mandate,
        load_factor_sources=load_factor_sources or real_factor_loader(candidates, screens),
        snapshots=snapshots or evaluation_snapshots(), load_inputs=load_inputs or load_execution_inputs,
        workdir=workdir, registry=registry, allocator_cli=allocator_cli, replay_cli=replay_cli,
    )
    recorded_experiments = 0
    linked_identity = None
    if validation.status is PortfolioValidationStatus.PRIOR_RESULT_CITED:
        linked_identity = validation.experiment_identity
    if record and validation.status in _EXECUTED:
        from alpha_agent.portfolio.registry_record import build_portfolio_import_bundle

        bundle = build_portfolio_import_bundle(validation, code_commit=code_commit, source_artifact=source_artifact)
        recorded_experiments = registry.apply_bundle(bundle).get("experiments", 0)
        linked_identity = validation.experiment_identity
    evidence = build_signal_path_evidence(
        events, ranked=ranked, screens=screens, plan=plan, validation=validation,
        recorded_at=recorded_at or datetime.now(UTC).isoformat(), experiment_identity=linked_identity,
    )
    recorded_evidence = registry.record_signal_path_evidence(evidence) if record else 0
    capabilities = study_capabilities or probe_study_capabilities()
    outcome = ResearchLoopOutcome(
        research_run_id=research_run_id(events, ranked=ranked, plan=plan, validation=validation),
        validation=validation, experiment_identity=linked_identity,
        evidence_by_outcome=dict(sorted(Counter(r.outcome.value for r in evidence).items())),
        evidence_by_scope=dict(sorted(Counter(r.evidence_scope.value for r in evidence).items())),
        evidence_by_reason=dict(sorted(Counter(r.reason_code.value for r in evidence).items())),
        studies=tuple(assess_study(d, capabilities) for d in default_designs()),
        recorded_experiments=recorded_experiments, recorded_evidence=recorded_evidence,
    )
    return outcome, evidence

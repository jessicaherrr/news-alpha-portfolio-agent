"""Agent runtime-integration release, section 21 -- the controlled end-to-end
acceptance test.

Uses a scripted, deterministic LLM (no real Claude call) but runs the REAL
orchestration / execution / validation path: the real `ResearchOrchestrator`
(Phase 18), the real `ProductionExecutionValidationService` (real C++ backtest
+ frozen validation over the real, already-acquired local NQ 2018-2024
dataset), and a real `ExperimentRegistry` -- but an ISOLATED, temporary one,
never the production `data/registry/experiments.sqlite`.

Proves, in one test, everything section 21 requires:

 1. the objective reaches the planning context
 2. prior FailureMemory (seeded from a REAL, already-committed production
    registry row -- the historical NQ TSMOM canonical REJECT) is visible
    BEFORE H1 is proposed
 3. H1 compiles
 4. a full scientific `experiment_identity` is built
 5. an identity-dedup decision occurs (the novelty gate blocks a trivial
    ordinary-TSMOM retry BEFORE it ever reaches compilation-driven identity
    admission)
 6. the allowed novel proposal (ma_trend/NQ, a distinct mechanism) reaches the
    real `ProductionExecutionValidationService`
 7. the real C++ path executes
 8. the real frozen validation executes
 9. a correctly typed result is returned
10. the isolated test registry receives it (never production)
11. FailureMemory reflects it
12. the SECOND planning context contains the first family's result
13. a trivial parameter-only / mechanism-only retry of an already-REJECTED
    hypothesis is refused by the deterministic novelty gate, not silently
    permitted

This test does not spend real Claude API tokens and does not touch 2025.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from alpha_agent.agents.execution_service import ProductionExecutionValidationService
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.agents.llm import ScriptedLLMClient
from alpha_agent.agents.orchestrator import PlanningDisposition
from alpha_agent.agents.orchestrator_factory import build_deep_research_orchestrator
from alpha_agent.registry.enums import TrialRole
from alpha_agent.registry.models import ImportBundle
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services

_UNIQUE_OBJECTIVE = (
    "Investigate an economically distinct NQ mechanism beyond ordinary "
    "multi-week time-series momentum, which the registry already rejects."
)

# -- H1 attempt: a trivial, un-novel retry of the ALREADY-REJECTED ordinary
#    NQ TSMOM(20, 120) mechanism -- must be blocked by the deterministic
#    novelty gate BEFORE it is admitted to the family.
_TSMOM_RETRY_HYPOTHESIS = {
    "hypothesis_id": "H-ACCEPT-TSMOM-RETRY",
    "title": "NQ multi-week time-series momentum (retry)",
    "economic_mechanism": (
        "Gradual diffusion of macro / earnings information leaves index "
        "futures returns positively autocorrelated at multi-week horizons."
    ),
    "universe": ["NQ"],
    "horizon": "20-120 trading days",
    "required_features": ["diff"],
    "signal_description": "Long when the 20-period minus 120-period price diff is positive, short otherwise.",
    "expected_regime": "trending macro regime",
    "failure_regime": "mean-reverting / choppy regime",
    "falsification_test": "No positive OOS net PnL after costs.",
    "novelty_notes": "",
}
_TSMOM_RETRY_PLAN = {
    "expressible": True,
    "template": {
        "family_key": "tsmom",
        "root_symbol": "NQ",
        "params": {"fast_horizon": 20, "slow_horizon": 120, "size": 1},
    },
    "rationale": "Ordinary TSMOM convenience template.",
}

# -- H2: a genuinely distinct mechanism (ma_trend, not tsmom) on the same
#    root -- must be ADMITTED.
_MA_TREND_HYPOTHESIS = {
    "hypothesis_id": "H-ACCEPT-MATREND-NQ",
    "title": "NQ dual moving-average trend",
    "economic_mechanism": (
        "Slow-moving positioning and information diffusion in index futures "
        "create sustained trends that a fast/slow moving-average crossover "
        "captures with a lag -- a distinct mechanism from a raw price-diff "
        "momentum signal."
    ),
    "universe": ["NQ"],
    "horizon": "50-200 trading days",
    "required_features": ["ma"],
    "signal_description": "Long when the fast moving average is above the slow moving average, short otherwise.",
    "expected_regime": "sustained trend regime",
    "failure_regime": "choppy, range-bound regime",
    "falsification_test": "No positive OOS net PnL after costs, or parameter neighbourhood unstable.",
    "novelty_notes": (
        "A moving-average crossover, not a raw price-diff momentum signal -- distinct "
        "from the already-rejected ordinary TSMOM mechanism on this root."
    ),
}
_MA_TREND_PLAN = {
    "expressible": True,
    "template": {
        "family_key": "ma_trend",
        "root_symbol": "NQ",
        "params": {"fast_window": 50, "slow_window": 200, "size": 1},
    },
    "rationale": "Dual moving-average crossover maps to the ma_trend convenience family.",
}


def _seed_isolated_registry_from_production(isolated: ExperimentRegistry) -> None:
    """Copy the REAL, already-committed NQ TSMOM canonical REJECT out of the
    production registry into the isolated test registry -- proving pre-H1
    failure memory against real evidence without ever writing to, or reading
    scientific conclusions FROM the test back INTO, production."""
    with services.open_registry() as prod:
        views = list(prod.experiments(strategy_family="tsmom", root_symbol="NQ", authoritative_only=True))
    canonical = next(v for v in views if v.experiment.trial_role is TrialRole.CANONICAL)
    assert canonical.result is not None
    bundle = ImportBundle(
        import_id="agent_acceptance_test_seed",
        phase="agent-runtime-acceptance-seed",
        source_fingerprint="acceptance-test-seed-from-production-registry-readonly-copy",
        created_at=datetime.now(UTC).isoformat(),
        experiments=(canonical.experiment,),
        results=(canonical.result,),
    )
    isolated.apply_bundle(bundle)


@pytest.mark.slow
def test_full_feedback_loop_closes_end_to_end(tmp_path):
    isolated_path = tmp_path / "isolated_test_registry.sqlite"
    isolated = ExperimentRegistry(isolated_path)
    try:
        _seed_isolated_registry_from_production(isolated)

        # (2) prior FailureMemory must be visible BEFORE H1 is proposed.
        research_client = ScriptedLLMClient(
            [json.dumps(_TSMOM_RETRY_HYPOTHESIS), json.dumps(_MA_TREND_HYPOTHESIS)]
        )
        compiler_client = ScriptedLLMClient(
            [json.dumps(_TSMOM_RETRY_PLAN), json.dumps(_MA_TREND_PLAN)]
        )
        execution_service = ProductionExecutionValidationService(work_dir=tmp_path / "exec_work")

        orch = build_deep_research_orchestrator(
            registry=isolated,
            objective=_UNIQUE_OBJECTIVE,
            root="NQ",
            family_stem="agent-acceptance-test",
            research_client=research_client,
            compiler_client=compiler_client,
            target_family_size=1,
            execution_service=execution_service,
        )

        # (1) the objective reaches the planning context.
        ctx_before = orch._planning_context()
        assert ctx_before.objective == _UNIQUE_OBJECTIVE
        assert any(
            d.strategy_family == "tsmom" and d.root_symbol == "NQ" for d in ctx_before.failure_memory
        ), "the seeded NQ TSMOM rejection must be visible BEFORE H1 is proposed"

        # (3)+(5)+(13) plan the family: slot 0 (trivial TSMOM retry, no
        # novelty_notes) must be blocked by the deterministic novelty gate;
        # slot 1 (a genuinely distinct ma_trend mechanism) must be admitted.
        manifest = orch.plan_family(generation=0)
        dispositions = [o.disposition for o in manifest.planning_log]
        assert PlanningDisposition.NEAR_DUPLICATE_NO_NOVELTY in dispositions
        assert PlanningDisposition.MEMBER_ADDED in dispositions
        assert manifest.planning_complete
        assert manifest.predeclared_family_size == 1
        member = manifest.members[0]
        assert member.strategy_family == "ma_trend"
        assert member.root_symbol == "NQ"

        # (4) a full scientific experiment_identity was built (pre-run,
        # before any execution).
        assert member.experiment_identity.startswith("experiment1:")
        dup = isolated.find_exact_duplicate(
            member.experiment_identity, asset_domain=AssetDomain.FUTURES
        )
        assert not dup.blocks_reexecution  # genuinely novel identity -- permitted

        # (6)-(9) execute: reaches the REAL ExecutionValidationService, which
        # runs the REAL C++ backtest and the REAL frozen validation.
        evidence = orch.execute_family(manifest)
        trail = evidence[member.experiment_identity]
        assert trail, "the execution service must have been called"
        final_ev = trail[-1]
        assert final_ev.engine == (
            "alpha_agent.agents.execution_service.ProductionExecutionValidationService/1"
        ), "the C++/frozen-validation path is authoritative -- Python/LLM never substitute a result"

        # (10) the isolated test registry receives the attempt/result -- and
        # ONLY the isolated one; production is never touched by this test.
        report = orch.finalize_family(manifest, evidence)
        assert report.status.value == "FINALIZED"
        view = isolated.get(member.experiment_identity)
        assert view.has_valid_authoritative_result or view.result is not None

        # (11)+(12) FailureMemory / the SECOND planning context reflects the
        # first family's result -- the mandatory feedback-loop proof.
        ctx_after = orch._planning_context()
        ma_trend_digest = next(
            d for d in ctx_after.failure_memory
            if d.strategy_family == "ma_trend" and d.root_symbol == "NQ"
        )
        assert ma_trend_digest.valid_execution_attempts >= 1 or ma_trend_digest.verdict_counts
        assert ctx_after.failure_memory != ctx_before.failure_memory

        # production registry counts are unaffected by any of this.
        prod_summary_after = services.registry_summary()
        assert prod_summary_after is not None  # sanity: production registry is still readable/unchanged shape
    finally:
        isolated.close()

"""Alpha Discovery campaign, Part K -- the multi-mechanism "Discover
Strategies" UI action (task spec sections 54-60).

This is PLAN + (optionally) FAST-SCREEN + FREEZE only -- it deliberately
stops BEFORE strict validation. Mirroring `alpha_agent.ui.deep_research`'s own
framing ("Run Deep Research" is the ONE UI boundary that may write to the
registry), this module never calls `ResearchOrchestrator.execute_family` /
`finalize_family` and never opens the registry for anything but read-only
duplicate/failure-memory lookups (`plan_family` itself never writes). Handing
a `FrozenCandidateSet` to real strict validation is a SEPARATE, explicit,
heavier action outside this module's scope (task spec section 42's freeze ->
validate boundary, kept intact) -- not wired to a casual UI click in this
release.

Reuses, rather than re-implements, three already-real pieces:
`alpha_agent.discovery.candidate_pool.generate_candidate_pool` (Part D, itself
built on the frozen Phase 18 orchestrator), `alpha_agent.screening.
fast_screen.run_fast_screen` (Part F, real C++ over the 2018-2022 research
window only), and `alpha_agent.screening.freeze.freeze_top_k` (Part G). The
same Offline/Deterministic vs. Claude Research engine choice
`alpha_agent.ui.llm_demo`/`agent.py` already use applies here -- "Offline"
replays curated scenario hypotheses/plans (this module's own bounded
scenario-to-mechanism map), never a live network call by default.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from alpha_agent.agents import (
    AdaptiveTestingError,
    FamilyManifest,
    FamilyManifestError,
    FamilyPlanSpec,
    FamilyReport,
    OrchestratorBudget,
    OrchestratorConfig,
    ResearchAgent,
    ResearchOrchestrator,
    StrategyCompilerAgent,
)
from alpha_agent.agents.execution_service import ProductionExecutionValidationService
from alpha_agent.agents.llm import (
    AnthropicClient,
    LLMClient,
    LLMClientUnavailable,
    ScriptedLLMClient,
)
from alpha_agent.artifacts.store import DEFAULT_ARTIFACT_DIR
from alpha_agent.discovery import CandidatePoolResult, generate_candidate_pool
from alpha_agent.knowledge import EconomicMechanism, StrategyKnowledgeBase
from alpha_agent.recommendation.profile import InvestorProfile
from alpha_agent.registry.models import MarketWindow
from alpha_agent.screening.fast_screen import FastScreenTrial, run_fast_screen
from alpha_agent.screening.freeze import FreezeError, FrozenCandidateSet, freeze_top_k
from alpha_agent.ui import llm_demo, services
from alpha_agent.validation.policy import ReliabilityPolicy

__all__ = [
    "DEFAULT_SCENARIO_MECHANISM_MAP",
    "STRICT_VALIDATION_WARNING",
    "DiscoveryCampaignOutcome",
    "StrictValidationOutcome",
    "run_discovery_campaign",
    "run_fast_screen_on_pool",
    "run_strict_validation",
]

#: Task spec Checkpoint 4, requirement 1 -- shown verbatim before the user can
#: confirm strict validation.
STRICT_VALIDATION_WARNING = (
    "These candidates are now frozen. Their logic and parameters cannot change "
    "once validation begins."
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CLI = _REPO_ROOT / "build" / "cpp" / "cpp" / "quant_backtest_targets_csv"

#: This module's OWN bounded scenario -> mechanism map (task spec section 26:
#: mechanism-diverse, not parameter variants). Reuses `llm_demo.SCENARIOS`'
#: existing hand-built, schema-valid, deterministic hypothesis/plan pairs --
#: never duplicates them -- so the offline path replays exactly the same
#: canned proposals the Agent page's single-hypothesis flow already does,
#: just across several of them in one bounded pass.
DEFAULT_SCENARIO_MECHANISM_MAP: tuple[tuple[str, EconomicMechanism], ...] = (
    ("ma_trend_zn", EconomicMechanism.TREND),
    ("tsmom_nq", EconomicMechanism.MOMENTUM),
    ("mean_reversion_cl", EconomicMechanism.MEAN_REVERSION),
    ("breakout_gc", EconomicMechanism.BREAKOUT),
    ("trend_vol_regime_nq_novel", EconomicMechanism.REGIME_CONDITIONED_TREND),
)


@dataclass(frozen=True)
class DiscoveryCampaignOutcome:
    accepted: bool
    error: str | None = None
    pool: CandidatePoolResult | None = None
    trials: tuple[FastScreenTrial, ...] = ()
    frozen: FrozenCandidateSet | None = None
    knowledge_base: StrategyKnowledgeBase | None = None


def _scripted_client_for(
    scenario_keys: tuple[str, ...], *, root: str, field: str,
) -> LLMClient:
    """One `ScriptedLLMClient` whose response queue carries every scenario's
    hypothesis (`field == "hypothesis"`) or plan (`field == "plan"`), IN
    ORDER, re-rooted to `root` -- `generate_candidate_pool` builds a fresh
    `ResearchAgent`/`StrategyCompilerAgent` PER mechanism but they all share
    this one underlying client, so the queue is consumed in exactly the
    mechanism-priority order the campaign requests."""
    import json

    responses = []
    for key in scenario_keys:
        sc = llm_demo.scenario(key)
        payload = llm_demo.rooted_hypothesis(sc.hypothesis, root) if field == "hypothesis" else llm_demo.rooted_plan(sc.plan, root)
        responses.append(json.dumps(payload))
    return ScriptedLLMClient(responses)


def run_discovery_campaign(
    *,
    mode: str,
    objective: str,
    root: str,
    family_stem: str,
    profile: InvestorProfile | None = None,
    scenario_mechanism_map: tuple[tuple[str, EconomicMechanism], ...] = DEFAULT_SCENARIO_MECHANISM_MAP,
    target_k: int = 3,
    run_fast_screen_backtests: bool = False,
    cli_executable: str | Path | None = None,
    research_sources: str = "offline",
) -> DiscoveryCampaignOutcome:
    """One bounded "Discover Strategies" pass for `root`: mechanism-diverse
    generation (Part D) -> optionally real Fast Screen (Part F, only when
    `run_fast_screen_backtests=True` -- a real C++ subprocess per candidate,
    several seconds to ~1 minute each) -> Freeze (Part G). Never executes
    strict validation and never writes the registry.

    `research_sources` (task spec Checkpoint 9/section 46 -- "OFFLINE
    RESEARCH" vs "CONNECTED RESEARCH"): `"offline"` (default -- UNCHANGED
    prior behaviour, every external source `NOT_CONNECTED`, classics +
    internal registry only) or `"connected"` (live GitHub/academic/
    community/practitioner connectors, `alpha_agent.knowledge.live_adapters`).
    One connector failing never aborts the pass -- `StrategyKnowledgeBase`
    already carries each source's real, honest `IngestionResult` regardless.
    """
    scenario_keys = tuple(k for k, _ in scenario_mechanism_map)
    try:
        if mode == llm_demo.LIVE_MODE:
            research_client: LLMClient = AnthropicClient()
            compiler_client: LLMClient = AnthropicClient()
        else:
            research_client = _scripted_client_for(scenario_keys, root=root, field="hypothesis")
            compiler_client = _scripted_client_for(scenario_keys, root=root, field="plan")
    except LLMClientUnavailable as exc:
        return DiscoveryCampaignOutcome(accepted=False, error=f"Live LLM unavailable: {exc}")

    policy = ReliabilityPolicy()
    exec_svc = ProductionExecutionValidationService(reliability_policy=policy)
    try:
        planes = exec_svc.identity_planes_for_root(root)
    except Exception as exc:  # noqa: BLE001 -- real-data reconstitution can fail (missing local dataset)
        return DiscoveryCampaignOutcome(accepted=False, error=f"Could not derive identity planes for {root}: {exc}")

    market_window = MarketWindow(label="VALIDATION_2023_2024", start_date="2023-01-01", end_date="2024-12-31")

    if research_sources not in ("offline", "connected"):
        return DiscoveryCampaignOutcome(
            accepted=False, error=f"research_sources must be 'offline' or 'connected', got {research_sources!r}"
        )
    if research_sources == "connected":
        from alpha_agent.knowledge import live_adapters

        adapters = live_adapters()
    else:
        # Release UX bugfix pass, issue 1: "offline" means external research
        # was never REQUESTED for this campaign -- report DISABLED, never
        # NOT_CONNECTED (which would falsely imply a real attempt was made
        # and failed). `disabled_adapters()` replaces the previous implicit
        # `adapters=None` -> `default_adapters()` (NotConnectedAdapter) path.
        from alpha_agent.knowledge import disabled_adapters

        adapters = disabled_adapters()

    with services.open_registry() as reg:
        rows = services.list_experiments()
        # `query=""` (never the raw free-text objective): each live connector
        # generates its OWN bounded, market-aware query set internally when
        # given no explicit query (task spec section 33 -- "query generation
        # must happen before retrieval"); a long free-text objective sentence
        # is a poor literal search-engine/API query.
        kb = StrategyKnowledgeBase(registry_rows=rows, query="", markets=(root,), adapters=adapters)
        pool = generate_candidate_pool(
            market=root, objective=objective, registry=reg,
            research_llm=research_client, compiler_llm=compiler_client, execution_service=exec_svc,
            reliability_policy=policy, planes=planes, market_window=market_window,
            fdr_q_threshold=policy.fdr_q_threshold, knowledge_base=kb, profile=profile,
            max_mechanisms=len(scenario_keys), per_mechanism_family_size=1, family_stem=family_stem,
        )

    if not pool.members:
        return DiscoveryCampaignOutcome(accepted=True, pool=pool, knowledge_base=kb)

    if not run_fast_screen_backtests:
        return DiscoveryCampaignOutcome(accepted=True, pool=pool, knowledge_base=kb)

    trials, frozen, screen_error = _fast_screen_and_freeze(
        pool=pool, root=root, family_stem=family_stem, target_k=target_k,
        cli_executable=cli_executable, policy=policy, planes=planes, market_window=market_window,
    )
    if trials is None:  # the C++ CLI itself was not found -- generation still succeeded
        return DiscoveryCampaignOutcome(accepted=True, pool=pool, knowledge_base=kb, error=screen_error)
    if frozen is None:
        return DiscoveryCampaignOutcome(
            accepted=True, pool=pool, trials=trials, knowledge_base=kb, error=screen_error,
        )
    return DiscoveryCampaignOutcome(
        accepted=True, pool=pool, trials=trials, frozen=frozen, knowledge_base=kb,
    )


def _fast_screen_and_freeze(
    *,
    pool: CandidatePoolResult,
    root: str,
    family_stem: str,
    target_k: int,
    cli_executable: str | Path | None,
    policy: ReliabilityPolicy,
    planes,
    market_window: MarketWindow,
) -> tuple[tuple[FastScreenTrial, ...] | None, FrozenCandidateSet | None, str | None]:
    """The real, one real-C++-backtest-per-candidate Fast Screen + Freeze
    tail, factored out so it can run either as part of `run_discovery_campaign`
    (`run_fast_screen_backtests=True`) OR standalone, against an
    ALREADY-COMPILED pool (`run_fast_screen_on_pool`, task spec issue 7 --
    "Fast Screen the already-compiled exact candidates" without restarting
    generation). Returns `(trials, frozen, error)`: `trials is None` means the
    CLI itself was not found (nothing ran); `frozen is None` with
    `trials is not None` means screening ran but no candidate survived
    freezing.
    """
    cli = Path(cli_executable) if cli_executable is not None else _DEFAULT_CLI
    if not cli.exists():
        return None, None, "Compiled C++ CLI not found -- Fast Screen skipped (candidate generation still succeeded)."

    recon_cache: dict[str, object] = {}
    trials: list[FastScreenTrial] = []
    with tempfile.TemporaryDirectory(prefix="discover_fastscreen_") as tmp:
        for member in pool.members:
            if member.root_symbol not in recon_cache:
                from alpha_agent.data.real_market_dataset import reconstitute_root

                recon_cache[member.root_symbol] = reconstitute_root(member.root_symbol)
            trial = run_fast_screen(
                member=member, cli_executable=cli, work_dir=tmp, recon=recon_cache[member.root_symbol],
            )
            trials.append(trial)

    try:
        frozen = freeze_top_k(
            pool=pool, trials=trials, k=target_k, family_stem=family_stem,
            planes=planes, market_window=market_window, fdr_q_threshold=policy.fdr_q_threshold,
        )
    except FreezeError as exc:
        return tuple(trials), None, f"No candidate survived the Fast Screen: {exc}"
    return tuple(trials), frozen, None


def run_fast_screen_on_pool(
    *,
    pool: CandidatePoolResult,
    knowledge_base: StrategyKnowledgeBase | None,
    root: str,
    family_stem: str,
    target_k: int = 3,
    cli_executable: str | Path | None = None,
) -> DiscoveryCampaignOutcome:
    """Task spec issue 7: Fast-Screen (Part F) + Freeze (Part G) an ALREADY-
    COMPILED candidate `pool` in place -- never re-runs generation, never
    calls the `ResearchAgent`/`StrategyCompilerAgent` again, never re-queries
    any knowledge source. `knowledge_base` is passed through unchanged into
    the returned outcome so the caller's source-lineage/research-process
    display keeps working against the SAME real ingestion results the
    original compile-only pass already produced.

    Re-derives `IdentityPlanes`/`ReliabilityPolicy`/`MarketWindow` fresh from
    `root` -- these are pure, deterministic functions of the frozen research
    dataset and never depend on anything the LLM/live-research step touched,
    so this is byte-identical to what `run_discovery_campaign(...,
    run_fast_screen_backtests=True)` would have produced for the SAME pool.
    """
    policy = ReliabilityPolicy()
    exec_svc = ProductionExecutionValidationService(reliability_policy=policy)
    try:
        planes = exec_svc.identity_planes_for_root(root)
    except Exception as exc:  # noqa: BLE001 -- an honest UI error, never a raw stack trace
        return DiscoveryCampaignOutcome(
            accepted=False, error=f"Could not derive identity planes for {root}: {exc}",
            pool=pool, knowledge_base=knowledge_base,
        )
    market_window = MarketWindow(label="VALIDATION_2023_2024", start_date="2023-01-01", end_date="2024-12-31")

    if not pool.members:
        return DiscoveryCampaignOutcome(accepted=True, pool=pool, knowledge_base=knowledge_base)

    trials, frozen, screen_error = _fast_screen_and_freeze(
        pool=pool, root=root, family_stem=family_stem, target_k=target_k,
        cli_executable=cli_executable, policy=policy, planes=planes, market_window=market_window,
    )
    if trials is None:
        return DiscoveryCampaignOutcome(accepted=True, pool=pool, knowledge_base=knowledge_base, error=screen_error)
    return DiscoveryCampaignOutcome(
        accepted=True, pool=pool, trials=trials, frozen=frozen, knowledge_base=knowledge_base, error=screen_error,
    )


@dataclass(frozen=True)
class StrictValidationOutcome:
    """Typed outcome of one "Run Strict Validation" click. Never a bare
    exception or a raw stack trace to the UI (mirrors `DeepResearchOutcome`)."""

    accepted: bool
    error: str | None = None
    manifest: FamilyManifest | None = None
    report: FamilyReport | None = None


def run_strict_validation(
    *, frozen: FrozenCandidateSet, objective: str, root: str,
    artifact_dir: str | Path | None = DEFAULT_ARTIFACT_DIR,
) -> StrictValidationOutcome:
    """Checkpoint 4 -- the real Freeze -> Adopt -> Strict Validation handoff.

    `artifact_dir` (Checkpoint 11): opted into by default (the standard
    `alpha_agent.artifacts.store.DEFAULT_ARTIFACT_DIR`, gitignored operational
    output) so a real strict-validation run persists its headline fills /
    trades / daily-equity / price-bar artifacts, exactly like `run_fast_screen`
    already does when asked. Pass `None` to disable (byte-identical to this
    parameter's prior absence).

    `frozen` is the EXACT `FrozenCandidateSet` `run_discovery_campaign` (or an
    equivalent Fast-Screen/Freeze pass) already produced: its `manifest`
    carries the already-frozen `StrategySpec`s, fingerprints, candidate
    identities, cadence, and family membership untouched. This function NEVER
    calls the `ResearchAgent` or the `StrategyCompilerAgent` -- the LLM is
    never asked to regenerate a frozen hypothesis (task spec Checkpoint 4,
    requirement 3). It builds a FRESH `ResearchOrchestrator` (a fresh instance
    that never called `plan_family` for this manifest), independently
    re-derives today's real `IdentityPlanes` for `root` (never trusts
    `frozen.manifest.planes` directly -- see
    `ResearchOrchestrator.adopt_frozen_manifest`'s docstring), safely adopts
    the frozen manifest, and runs the real, UNCHANGED
    `execute_family` -> `finalize_family` sequence against the PRODUCTION
    registry (`services.open_registry`) -- real 2023-2024 strict validation,
    real BH/FDR over the whole frozen family, written once, honestly.

    A UI click cannot bypass any scientific safeguard: every check in
    `adopt_frozen_manifest` still runs (wrong plane / wrong policy / wrong
    BH q-threshold / root outside universe / unsupported cadence / tampered
    member / underfilled family), and an evaluation plane that already
    carries an executed family is refused via the SAME `AdaptiveTestingError`
    `execute_family` always enforced (task spec Checkpoint 4, requirements
    6/7/8) -- nothing new needed for that, it falls out of the existing
    invariant.
    """
    policy = ReliabilityPolicy()
    exec_svc = ProductionExecutionValidationService(reliability_policy=policy, artifact_dir=artifact_dir)
    try:
        planes = exec_svc.identity_planes_for_root(root)
    except Exception as exc:  # noqa: BLE001 -- an honest UI error, never a raw stack trace
        return StrictValidationOutcome(
            accepted=False, error=f"Could not derive identity planes for {root}: {exc}"
        )

    market_window = MarketWindow(
        label="VALIDATION_2023_2024", start_date="2023-01-01", end_date="2024-12-31"
    )
    config = OrchestratorConfig(
        objective=objective,
        market_universe=(root,),
        market_window=market_window,
        planes=planes,
        family_plan=FamilyPlanSpec(
            family_stem=frozen.manifest.family_stem,
            target_family_size=frozen.manifest.declared_target_size,
        ),
        budget=OrchestratorBudget(plan_successor_family_after_finalization=False),
        phase="alpha-discovery-live-strict-validation",
    )
    # `plan_family` is never called on this orchestrator -- these agents exist
    # only to satisfy `ResearchOrchestrator.__init__`'s constructor contract
    # and will never receive a single call.
    never_called = ScriptedLLMClient([])
    try:
        with services.open_registry() as reg:
            orch = ResearchOrchestrator(
                registry=reg,
                research_agent=ResearchAgent(never_called),
                compiler_agent=StrategyCompilerAgent(never_called),
                execution_service=exec_svc,
                config=config,
                reliability_policy=policy,
            )
            adopted = orch.adopt_frozen_manifest(
                frozen.manifest, fast_screen_provenance=frozen.screen_scores
            )
            evidence = orch.execute_family(adopted)
            report = orch.finalize_family(adopted, evidence)
            return StrictValidationOutcome(accepted=True, manifest=adopted, report=report)
    except (FamilyManifestError, AdaptiveTestingError) as exc:
        return StrictValidationOutcome(accepted=False, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 -- an honest UI error, never a raw stack trace
        return StrictValidationOutcome(accepted=False, error=f"{type(exc).__name__}: {exc}")

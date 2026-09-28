"""Production ``ExecutionValidationService`` (Agent runtime-integration
release, sections 6/7).

``ResearchOrchestrator`` (Phase 18) already depends on the
``ExecutionValidationService`` protocol; until now the only implementation was
``ScriptedExecutionValidationService``, a test double. This module composes
ONLY existing, frozen, real implementations -- it reimplements no PnL, no
fill, no cost, no risk, and no validation math:

    StrategySpec (already compiled by StrategyCompilerAgent)
        -> the already-acquired real local 2018-2024 dataset
           (`alpha_agent.data.real_market_dataset.reconstitute_root` -- no
           network, no Databento call, ever)
        -> feature / target-schedule preparation, reusing the EXACT extracted
           daily / native-1m schedule builders `phase_13_5c_matrix.py` already
           uses for the committed 107-hypothesis matrix
           (`daily_baseline_schedule` / `native_1m_schedule`, byte-identical
           to the private methods they replaced -- proven by the unmodified
           Phase 13.5C test suite)
        -> the frozen `alpha_agent.validation.runner.CliBacktestRunner` ->
           `quant_backtest_targets_csv` real C++ backtest path (roll-aware,
           via the same `RollAwareCliRunner` the matrix uses)
        -> the frozen `alpha_agent.validation.engine.ValidationEngine` /
           `assemble_report` / `evaluate_policy` implementation
        -> a typed `TrialEvidence`

SCOPE -- the smallest deterministic, architecture-consistent slice for this
release (CLAUDE.md sections 12 / 25):

* ONE root per orchestrator run. `DatasetIdentity` / `ValidationSpec` in this
  codebase are inherently per-root (confirmed against the live registry: the
  same root's rows share one `dataset_fingerprint`, different roots do not),
  and Phase 18.2's "one evaluation plane, one outcome-adaptive campaign"
  concept already assumes one frozen dataset/split/validation/policy plane
  per orchestrator config.
* The five already-frozen Phase 11 baseline / Silver-Bullet families
  (tsmom / ma_trend / breakout / mean_reversion / silver_bullet) -- exactly
  what `StrategyCompilerAgent`'s TEMPLATE build mode produces and what the
  existing 107-hypothesis matrix already validated end to end on real data. A
  full custom BLUEPRINT family the compiler did not template is a genuine,
  honestly reported capability gap: `INVALID_EXECUTION` / `SOFTWARE_DEFECT`,
  never a scientific verdict. Extending this to arbitrary blueprints is
  future work, not silently faked here.
* No parameter neighbourhood / ablations for a fresh hypothesis (there is no
  predeclared neighbourhood for a brand-new proposal in this release) -- the
  per-trial validation runs the CANONICAL spec only. The frozen
  `ReliabilityPolicy`'s parameter-stability / ablation gates simply are not
  evaluated (`None` inputs to `evaluate_policy`, exactly as the frozen engine
  already handles an absent neighbourhood/ablation set) -- never approximated,
  never fabricated.
* No roll-close-marks OPTIMISATION beyond what the matrix itself used for the
  same root/window: this service reuses the identical `RollAwareCliRunner` +
  auxiliary same-timestamp roll marks the 107-hypothesis matrix used, so a
  replay of an already-committed experiment reproduces the same numbers
  (CLAUDE.md's "Auxiliary roll-close marks" section: empty marks = the frozen
  `RejectDefer` reference path; this service supplies the SAME real marks the
  matrix did, not a different default).

The locked 2025 holdout is never reachable from this service: `run()` refuses
`holdout=True` outright (the iteration loop never sets it; this is a second,
independent guard), and every dataset load is bounded to
`< HOLDOUT_START_NS` before it ever reaches a DataFrame this module touches.
"""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

from alpha_agent.agents.orchestrator import (
    FamilyManifest,
    FamilyMember,
    HoldoutIsolationError,
    NonFamilyVerdict,
    TrialEvidence,
)
from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.data.real_market_dataset import (
    HOLDOUT_START_NS,
    RESEARCH_WINDOW,
    ReconstitutedRoot,
    _iso_ns,
    daily_signal_series,
    reconstitute_root,
    roll_close_marks,
    write_roll_close_marks_csv,
)
from alpha_agent.execution.capability import assess_static_capability
from alpha_agent.registry.enums import AttemptStatus, InvalidationClass
from alpha_agent.strategy.candidates_phase_13_5c import SIGNAL_CADENCE
from alpha_agent.strategy.spec import StrategySpec
from alpha_agent.validation.engine import ValidationEngine
from alpha_agent.validation.enums import SplitRole, Verdict
from alpha_agent.validation.phase_13_5c_matrix import (
    CAPITAL_BASE_USD,
    RollAwareCliRunner,
    build_split_plan,
    daily_baseline_schedule,
    dataset_identity_for_root,
    frozen_bootstrap_config,
    frozen_cost_plan,
    frozen_null_config,
    frozen_walk_forward,
    native_1m_schedule,
)
from alpha_agent.validation.policy import ReliabilityPolicy, evaluate_policy
from alpha_agent.validation.report import ValidationReport
from alpha_agent.validation.runner import CliBacktestRunner
from alpha_agent.validation.spec import ValidationSpec, validation_protocol_fingerprint

__all__ = [
    "SUPPORTED_FAMILIES",
    "ExecutionServiceError",
    "ProductionExecutionValidationService",
    "UnsupportedFamily",
]

#: Alpha Discovery Part E hardening: execution is now gated by CAPABILITY
#: (signal cadence + registered features -- `alpha_agent.execution.
#: capability`), not by a family-NAME allowlist. `SIGNAL_CADENCE` (the same
#: frozen mapping the Phase 13.5C matrix and the compiler agent's TEMPLATE
#: mode already use) is kept only as the FALLBACK cadence source for a
#: `FamilyMember` whose `signal_cadence` field was never populated (e.g. a
#: hand-built `FamilyMember` from an older call site / test) -- see
#: `_resolve_cadence`. A member built by the real `ResearchOrchestrator`
#: today always has `signal_cadence` populated already, so for the five
#: legacy families this changes nothing observable: same cadence value, same
#: outcome, exact byte-identical replay (proven by
#: `tests/python/test_execution_service.py::
#: test_replay_matches_committed_nq_tsmom_canonical_result`).
#:
#: Public, read-only capability query (which family NAMES are known to map to
#: a supported cadence today) -- never a second source of truth for the gate
#: itself, which always re-derives the ACTUAL cadence per member.
SUPPORTED_FAMILIES: frozenset[str] = frozenset(SIGNAL_CADENCE)


def _resolve_cadence(member: FamilyMember) -> str:
    """Thin wrapper over the shared `alpha_agent.execution.capability.
    resolve_cadence` -- kept as a module-level function here (rather than
    inlined at the one call site) only because it is already covered by name
    in `tests/python/test_alpha_discovery_part_e_capability.py`."""
    from alpha_agent.execution.capability import resolve_cadence

    return resolve_cadence(signal_cadence=member.signal_cadence, strategy_family=member.strategy_family)

_DEFAULT_CLI_RELATIVE = "build/cpp/cpp/quant_backtest_targets_csv"
_ENGINE_ID = "alpha_agent.agents.execution_service.ProductionExecutionValidationService/1"


class ExecutionServiceError(RuntimeError):
    """Base class for this module's own errors. Every one of them is caught by
    `ProductionExecutionValidationService.run` and turned into a typed
    `INVALID_EXECUTION` `TrialEvidence` -- it never escapes as an uncaught
    exception and it never becomes a scientific verdict."""


class UnsupportedFamily(ExecutionServiceError):
    """The proposed strategy's family is not one this release's execution
    service can run. A genuine capability gap, not a scientific outcome."""


class ProtocolMismatch(ExecutionServiceError):
    """The manifest's declared validation-protocol fingerprint does not match
    what this service is actually configured to run. Refuse rather than
    silently execute under a different protocol than the one the frozen
    family was planned against."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _CompiledSpecAdapter:
    """A `StrategyFamilyAdapter` (see `validation.engine`) for ONE already-
    compiled `StrategySpec` -- unlike `_Phase135cAdapter` (which looks a
    family's typed factory up by name to build a spec per candidate
    parameter set), this adapter already HAS the exact compiled spec the
    `StrategyCompilerAgent` produced and simply returns it; there is no
    parameter neighbourhood in this release, so `params` is never consulted.
    """

    def __init__(
        self,
        spec: StrategySpec,
        canonical_params: dict,
        *,
        cadence: str,
        signal_1m,
        calendar: SessionCalendar,
        root: str,
    ):
        self.strategy_key = spec.strategy_id
        self.canonical_params = dict(canonical_params)
        self._spec = spec
        self._cadence = cadence
        self._root = root
        from alpha_agent.schemas.market_data import PriceDomain

        self._price_domain = PriceDomain
        if cadence == "native_1m":
            tday, sess = calendar.classify_series(signal_1m["ts_event_ns"], root)
            self._sig = signal_1m.assign(
                trading_day=tday.astype(str), session=sess.astype(str)
            ).reset_index(drop=True)
        else:
            self._daily = daily_signal_series(signal_1m, root, calendar=calendar)

    def spec_for(self, params: dict) -> StrategySpec:
        return self._spec

    def schedule_for(
        self, spec: StrategySpec, bars, *, emit_from_ts_ns: int
    ) -> TargetSchedule | None:
        lo = int(bars["ts_event_ns"].iloc[0])
        hi = int(bars["ts_event_ns"].iloc[-1])
        if self._cadence == "native_1m":
            return native_1m_schedule(
                spec, self._sig, lo, hi, emit_from_ts_ns,
                root=self._root, price_domain=self._price_domain,
            )
        return daily_baseline_schedule(
            spec, self._daily, lo, hi, emit_from_ts_ns,
            root=self._root, price_domain=self._price_domain,
        )


def _non_family_outcome(
    report: ValidationReport, policy: ReliabilityPolicy
) -> tuple[NonFamilyVerdict, tuple[str, ...], dict]:
    """The frozen `ReliabilityPolicy` re-applied EXACTLY the way the already-
    frozen `phase_13_5c_matrix.global_verdict` re-applies it (a substituted
    `fdr_result`, everything else from the per-trial engine report unchanged)
    -- except here the FDR/BH gate is force-satisfied rather than substituted
    with a different family, because `ResearchOrchestrator` computes its OWN
    cross-hypothesis family BH separately over every member's raw
    `trial_p_value`. This function answers ONLY "did every OTHER frozen gate
    pass" -- it is not, and must never become, the final verdict."""
    idx = report.canonical_trial_index
    forced_decisions = tuple(
        d.model_copy(update={"rejected": True}) if i == idx else d
        for i, d in enumerate(report.fdr_result.decisions)
    )
    forced_fdr = report.fdr_result.model_copy(update={"decisions": forced_decisions})
    outcome = evaluate_policy(
        policy,
        oos_metrics=report.oos_metrics,
        walk_forward=report.walk_forward,
        null_results=list(report.null_results),
        fdr_result=forced_fdr,
        canonical_trial_index=idx,
        dsr_result=report.dsr_result,
        cost_report=report.cost_stress,
        parameter_stability=report.parameter_stability,
        ablation_report=report.ablation_report,
        regime_result=report.regime_result,
        cross_market=report.cross_market,
        # Validation-safety fix: unlike the frozen Phase 13.5C matrix (whose
        # canonical trials always carry a real predeclared neighbourhood) and
        # unlike the Phase 15 ML adjudication path (which never has a
        # per-hyperparameter-point-economics concept at all, by design), a
        # runtime-agent single-strategy proposal genuinely SHOULD have one --
        # its absence here is missing required evidence, not evidence that
        # legitimately does not apply. See `evaluate_policy`'s docstring.
        parameter_stability_required=True,
    )
    reasons = tuple(c.value for c in outcome.reason_codes)
    om = report.oos_metrics
    trial_p = float(report.fdr_result.decisions[idx].p_value)
    metrics = {
        "gross_pnl_usd": om.oos_gross_pnl_usd,
        "costs_usd": om.oos_costs_usd,
        "net_pnl_usd": om.oos_net_pnl_usd,
        "daily_sharpe": om.daily_sharpe,
        "annualized_sharpe": om.annualized_sharpe,
        "gating_null_p": trial_p,
        "dsr_probability": (
            report.dsr_result.deflated_sharpe_ratio if report.dsr_result.is_valid else None
        ),
        "fold_consistency": report.walk_forward.fold_consistency,
        "n_trades": om.n_trades,
        "n_fills": om.n_fills,
        "n_oos_days": om.n_trading_days,
    }
    if outcome.verdict is Verdict.PASS:
        return NonFamilyVerdict.ELIGIBLE, (), metrics
    if outcome.verdict is Verdict.REJECT:
        return NonFamilyVerdict.REJECT, reasons, metrics
    return NonFamilyVerdict.INCONCLUSIVE, reasons, metrics


class ProductionExecutionValidationService:
    """The real `ExecutionValidationService` `ResearchOrchestrator` (Phase 18)
    was always designed to be handed. See module docstring for exact scope.
    """

    def __init__(
        self,
        *,
        reliability_policy: ReliabilityPolicy | None = None,
        cli_executable: str | Path | None = None,
        work_dir: str | Path | None = None,
        repo_root: str | Path | None = None,
        artifact_dir: str | Path | None = None,
    ):
        self._policy = reliability_policy or ReliabilityPolicy()
        root = Path(repo_root) if repo_root is not None else _repo_root()
        self._cli = Path(cli_executable) if cli_executable is not None else root / _DEFAULT_CLI_RELATIVE
        self._work_dir = (
            Path(work_dir) if work_dir is not None
            else Path(tempfile.mkdtemp(prefix="agent_execution_service_"))
        )
        self._calendar = default_calendar()
        self._recon_cache: dict[str, ReconstitutedRoot] = {}
        # Alpha Discovery Part I: opt-in real artifact persistence. `None` (the
        # default -- every existing caller) preserves the exact prior
        # behaviour byte-for-byte: no trades/fills export requested from the
        # CLI, `TrialEvidence.source_artifact(_sha256)` stay `None`, exactly
        # as before this parameter existed.
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    # -- ExecutionValidationService protocol -------------------------------
    def run(
        self,
        *,
        member: FamilyMember,
        manifest: FamilyManifest,
        attempt_ordinal: int,
        holdout: bool,
    ) -> TrialEvidence:
        if holdout:
            raise HoldoutIsolationError(
                "ProductionExecutionValidationService is the ITERATION service; the "
                "locked 2025 holdout is never touched from here"
            )
        try:
            return self._run_valid(member=member, manifest=manifest, attempt_ordinal=attempt_ordinal)
        except ExecutionServiceError as exc:
            return self._invalid(exc, InvalidationClass.SOFTWARE_DEFECT, member)
        except FileNotFoundError as exc:
            return self._invalid(exc, InvalidationClass.DATA_PIPELINE_DEFECT, member)
        except Exception as exc:  # noqa: BLE001 -- an engineering failure must never
            # become a scientific REJECT (CLAUDE.md registry-write-discipline section);
            # preserved verbatim (with a full traceback) as typed evidence instead of
            # propagating as an uncaught crash of the whole orchestration run.
            return self._invalid(exc, InvalidationClass.SOFTWARE_DEFECT, member)

    def _invalid(self, exc: Exception, cls: InvalidationClass, member: FamilyMember) -> TrialEvidence:
        return TrialEvidence(
            status=AttemptStatus.INVALID_EXECUTION,
            invalidation_class=cls,
            invalidation_detail=f"{type(exc).__name__}: {exc}",
            invalidation_evidence={
                "experiment_identity": member.experiment_identity,
                "traceback": "".join(traceback.format_exception(exc))[-4000:],
            },
            engine=_ENGINE_ID,
        )

    # -- public: identity-plane derivation -----------------------------
    def identity_planes_for_root(self, root: str):
        """The `IdentityPlanes` this service will actually run under for
        `root` -- computed via the EXACT same dataset reconstitution
        (sharing this instance's cache, so building the planes then calling
        `run()` never reconstitutes twice) and the exact same
        `validation_protocol_fingerprint` derivation `_run_valid` asserts
        against. A caller (e.g. `alpha_agent.agents.orchestrator_factory`)
        MUST use this rather than re-deriving the planes independently, or the
        two could silently drift apart."""
        from alpha_agent.agents.orchestrator import IdentityPlanes

        recon = self._reconstitute(root)
        span_lo = _iso_ns(RESEARCH_WINDOW[0])
        span_hi = HOLDOUT_START_NS
        exec_bars = recon.canonical_bars
        exec_bars = exec_bars[
            (exec_bars["ts_event_ns"] >= span_lo) & (exec_bars["ts_event_ns"] < span_hi)
        ][["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]]
        exec_bars = exec_bars.sort_values("ts_event_ns").reset_index(drop=True)

        dataset = dataset_identity_for_root(recon, exec_bars)
        split_plan = build_split_plan()
        policy_fp = self._policy.identity()
        protocol_fp = validation_protocol_fingerprint(
            dataset=dataset, split_plan=split_plan, walk_forward=frozen_walk_forward(),
            cost_stress=frozen_cost_plan(), null_test=frozen_null_config(),
            bootstrap=frozen_bootstrap_config(), minimum_sample=self._policy.minimum_sample,
            reliability_policy_fingerprint=policy_fp,
        )
        from alpha_agent.registry.phase_13_5c_import import config_identities_for_orchestrator

        config_ids = config_identities_for_orchestrator(_repo_root())
        return IdentityPlanes(
            dataset_fingerprint=dataset.identity(),
            split_identity=split_plan.split_fingerprint(),
            validation_spec_fingerprint=protocol_fp,
            reliability_policy_fingerprint=policy_fp,
            execution_config_identity=config_ids["execution_config_identity"],
            cost_config_identity=config_ids["cost_config_identity"],
            risk_identity=config_ids["risk_identity"],
        )

    # -- internals ----------------------------------------------------------
    def _reconstitute(self, root: str) -> ReconstitutedRoot:
        if root not in self._recon_cache:
            self._recon_cache[root] = reconstitute_root(root, calendar=self._calendar)
        return self._recon_cache[root]

    def _run_valid(
        self, *, member: FamilyMember, manifest: FamilyManifest, attempt_ordinal: int
    ) -> TrialEvidence:
        cadence = _resolve_cadence(member)
        required_feature_kinds = [f.spec.kind for f in member.strategy_spec.features]
        capability = assess_static_capability(
            signal_cadence=cadence, required_features=required_feature_kinds
        )
        if not capability.supported:
            raise UnsupportedFamily(
                f"strategy_family {member.strategy_family!r} (signal_cadence={cadence!r}) is not "
                f"executable by this release's ExecutionValidationService: "
                f"{capability.reason.value} -- {capability.detail}. A full custom blueprint whose "
                "compiler build does not yet declare a real, supported signal cadence is a genuine "
                "capability gap, not a scientific outcome -- propose a template hypothesis instead"
            )
        root = member.root_symbol
        recon = self._reconstitute(root)

        span_lo = _iso_ns(RESEARCH_WINDOW[0])
        span_hi = HOLDOUT_START_NS  # strictly < 2025-01-01, mirrors the frozen matrix

        exec_bars = recon.canonical_bars
        exec_bars = exec_bars[
            (exec_bars["ts_event_ns"] >= span_lo) & (exec_bars["ts_event_ns"] < span_hi)
        ][["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]]
        exec_bars = exec_bars.sort_values("ts_event_ns").reset_index(drop=True)

        sig_1m = recon.forward_adjusted
        sig_1m = sig_1m[(sig_1m["ts_event_ns"] >= span_lo) & (sig_1m["ts_event_ns"] < span_hi)]
        sig_1m = sig_1m.sort_values("ts_event_ns").reset_index(drop=True)

        run_dir = self._work_dir / f"{member.experiment_identity.split(':')[-1][:16]}__attempt{attempt_ordinal}"
        run_dir.mkdir(parents=True, exist_ok=True)

        marks = roll_close_marks(recon, span_lo, span_hi)
        marks_path, _marks_sha = write_roll_close_marks_csv(marks, run_dir / "roll_close_marks.csv")
        roll_ts = tuple(
            int(rl.effective_ts_ns) for rl in recon.rolls
            if span_lo <= int(rl.effective_ts_ns) < span_hi
        )

        adapter = _CompiledSpecAdapter(
            member.strategy_spec, member.params, cadence=cadence,
            signal_1m=sig_1m, calendar=self._calendar, root=root,
        )

        # single source of truth for the identity planes -- shares this
        # instance's reconstitution cache, so this never reconstitutes twice.
        planes = self.identity_planes_for_root(root)
        dataset = dataset_identity_for_root(recon, exec_bars)
        split_plan = build_split_plan()
        walk_forward = frozen_walk_forward()
        cost_stress = frozen_cost_plan()
        null_test = frozen_null_config()
        bootstrap = frozen_bootstrap_config()
        minimum_sample = self._policy.minimum_sample
        policy_fp = self._policy.identity()
        protocol_fp = planes.validation_spec_fingerprint

        if protocol_fp != manifest.planes.validation_spec_fingerprint:
            raise ProtocolMismatch(
                f"this service's validation protocol {protocol_fp} does not match the "
                f"frozen manifest's declared {manifest.planes.validation_spec_fingerprint} "
                "-- refusing to execute under a different protocol than the one the "
                "family was planned against"
            )
        if policy_fp != manifest.planes.reliability_policy_fingerprint:
            raise ProtocolMismatch(
                f"this service's ReliabilityPolicy identity {policy_fp} does not match "
                f"the frozen manifest's {manifest.planes.reliability_policy_fingerprint}"
            )

        spec = ValidationSpec(
            label=f"agent__{root}__{member.strategy_family}",
            strategy_fingerprint=member.strategy_fingerprint,
            strategy_key=member.strategy_family,
            dataset=dataset,
            split_plan=split_plan,
            walk_forward=walk_forward,
            capital_base_usd=CAPITAL_BASE_USD,
            cost_stress=cost_stress,
            null_test=null_test,
            bootstrap=bootstrap,
            trial_family_id=f"agent-single-trial:{member.strategy_fingerprint}",
            parameter_neighbourhood=None,
            minimum_sample=minimum_sample,
            reliability_policy_fingerprint=policy_fp,
        )
        runner = RollAwareCliRunner(
            CliBacktestRunner(str(self._cli), run_dir, roll_close_marks_path=marks_path),
            roll_ts,
        )
        # Alpha Discovery live-research campaign, Checkpoint 11: request a
        # real headline trades/fills export ONLY when artifact persistence is
        # opted into -- `ValidationEngine.headline_trades_out_path` /
        # `headline_fills_out_path` are threaded to ONLY its single headline
        # `_run_span` call, never to a fold / null / cost-stress / ablation
        # run, so no internal-only path can ever be persisted as if it were
        # the headline strategy.
        headline_trades_path = run_dir / "headline_trades.csv" if self._artifact_dir is not None else None
        headline_fills_path = run_dir / "headline_fills.csv" if self._artifact_dir is not None else None
        engine = ValidationEngine(
            spec, self._policy, runner, adapter, exec_bars, recon.contracts,
            calendar=self._calendar, oos_split_role=SplitRole.VALIDATION,
            headline_trades_out_path=headline_trades_path, headline_fills_out_path=headline_fills_path,
        )
        report = engine.run_research()

        non_family_verdict, reason_codes, metrics = _non_family_outcome(report, self._policy)
        trial_p = float(report.fdr_result.decisions[report.canonical_trial_index].p_value)

        # Alpha Discovery Part I / live-research campaign Checkpoint 11: bind
        # the REAL, already-computed headline OOS daily-equity trace
        # (`report.oos_daily` -- the frozen ValidationEngine's own official
        # output, read here, never re-derived) AND, now, the real headline
        # fills/trades CSV `ValidationEngine` just wrote (requested ONLY for
        # its single headline run, above) to this experiment identity/attempt
        # for investor visualizations. Opt-in only
        # (`self._artifact_dir is not None`); every existing caller that
        # never sets `artifact_dir` gets byte-identical prior behaviour, and
        # `persist_run_artifacts` already silently omits a fills/trades
        # artifact whose file was never actually written (e.g. a headline
        # window with zero fills).
        source_artifact = None
        source_artifact_sha256 = None
        if self._artifact_dir is not None:
            from alpha_agent.artifacts.store import manifest_path_for, persist_run_artifacts

            # the real executed price bars for exactly the headline OOS
            # window (never the whole multi-year research span) -- bounded
            # by the headline daily trace's own first/last timestamp, plus a
            # one-day buffer so the final trading day's own bars are not
            # truncated.
            headline_price_bars = None
            if report.oos_daily.n_days > 0:
                lo_ts = int(report.oos_daily.ts_ns[0])
                hi_ts = int(report.oos_daily.ts_ns[-1]) + 86_400_000_000_000
                headline_price_bars = exec_bars[
                    (exec_bars["ts_event_ns"] >= lo_ts) & (exec_bars["ts_event_ns"] <= hi_ts)
                ]
            bundle = persist_run_artifacts(
                experiment_identity=member.experiment_identity, attempt_ordinal=attempt_ordinal,
                strategy_fingerprint=member.strategy_fingerprint, daily=report.oos_daily,
                fills_csv_path=headline_fills_path, trades_csv_path=headline_trades_path,
                price_bars=headline_price_bars, out_dir=self._artifact_dir,
            )
            source_artifact = str(manifest_path_for(bundle, out_dir=self._artifact_dir))
            source_artifact_sha256 = bundle.bundle_sha256

        return TrialEvidence(
            status=AttemptStatus.VALID,
            trial_p_value=trial_p,
            non_family_verdict=non_family_verdict,
            non_family_reason_codes=reason_codes,
            metrics=metrics,
            engine=_ENGINE_ID,
            target_schedule_hash=report.target_schedule_hash,
            report_fingerprint=report.report_fingerprint(),
            source_artifact=source_artifact,
            source_artifact_sha256=source_artifact_sha256,
            evidence_completeness="full",
            produced_under_validation_spec_fingerprint=protocol_fp,
            produced_under_reliability_policy_fingerprint=policy_fp,
        )

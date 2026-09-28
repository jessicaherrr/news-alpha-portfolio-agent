"""Deterministic import of the corrected Phase 13.5C research history (section 13).

Nothing is recomputed. Every number written into the registry is read out of a
committed Phase 13.5C artifact; this module only re-keys that evidence into the
registry's identity model. It is strictly offline: no Databento client, no
network, no market-data acquisition, no 2025 access.

What is imported
----------------
* 21 canonical + 86 predeclared-neighbour experiments = **107 unique statistical
  hypotheses**, the corrected (``d40050b``) authoritative lineage;
* 21 pre-correction canonical experiments (``52222c3``), retained as historical
  lineage and marked superseded by explicit ``SUPERSEDES`` / ``CORRECTS`` edges;
* the failure memory: every canonical rejection's typed reason codes, plus the
  roll-execution data-plumbing lesson, the fractional-Treasury economics defect
  and its correction, and the checkpoint supersession;
* 21 data-quality sensitivity reruns and 5 cross-market summaries as **evidence,
  not experiments** -- they never enter the 107.

Feature provenance is a variant's own, never the canonical trial's
------------------------------------------------------------------
A predeclared neighbour changes parameters, and for most families that changes
its ``FeatureSpec``s too -- a TSMOM neighbour with ``fast_horizon=10`` depends on
a different feature than the canonical ``fast_horizon=20``. Copying the canonical
feature set would both misstate provenance and, because
``feature_spec_fingerprint`` is a legitimate PRE-RUN identity input, give the
neighbour the wrong ``experiment_identity`` -- so an agent computing the
neighbour's true identity would fail exact-duplicate detection.

The importer therefore *reconstructs* each variant's ``StrategySpec`` from its
frozen parameters using the frozen factories
(:func:`alpha_agent.strategy.candidates_phase_13_5c.spec_for_params`), proves the
reconstruction is faithful by requiring its fingerprint to equal the committed
``neighbour_fingerprints`` entry, and derives that spec's own feature
fingerprints with the frozen Phase 13.5C rule. Reconstruction reads no market
data, runs no backtest and inspects no performance.

Where a parameter changes only DSL thresholds and not a ``FeatureSpec`` (the
mean-reversion ``entry_z`` / ``exit_z`` neighbours), the reconstructed feature
set legitimately equals the canonical one. That is derived, not assumed.

Provenance is never fabricated
------------------------------
``target_schedule_hash`` and ``report_fingerprint`` are post-run provenance, not
identity. They are recorded **only** where a committed artifact actually holds
them:

* the 21 canonical trials take both from their own
  ``<ROOT>__<FAMILY>__validation_report.json``;
* **all 86 neighbours** get ``target_schedule_hash = None`` and
  ``report_fingerprint = None``. No per-neighbour ``ValidationReport`` was ever
  written; their statistics come from ``TRIAL_FAMILY.csv``, and each result
  records that in ``source_artifact`` / ``source_artifact_sha256``;
* the 21 pre-correction rows get ``target_schedule_hash = None`` and
  ``report_fingerprint = None`` too -- the corrected report's values are
  evidence for the *corrected* run and are never carried over.

``CONTRACT_ECONOMICS_CORRECTION.json`` does list 20 ZN schedule hashes
(4 canonical + 16 neighbours, pre and post), and it is tempting to treat those
as the missing neighbour provenance. They are **not**: the invariance script
builds them with ``emit_from_ts_ns`` at 2018-01-01 over the whole 2018-2024
span, whereas a ``ValidationReport``'s ``target_schedule_hash`` covers the
headline VALIDATION-window schedule. They are hashes of a different schedule
computed for a different purpose (proving the economics correction left ZN's
signals untouched), and attaching them to a neighbour's headline run would be
the same class of error as copying the canonical hash. The importer proves the
mismatch is real rather than assuming it -- see
:meth:`Phase135cSources.assert_internally_consistent`.

Why the pre-correction runs are separate experiments rather than an overwrite
-----------------------------------------------------------------------------
BH q-values and the DSR benchmark are computed jointly over the whole 107-trial
family, so ZN's defective point value changed the global statistics of *every*
trial, not just ZN's. The family-wide contract economics therefore belong to the
execution configuration, which is part of ``experiment_identity``: the corrected
and defective runs are genuinely different experiments, and the registry records
both instead of silently replacing one with the other.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from alpha_agent.registry.enums import (
    AssetDomain,
    ExperimentStatus,
    FailureClass,
    FailureScope,
    RegistryVerdict,
    RelationType,
    TrialRole,
)
from alpha_agent.registry.identity import (
    cost_config_identity,
    execution_config_identity,
    experiment_identity,
    friendly_experiment_id,
    parameter_variant_identity,
    risk_config_identity,
)
from alpha_agent.registry.models import (
    CrossMarketEvidenceRecord,
    ExperimentRecord,
    FailureRecord,
    ImportBundle,
    LineageEdge,
    MarketWindow,
    ResultRecord,
    SensitivityEvidence,
)
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.candidates_phase_13_5c import (
    feature_fingerprints,
    spec_for_params,
)
from alpha_agent.validation.fingerprint import fingerprint

PHASE = "13.5C"
PRECORRECTION_PHASE = "13.5C-precorrection"
CORRECTED_COMMIT = "d40050b"
SUPERSEDED_COMMIT = "52222c3"
PROVENANCE_COMMIT = "bdc231a"
ROLL_MARKS_COMMIT = "52222c3"

EXPECTED_UNIQUE_TRIALS = 107
EXPECTED_CANONICAL = 21
EXPECTED_NEIGHBOURS = 86
#: No committed artifact records a neighbour's headline-window schedule hash,
#: so every neighbour's is NULL. Never back-fill it.
EXPECTED_NEIGHBOURS_WITH_SCHEDULE_HASH = 0
EXPECTED_VERDICTS = {"PASS": 0, "REJECT": 19, "INCONCLUSIVE": 2}

#: Inclusive market windows. The headline OOS window ends 2024-12-31: the
#: Phase 13.5C ``2025-01-01`` boundary is EXCLUSIVE and is never stored as a
#: date value here (see :mod:`alpha_agent.registry.holdout_guard`).
VALIDATION_WINDOW = MarketWindow(
    label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31"
)
SPLIT_LABEL = "VALIDATION_2023_2024"

SOURCE_FILES = (
    "REAL_MARKET_REPORT.json",
    "VERDICT_TABLE.csv",
    "TRIAL_FAMILY.csv",
    "CROSS_MARKET_SUMMARY.csv",
    "DATA_QUALITY_SENSITIVITY.csv",
    "CONTRACT_ECONOMICS_QA.json",
    "CONTRACT_ECONOMICS_CORRECTION.json",
    "roll_qa.json",
)

#: Phase 13 ``ReasonCode`` -> Phase 14 ``FailureClass``. Phase 13's typed reason
#: codes stay authoritative (section 5); this is a pure classification.
REASON_TO_FAILURE_CLASS: dict[str, FailureClass] = {
    "null_hypothesis_not_rejected": FailureClass.NULL_NOT_REJECTED,
    "fdr_qvalue_above_threshold": FailureClass.FDR_NOT_PASSED,
    "deflated_sharpe_below_threshold": FailureClass.DSR_NOT_PASSED,
    "negative_oos_net_pnl": FailureClass.SCIENTIFIC_REJECTION,
    "fold_consistency_below_threshold": FailureClass.SCIENTIFIC_REJECTION,
    "cost_stress_degradation_exceeds_limit": FailureClass.COST_FRAGILITY,
    "parameter_neighbourhood_unstable": FailureClass.PARAMETER_INSTABILITY,
    "performance_concentrated_in_one_regime": FailureClass.SCIENTIFIC_REJECTION,
    "performance_concentrated_in_one_root": FailureClass.CROSS_MARKET_WEAKNESS,
    "insufficient_oos_days": FailureClass.STATISTICAL_INCONCLUSIVE,
    "insufficient_trades": FailureClass.INSUFFICIENT_TRADES,
    "insufficient_folds": FailureClass.STATISTICAL_INCONCLUSIVE,
    "insufficient_nonzero_observations": FailureClass.STATISTICAL_INCONCLUSIVE,
    "no_daily_trace": FailureClass.STATISTICAL_INCONCLUSIVE,
    "regime_not_evaluated": FailureClass.STATISTICAL_INCONCLUSIVE,
    "cross_market_not_evaluated": FailureClass.STATISTICAL_INCONCLUSIVE,
}


TRIAL_FAMILY_ARTIFACT = "outputs/phase_13_5c/TRIAL_FAMILY.csv"
CORRECTION_ARTIFACT = "outputs/phase_13_5c/CONTRACT_ECONOMICS_CORRECTION.json"

NEIGHBOUR_EVIDENCE_NOTE = (
    "partial: a predeclared neighbour has no ValidationReport of its own. The "
    "committed Phase 13.5C evidence retains its strategy fingerprint, p-value, "
    "BH q-value and rejection at q (TRIAL_FAMILY.csv) plus the canonical "
    "parameter-stability aggregate; per-neighbour PnL/Sharpe and "
    "report_fingerprint were not retained"
)
NEIGHBOUR_NO_SCHEDULE_NOTE = (
    "; target_schedule_hash was not retained per neighbour in the committed "
    "Phase 13.5C evidence and is deliberately NULL rather than inherited from "
    "the canonical trial. (CONTRACT_ECONOMICS_CORRECTION.json lists ZN variant "
    "schedule hashes, but those are full-span 2018-2024 invariance-check "
    "schedules, not this headline VALIDATION-window run's.)"
)


class Phase135cArtifactError(RuntimeError):
    """A Phase 13.5C artifact is missing or internally inconsistent in a way
    that would change the research interpretation. Never patched over."""


# --------------------------------------------------------------------------
# artifact loading
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    if not path.exists():
        raise Phase135cArtifactError(f"missing Phase 13.5C artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise Phase135cArtifactError(f"missing Phase 13.5C artifact: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _i(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


class Phase135cSources:
    """Every committed Phase 13.5C artifact the import reads, loaded once."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.out_dir = repo_root / "outputs" / "phase_13_5c"
        self.manifest_path = (
            repo_root / "data" / "manifests" / "phase_13_5c" / "candidate_manifest.json"
        )
        self.manifest = _json(self.manifest_path)["manifest"]
        self.manifest_fingerprint = _json(self.manifest_path)["manifest_fingerprint"]
        self.report = _json(self.out_dir / "REAL_MARKET_REPORT.json")
        self.trial_family = _csv(self.out_dir / "TRIAL_FAMILY.csv")
        self.cross_market = _csv(self.out_dir / "CROSS_MARKET_SUMMARY.csv")
        self.sensitivity = _csv(self.out_dir / "DATA_QUALITY_SENSITIVITY.csv")
        self.economics_qa = _json(self.out_dir / "CONTRACT_ECONOMICS_QA.json")
        self.correction = _json(self.out_dir / "CONTRACT_ECONOMICS_CORRECTION.json")
        self.roll_qa = _json(self.out_dir / "roll_qa.json")
        self._validation_reports: dict[tuple[str, str], dict] = {}

    def validation_report(self, root: str, family: str) -> dict:
        key = (root, family)
        if key not in self._validation_reports:
            path = self.out_dir / f"{root}__{family.upper()}__validation_report.json"
            self._validation_reports[key] = _json(path)
        return self._validation_reports[key]

    def artifact_sha256(self, name: str) -> str:
        """SHA-256 of one committed artifact, by repo-relative path."""
        return _sha256(self.repo_root / name)

    def zn_schedule_hashes(self, *, corrected: bool) -> dict[str, str]:
        key = (
            "zn_target_schedule_hashes_post" if corrected
            else "zn_target_schedule_hashes_pre"
        )
        return dict(self.correction[key])

    def source_fingerprint(self) -> str:
        payload = {
            "candidate_manifest.json": _sha256(self.manifest_path),
            **{name: _sha256(self.out_dir / name) for name in SOURCE_FILES},
        }
        for trial in self.manifest["trials"]:
            name = (
                f"{trial['root_symbol']}__{trial['family_key'].upper()}"
                "__validation_report.json"
            )
            payload[name] = _sha256(self.out_dir / name)
        return fingerprint("p14source1", payload)

    def assert_internally_consistent(self) -> None:
        """Refuse to import artifacts that disagree with each other."""
        rep = self.report
        if rep.get("unique_statistical_trial_count") != EXPECTED_UNIQUE_TRIALS:
            raise Phase135cArtifactError(
                f"REAL_MARKET_REPORT unique_statistical_trial_count="
                f"{rep.get('unique_statistical_trial_count')}, expected "
                f"{EXPECTED_UNIQUE_TRIALS}"
            )
        if len(self.trial_family) != EXPECTED_UNIQUE_TRIALS:
            raise Phase135cArtifactError(
                f"TRIAL_FAMILY.csv has {len(self.trial_family)} rows, expected "
                f"{EXPECTED_UNIQUE_TRIALS}"
            )
        counts = dict(rep.get("verdict_counts") or {})
        got = {k: int(counts.get(k, 0)) for k in EXPECTED_VERDICTS}
        if got != EXPECTED_VERDICTS:
            raise Phase135cArtifactError(
                f"corrected canonical verdict counts {got} != {EXPECTED_VERDICTS}"
            )
        if self.correction["corrected_verdict_counts"] != EXPECTED_VERDICTS:
            raise Phase135cArtifactError(
                "CONTRACT_ECONOMICS_CORRECTION corrected_verdict_counts disagrees with "
                "the expected corrected headline"
            )
        if self.correction["candidate_manifest_fingerprint"] != self.manifest_fingerprint:
            raise Phase135cArtifactError(
                "candidate manifest fingerprint disagrees between the frozen manifest "
                "and the correction artifact"
            )
        if rep.get("frozen_candidate_manifest_fingerprint") != self.manifest_fingerprint:
            raise Phase135cArtifactError(
                "candidate manifest fingerprint disagrees between the frozen manifest "
                "and REAL_MARKET_REPORT.json"
            )
        if rep.get("software_or_data_failures"):
            raise Phase135cArtifactError(
                f"Phase 13.5C reported software/data failures: "
                f"{rep['software_or_data_failures']}"
            )
        if not rep.get("holdout_2025_never_requested_or_accessed", False):
            raise Phase135cArtifactError(
                "REAL_MARKET_REPORT does not assert the 2025 holdout was never accessed"
            )
        if not self.economics_qa.get("pass", False):
            raise Phase135cArtifactError("CONTRACT_ECONOMICS_QA did not pass")
        if not self.roll_qa.get("pass", False):
            raise Phase135cArtifactError("roll_qa did not pass")
        # The correction artifact's own invariance claim must hold.
        post = self.zn_schedule_hashes(corrected=True)
        if self.zn_schedule_hashes(corrected=False) != post:
            raise Phase135cArtifactError(
                "ZN pre/post correction target schedule hashes disagree, contradicting "
                "the artifact's own zn_target_schedule_hashes_identical invariance check"
            )
        # ...and they must NOT be mistaken for the headline-run schedule hashes.
        # The invariance script emits from 2018-01-01 across the whole span; a
        # ValidationReport fingerprints the headline VALIDATION-window schedule.
        # Proving they differ is what licenses leaving neighbour provenance NULL
        # instead of back-filling it from this artifact.
        collisions = [
            family for family in ("breakout", "ma_trend", "mean_reversion", "tsmom")
            if post.get(f"{family}/canonical")
            == self.validation_report("ZN", family)["fingerprints"].get(
                "target_schedule_hash"
            )
        ]
        if collisions:
            raise Phase135cArtifactError(
                f"ZN {collisions} canonical schedule hashes from "
                "CONTRACT_ECONOMICS_CORRECTION.json now match the validation reports'. "
                "The two were computed over different windows; if that changed, revisit "
                "whether neighbour schedule provenance can be sourced from the "
                "correction artifact after all."
            )


# --------------------------------------------------------------------------
# configuration identities
# --------------------------------------------------------------------------
def _economics_from_qa(sources: Phase135cSources) -> dict[str, dict[str, float]]:
    return {
        r["root"]: {
            "point_value_usd": float(r["derived_point_value_usd"]),
            "tick_size": float(r["derived_tick_size"]),
            "tick_value_usd": float(r["derived_tick_value_usd"]),
        }
        for r in sources.economics_qa["roots"]
    }


def _precorrection_economics(sources: Phase135cSources) -> dict[str, dict[str, float]]:
    econ = {k: dict(v) for k, v in _economics_from_qa(sources).items()}
    pre = sources.correction["zn_contract_economics_pre"]
    econ["ZN"] = {
        "point_value_usd": float(pre["point_value_usd"]),
        "tick_size": float(pre["tick_size"]),
        "tick_value_usd": float(pre["tick_value_usd"]),
    }
    return econ


CORRECTED_ECONOMICS_RULE = (
    "fractionally-quoted (non-sentinel main_fraction) => price_scale = 0.01 "
    "(percent of par); decimal products unchanged"
)
DEFECTIVE_ECONOMICS_RULE = (
    "DEFECTIVE: min_price_increment_amount / display_factor used as the USD tick "
    "value on the fractional path"
)


def _config_identities(sources: Phase135cSources) -> dict[str, str]:
    m = sources.manifest
    corrected = execution_config_identity(
        execution_assumptions=m["execution_assumptions"],
        contract_economics=_economics_from_qa(sources),
        economics_rule=CORRECTED_ECONOMICS_RULE,
    )
    defective = execution_config_identity(
        execution_assumptions=m["execution_assumptions"],
        contract_economics=_precorrection_economics(sources),
        economics_rule=DEFECTIVE_ECONOMICS_RULE,
    )
    cost = cost_config_identity(
        base_commission_per_contract_usd=float(
            m["execution_assumptions"]["commission_per_contract_usd"]
        ),
        base_slippage_ticks=float(m["execution_assumptions"]["slippage_ticks"]),
        base_spread_ticks=float(m["execution_assumptions"]["spread_ticks"]),
        scenarios=list(m["cost_scenarios"]),
    )
    risk = risk_config_identity(m["risk_assumptions"])
    return {
        "execution_corrected": corrected,
        "execution_defective": defective,
        "cost": cost,
        "risk": risk,
    }


def config_identities_for_orchestrator(repo_root: Path) -> dict[str, str]:
    """Public, additive reuse of the SAME execution/cost/risk config identities
    already computed for the 107-hypothesis Phase 13.5C import (Agent
    runtime-integration release, section 6/7): a new orchestrator-driven
    hypothesis on an already-used root shares these identity components with
    existing registry rows instead of inventing a second, parallel numbering.
    Reads only already-committed local artifacts (the frozen candidate
    manifest + contract-economics QA) -- no market-data or Databento access.
    Always the CORRECTED economics rule; the defective/pre-correction variant
    is import-history only and is never produced by new research.
    """
    sources = Phase135cSources(repo_root)
    ids = _config_identities(sources)
    return {
        "execution_config_identity": ids["execution_corrected"],
        "cost_config_identity": ids["cost"],
        "risk_identity": ids["risk"],
    }


# --------------------------------------------------------------------------
# record construction
# --------------------------------------------------------------------------
class ReconstructedVariant(BaseModel):
    """One frozen variant rebuilt from its parameters, with its OWN semantics."""

    model_config = {"frozen": True, "extra": "forbid"}

    family_key: str
    root_symbol: str
    variant_label: str                    # "canonical" | "neighbour_<i>"
    params: dict
    strategy_id: str                      # the reconstructed spec's own id
    strategy_fingerprint: str
    feature_fingerprints: tuple[str, ...]

    def feature_spec_fingerprint(self) -> str:
        return fingerprint("featset1", {"features": list(self.feature_fingerprints)})


def reconstruct_variant(
    *, family_key: str, root_symbol: str, variant_label: str, params: dict,
    expected_strategy_fingerprint: str,
) -> ReconstructedVariant:
    """Rebuild one frozen variant and PROVE the reconstruction is faithful.

    Configuration reconstruction only -- no market data, no backtest, no
    performance. A fingerprint disagreement means the frozen factories no longer
    produce the spec the candidate manifest was frozen from, which is a genuine
    provenance blocker, not something to work around.
    """
    spec = spec_for_params(family_key, params)
    got = strategy_fingerprint(spec)
    if got != expected_strategy_fingerprint:
        raise Phase135cArtifactError(
            f"reconstructed {family_key}/{root_symbol}/{variant_label} StrategySpec "
            f"fingerprint {got} != the committed candidate-manifest fingerprint "
            f"{expected_strategy_fingerprint}; the frozen factories no longer rebuild "
            "the frozen hypothesis -- refusing to derive feature provenance from it"
        )
    return ReconstructedVariant(
        family_key=family_key, root_symbol=root_symbol, variant_label=variant_label,
        params=dict(params), strategy_id=spec.strategy_id, strategy_fingerprint=got,
        feature_fingerprints=tuple(feature_fingerprints(spec)),
    )


def _strategy_spec_json(
    *, family: str, params: dict, feature_fingerprints: list[str],
    signal_cadence: str, execution_cadence: str, strategy_id: str,
) -> dict:
    """The registry's structural view of a StrategySpec.

    Structured, typed, and small: the full spec is reproducible from
    ``strategy_fingerprint`` plus the frozen candidate manifest, so the registry
    stores the shape and the parameters, not a duplicated Phase 10 artifact.
    """
    return {
        "schema": "registry-strategy-spec/1",
        "strategy_family": family,
        "strategy_id": strategy_id,
        "params": dict(params),
        "parameter_names": sorted(str(k) for k in params),
        "feature_fingerprints": list(feature_fingerprints),
        "signal_cadence": signal_cadence,
        "execution_cadence": execution_cadence,
    }


def _trial_stats(sources: Phase135cSources) -> dict[str, dict[str, Any]]:
    """label -> {p_value, bh_q_value, rejected_at_q, strategy_fingerprint}."""
    out: dict[str, dict[str, Any]] = {}
    for row in sources.trial_family:
        out[row["label"]] = {
            "p_value": _f(row["p_value"]),
            "bh_q_value": _f(row["bh_q_value"]),
            "rejected_at_q": row["rejected_at_q"].strip().lower() == "true",
            "strategy_fingerprint": row["strategy_fingerprint"],
        }
    return out


def build_bundle(repo_root: Path) -> ImportBundle:
    """Assemble the complete, validated Phase 13.5C import bundle."""
    sources = Phase135cSources(repo_root)
    sources.assert_internally_consistent()
    ids = _config_identities(sources)
    stats = _trial_stats(sources)
    created_at = str(sources.report["generated_at"])
    trial_family_sha = sources.artifact_sha256(TRIAL_FAMILY_ARTIFACT)
    correction_sha = sources.artifact_sha256(CORRECTION_ARTIFACT)

    experiments: list[ExperimentRecord] = []
    results: list[ResultRecord] = []
    failures: list[FailureRecord] = []
    lineage: list[LineageEdge] = []
    sensitivity: list[SensitivityEvidence] = []
    cross_market: list[CrossMarketEvidenceRecord] = []

    canonical_identity_by_key: dict[tuple[str, str], str] = {}
    superseded_identity_by_key: dict[tuple[str, str], str] = {}

    for trial in sources.manifest["trials"]:
        family = trial["family_key"]
        root = trial["root_symbol"]
        report = sources.validation_report(root, family)
        fp = report["fingerprints"]

        canonical_variant_spec = reconstruct_variant(
            family_key=family, root_symbol=root, variant_label="canonical",
            params=trial["canonical_params"],
            expected_strategy_fingerprint=trial["strategy_fingerprint"],
        )
        if list(canonical_variant_spec.feature_fingerprints) != list(
            trial["feature_fingerprints"]
        ):
            raise Phase135cArtifactError(
                f"reconstructed {root}/{family} canonical feature fingerprints disagree "
                "with the frozen candidate manifest"
            )

        common = {
            # Phase 6 ETF Research Pilot: every Phase 13.5C row is a known
            # Futures experiment -- a stated fact about what this importer
            # imports, not an inference. Shared via **common so all three
            # ExperimentRecord construction sites below (canonical, corrected
            # neighbour, pre-correction superseded) say it explicitly.
            "asset_domain": AssetDomain.FUTURES,
            "candidate_manifest_fingerprint": sources.manifest_fingerprint,
            "dataset_fingerprint": fp["dataset_fingerprint"],
            "split_identity": fp["split_fingerprint"],
            "market_window": VALIDATION_WINDOW,
            "validation_spec_fingerprint": fp["validation_fingerprint"],
            "reliability_policy_fingerprint": fp["reliability_policy_fingerprint"],
            "cost_config_identity": ids["cost"],
            "risk_identity": ids["risk"],
        }
        # Each variant carries its OWN feature-spec identity -- never a shared
        # one lifted from the canonical trial (Phase 14.2).
        canonical_feature_fp = canonical_variant_spec.feature_spec_fingerprint()

        # ---- corrected canonical --------------------------------------
        canonical_variant = parameter_variant_identity(trial["canonical_params"])
        canonical_identity = experiment_identity(
            strategy_fingerprint=trial["strategy_fingerprint"],
            strategy_family=family,
            root_symbol=root,
            parameter_variant_identity=canonical_variant,
            dataset_fingerprint=common["dataset_fingerprint"],
            split_identity=common["split_identity"],
            validation_spec_fingerprint=common["validation_spec_fingerprint"],
            reliability_policy_fingerprint=common["reliability_policy_fingerprint"],
            execution_config_identity=ids["execution_corrected"],
            cost_config_identity=ids["cost"],
            risk_identity=ids["risk"],
            feature_spec_fingerprint=canonical_feature_fp,
        )
        canonical_identity_by_key[(root, family)] = canonical_identity
        canonical_id = friendly_experiment_id(
            root_symbol=root, strategy_family=family,
            variant_label="canonical", split_label=SPLIT_LABEL,
        )
        experiments.append(
            ExperimentRecord(
                experiment_identity=canonical_identity,
                experiment_id=canonical_id,
                display_name=f"{root} / {family.upper()} / canonical / {SPLIT_LABEL}",
                created_at=created_at,
                phase=PHASE,
                status=ExperimentStatus.COMPLETED,
                code_commit=CORRECTED_COMMIT,
                root_symbol=root,
                strategy_family=family,
                strategy_fingerprint=trial["strategy_fingerprint"],
                strategy_id=trial["strategy_id"],
                strategy_spec_json=_strategy_spec_json(
                    family=family, params=trial["canonical_params"],
                    feature_fingerprints=list(
                        canonical_variant_spec.feature_fingerprints
                    ),
                    signal_cadence=trial["signal_cadence"],
                    execution_cadence=trial["execution_cadence"],
                    strategy_id=canonical_variant_spec.strategy_id,
                ),
                feature_spec_fingerprint=canonical_feature_fp,
                target_schedule_hash=fp.get("target_schedule_hash"),
                execution_config_identity=ids["execution_corrected"],
                trial_role=TrialRole.CANONICAL,
                parameter_variant_identity=canonical_variant,
                parameter_variant_label="canonical",
                report_fingerprint=fp["report_fingerprint"],
                notes="corrected Phase 13.5C matrix; headline-adjudicated trial",
                **common,
            )
        )
        report_artifact = (
            f"outputs/phase_13_5c/{root}__{family.upper()}__validation_report.json"
        )
        results.append(
            _canonical_result(
                canonical_identity, report, stats, family, root,
                source_artifact=report_artifact,
                source_artifact_sha256=sources.artifact_sha256(report_artifact),
            )
        )
        failures.extend(
            _scientific_failures(
                canonical_identity, canonical_id, report, family, root, created_at
            )
        )

        # ---- corrected neighbours -------------------------------------
        for i, (params, nfp) in enumerate(
            zip(trial["neighbour_params"], trial["neighbour_fingerprints"], strict=True)
        ):
            label = f"neighbour:{family}/{root}/{i}"
            if label not in stats:
                raise Phase135cArtifactError(
                    f"TRIAL_FAMILY.csv has no row for {label}"
                )
            if stats[label]["strategy_fingerprint"] != nfp:
                raise Phase135cArtifactError(
                    f"{label}: TRIAL_FAMILY.csv strategy fingerprint disagrees with the "
                    "frozen candidate manifest"
                )
            neighbour_spec = reconstruct_variant(
                family_key=family, root_symbol=root,
                variant_label=f"neighbour_{i}", params=params,
                expected_strategy_fingerprint=nfp,
            )
            neighbour_feature_fp = neighbour_spec.feature_spec_fingerprint()
            variant = parameter_variant_identity(params)
            identity = experiment_identity(
                strategy_fingerprint=nfp,
                strategy_family=family,
                root_symbol=root,
                parameter_variant_identity=variant,
                dataset_fingerprint=common["dataset_fingerprint"],
                split_identity=common["split_identity"],
                validation_spec_fingerprint=common["validation_spec_fingerprint"],
                reliability_policy_fingerprint=common["reliability_policy_fingerprint"],
                execution_config_identity=ids["execution_corrected"],
                cost_config_identity=ids["cost"],
                risk_identity=ids["risk"],
                feature_spec_fingerprint=neighbour_feature_fp,
            )
            neighbour_id = friendly_experiment_id(
                root_symbol=root, strategy_family=family,
                variant_label=f"neighbour_{i}", split_label=SPLIT_LABEL,
            )
            experiments.append(
                ExperimentRecord(
                    experiment_identity=identity,
                    experiment_id=neighbour_id,
                    display_name=f"{root} / {family.upper()} / neighbour_{i} / {SPLIT_LABEL}",
                    created_at=created_at,
                    phase=PHASE,
                    status=ExperimentStatus.COMPLETED,
                    code_commit=CORRECTED_COMMIT,
                    root_symbol=root,
                    strategy_family=family,
                    strategy_fingerprint=nfp,
                    # the reconstructed spec's OWN id, not a synthetic label
                    strategy_id=neighbour_spec.strategy_id,
                    strategy_spec_json=_strategy_spec_json(
                        family=family, params=params,
                        feature_fingerprints=list(neighbour_spec.feature_fingerprints),
                        signal_cadence=trial["signal_cadence"],
                        execution_cadence=trial["execution_cadence"],
                        strategy_id=neighbour_spec.strategy_id,
                    ),
                    feature_spec_fingerprint=neighbour_feature_fp,
                    # no committed artifact records this neighbour's compiled
                    # headline schedule; NULL, never the canonical trial's.
                    target_schedule_hash=None,
                    execution_config_identity=ids["execution_corrected"],
                    trial_role=TrialRole.NEIGHBOUR,
                    parameter_variant_identity=variant,
                    parameter_variant_label=f"neighbour_{i}",
                    parent_experiment_identity=canonical_identity,
                    # no per-neighbour ValidationReport was ever written
                    report_fingerprint=None,
                    notes=(
                        "predeclared parameter neighbour: a full BH/FDR trial, "
                        "not headline-adjudicated"
                    ),
                    **common,
                )
            )
            s = stats[label]
            results.append(
                ResultRecord(
                    experiment_identity=identity,
                    headline_verdict=RegistryVerdict.NOT_ADJUDICATED,
                    reason_codes=(),
                    bh_p_value=s["p_value"],
                    bh_q=s["bh_q_value"],
                    bh_rejected_at_q=s["rejected_at_q"],
                    parameter_stability={
                        "role": "neighbour_of_canonical",
                        "canonical_experiment_id": canonical_id,
                    },
                    holdout_eligible=False,
                    evidence_completeness=(
                        NEIGHBOUR_EVIDENCE_NOTE + NEIGHBOUR_NO_SCHEDULE_NOTE
                    ),
                    source_artifact=TRIAL_FAMILY_ARTIFACT,
                    source_artifact_sha256=trial_family_sha,
                )
            )
            lineage.append(
                LineageEdge(
                    source_experiment_identity=identity,
                    target_experiment_identity=canonical_identity,
                    relation_type=RelationType.PARAMETER_NEIGHBOUR_OF,
                    note=f"predeclared +/- one grid step neighbour {i}",
                )
            )

        # ---- pre-correction canonical (historical, superseded) --------
        pre = sources.correction["global_statistics_superseded"]["per_trial"].get(
            f"{root}/{family}"
        )
        if pre is None:
            raise Phase135cArtifactError(
                f"CONTRACT_ECONOMICS_CORRECTION has no pre-correction record for "
                f"{root}/{family}"
            )
        sup_identity = experiment_identity(
            strategy_fingerprint=trial["strategy_fingerprint"],
            strategy_family=family,
            root_symbol=root,
            parameter_variant_identity=canonical_variant,
            dataset_fingerprint=common["dataset_fingerprint"],
            split_identity=common["split_identity"],
            validation_spec_fingerprint=common["validation_spec_fingerprint"],
            reliability_policy_fingerprint=common["reliability_policy_fingerprint"],
            execution_config_identity=ids["execution_defective"],
            cost_config_identity=ids["cost"],
            risk_identity=ids["risk"],
            feature_spec_fingerprint=canonical_feature_fp,
        )
        superseded_identity_by_key[(root, family)] = sup_identity
        sup_id = friendly_experiment_id(
            root_symbol=root, strategy_family=family, variant_label="canonical",
            split_label=SPLIT_LABEL, lineage_tag="precorrection_52222c3",
        )
        experiments.append(
            ExperimentRecord(
                experiment_identity=sup_identity,
                experiment_id=sup_id,
                display_name=(
                    f"{root} / {family.upper()} / canonical / {SPLIT_LABEL} "
                    f"(pre-correction {SUPERSEDED_COMMIT})"
                ),
                created_at=f"unrecorded (pre-correction checkpoint {SUPERSEDED_COMMIT})",
                phase=PRECORRECTION_PHASE,
                status=ExperimentStatus.COMPLETED,
                code_commit=SUPERSEDED_COMMIT,
                root_symbol=root,
                strategy_family=family,
                strategy_fingerprint=trial["strategy_fingerprint"],
                strategy_id=trial["strategy_id"],
                strategy_spec_json=_strategy_spec_json(
                    family=family, params=trial["canonical_params"],
                    feature_fingerprints=list(
                        canonical_variant_spec.feature_fingerprints
                    ),
                    signal_cadence=trial["signal_cadence"],
                    execution_cadence=trial["execution_cadence"],
                    strategy_id=canonical_variant_spec.strategy_id,
                ),
                feature_spec_fingerprint=canonical_feature_fp,
                # the corrected report's schedule hash and report fingerprint
                # are evidence for the CORRECTED run, never carried over here.
                target_schedule_hash=None,
                execution_config_identity=ids["execution_defective"],
                trial_role=TrialRole.CANONICAL,
                parameter_variant_identity=canonical_variant,
                parameter_variant_label="canonical",
                report_fingerprint=None,
                notes=(
                    "pre-correction Phase 13.5C checkpoint; retained as historical "
                    "lineage and SUPERSEDED for performance interpretation"
                ),
                **common,
            )
        )
        results.append(
            ResultRecord(
                experiment_identity=sup_identity,
                headline_verdict=RegistryVerdict(pre["verdict_pre"]),
                reason_codes=(),
                net_pnl_usd=_f(pre["net_pnl_pre"]),
                bh_q=_f(pre["bh_q_pre"]),
                dsr_probability=_f(pre["dsr_pre"]),
                holdout_eligible=False,
                evidence_completeness=(
                    "partial: the retained pre-correction artifact "
                    "(CONTRACT_ECONOMICS_CORRECTION.json) records the superseded "
                    "verdict, net PnL, global BH q-value and DSR per canonical trial; "
                    "per-trial reason codes, report_fingerprint and the 86 "
                    "pre-correction neighbour statistics were not retained"
                    "; target_schedule_hash is NULL -- the corrected report's hash "
                    "is evidence for the corrected run, and the correction artifact's "
                    "ZN hashes are full-span invariance-check schedules, not this run's"
                ),
                source_artifact=CORRECTION_ARTIFACT,
                source_artifact_sha256=correction_sha,
            )
        )
        for relation in (RelationType.SUPERSEDES, RelationType.CORRECTS):
            lineage.append(
                LineageEdge(
                    source_experiment_identity=canonical_identity,
                    target_experiment_identity=sup_identity,
                    relation_type=relation,
                    note=(
                        f"{CORRECTED_COMMIT} corrects the fractional-Treasury contract "
                        f"economics defect in {SUPERSEDED_COMMIT}; the 107-trial BH/DSR "
                        "family is computed jointly, so every trial's global statistics "
                        "are superseded"
                    ),
                )
            )
        failures.append(
            _superseded_failure(sup_identity, sup_id, root, family, pre)
        )
        if root == "ZN":
            failures.append(
                _zn_experiment_failure(sup_identity, sup_id, family, sources)
            )

    # ---- evidence that is NOT an experiment ---------------------------
    sensitivity.extend(_sensitivity_evidence(sources, canonical_identity_by_key))
    cross_market.extend(_cross_market_evidence(sources, canonical_identity_by_key))

    # ---- system-level engineering / procedural lessons ----------------
    failures.extend(_system_failures(sources))

    _assert_counts(experiments)

    source_fp = sources.source_fingerprint()
    metadata = {
        "phase": PHASE,
        "corrected_commit": CORRECTED_COMMIT,
        "superseded_commit": SUPERSEDED_COMMIT,
        "provenance_commit": PROVENANCE_COMMIT,
        "candidate_manifest_fingerprint": sources.manifest_fingerprint,
        "bh_fdr_family_fingerprint": sources.report["bh_fdr_family_fingerprint"],
        "unique_statistical_trial_count": EXPECTED_UNIQUE_TRIALS,
        "canonical_trial_count": EXPECTED_CANONICAL,
        "neighbour_trial_count": EXPECTED_NEIGHBOURS,
        "completed_cpp_execution_count": int(
            sources.report["completed_cpp_execution_count"]
        ),
        "data_quality_sensitivity_reruns": int(
            sources.report["data_quality_sensitivity"]["n_sensitivity_runs"]
        ),
        "data_quality_sensitivity_in_bh_denominator": bool(
            sources.report["data_quality_sensitivity"]["in_bh_fdr_denominator"]
        ),
        "cross_market_summaries": len(sources.cross_market),
        "corrected_verdict_counts": EXPECTED_VERDICTS,
        "holdout_2025_never_accessed": bool(
            sources.report["holdout_2025_never_requested_or_accessed"]
        ),
        "locked_holdout_2025": str(sources.report["date_roles"]["locked_holdout_2025"]),
        "no_new_databento_spend": bool(sources.report["no_new_databento_spend"]),
        "research_train_window": str(sources.report["date_roles"]["research_train"]),
        "headline_validation_window": str(
            sources.report["date_roles"]["headline_validation"]
        ),
    }
    return ImportBundle(
        import_id=fingerprint("p14import1", {"phase": PHASE, "source": source_fp}),
        phase=PHASE,
        source_fingerprint=source_fp,
        created_at=created_at,
        experiments=tuple(experiments),
        results=tuple(results),
        failures=tuple(failures),
        lineage=tuple(lineage),
        sensitivity=tuple(sensitivity),
        cross_market=tuple(cross_market),
        metadata=metadata,
    )


def _canonical_result(
    identity: str, report: dict, stats: dict, family: str, root: str,
    *, source_artifact: str, source_artifact_sha256: str,
) -> ResultRecord:
    headline = report["headline_verdict_global_family"]
    oos = report["oos_metrics"]
    label = f"canonical:{family}/{root}"
    s = stats.get(label, {})
    return ResultRecord(
        experiment_identity=identity,
        headline_verdict=RegistryVerdict(headline["verdict"]),
        reason_codes=tuple(headline["reason_codes"]),
        gross_pnl_usd=_f(oos.get("gross_pnl_usd")),
        costs_usd=_f(oos.get("costs_usd")),
        net_pnl_usd=_f(oos.get("net_pnl_usd")),
        daily_sharpe=_f(oos.get("daily_sharpe")),
        annualized_sharpe=_f(oos.get("annualized_sharpe")),
        gating_null_p=_f(report.get("gating_null_centered_block_bootstrap_p")),
        bh_p_value=_f(s.get("p_value")),
        bh_q=_f(headline.get("bh_q_value_over_107")),
        bh_rejected_at_q=bool(s["rejected_at_q"]) if "rejected_at_q" in s else None,
        dsr_probability=_f(headline.get("dsr_probability_over_107_effective_trials")),
        fold_consistency=_f((report.get("walk_forward") or {}).get("fold_consistency")),
        n_trades=_i(oos.get("n_trades")),
        n_fills=_i(oos.get("n_fills")),
        n_oos_days=_i(oos.get("n_trading_days")),
        parameter_stability=report.get("parameter_stability") or {},
        cost_stress=report.get("cost_stress") or {},
        bootstrap_evidence=report.get("bootstrap_annualized_sharpe_ci") or {},
        regime_evidence=report.get("regime_evidence_volatility") or {},
        cross_market_reference=report.get("cross_market_evidence") or {},
        holdout_eligible=headline["verdict"] == RegistryVerdict.PASS.value,
        evidence_completeness="full",
        source_artifact=source_artifact,
        source_artifact_sha256=source_artifact_sha256,
    )


def _scientific_failures(
    identity: str, experiment_id: str, report: dict, family: str, root: str,
    created_at: str,
) -> list[FailureRecord]:
    """One typed failure record per distinct failure class in the verdict.

    Phase 13's reason codes stay authoritative; the class is a classification of
    them, and the free text supplements -- never replaces -- the typed code.
    """
    headline = report["headline_verdict_global_family"]
    verdict = headline["verdict"]
    if verdict == RegistryVerdict.PASS.value:
        return []
    out: list[FailureRecord] = []
    seen: set[FailureClass] = set()
    for code in headline["reason_codes"]:
        cls = REASON_TO_FAILURE_CLASS.get(code, FailureClass.SCIENTIFIC_REJECTION)
        if cls in seen:
            continue
        seen.add(cls)
        codes = [
            c for c in headline["reason_codes"]
            if REASON_TO_FAILURE_CLASS.get(c, FailureClass.SCIENTIFIC_REJECTION) is cls
        ]
        out.append(
            FailureRecord(
                failure_id=f"F__{experiment_id}__{cls.value}",
                scope=FailureScope.EXPERIMENT,
                experiment_identity=identity,
                failure_class=cls,
                failure_code=";".join(codes),
                summary=(
                    f"{root} {family} canonical: {verdict} under the frozen Phase 13 "
                    f"ReliabilityPolicy ({', '.join(codes)})"
                ),
                mechanism=(
                    "Adjudicated by the frozen Phase 13 reliability policy over the "
                    "corrected 107-trial multiple-testing family; no threshold was "
                    "changed after the result was seen."
                ),
                evidence={
                    "reason_codes": list(headline["reason_codes"]),
                    "verdict": verdict,
                    "daily_sharpe": report["oos_metrics"]["daily_sharpe"],
                    "net_pnl_usd": report["oos_metrics"]["net_pnl_usd"],
                    "bh_q_value_over_107": headline.get("bh_q_value_over_107"),
                    "dsr_probability_over_107_effective_trials": headline.get(
                        "dsr_probability_over_107_effective_trials"
                    ),
                    "gating_null_p": report.get(
                        "gating_null_centered_block_bootstrap_p"
                    ),
                    "n_trades": report["oos_metrics"]["n_trades"],
                    "oos_window": report["oos_window"],
                },
                action_taken=(
                    "recorded as first-class evidence; the candidate is NOT eligible "
                    "for the locked holdout"
                ),
                resolved=True,
                resolution_commit=CORRECTED_COMMIT,
                created_at=created_at,
                root_symbol=root,
                strategy_family=family,
            )
        )
    return out


def _superseded_failure(
    identity: str, experiment_id: str, root: str, family: str, pre: dict
) -> FailureRecord:
    return FailureRecord(
        failure_id=f"F__{experiment_id}__{FailureClass.SUPERSEDED_RESULT.value}",
        scope=FailureScope.EXPERIMENT,
        experiment_identity=identity,
        failure_class=FailureClass.SUPERSEDED_RESULT,
        failure_code="precorrection_global_statistics_superseded",
        summary=(
            f"{root} {family} pre-correction result ({SUPERSEDED_COMMIT}) is superseded "
            f"by {CORRECTED_COMMIT}"
        ),
        mechanism=(
            "The defective ZN point value entered the joint 107-trial family, so the "
            "pre-correction BH q-values and DSR of EVERY trial -- not only ZN's -- are "
            "superseded. ZN's commission-dominated Sharpes inflated the cross-trial "
            "Sharpe variance that sets the DSR benchmark, pinning DSR to 0.0 for all 21 "
            "canonical trials."
        ),
        evidence={
            "bh_q_pre": pre["bh_q_pre"], "bh_q_post": pre["bh_q_post"],
            "dsr_pre": pre["dsr_pre"], "dsr_post": pre["dsr_post"],
            "net_pnl_pre": pre["net_pnl_pre"], "net_pnl_post": pre["net_pnl_post"],
            "verdict_pre": pre["verdict_pre"], "verdict_post": pre["verdict_post"],
            "superseded_commit": SUPERSEDED_COMMIT,
            "corrected_commit": CORRECTED_COMMIT,
        },
        action_taken=(
            "retained in git history and in the registry as historical lineage; never "
            "deleted, never presented as valid"
        ),
        resolved=True,
        resolution_commit=CORRECTED_COMMIT,
        superseded_by=CORRECTED_COMMIT,
        created_at=f"unrecorded (pre-correction checkpoint {SUPERSEDED_COMMIT})",
        root_symbol=root,
        strategy_family=family,
    )


def _zn_experiment_failure(
    identity: str, experiment_id: str, family: str, sources: Phase135cSources
) -> FailureRecord:
    d = sources.correction["defect"]
    return FailureRecord(
        failure_id=f"F__{experiment_id}__{FailureClass.CONTRACT_ECONOMICS_FAILURE.value}",
        scope=FailureScope.EXPERIMENT,
        experiment_identity=identity,
        failure_class=FailureClass.CONTRACT_ECONOMICS_FAILURE,
        failure_code="fractional_treasury_point_value_misderived",
        summary=(
            f"ZN {family} pre-correction run used point value "
            f"{d['old_point_value_usd']} instead of {d['corrected_point_value_usd']} "
            "(15,625x too small)"
        ),
        mechanism=str(d["description"]),
        evidence={
            "old_point_value_usd": d["old_point_value_usd"],
            "old_tick_value_usd": d["old_tick_value_usd"],
            "corrected_point_value_usd": d["corrected_point_value_usd"],
            "corrected_tick_value_usd": d["corrected_tick_value_usd"],
            "source_definition_fields_zn": d["source_definition_fields_zn"],
        },
        action_taken=(
            "generic percent-of-par quote-convention rule; run re-executed under "
            f"{CORRECTED_COMMIT}"
        ),
        resolved=True,
        resolution_commit=CORRECTED_COMMIT,
        superseded_by=CORRECTED_COMMIT,
        created_at=f"unrecorded (pre-correction checkpoint {SUPERSEDED_COMMIT})",
        root_symbol="ZN",
        strategy_family=family,
    )


def _system_failures(sources: Phase135cSources) -> list[FailureRecord]:
    """The engineering / data-integrity lessons a future agent must not relearn
    the hard way (section 6)."""
    created_at = str(sources.report["generated_at"])
    d = sources.correction["defect"]
    roll_roots = {r["root"]: r for r in sources.roll_qa["roots"]}
    return [
        FailureRecord(
            failure_id="F__SYSTEM__ROLL_CLOSE_MARK_DATA_PLUMBING",
            scope=FailureScope.SYSTEM,
            failure_class=FailureClass.ROLL_DATA_FAILURE,
            failure_code="roll_close_leg_missing_outgoing_same_timestamp_close",
            summary=(
                "The clean continuous-front primary MarketEvent stream carried no "
                "same-timestamp close for the OUTGOING contract, so the C++ RejectDefer "
                "roll close-leg had nothing real to price against."
            ),
            mechanism=(
                "A position held across a real futures roll must close the outgoing "
                "contract at a genuine same-timestamp market price. The primary stream "
                "is front-contract only, so the outgoing close was simply absent and the "
                "engine fail-loud refused to invent a retroactive expiry fill."
            ),
            evidence={
                "rejected_solution": (
                    "feed the roll-overlap bars as ordinary MarketEvents -- REJECTED: "
                    "equal-timestamp ordering by instrument_id could change "
                    "active-contract state, bars_seen and latency, i.e. it would alter "
                    "the frozen execution semantics"
                ),
                "accepted_correction": (
                    "auxiliary roll-close marks keyed by (instrument_id, ts_event_ns), "
                    "consulted ONLY by the roll close-leg"
                ),
                "invariants": [
                    (
                        "auxiliary marks are never MarketEvents: never in the event "
                        "stream, MarketState, bars_seen, Strategy::decide, BarHistory, "
                        "latency or daily-equity sampling"
                    ),
                    "empty marks (the default) => byte-identical frozen RejectDefer path",
                    "rolls_priced_auxiliary_marks <= rolls_priced_contemporaneous",
                ],
                "frozen_failure_message_without_marks": roll_roots["ES"]["without_marks"],
                "roll_qa_pass": bool(sources.roll_qa["pass"]),
                "roots_verified": sorted(roll_roots),
                "rolls_priced": {k: v["n_rolls_priced"] for k, v in sorted(roll_roots.items())},
            },
            action_taken=(
                "additive EngineConfig::roll.close_marks C++ interface + deterministic "
                "Python builder from the acquired roll-overlap raw bars; roll QA gate "
                "(outputs/phase_13_5c/roll_qa.json)"
            ),
            resolved=True,
            resolution_commit=ROLL_MARKS_COMMIT,
            created_at=created_at,
        ),
        FailureRecord(
            failure_id="F__SYSTEM__ZN_FRACTIONAL_CONTRACT_ECONOMICS",
            scope=FailureScope.SYSTEM,
            failure_class=FailureClass.CONTRACT_ECONOMICS_FAILURE,
            failure_code="fractional_treasury_point_value_misderived",
            summary=(
                "Fractionally quoted CBOT Treasury economics were derived by treating "
                "min_price_increment_amount / display_factor as the USD tick value, "
                "giving ZN a point value of 0.064 instead of 1000 (15,625x too small)."
            ),
            mechanism=str(d["description"]),
            evidence={
                "affected_root": d["affected_root"],
                "n_affected_of_107": d["n_affected_of_107"],
                "affected_statistical_trials": list(d["affected_statistical_trials"]),
                "defective_economics": {
                    "point_value_usd": d["old_point_value_usd"],
                    "tick_value_usd": d["old_tick_value_usd"],
                },
                "corrected_economics": {
                    "point_value_usd": d["corrected_point_value_usd"],
                    "tick_value_usd": d["corrected_tick_value_usd"],
                    "price_scale": 0.01,
                },
                "source_definition_fields_zn": d["source_definition_fields_zn"],
                "correction_rule": str(sources.correction["correction_rule"]),
                "invariance_checks": dict(sources.correction["invariance_checks"]),
                "global_impact": (
                    "20 of the 107 hypotheses were defective and entered the joint "
                    "BH/DSR family, so the pre-correction global q-values and DSR of "
                    "every trial are superseded"
                ),
                "no_root_specific_hardcoding": True,
                "qa_gate": "scripts/phase_13_5c_contract_economics_qa.py",
            },
            action_taken=(
                "generic quote-convention rule (percent-of-par for non-sentinel "
                "main_fraction); full matrix re-executed; QA gate now blocks the "
                "research matrix on any point-value mismatch"
            ),
            resolved=True,
            resolution_commit=CORRECTED_COMMIT,
            created_at=created_at,
        ),
        FailureRecord(
            failure_id="F__SYSTEM__PRECORRECTION_MATRIX_SUPERSEDED",
            scope=FailureScope.SYSTEM,
            failure_class=FailureClass.SUPERSEDED_RESULT,
            failure_code="phase_13_5c_precorrection_checkpoint_superseded",
            summary=(
                f"Commit {SUPERSEDED_COMMIT} is the pre-correction Phase 13.5C "
                f"checkpoint. It is retained in git history and SUPERSEDED for "
                f"performance interpretation by {CORRECTED_COMMIT}."
            ),
            mechanism=str(sources.correction["superseded_note"]),
            evidence={
                "superseded_commit": SUPERSEDED_COMMIT,
                "corrected_commit": CORRECTED_COMMIT,
                "provenance_commit": PROVENANCE_COMMIT,
                "bh_q_values_changed": sources.correction[
                    "global_statistics_superseded"
                ]["bh_q_values_changed"],
                "dsr_values_changed": sources.correction[
                    "global_statistics_superseded"
                ]["dsr_values_changed"],
                "corrected_verdict_counts": dict(
                    sources.correction["corrected_verdict_counts"]
                ),
                "unique_trial_count_unchanged": int(sources.correction["unique_trial_count"]),
                "candidate_manifest_fingerprint_unchanged": bool(
                    sources.correction["invariance_checks"][
                        "candidate_manifest_fingerprint_unchanged"
                    ]
                ),
            },
            action_taken=(
                "the pre-correction canonical experiments are imported as historical "
                "lineage with explicit SUPERSEDES / CORRECTS edges; nothing is deleted"
            ),
            resolved=True,
            resolution_commit=CORRECTED_COMMIT,
            created_at=created_at,
        ),
    ]


def _sensitivity_evidence(
    sources: Phase135cSources, canonical_by_key: dict[tuple[str, str], str]
) -> list[SensitivityEvidence]:
    """Degraded-vendor-day reruns: robustness evidence, never a new hypothesis."""
    out: list[SensitivityEvidence] = []
    for row in sources.sensitivity:
        root, family = row["root"], row["family"]
        identity = canonical_by_key.get((root, family))
        if identity is None:
            raise Phase135cArtifactError(
                f"sensitivity row references unknown canonical trial {root}/{family}"
            )
        out.append(
            SensitivityEvidence(
                evidence_id=f"SENS__{root}__{family.upper()}__DEGRADED_DAYS_EXCLUDED",
                experiment_identity=identity,
                kind="degraded_vendor_days_excluded",
                baseline_verdict=RegistryVerdict(row["canonical_verdict"]),
                rerun_verdict=RegistryVerdict(row["excluded_verdict"])
                if row["excluded_verdict"] in RegistryVerdict.__members__
                else RegistryVerdict.NOT_ADJUDICATED,
                verdict_changed=row["verdict_changed"].strip().lower() == "true",
                metrics={
                    "canonical_daily_sharpe": _f(row["canonical_daily_sharpe"]),
                    "canonical_net_pnl_usd": _f(row["canonical_net_pnl_usd"]),
                    "excluded_daily_sharpe": _f(row["excluded_daily_sharpe"]),
                    "excluded_net_pnl_usd": _f(row["excluded_net_pnl_usd"]),
                    "daily_sharpe_delta": _f(row["daily_sharpe_delta"]),
                    "net_pnl_delta": _f(row["net_pnl_delta"]),
                    "degraded_vendor_dates": list(
                        sources.report["data_quality_sensitivity"]["degraded_vendor_dates"]
                    ),
                },
                in_bh_fdr_denominator=False,
            )
        )
    return out


def _cross_market_evidence(
    sources: Phase135cSources, canonical_by_key: dict[tuple[str, str], str]
) -> list[CrossMarketEvidenceRecord]:
    """Descriptive aggregation over per-root canonical results -- NOT trials."""
    out: list[CrossMarketEvidenceRecord] = []
    for row in sources.cross_market:
        family = row["family"]
        refs = tuple(
            identity
            for (root, fam), identity in sorted(canonical_by_key.items())
            if fam == family
        )
        out.append(
            CrossMarketEvidenceRecord(
                evidence_id=f"XMKT__{family.upper()}",
                strategy_family=family,
                status=row["status"],
                n_roots=int(row["n_roots"]),
                n_positive_roots=int(row["n_positive_roots"]),
                max_single_root_pnl_share=float(row["max_single_root_pnl_share"]),
                herfindahl_pnl=float(row["herfindahl_pnl"]),
                concentrated_in_one_root=row["concentrated_in_one_root"].strip().lower()
                == "true",
                referenced_experiment_identities=refs,
            )
        )
    return out


def _assert_counts(experiments: list[ExperimentRecord]) -> None:
    authoritative = [e for e in experiments if e.phase == PHASE]
    canonical = [e for e in authoritative if e.trial_role is TrialRole.CANONICAL]
    neighbour = [e for e in authoritative if e.trial_role is TrialRole.NEIGHBOUR]
    if len(canonical) != EXPECTED_CANONICAL:
        raise Phase135cArtifactError(
            f"built {len(canonical)} canonical experiments, expected {EXPECTED_CANONICAL}"
        )
    if len(neighbour) != EXPECTED_NEIGHBOURS:
        raise Phase135cArtifactError(
            f"built {len(neighbour)} neighbour experiments, expected {EXPECTED_NEIGHBOURS}"
        )
    if len(authoritative) != EXPECTED_UNIQUE_TRIALS:
        raise Phase135cArtifactError(
            f"built {len(authoritative)} unique statistical hypotheses, expected "
            f"{EXPECTED_UNIQUE_TRIALS}"
        )
    identities = {e.experiment_identity for e in experiments}
    if len(identities) != len(experiments):
        raise Phase135cArtifactError("experiment identities are not unique")
    _assert_provenance(experiments)


def _assert_provenance(experiments: list[ExperimentRecord]) -> None:
    """Refuse to write provenance no committed artifact supports (section 3/4).

    This is the Phase 14.1 blocker turned into a gate: a neighbour must never
    silently inherit the canonical trial's schedule hash or report fingerprint.
    """
    by_identity = {e.experiment_identity: e for e in experiments}
    canonical = [e for e in experiments if e.trial_role is TrialRole.CANONICAL]
    neighbours = [e for e in experiments if e.trial_role is TrialRole.NEIGHBOUR]

    corrected_canonical = [e for e in canonical if e.phase == PHASE]
    missing = [
        e.experiment_id for e in corrected_canonical
        if not e.target_schedule_hash or not e.report_fingerprint
    ]
    if missing:
        raise Phase135cArtifactError(
            f"canonical experiments missing their supported provenance: {missing}"
        )

    fabricated = [e.experiment_id for e in neighbours if e.report_fingerprint]
    if fabricated:
        raise Phase135cArtifactError(
            "no per-neighbour ValidationReport exists; refusing to record a "
            f"report_fingerprint for {fabricated}"
        )

    with_schedule = [e for e in neighbours if e.target_schedule_hash]
    if len(with_schedule) != EXPECTED_NEIGHBOURS_WITH_SCHEDULE_HASH:
        raise Phase135cArtifactError(
            f"{len(with_schedule)} neighbours carry a target_schedule_hash, expected "
            f"{EXPECTED_NEIGHBOURS_WITH_SCHEDULE_HASH}: no committed artifact records a "
            "neighbour's headline-window schedule, so it must stay NULL"
        )
    inherited = sorted(
        e.experiment_id for e in neighbours
        if e.parent_experiment_identity
        and e.target_schedule_hash is not None
        and by_identity[e.parent_experiment_identity].target_schedule_hash
        == e.target_schedule_hash
    )
    if inherited:
        raise Phase135cArtifactError(
            f"neighbours inherited the canonical target_schedule_hash: {inherited}"
        )
    shared = sorted(
        e.experiment_id for e in neighbours
        if e.parent_experiment_identity
        and e.feature_spec_fingerprint
        == by_identity[e.parent_experiment_identity].feature_spec_fingerprint
        and list(e.strategy_spec_json["feature_fingerprints"])
        != list(
            by_identity[e.parent_experiment_identity].strategy_spec_json[
                "feature_fingerprints"
            ]
        )
    )
    if shared:
        raise Phase135cArtifactError(
            f"neighbour feature_spec_fingerprint disagrees with its own feature "
            f"fingerprints: {shared}"
        )
    for e in neighbours:
        derived = fingerprint(
            "featset1",
            {"features": list(e.strategy_spec_json["feature_fingerprints"])},
        )
        if e.feature_spec_fingerprint != derived:
            raise Phase135cArtifactError(
                f"{e.experiment_id}: feature_spec_fingerprint is not the featset "
                "fingerprint of its own feature fingerprints"
            )
    superseded = [e for e in canonical if e.phase == PRECORRECTION_PHASE]
    carried = sorted(
        e.experiment_id for e in superseded
        if e.target_schedule_hash is not None or e.report_fingerprint is not None
    )
    if carried:
        raise Phase135cArtifactError(
            "pre-correction rows must not carry the corrected run's provenance: "
            f"{carried}"
        )


def import_phase_13_5c(
    registry: ExperimentRegistry, repo_root: Path
) -> tuple[ImportBundle, dict[str, int]]:
    """Import the corrected Phase 13.5C history, transactionally and idempotently."""
    bundle = build_bundle(repo_root)
    counts = registry.apply_bundle(bundle)
    return bundle, counts

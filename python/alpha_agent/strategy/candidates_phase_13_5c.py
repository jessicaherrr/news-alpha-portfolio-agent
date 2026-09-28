"""Phase 13.5C -- FROZEN candidate manifest.

Materialised and committed **before** any real-market performance is observed
(prompt sections 4, 22). It records, for every canonical strategy x root trial:

* the canonical :class:`~alpha_agent.strategy.spec.StrategySpec` fingerprint,
* the canonical parameters,
* a PRE-DECLARED parameter neighbourhood (never grown after results),
* the feature fingerprints,
* root applicability,
* the execution / risk / cost assumptions actually used by the 13.5C driver,
* the multiple-testing family id.

Canonical trials = 20 baseline (4 families x 5 roots) + 1 Silver Bullet (NQ) = 21.
Silver Bullet on ES/CL/GC/ZN is ``NOT_EVALUATED`` -- the Phase 12 benchmark was
frozen as an NQ hypothesis including its NQ session/window semantics; a
root-specific window is a NEW hypothesis for a later phase.

Nothing in this module reads market data or computes a statistic.
"""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.strategy.baselines.factories import (
    make_breakout_spec,
    make_ma_trend_spec,
    make_mean_reversion_spec,
    make_tsmom_spec,
)
from alpha_agent.strategy.baselines.params import (
    BreakoutParams,
    MaTrendParams,
    MeanReversionParams,
    TsmomParams,
)
from alpha_agent.strategy.baselines.silver_bullet import (
    SILVER_BULLET_BENCHMARK,
    make_silver_bullet_spec,
)

PHASE = "13.5C"
MANIFEST_SCHEMA_VERSION = "phase-13-5c-candidate-manifest/1"

ROOTS: tuple[str, ...] = ("ES", "NQ", "CL", "GC", "ZN")

# --- signal / execution cadence (prompt correction 1) ----------------------
# The 4 baselines: daily SIGNAL (causal trading_day aggregation of the 1m
# forward-adjusted continuous) -> daily decision stamped at that trading day's
# last eligible 1m event -> C++ executes on NATIVE 1-MINUTE raw-contract bars.
# Silver Bullet: native 1m signal + native 1m execution.
SIGNAL_CADENCE = {
    "tsmom": "daily_trading_day",
    "ma_trend": "daily_trading_day",
    "breakout": "daily_trading_day",
    "mean_reversion": "daily_trading_day",
    "silver_bullet": "native_1m",
}
EXECUTION_CADENCE = "native_1m_raw_contract"       # all families

# --- execution / risk / cost assumptions (what the driver actually uses) ----
EXECUTION_ASSUMPTIONS = {
    "bars": "native 1-minute RAW_CONTRACT (layer 2); C++ ActiveContractResolver",
    "commission_per_contract_usd": 2.0,
    "slippage_ticks": 0.0,
    "spread_ticks": 0.0,
    "latency_bars": 0,
    "fill_timing": "decision at bar T -> fill at the next eligible 1m event (T+1+latency)",
    "official_pnl_source": "cpp_portfolio_accountant_daily_equity_trace",
}
# TRUTHFUL risk identity (prompt correction 6): the frozen reference CLI
# `quant_backtest_targets_csv` constructs `quant::PassThroughRiskManager`
# (cpp/apps/backtest_targets_csv.cpp: "frozen reference path (Phase 19 wires
# RiskConfig)"). Phase 08 HARD RISK LIMITS ARE NOT ENFORCED on this path.
RISK_ASSUMPTIONS = {
    "risk_manager": "PassThroughRiskManager",
    "cpp_reference_cli": "quant_backtest_targets_csv",
    "hard_risk_limits_enforced": False,
    "limitation": (
        "Phase 08 MaxContractsRiskManager / hard risk limits are NOT wired into "
        "the Phase 13 / 13.5C CLI execution path; RiskConfig wiring is deferred to "
        "Phase 19 (pybind boundary). Every target is passed through unchanged. "
        "The candidate `size` is a small integer target (1) by construction."
    ),
}
COST_SCENARIOS = (
    {"label": "baseline_1_0x", "kind": "multiplier", "multiplier": 1.0},
    {"label": "stress_1_5x", "kind": "multiplier", "multiplier": 1.5},
    {"label": "stress_2_0x", "kind": "multiplier", "multiplier": 2.0},
)

MT_FAMILY_GLOBAL = "phase_13_5c.all"

SILVER_BULLET_NOT_EVALUATED_ROOTS = ("ES", "CL", "GC", "ZN")
SB_NOT_EVALUATED_REASON = "benchmark_not_predeclared_for_root"


# ==========================================================================
# Canonical parameters -- textbook values, frozen BEFORE any performance.
# Baseline horizons/windows are in TRADING DAYS (the signal cadence).
# ==========================================================================
CANONICAL_PARAMS: dict[str, dict] = {
    "tsmom": {"fast_horizon": 20, "slow_horizon": 120, "size": 1},
    "ma_trend": {"fast_window": 50, "slow_window": 200, "size": 1},
    "breakout": {"lookback": 55, "size": 1},
    "mean_reversion": {"zscore_window": 20, "entry_z": 2.0, "exit_z": 0.5, "size": 1},
}

# PRE-DECLARED neighbourhood deltas -- one parameter varied at a time, +/- one
# grid step, every value inside the declared param_grid_ranges
# (strategy/baselines/families.py). NOT grown after results.
NEIGHBOUR_DELTAS: dict[str, tuple[dict, ...]] = {
    "tsmom": (
        {"fast_horizon": 10},
        {"fast_horizon": 40},
        {"slow_horizon": 90},
        {"slow_horizon": 180},
    ),
    "ma_trend": (
        {"fast_window": 25},
        {"fast_window": 75},
        {"slow_window": 150},
        {"slow_window": 250},
    ),
    "breakout": (
        {"lookback": 35},
        {"lookback": 85},
    ),
    "mean_reversion": (
        {"zscore_window": 15},
        {"zscore_window": 30},
        {"entry_z": 1.5},
        {"entry_z": 2.5},
        {"exit_z": 0.25},
        {"exit_z": 0.75},
    ),
}

# Silver Bullet (NQ only) -- neighbours over declared SILVER_BULLET_GRID_RANGES.
# The frozen session window / sweep geometry / FVG / exits are UNCHANGED here;
# only displacement strength, retracement depth and the liquidity lookback vary,
# all predeclared, all inside the declared ranges.
SB_NEIGHBOUR_DELTAS: tuple[dict, ...] = (
    {"displacement_atr_multiple": 1.25},
    {"displacement_atr_multiple": 2.0},
    {"retracement_fraction": 0.3},
    {"retracement_fraction": 0.7},
    {"liquidity_lookback": 15},
    {"liquidity_lookback": 30},
)

_BASELINE_FACTORY = {
    "tsmom": (make_tsmom_spec, TsmomParams),
    "ma_trend": (make_ma_trend_spec, MaTrendParams),
    "breakout": (make_breakout_spec, BreakoutParams),
    "mean_reversion": (make_mean_reversion_spec, MeanReversionParams),
}
BASELINE_FAMILIES = tuple(_BASELINE_FACTORY)


def baseline_params(family: str, root: str) -> dict:
    """Full canonical parameter dict (factory kwargs) for one baseline x root."""
    return {"root_symbol": root, **CANONICAL_PARAMS[family]}


def baseline_neighbour_params(family: str, root: str) -> tuple[dict, ...]:
    base = baseline_params(family, root)
    return tuple({**base, **delta} for delta in NEIGHBOUR_DELTAS[family])


def baseline_spec(family: str, root: str):
    factory, model = _BASELINE_FACTORY[family]
    return factory(model(**baseline_params(family, root)))


def silver_bullet_params(root: str = "NQ") -> dict:
    p = SILVER_BULLET_BENCHMARK.model_dump()
    p["root_symbol"] = root
    return p


def silver_bullet_neighbour_params() -> tuple[dict, ...]:
    base = silver_bullet_params("NQ")
    return tuple({**base, **delta} for delta in SB_NEIGHBOUR_DELTAS)


def _feature_fingerprints(spec) -> list[str]:
    return [
        "feat1:" + hashlib.sha256(decl.spec.canonical_json().encode()).hexdigest()
        for decl in spec.features
    ]


def feature_fingerprints(spec) -> list[str]:
    """The frozen Phase 13.5C feature-fingerprint rule, exposed for consumers
    that must derive a *variant's own* feature set with exactly this rule.

    Additive alias only -- the rule itself is unchanged and this changes no
    manifest content or fingerprint.
    """
    return _feature_fingerprints(spec)


def spec_for_params(family_key: str, params: dict):
    """Deterministically rebuild the ``StrategySpec`` for any frozen variant --
    canonical or predeclared neighbour -- from its full parameter dict.

    Configuration reconstruction only: it reads no market data, runs no
    backtest, and inspects no performance. The factories and parameter models
    are the frozen Phase 10/11/12 ones, so a reconstructed spec's fingerprint
    must equal the fingerprint frozen in the candidate manifest.
    """
    if family_key == "silver_bullet":
        from alpha_agent.strategy.baselines.silver_bullet import SilverBulletParams

        return make_silver_bullet_spec(SilverBulletParams(**params))
    factory, model = _BASELINE_FACTORY[family_key]
    return factory(model(**params))


# ==========================================================================
# manifest models
# ==========================================================================
class CandidateTrial(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    family_key: str
    root_symbol: str
    canonical_params: dict
    strategy_fingerprint: str
    strategy_id: str
    feature_fingerprints: tuple[str, ...]
    neighbour_params: tuple[dict, ...]
    neighbour_fingerprints: tuple[str, ...]
    signal_cadence: str
    execution_cadence: str = EXECUTION_CADENCE
    multiple_testing_family_id: str
    applicability: str = "EVALUATED"
    applicability_reason: str = ""


class CandidateManifest(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = MANIFEST_SCHEMA_VERSION
    phase: str = PHASE
    roots: tuple[str, ...] = ROOTS
    trials: tuple[CandidateTrial, ...]
    not_evaluated: tuple[dict, ...]          # {family_key, root_symbol, reason}
    execution_assumptions: dict = Field(default_factory=lambda: dict(EXECUTION_ASSUMPTIONS))
    risk_assumptions: dict = Field(default_factory=lambda: dict(RISK_ASSUMPTIONS))
    cost_scenarios: tuple[dict, ...] = COST_SCENARIOS
    global_multiple_testing_family_id: str = MT_FAMILY_GLOBAL
    # Part 1.1 correction: the UNIQUE BH/FDR trial family = every canonical +
    # every predeclared neighbour strategy fingerprint, de-duplicated by
    # (strategy_fingerprint, root). Cross-market roots are the SAME per-root
    # results summarised -- NOT additional trials. The data-quality sensitivity
    # pass is a robustness rerun of the same hypotheses -- NOT trials.
    bh_fdr_trial_family: dict = Field(default_factory=dict)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    def manifest_fingerprint(self) -> str:
        return "p135c-manifest:" + hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()

    def n_canonical_trials(self) -> int:
        return len(self.trials)


def build_candidate_manifest() -> CandidateManifest:
    trials: list[CandidateTrial] = []

    for family in BASELINE_FAMILIES:
        for root in ROOTS:
            spec = baseline_spec(family, root)
            neigh = baseline_neighbour_params(family, root)
            factory, model = _BASELINE_FACTORY[family]
            neigh_fps = tuple(
                strategy_fingerprint(factory(model(**np))) for np in neigh
            )
            trials.append(
                CandidateTrial(
                    family_key=family,
                    root_symbol=root,
                    canonical_params=baseline_params(family, root),
                    strategy_fingerprint=strategy_fingerprint(spec),
                    strategy_id=spec.strategy_id,
                    feature_fingerprints=tuple(_feature_fingerprints(spec)),
                    neighbour_params=neigh,
                    neighbour_fingerprints=neigh_fps,
                    signal_cadence=SIGNAL_CADENCE[family],
                    multiple_testing_family_id=f"phase_13_5c.{family}",
                )
            )

    # Silver Bullet -- NQ only.
    sb_spec = make_silver_bullet_spec(SILVER_BULLET_BENCHMARK)
    sb_neigh = silver_bullet_neighbour_params()
    from alpha_agent.strategy.baselines.silver_bullet import SilverBulletParams

    sb_neigh_fps = tuple(
        strategy_fingerprint(make_silver_bullet_spec(SilverBulletParams(**np)))
        for np in sb_neigh
    )
    trials.append(
        CandidateTrial(
            family_key="silver_bullet",
            root_symbol="NQ",
            canonical_params=silver_bullet_params("NQ"),
            strategy_fingerprint=strategy_fingerprint(sb_spec),
            strategy_id=sb_spec.strategy_id,
            feature_fingerprints=tuple(_feature_fingerprints(sb_spec)),
            neighbour_params=sb_neigh,
            neighbour_fingerprints=sb_neigh_fps,
            signal_cadence=SIGNAL_CADENCE["silver_bullet"],
            multiple_testing_family_id="phase_13_5c.silver_bullet",
        )
    )

    not_evaluated = tuple(
        {
            "family_key": "silver_bullet",
            "root_symbol": r,
            "reason": SB_NOT_EVALUATED_REASON,
            "note": (
                "SILVER_BULLET_BENCHMARK is a frozen NQ hypothesis (NQ session / "
                "window semantics). A root-specific Silver Bullet window is a NEW "
                "hypothesis with its own StrategySpec fingerprint, for a later phase."
            ),
        }
        for r in SILVER_BULLET_NOT_EVALUATED_ROOTS
    )

    manifest = CandidateManifest(trials=tuple(trials), not_evaluated=not_evaluated)
    # recompute + assert the unique BH/FDR trial family from semantic identities
    from alpha_agent.validation.phase_13_5c_trials import unique_trial_identities

    b = unique_trial_identities(manifest)
    return manifest.model_copy(
        update={
            "bh_fdr_trial_family": {
                "family_id": MT_FAMILY_GLOBAL,
                "unique_trial_count": b.unique_trial_count,
                "canonical": b.canonical,
                "neighbour": b.neighbour,
                "ablation": b.ablation,
                "cross_market_summarised_not_trials": b.cross_market_excluded,
                "data_quality_sensitivity_in_denominator": False,
                "family_composition_fingerprint": b.identity(),
                "note": (
                    "unique statistical trials = canonical + predeclared neighbours, "
                    "de-duplicated by (strategy_fingerprint, root). Cross-market roots "
                    "are the same per-root canonical results summarised, not new "
                    "hypotheses. BH/FDR, q-values, DSR effective-trial count and the "
                    "MultipleTestingFamily fingerprint all use this count."
                ),
            }
        }
    )


_PERFORMANCE_KEYS = {
    "pnl", "net_pnl", "gross_pnl", "sharpe", "daily_sharpe", "annualized_sharpe",
    "p_value", "q_value", "dsr", "verdict", "return", "returns", "drawdown",
    "reason_codes", "fills", "trades",
}


def assert_no_performance_fields(manifest: CandidateManifest) -> None:
    """The manifest is frozen BEFORE performance: no result field may appear."""
    blob = manifest.canonical_json().lower()
    hits = sorted(k for k in _PERFORMANCE_KEYS if f'"{k}"' in blob)
    if hits:
        raise AssertionError(f"candidate manifest leaks performance keys: {hits}")


def freeze_candidate_manifest(path: str) -> CandidateManifest:
    """Write the deterministic frozen manifest + its fingerprint to ``path``."""
    from pathlib import Path

    m = build_candidate_manifest()
    assert_no_performance_fields(m)
    payload = {
        "manifest_fingerprint": m.manifest_fingerprint(),
        "n_canonical_trials": m.n_canonical_trials(),
        "manifest": m.model_dump(mode="json"),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return m

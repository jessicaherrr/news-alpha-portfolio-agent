"""Closed enumerations for the Phase 13 reliability-aware validation framework.

Every semantic choice in the framework is one of these values -- there are no
free-form strings driving behaviour and no arbitrary callables. All enums are
plain ``str`` enums so they serialise stably into a validation fingerprint.
"""
from __future__ import annotations

from enum import Enum


class SplitRole(str, Enum):
    """The role a chronological data window plays. A closed set (section 5)."""

    TRAIN = "train"                # discovery / parameter choice / hypothesis generation
    VALIDATION = "validation"      # research validation before the strategy is frozen
    LOCKED_HOLDOUT = "locked_holdout"  # one final evaluation, never inspected earlier


class Verdict(str, Enum):
    """The only three outcomes of a validation run (section 19). REJECT is a
    normal, successful system outcome."""

    PASS = "PASS"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


class ReasonCode(str, Enum):
    """Typed reason codes attached to a verdict. No free text drives semantics
    (section 22)."""

    # -- INCONCLUSIVE (insufficient evidence, section 8) --
    INSUFFICIENT_OOS_DAYS = "insufficient_oos_days"
    INSUFFICIENT_TRADES = "insufficient_trades"
    INSUFFICIENT_FOLDS = "insufficient_folds"
    INSUFFICIENT_NONZERO_OBS = "insufficient_nonzero_observations"
    NO_DAILY_TRACE = "no_daily_trace"
    REGIME_NOT_EVALUATED = "regime_not_evaluated"
    CROSS_MARKET_NOT_EVALUATED = "cross_market_not_evaluated"
    PARAMETER_STABILITY_NOT_EVALUATED = "parameter_stability_not_evaluated"
    ABLATION_NOT_EVALUATED = "ablation_not_evaluated"

    # -- REJECT (clear statistical failure) --
    NULL_NOT_REJECTED = "null_hypothesis_not_rejected"
    FDR_NOT_SIGNIFICANT = "fdr_qvalue_above_threshold"
    DSR_BELOW_THRESHOLD = "deflated_sharpe_below_threshold"
    NEGATIVE_OOS_NET_PNL = "negative_oos_net_pnl"
    FOLD_CONSISTENCY_FAILED = "fold_consistency_below_threshold"
    COST_STRESS_FAILED = "cost_stress_degradation_exceeds_limit"
    PARAMETER_UNSTABLE = "parameter_neighbourhood_unstable"
    REGIME_CONCENTRATED = "performance_concentrated_in_one_regime"
    CROSS_MARKET_CONCENTRATED = "performance_concentrated_in_one_root"

    # -- PASS --
    ALL_GATES_SATISFIED = "all_required_gates_satisfied"


class BootstrapMethod(str, Enum):
    """Time-series-aware resampling (section 10). Naive IID is deliberately absent
    as a main-path option."""

    MOVING_BLOCK = "moving_block"
    STATIONARY = "stationary"          # Politis & Romano (1994)


class NullMethod(str, Enum):
    """Market-appropriate empirical null tests (section 11)."""

    # Causal null: forward-only schedule shift by a FIXED family max shift K,
    # within an allowed causal segment, from a COMMON-SUPPORT control schedule
    # (rows too close to a segment end to survive any k <= K are pre-dropped from
    # BOTH the control and every shifted replicate). No wrap. It is
    # DIAGNOSTIC_ONLY -- the ReliabilityPolicy never gates on it (Phase 13.2).
    SCHEDULE_TIME_SHIFT = "schedule_time_shift"
    # Research-only permutation null: a circular rotation of the schedule that
    # DOES wrap late rows to earlier timestamps. Never the default; kept explicit
    # so it is not silently conflated with the causal shift (Phase 13.1 section 9).
    CIRCULAR_SCHEDULE_PERMUTATION = "circular_schedule_permutation"
    # The OFFICIAL gating null: block bootstrap of the mean-removed daily returns.
    CENTERED_BLOCK_BOOTSTRAP = "centered_block_bootstrap"


class NullRole(str, Enum):
    """Whether a null result gates the verdict or is diagnostic only
    (Phase 13.2 section 4)."""

    GATING = "gating"
    DIAGNOSTIC = "diagnostic"


class RegimeKind(str, Enum):
    """Predeclared causal regime partitions (section 17). Thresholds, if learned,
    come only from allowed past/train data."""

    VOLATILITY = "volatility"
    TREND = "trend"
    SESSION = "session"


class CostScenarioKind(str, Enum):
    """How a cost-stress scenario scales the reference execution assumptions
    (section 15). Every scenario reruns the C++ engine path."""

    MULTIPLIER = "multiplier"          # scale commission + slippage + spread by a factor
    ABSOLUTE = "absolute"              # explicit commission / slippage / spread values


class AblationKind(str, Enum):
    """Named, predeclared strategy ablations (section 16). Each is its own
    ``StrategySpec`` fingerprint and its own trial."""

    DROP_TIME_WINDOW = "drop_time_window"
    DROP_DISPLACEMENT = "drop_displacement"
    DROP_RETRACEMENT = "drop_retracement"
    DROP_SLOW_HORIZON = "drop_slow_horizon"
    SINGLE_HORIZON = "single_horizon"
    CUSTOM = "custom"


class PortfolioStatePolicy(str, Enum):
    """How an out-of-sample walk-forward fold starts (sections 6, 7). The
    reliability-safe default carries no economic position from training."""

    FLAT_START = "flat_start"          # documented default: no inherited position/PnL
    CARRY_WARMUP_ONLY = "carry_warmup_only"  # historical feature warmup, still flat book


class EvidenceStatus(str, Enum):
    """Whether an optional evidence family was actually evaluated (section 17/18)."""

    EVALUATED = "evaluated"
    NOT_EVALUATED = "not_evaluated"

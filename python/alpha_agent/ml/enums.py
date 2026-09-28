"""Closed enumerations for the Phase 15 ML meta-labeling layer.

Every semantic choice an ML experiment can make is a closed set. Free text may
supplement a typed code but never *is* one, and an LLM never picks a value here:
the manifest is frozen before any performance is observed.
"""
from __future__ import annotations

from enum import Enum


class MetaLabelAction(str, Enum):
    """The MVP meta-label action space -- deliberately closed (prompt 15A s.4).

    The model may NEVER invent a position size. ``TAKE`` preserves the primary
    strategy's own target; ``SKIP`` substitutes the neutral target. A predicted
    probability is stored for calibration/audit and is never turned into
    continuous leverage.
    """

    TAKE = "TAKE"
    SKIP = "SKIP"


class LabelKind(str, Enum):
    """How the supervised target is defined.

    ``PRIMARY_EPISODE_NET_PNL_SIGN`` is the frozen Phase 15 definition: the sign
    of the C++ Fill-derived net PnL of the primary strategy's own closed trade
    episode. It introduces no arbitrary future-return horizon and needs no
    change to primary strategy semantics.
    """

    PRIMARY_EPISODE_NET_PNL_SIGN = "PRIMARY_EPISODE_NET_PNL_SIGN"


class EpisodeExclusionReason(str, Enum):
    """Why a primary episode produced no labelled event. Never silently dropped."""

    UNTERMINATED_AT_CORPUS_END = "UNTERMINATED_AT_CORPUS_END"
    NO_ATTRIBUTED_FILLS = "NO_ATTRIBUTED_FILLS"
    FEATURES_MISSING_AT_DECISION = "FEATURES_MISSING_AT_DECISION"
    OUTSIDE_DEVELOPMENT_CORPUS = "OUTSIDE_DEVELOPMENT_CORPUS"


class MLModelFamily(str, Enum):
    """The predeclared, deliberately small model set (prompt 15A s.8).

    No AutoML, no neural network, no adaptive family expansion. ``SYNTHETIC_*``
    is a deterministic pure-numpy stand-in used ONLY by the software tests; the
    manifest refuses it for a research candidate.
    """

    LOGISTIC_L2 = "LOGISTIC_L2"                     # sklearn LogisticRegression
    HIST_GRADIENT_BOOSTING = "HIST_GRADIENT_BOOSTING"  # sklearn HistGradientBoostingClassifier
    SYNTHETIC_DETERMINISTIC = "SYNTHETIC_DETERMINISTIC"  # tests only -- never research


class ModelSelectionObjective(str, Enum):
    """The predeclared INNER-loop selection objective.

    Closed on purpose. The inner loop never selects on economics -- only on a
    proper scoring rule -- so the C++ economic evaluation stays a genuinely
    out-of-sample question. The objective is part of the frozen model-search
    PROCEDURE and therefore part of pre-run experiment identity: two studies that
    search the same grid under different objectives are different hypotheses.
    """

    MEAN_INNER_FOLD_LOG_LOSS_MINIMISED = "mean_inner_fold_log_loss_minimised"


class ModelSelectionTieBreak(str, Enum):
    """How the inner loop breaks an objective tie. Predeclared, never a result.

    ``FROZEN_GRID_DECLARATION_ORDER_FIRST``: the earliest point of the frozen
    grid wins. Deterministic and knowable before any fit, which is exactly why it
    belongs in identity rather than in provenance -- a different tie-break can
    select a different configuration from the same grid on the same data, so it
    is a different search procedure.
    """

    FROZEN_GRID_DECLARATION_ORDER_FIRST = "frozen_grid_declaration_order_first"


class PreprocessingKind(str, Enum):
    """Fitted transforms. Every one of these is fit on a TRAIN fold only."""

    NONE = "NONE"
    MEDIAN_IMPUTE = "MEDIAN_IMPUTE"
    MEDIAN_IMPUTE_STANDARDIZE = "MEDIAN_IMPUTE_STANDARDIZE"


class RegimeSpecKind(str, Enum):
    """A causal regime representation.

    ``DETERMINISTIC_CAUSAL_*`` needs no fitting at all. ``LEARNED_*`` is fitted,
    and therefore may only ever be fit on a train fold.
    """

    NONE = "NONE"
    DETERMINISTIC_CAUSAL_VOL_TREND = "DETERMINISTIC_CAUSAL_VOL_TREND"
    LEARNED_TRAIN_FOLD_KMEANS = "LEARNED_TRAIN_FOLD_KMEANS"


class MLTrialRole(str, Enum):
    """The role a Phase 15 configuration plays in its multiple-testing family."""

    HEADLINE = "HEADLINE"          # a BH/FDR trial
    ABLATION = "ABLATION"          # a BH/FDR trial (a genuinely new configuration)
    PLACEBO = "PLACEBO"            # a null control, NOT in the denominator
    DIAGNOSTIC = "DIAGNOSTIC"      # descriptive rerun, NOT in the denominator


class MLRefusalReason(str, Enum):
    """Why a training scope was REFUSED rather than fitted (prompt 15A.1 s.5).

    A refusal is a typed, first-class outcome recorded as evidence. The event
    gates are predeclared and frozen -- ``min_train_events = 100`` and
    ``min_test_events = 20`` -- and are never lowered after real counts are seen.
    """

    INSUFFICIENT_TRAIN_EVENTS = "INSUFFICIENT_TRAIN_EVENTS"
    INSUFFICIENT_TEST_EVENTS = "INSUFFICIENT_TEST_EVENTS"
    NO_USABLE_INNER_FOLD = "NO_USABLE_INNER_FOLD"


class FoldKind(str, Enum):
    OUTER = "OUTER"
    INNER = "INNER"


class DevelopmentSplitRole(str, Enum):
    """Phase 15's OWN truthful data roles (prompt 15A s.2).

    Phase 13's history is unchanged, but Phase 15 is a NEW research program
    created AFTER Phase 13 results were observed, so 2023-2024 is NOT an
    untouched out-of-sample set for it.
    """

    PHASE_15_DEVELOPMENT_CORPUS = "PHASE_15_DEVELOPMENT_CORPUS"   # 2018-01-01..2024-12-31
    LOCKED_FINAL_HOLDOUT = "LOCKED_FINAL_HOLDOUT"                 # 2025, never accessed

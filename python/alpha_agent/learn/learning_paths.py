"""Phase 4 -- Learning Paths: a small, curated, hand-authored ordered sequence
of Concepts/Strategies for a beginner. Fixed content, not generated; every
step's `ref_id` is validated against the real Concept/Strategy libraries in
`test_phase4_learn_content.py` so a path can never silently point at a
retired id."""
from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["LEARNING_PATHS", "LearningPath", "LearningPathStep", "get_learning_path", "list_learning_paths"]


class LearningPathStep(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: str  # "concept" | "strategy"
    ref_id: str
    why: str


class LearningPath(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    path_id: str
    title: str
    summary: str
    steps: tuple[LearningPathStep, ...] = Field(default_factory=tuple)


_PATHS: tuple[LearningPath, ...] = (
    LearningPath(
        path_id="idea_to_execution",
        title="From Economic Idea to Real Execution",
        summary="Follow one strategy family all the way from its economic story to the C++ engine that trades it.",
        steps=(
            LearningPathStep(kind="strategy", ref_id="tsmom", why="Start with the clearest single mechanism: persistent directional drift."),
            LearningPathStep(kind="concept", ref_id="look_ahead_bias", why="Understand the one rule every execution path must obey."),
            LearningPathStep(kind="concept", ref_id="contract_economics", why="See how a fill turns into real dollars."),
            LearningPathStep(kind="concept", ref_id="positive_oos_pnl", why="The most basic bar a strategy must clear."),
        ),
    ),
    LearningPath(
        path_id="why_backtests_lie",
        title="Why Most Backtests Lie: Validation Rigor",
        summary="The gates that separate a real, repeatable edge from an overfit backtest.",
        steps=(
            LearningPathStep(kind="concept", ref_id="bootstrap_null", why="First question: could luck alone explain this?"),
            LearningPathStep(kind="concept", ref_id="bh_fdr", why="Testing many ideas inflates the odds of a false positive."),
            LearningPathStep(kind="concept", ref_id="dsr", why="Searching many parameters inflates the best one's Sharpe."),
            LearningPathStep(kind="concept", ref_id="parameter_stability", why="A real edge looks like a plateau, not a spike."),
            LearningPathStep(kind="concept", ref_id="cost_stress", why="Real trading costs can erase a thin edge."),
        ),
    ),
    LearningPath(
        path_id="reading_a_failed_experiment",
        title="Reading a Failed Experiment",
        summary="What to actually look at when a strategy gets REJECT or INCONCLUSIVE.",
        steps=(
            LearningPathStep(kind="concept", ref_id="walk_forward", why="Check whether the edge was consistent across time, not just total."),
            LearningPathStep(kind="concept", ref_id="regime_robustness", why="Check whether one unusual period carried the whole result."),
            LearningPathStep(kind="concept", ref_id="cross_market_evidence", why="Check whether the mechanism generalizes beyond one root."),
        ),
    ),
)

LEARNING_PATHS: dict[str, LearningPath] = {p.path_id: p for p in _PATHS}


def list_learning_paths() -> tuple[LearningPath, ...]:
    return _PATHS


def get_learning_path(path_id: str) -> LearningPath | None:
    return LEARNING_PATHS.get(path_id)

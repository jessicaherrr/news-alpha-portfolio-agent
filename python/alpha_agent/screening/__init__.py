"""Alpha Discovery campaign, Part F/G -- the research-only C++ Fast Screen
and the Freeze-before-strict-validation handoff. See
`alpha_agent.screening.fast_screen` / `.freeze` for the module-level
boundary docstrings.
"""
from __future__ import annotations

from alpha_agent.screening.fast_screen import (
    FastScreenStatus,
    FastScreenTrial,
    ResearchScreenScore,
    ResearchWindowViolation,
    assert_research_window_only,
    rank_fast_screen_trials,
    run_fast_screen,
    score_fast_screen_trial,
    select_top_k,
)
from alpha_agent.screening.freeze import (
    DEFAULT_FREEZE_DIR,
    FreezeError,
    FrozenCandidateSet,
    freeze_top_k,
    read_frozen_manifest,
    write_frozen_manifest,
)

__all__ = [
    "DEFAULT_FREEZE_DIR",
    "FastScreenStatus",
    "FastScreenTrial",
    "FreezeError",
    "FrozenCandidateSet",
    "ResearchScreenScore",
    "ResearchWindowViolation",
    "assert_research_window_only",
    "freeze_top_k",
    "rank_fast_screen_trials",
    "read_frozen_manifest",
    "run_fast_screen",
    "score_fast_screen_trial",
    "select_top_k",
    "write_frozen_manifest",
]

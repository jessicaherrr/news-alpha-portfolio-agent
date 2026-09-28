"""Alpha Discovery campaign, Part F -- Research-only Fast Screen tests (task
spec section 74).

Two layers: fast, synthetic unit tests for the scoring/ranking machinery
(no C++, no real data), and one real-data integration test
(`test_run_fast_screen_real_nq_tsmom_stays_inside_the_research_window`) that
drives the actual compiled CLI over the real already-acquired NQ 2018-2024
dataset -- restricted, and asserted to be restricted, to 2018-2022 only.
"""
from __future__ import annotations

import numpy as np
import pytest
from alpha_agent.screening.fast_screen import (
    _RESEARCH_HI_NS,
    FastScreenStatus,
    FastScreenTrial,
    ResearchWindowViolation,
    assert_research_window_only,
    rank_fast_screen_trials,
    rank_pool_members,
    run_fast_screen,
    score_fast_screen_trial,
    select_top_k,
)
from alpha_agent.validation.runner import synthetic_backtest_run

pytest_plugins = ()


def test_assert_research_window_only_passes_on_valid_and_empty_frames():
    import pandas as pd

    ok = pd.DataFrame({"ts_event_ns": [1, 2, _RESEARCH_HI_NS - 1]})
    assert_research_window_only(ok)  # no raise
    assert_research_window_only(pd.DataFrame({"ts_event_ns": []}))  # empty -- no raise


def test_assert_research_window_only_raises_on_validation_window_timestamp():
    import pandas as pd

    bad = pd.DataFrame({"ts_event_ns": [1, _RESEARCH_HI_NS]})  # exactly the boundary -- must reject
    with pytest.raises(ResearchWindowViolation):
        assert_research_window_only(bad)


def test_assert_research_window_only_raises_well_beyond_the_boundary():
    import pandas as pd

    bad = pd.DataFrame({"ts_event_ns": [_RESEARCH_HI_NS + 10_000_000_000]})
    with pytest.raises(ResearchWindowViolation):
        assert_research_window_only(bad)


# ---------------------------------------------------------------------------
# ResearchScreenScore -- synthetic BacktestRun, no C++
# ---------------------------------------------------------------------------


def test_score_bounded_0_100():
    run = synthetic_backtest_run(np.array([100.0, -50.0, 200.0, 0.0, 150.0]), n_trades=5, n_fills=10)
    score = score_fast_screen_trial(run)
    assert 0.0 <= score.total <= 100.0


def test_score_missing_data_never_creates_positive_default():
    run = synthetic_backtest_run(np.array([]), n_trades=0, n_fills=0)
    score = score_fast_screen_trial(run)
    assert score.total == 0.0
    assert "active_days" in score.missing_inputs or "daily_sharpe" in score.missing_inputs


def test_score_higher_sharpe_scores_higher_holding_other_things_similar():
    strong = synthetic_backtest_run(
        np.array([100.0] * 20 + [-10.0] * 5), n_trades=25, n_fills=50, costs_usd=50.0
    )
    weak = synthetic_backtest_run(
        np.array([10.0] * 10 + [-10.0] * 15), n_trades=25, n_fills=50, costs_usd=50.0
    )
    assert score_fast_screen_trial(strong).sharpe_component > score_fast_screen_trial(weak).sharpe_component


def test_score_high_cost_burden_scores_lower():
    low_cost = synthetic_backtest_run(np.array([100.0] * 10), n_trades=10, n_fills=10, costs_usd=10.0)
    high_cost = synthetic_backtest_run(np.array([100.0] * 10), n_trades=10, n_fills=10, costs_usd=900.0)
    assert score_fast_screen_trial(low_cost).cost_component > score_fast_screen_trial(high_cost).cost_component


def test_score_never_imports_promise_module():
    """Architectural separation (task spec section 39): ResearchScreenScore
    must never share code with the post-validation Research Promise score.
    An AST import check, not a text-grep (this module's own docstring
    legitimately NAMES that module in prose, to explain why it is distinct)."""
    import ast
    import inspect

    import alpha_agent.screening.fast_screen as mod

    tree = ast.parse(inspect.getsource(mod))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
    assert not any(m.startswith("alpha_agent.recommendation") for m in imported_modules)


def test_score_has_no_p_value_q_value_dsr_or_verdict_fields():
    run = synthetic_backtest_run(np.array([100.0, 50.0]), n_trades=2, n_fills=2)
    score = score_fast_screen_trial(run)
    dumped = score.model_dump()
    for forbidden in ("p_value", "q_value", "dsr", "verdict", "bh_q"):
        assert forbidden not in dumped


# ---------------------------------------------------------------------------
# ranking / top-K selection
# ---------------------------------------------------------------------------


def _trial(eid: str, *, status=FastScreenStatus.SCREENED, total: float | None = None) -> FastScreenTrial:
    from alpha_agent.screening.fast_screen import ResearchScreenScore

    score = None
    if status is FastScreenStatus.SCREENED:
        score = ResearchScreenScore(
            sharpe_component=total or 0.0, cost_component=0.0, drawdown_component=0.0,
            activity_component=0.0, total=total or 0.0,
        )
    return FastScreenTrial(
        experiment_identity=eid, strategy_family="tsmom", root_symbol="NQ", status=status, score=score,
    )


def test_ranking_deterministic_and_score_descending():
    trials = [_trial("A", total=50.0), _trial("B", total=90.0), _trial("C", total=70.0)]
    ranked = rank_fast_screen_trials(trials)
    assert [t.experiment_identity for t in ranked] == ["B", "C", "A"]


def test_ranking_tie_break_by_experiment_identity():
    trials = [_trial("Z", total=50.0), _trial("A", total=50.0), _trial("M", total=50.0)]
    forward = rank_fast_screen_trials(trials)
    backward = rank_fast_screen_trials(list(reversed(trials)))
    assert [t.experiment_identity for t in forward] == [t.experiment_identity for t in backward] == ["A", "M", "Z"]


def test_unsupported_trials_never_outrank_a_screened_trial():
    trials = [_trial("A", status=FastScreenStatus.UNSUPPORTED), _trial("B", total=1.0)]
    ranked = rank_fast_screen_trials(trials)
    assert ranked[0].experiment_identity == "B"


def test_pool_members_follow_the_one_fast_screen_ranking_rule():
    """News Alpha Phase F cleanup: the Discover page no longer carries its own
    sort key -- it shows `rank_pool_members`, which reuses this module's rule."""
    from types import SimpleNamespace

    members = [SimpleNamespace(experiment_identity=e) for e in ("U1", "A", "Z", "U2", "M")]
    trials = [_trial("A", total=50.0), _trial("Z", total=90.0), _trial("M", total=50.0),
              _trial("U2", status=FastScreenStatus.UNSUPPORTED)]
    ranked = [m.experiment_identity for m in rank_pool_members(members, trials)]
    screened = [t.experiment_identity for t in rank_fast_screen_trials(trials) if t.score is not None]
    assert ranked[:3] == screened == ["Z", "A", "M"]
    assert ranked[3:] == ["U1", "U2"]  # unscored members keep pool order, after every screened one


def test_select_top_k_only_returns_screened_trials():
    trials = [
        _trial("A", total=90.0), _trial("B", status=FastScreenStatus.UNSUPPORTED),
        _trial("C", total=80.0), _trial("D", status=FastScreenStatus.INSUFFICIENT_DATA),
    ]
    top2 = select_top_k(trials, 2)
    assert [t.experiment_identity for t in top2] == ["A", "C"]


def test_select_top_k_can_return_fewer_than_k():
    trials = [_trial("A", total=90.0)]
    assert len(select_top_k(trials, 5)) == 1


def test_select_top_k_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        select_top_k([_trial("A", total=1.0)], 0)


# ---------------------------------------------------------------------------
# real C++ integration (slow-ish: one real subprocess backtest)
# ---------------------------------------------------------------------------


def _real_cli_path() -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return str(root / "build" / "cpp" / "cpp" / "quant_backtest_targets_csv")


@pytest.mark.skipif(
    not __import__("pathlib").Path(_real_cli_path()).exists(), reason="compiled CLI not present in this checkout",
)
def test_run_fast_screen_real_nq_tsmom_stays_inside_the_research_window(tmp_path):
    from alpha_agent.agents.orchestrator import FamilyMember
    from alpha_agent.registry.identity import experiment_identity, parameter_variant_identity
    from alpha_agent.strategy import strategy_fingerprint
    from alpha_agent.strategy.baselines.factories import make_tsmom_spec
    from alpha_agent.strategy.baselines.params import TsmomParams
    from alpha_agent.validation.policy import ReliabilityPolicy

    params = {"fast_horizon": 20, "slow_horizon": 120, "size": 1, "root_symbol": "NQ"}
    spec = make_tsmom_spec(TsmomParams(**params))
    fp = strategy_fingerprint(spec)
    pvi = parameter_variant_identity(params)
    identity = experiment_identity(
        strategy_fingerprint=fp, strategy_family="tsmom", root_symbol="NQ",
        parameter_variant_identity=pvi, dataset_fingerprint="valdataset2:fastscreen",
        split_identity="splitplan1:fastscreen", validation_spec_fingerprint="validationprotocol1:fastscreen",
        reliability_policy_fingerprint=ReliabilityPolicy().identity(),
        execution_config_identity="execconfig1:fastscreen", cost_config_identity="costconfig1:fastscreen",
        risk_identity="riskconfig1:fastscreen", feature_spec_fingerprint="featset1:fastscreen",
    )
    member = FamilyMember(
        ordinal=0, experiment_identity=identity, strategy_fingerprint=fp, strategy_id=spec.strategy_id,
        strategy_family="tsmom", root_symbol="NQ", params=params, parameter_variant_identity=pvi,
        feature_spec_fingerprint="featset1:fastscreen", hypothesis_id="H-FASTSCREEN",
        hypothesis_title="fast screen replay", strategy_spec=spec,
        strategy_spec_json={"schema": "registry-strategy-spec/1"},
    )
    trial = run_fast_screen(member=member, cli_executable=_real_cli_path(), work_dir=tmp_path)
    assert trial.status is FastScreenStatus.SCREENED
    assert trial.score is not None
    assert 0.0 <= trial.score.total <= 100.0
    assert trial.metrics["n_days"] > 0
    # every metric this module reports must have been produced with the C++
    # engine bounded to the RESEARCH window -- proven structurally by the two
    # in-function assert_research_window_only calls never raising.

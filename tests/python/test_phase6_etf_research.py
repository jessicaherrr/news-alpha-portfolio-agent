"""Phase 6 ETF pilot -- the real end-to-end tsmom hypothesis through the
REAL, unmodified C++ engine + validation stack, over real acquired data.

Requires the compiled C++ core (build/cpp/cpp/quant_backtest_targets_csv)
and the real acquired Phase 6 raw artifacts; skips cleanly if either is
absent, matching this repo's existing convention for tests that exercise the
real C++ boundary.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from alpha_agent.etf.calendar import etf_calendar
from alpha_agent.etf.research import DEFAULT_CLI, etf_split_plan, etf_walk_forward, run_etf_research
from alpha_agent.etf.strategy_family import ETF_TSMOM_FAMILY, etf_tsmom_adapter, etf_tsmom_spec
from alpha_agent.etf.universe import PILOT_UNIVERSE
from alpha_agent.validation.report import Verdict

_REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (_REPO_ROOT / "data/raw/databento/ARCX.PILLAR").exists()
    or not (_REPO_ROOT / DEFAULT_CLI).exists(),
    reason="requires the real Phase 6 ETF raw artifacts and the compiled C++ core",
)


def test_etf_tsmom_spec_uses_the_dsl_virtual_root_not_the_bare_ticker():
    spec = etf_tsmom_spec("XLE")
    assert spec.root_symbol == "EXLE"


def test_etf_tsmom_spec_features_are_session_continuous():
    """Weekend/holiday gaps in a real daily series are normal calendar
    structure, not a data defect -- must never trip RESET_ON_GAP (the exact
    class of bug behind the real, documented Phase 15B incident)."""
    from alpha_agent.features.enums import SessionPolicy

    spec = etf_tsmom_spec("XLE")
    for decl in spec.features:
        assert decl.spec.session_policy == SessionPolicy.CONTINUOUS


def test_etf_tsmom_family_label_is_distinct_from_futures_tsmom():
    assert ETF_TSMOM_FAMILY == "etf_tsmom"
    assert ETF_TSMOM_FAMILY != "tsmom"


def test_etf_calendar_covers_the_whole_pilot_universe():
    cal = etf_calendar()
    for ticker in PILOT_UNIVERSE:
        # must not raise SessionCalendarMissing
        cal.calendar_name_for(ticker)


def test_etf_split_plan_stays_inside_the_real_pre_holdout_window():
    from alpha_agent.data.databento_source import HOLDOUT_START

    plan = etf_split_plan("XLE")
    holdout_start_ns = int(__import__("pandas").Timestamp(HOLDOUT_START, tz="UTC").value)
    for w in plan.windows:
        assert w.start_ts_ns < holdout_start_ns
        assert w.end_ts_ns <= holdout_start_ns


def test_etf_walk_forward_has_enough_folds():
    wf = etf_walk_forward()
    assert wf.n_folds >= 3


# ---------------------------------------------------------------------------
# The real end-to-end run -- slow (shells the real C++ CLI many times for
# folds/nulls/bootstrap/cost-stress), so kept to a single canonical ticker.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_etf_research_produces_a_real_deterministic_verdict():
    artifact = run_etf_research("XLE", work_dir="outputs/phase_6/_scratch_test")
    assert artifact.root_symbol == "XLE"
    assert artifact.strategy_family == ETF_TSMOM_FAMILY
    # No fake PASS requirement -- any real, deterministic verdict is valid.
    assert artifact.report.verdict in (Verdict.PASS, Verdict.REJECT, Verdict.INCONCLUSIVE)
    assert artifact.report.reason_codes is not None
    assert "distribution" in artifact.corporate_action_disclosure.lower()


@pytest.mark.slow
def test_run_etf_research_is_deterministic_across_two_runs():
    a1 = run_etf_research("XLE", work_dir="outputs/phase_6/_scratch_test_a")
    a2 = run_etf_research("XLE", work_dir="outputs/phase_6/_scratch_test_b")
    assert a1.strategy_fingerprint == a2.strategy_fingerprint
    assert a1.validation_fingerprint == a2.validation_fingerprint
    assert a1.report.verdict == a2.report.verdict
    assert a1.report.reason_codes == a2.report.reason_codes


@pytest.mark.slow
def test_run_etf_research_never_touches_the_locked_holdout():
    """A defense-in-depth assertion: the dataset's own last_ts_ns must never
    reach 2025-01-01, mirroring the same bright-line check the acquisition
    layer already enforces."""
    from alpha_agent.data.databento_source import HOLDOUT_START

    artifact = run_etf_research("XLE", work_dir="outputs/phase_6/_scratch_test_c")
    holdout_start_ns = int(__import__("pandas").Timestamp(HOLDOUT_START, tz="UTC").value)
    assert artifact.dataset_identity.last_ts_ns < holdout_start_ns


def test_etf_tsmom_adapter_canonical_params_reference_the_real_ticker():
    adapter = etf_tsmom_adapter("GLD")
    assert adapter.canonical_params["root_symbol"] == "GLD"
    assert adapter.strategy_key == ETF_TSMOM_FAMILY

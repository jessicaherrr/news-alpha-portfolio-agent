"""Product UI Polish pass -- Paper user workflow (task spec sections 8-13) and
the Research -> Paper eligibility handoff (section 12).

Two layers, mirroring `test_phase_21_paper_trading.py`'s own split:

* `alpha_agent.ui.paper_actions.start_paper_run` -- the new, single write
  boundary from the UI into `PaperTradingEngine.start_run` -- exercised
  against the REAL compiled C++ CLI and REAL already-acquired NQ data (no
  network, no 2025), reusing the same PASS-seeded tmp-registry fixture
  pattern `test_phase_21_paper_trading.py` already established.
* the Paper Trading page and Research page's "Test in Paper" handoff --
  exercised with `streamlit.testing.v1.AppTest`, `services`/`paper_actions`
  monkeypatched so these stay fast, offline, and independent of real
  registry/ledger state.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import paper_actions, services
from test_phase_21_paper_trading import _experiment_record, _mr_fingerprint, _pass_result

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_CLI = REPO_ROOT / "build" / "cpp" / "cpp" / "quant_paper_trading_targets_csv"

pytestmark = pytest.mark.skipif(
    not PAPER_CLI.exists(), reason="quant_paper_trading_targets_csv not built (cmake --build build/cpp)"
)


# ---------------------------------------------------------------------------
# alpha_agent.ui.paper_actions -- the one write boundary
# ---------------------------------------------------------------------------


def test_start_paper_run_creates_a_real_run_for_a_genuinely_eligible_experiment(tmp_path, monkeypatch):
    fp = _mr_fingerprint()
    record = _experiment_record(strategy_fp=fp)
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        reg.insert_experiment(record, _pass_result(record.experiment_identity))

    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "experiments.sqlite")
    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", tmp_path / "paper_ledger.sqlite")

    result = paper_actions.start_paper_run(record.experiment_id)

    assert result.ok, result.error
    assert result.run_id
    assert result.error is None

    runs = services.list_paper_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == result.run_id
    assert runs[0]["root_symbol"] == "NQ"
    assert runs[0]["strategy_family"] == "mean_reversion"


def test_start_paper_run_refuses_an_ineligible_experiment_and_writes_nothing(tmp_path, monkeypatch):
    """No registry entry at all -> a typed refusal, never a fabricated run,
    and the ledger gains no run as a side effect of the attempt."""
    with ExperimentRegistry(tmp_path / "experiments.sqlite"):
        pass  # empty registry

    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "experiments.sqlite")
    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", tmp_path / "paper_ledger.sqlite")

    result = paper_actions.start_paper_run("does-not-exist")

    assert not result.ok
    assert result.run_id is None
    assert result.error is not None
    assert services.list_paper_runs() == []


def test_start_paper_run_refuses_a_non_pass_experiment(tmp_path, monkeypatch):
    from alpha_agent.registry.enums import RegistryVerdict
    from alpha_agent.registry.models import ResultRecord

    fp = _mr_fingerprint()
    record = _experiment_record(strategy_fp=fp)
    reject_result = ResultRecord(
        experiment_identity=record.experiment_identity, headline_verdict=RegistryVerdict.REJECT,
        reason_codes=("cost_sensitivity_failed",), net_pnl_usd=-100.0, daily_sharpe=-0.1,
        annualized_sharpe=-1.0, n_trades=10,
    )
    with ExperimentRegistry(tmp_path / "experiments.sqlite") as reg:
        reg.insert_experiment(record, reject_result)

    monkeypatch.setattr(services, "REGISTRY_PATH", tmp_path / "experiments.sqlite")
    monkeypatch.setattr(services, "PAPER_LEDGER_PATH", tmp_path / "paper_ledger.sqlite")

    result = paper_actions.start_paper_run(record.experiment_id)
    assert not result.ok
    assert "passed frozen validation" in result.error
    assert services.list_paper_runs() == []


# ---------------------------------------------------------------------------
# Paper Trading page -- empty state, eligibility gating, start workflow
# ---------------------------------------------------------------------------

_FAKE_ELIGIBLE = [
    {
        "experiment_id": "NQ__MEAN_REVERSION__CANONICAL__TEST", "experiment_identity": "ident-1",
        "root_symbol": "NQ", "strategy_family": "mean_reversion", "params": {"root_symbol": "NQ"},
        "backtest_daily_sharpe": 0.12, "backtest_net_pnl_usd": 5000.0, "backtest_n_trades": 42,
    }
]


def test_paper_page_shows_selector_and_start_button_when_eligible(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(services, "paper_eligible_experiments", lambda: _FAKE_ELIGIBLE)
    monkeypatch.setattr(services, "list_paper_runs", list)

    at = AppTest.from_string("from alpha_agent.ui.views.paper_trading import render\nrender()\n")
    at.run(timeout=90)
    assert not list(at.exception)

    full_text = " ".join(m.value for m in at.markdown)
    assert "No active paper test" not in full_text  # eligible -> not the empty state
    assert at.selectbox(key="pt-start-select")
    assert at.button(key="pt-start-button").label == "Start Paper Test"


def test_paper_page_start_button_calls_paper_actions_with_the_selected_experiment(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(services, "paper_eligible_experiments", lambda: _FAKE_ELIGIBLE)
    monkeypatch.setattr(services, "list_paper_runs", list)

    calls: list[str] = []

    def fake_start(experiment_id, **kwargs):
        calls.append(experiment_id)
        return paper_actions.StartPaperRunResult(ok=False, error="synthetic refusal for test")

    monkeypatch.setattr(paper_actions, "start_paper_run", fake_start)

    at = AppTest.from_string("from alpha_agent.ui.views.paper_trading import render\nrender()\n")
    at.run(timeout=90)
    at.button(key="pt-start-button").click().run(timeout=90)
    assert not list(at.exception)

    assert calls == ["NQ__MEAN_REVERSION__CANONICAL__TEST"]
    assert any("synthetic refusal for test" in e.value for e in at.error)


def test_paper_page_eligibility_is_never_decided_by_the_ui_itself():
    """Static guard: `paper_trading.py` must call the real, registry-derived
    `services.paper_eligible_experiments()` to decide whether to show the
    Start workflow -- it must never hardcode or independently compute an
    eligibility boolean."""
    from alpha_agent.ui.views import paper_trading as pt_view

    with open(pt_view.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert "services.paper_eligible_experiments()" in text


# ---------------------------------------------------------------------------
# Research -> Paper handoff (section 12)
# ---------------------------------------------------------------------------


def test_research_details_shows_test_in_paper_action_when_eligible(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(services, "paper_eligible_experiment", lambda exp_id: exp_id == "EXP-ELIGIBLE")

    script = (
        "import streamlit as st\n"
        "from alpha_agent.ui.views import research\n"
        "st.session_state['research_details_target'] = {"
        "'source': 'registry_lookup', 'objective': None, 'root': 'NQ', 'family': 'mean_reversion',"
        "'hypothesis': None, 'compiled': None, 'evidence': None, 'run_outcome': None,"
        "'experiment_id': 'EXP-ELIGIBLE'}\n"
        "research.render()\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=90)
    assert not list(at.exception)
    buttons = [b for b in at.button if b.key == "rd-test-in-paper"]
    assert buttons, "expected a 'Test in Paper' action when the experiment is genuinely paper-eligible"
    assert not buttons[0].disabled


def test_research_details_disables_test_in_paper_when_not_eligible(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(services, "paper_eligible_experiment", lambda exp_id: False)

    script = (
        "import streamlit as st\n"
        "from alpha_agent.ui.views import research\n"
        "st.session_state['research_details_target'] = {"
        "'source': 'registry_lookup', 'objective': None, 'root': 'NQ', 'family': 'mean_reversion',"
        "'hypothesis': None, 'compiled': None, 'evidence': None, 'run_outcome': None,"
        "'experiment_id': 'EXP-NOT-ELIGIBLE'}\n"
        "research.render()\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=90)
    assert not list(at.exception)
    buttons = [b for b in at.button if b.key == "rd-test-in-paper"]
    assert buttons
    assert buttons[0].disabled

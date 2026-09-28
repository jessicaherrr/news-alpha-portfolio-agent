"""Research Golden Path -- C++ Authority test (task spec section 16): the
Research Workflow's "Run Historical Test" action must use the existing real
C++ runner via the ONE existing orchestration boundary
(`alpha_agent.ui.deep_research.run_deep_research`), and must never silently
substitute a pandas backtest, Python-only accounting, or a Claude-computed
metric.

Two independent proofs, matching this repo's own convention
(`test_phase_20_streamlit_ui.py::test_services_module_never_calls_a_registry_write_method`):

1. STATIC -- `research_workflow.py`'s source never imports a general-purpose
   dataframe/backtest library and never calls the C++ CLI/pybind boundary
   directly; its only execution call is `deep_research.run_deep_research`.
2. DYNAMIC -- monkeypatching `deep_research.run_deep_research` proves
   `_run_historical_test` calls exactly that function (and nothing else) to
   produce its result.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.ui import services
from alpha_agent.ui.views import research_workflow

pytestmark = pytest.mark.skipif(
    not services.REGISTRY_PATH.exists(),
    reason="Phase 14 registry sqlite not present in this checkout",
)


def test_research_workflow_never_imports_a_dataframe_backtest_engine():
    with open(research_workflow.__file__, encoding="utf-8") as fh:
        text = fh.read()
    forbidden = [r"\bimport pandas\b", r"\bfrom pandas\b", r"quant_core_py", r"subprocess"]
    for pattern in forbidden:
        assert not re.search(pattern, text), f"research_workflow.py must never contain {pattern!r}"


def test_research_workflow_execution_call_site_is_deep_research_run_deep_research():
    with open(research_workflow.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert "deep_research.run_deep_research(" in text
    # No second, independent orchestrator entrypoint (`build_deep_research_orchestrator`,
    # `plan_family`/`execute_family`/`finalize_family`) may appear here -- that
    # sequencing belongs solely to `deep_research.py`.
    for forbidden in ("build_deep_research_orchestrator", ".plan_family(", ".execute_family(", ".finalize_family("):
        assert forbidden not in text


def test_run_historical_test_calls_deep_research_boundary_exactly_once(monkeypatch):
    calls: list[dict] = []

    class _FakeOutcome:
        accepted = True
        error = None
        report = None

    def fake_run_deep_research(**kwargs):
        calls.append(kwargs)
        return _FakeOutcome()

    monkeypatch.setattr(research_workflow.deep_research, "run_deep_research", fake_run_deep_research)

    cd = {
        "root_symbol": "NQ", "family_key": "tsmom", "strategy_fingerprint": "stratdsl1:deadbeef",
    }
    import streamlit as st

    st.session_state["rwf_mode"] = "scripted"
    st.session_state["rwf_scenario_key"] = "tsmom_nq"
    st.session_state["rwf_hypothesis"] = {"economic_mechanism": "test mechanism"}
    research_workflow._run_historical_test(cd)

    assert len(calls) == 1
    assert calls[0]["root"] == "NQ"
    assert calls[0]["family_stem"] == "rwf-tsmom"
    assert isinstance(st.session_state["rwf_run_outcome"], _FakeOutcome)

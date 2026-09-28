"""Phase 4 -- Learn page UI (real `streamlit.testing.v1.AppTest` renders,
mirroring `test_alpha_library_ui.py` / `test_sidebar_nav.py`'s own patterns).

Proves: the page renders with its four sub-views, the Three-Lens explainer
works for Concepts and Strategies, "Learn Why" deep-links from another page
land on the right Concept, My Research renders a real Failure Autopsy from
the real registry, and nothing here ever calls Claude or writes the registry
just by rendering.
"""
from __future__ import annotations

import re

import pytest
from alpha_agent.ui import services
from alpha_agent.ui.views import learn


def _learn_app(focus: str | None = "concepts"):
    """Learn opened on ``focus`` -- a Guides section (the Phase 4 explanation
    layer lives under Learn -> Guides since the Research Thread workspace) or
    a top-level tab; ``None`` = the default landing (My Alpha)."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from alpha_agent.ui.views.learn import render\nrender()\n")
    if focus is not None:
        at.session_state["learn_landing_focus"] = focus
    at.run(timeout=90)
    assert not list(at.exception), list(at.exception)
    return at


# ---------------------------------------------------------------------------
# basic render + structure
# ---------------------------------------------------------------------------


def test_learn_page_renders_its_four_tabs_and_the_four_guides():
    at = _learn_app(None)
    # `at.tabs` is a flat list across the whole tree -- assert the four
    # top-level sections are present, not that they are the ONLY tabs.
    labels = {t.label for t in at.tabs}
    assert {"My Alpha", "Research Map", "Community", "Guides"}.issubset(labels)
    assert at.session_state["learn-tabs"] == "My Alpha"  # the default landing
    guides = _learn_app("concepts")
    assert guides.session_state["learn-tabs"] == "Guides"
    assert set(guides.button_group(key="learn-guide-section").options) >= {"Concepts", "Strategies",
                                                                           "Failure autopsy", "Learning paths"}


def test_learn_page_never_calls_a_registry_write_method_static_guard():
    forbidden = [r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(", r"\.apply_bundle\(", r"INSERT OR REPLACE"]
    sources = [learn.__file__]
    import pathlib

    from alpha_agent import learn as learn_pkg

    sources.extend(str(p) for p in pathlib.Path(learn_pkg.__file__).parent.glob("*.py"))
    for src in sources:
        with open(src, encoding="utf-8") as fh:
            text = fh.read()
        for pattern in forbidden:
            assert not re.search(pattern, text), f"{src} must never call {pattern!r}"


def test_learn_never_calls_claude_or_market_data_by_default(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("Learn must never call Claude just by rendering")

    from alpha_agent.ui import services as ui_services

    monkeypatch.setattr(ui_services, "learn_failure_autopsy_narrative", _boom, raising=False)
    at = _learn_app()
    assert not list(at.exception)


# ---------------------------------------------------------------------------
# Concepts -- Three-Lens
# ---------------------------------------------------------------------------


def test_concepts_tab_default_renders_three_lens_tabs():
    at = _learn_app()
    tab_labels = {t.label for t in at.tabs}
    assert {"Intuition", "Quant", "Implementation"}.issubset(tab_labels)


def test_selecting_a_different_concept_updates_the_focus_and_content():
    at = _learn_app()
    at.selectbox(key="learn_focus_concept").select("dsr").run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["learn_focus_concept"] == "dsr"
    dsr = services.learn_concept("dsr")
    body_text = " ".join(m.value for m in at.markdown)
    assert dsr["one_line"] in body_text


# ---------------------------------------------------------------------------
# Strategies -- idea -> execution chain
# ---------------------------------------------------------------------------


def test_strategies_tab_renders_the_five_step_chain():
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "strategies"
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    for stage in ("Economic idea", "Equations & timing", "Python feature", "StrategySpec", "C++ execution"):
        assert stage in full_text


def test_strategy_family_selector_offers_the_four_compiler_families():
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "strategies"
    at.run(timeout=90)
    sb = at.selectbox(key="learn_focus_family")
    expected_names = {services.learn_strategy_explainer(k)["name"] for k in ("tsmom", "ma_trend", "breakout", "mean_reversion")}
    assert set(sb.options) == expected_names


# ---------------------------------------------------------------------------
# My Research -- Failure Autopsy (real registry)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_my_research_tab_renders_a_real_failure_autopsy():
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "my_research"
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    assert "What Failed, and Why It Matters" in full_text
    assert "What a Next Experiment Would Need" in full_text


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_my_research_tab_shows_the_quant_pipeline_walkthrough():
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "my_research"
    at.run(timeout=90)
    full_text = " ".join(m.value for m in at.markdown)
    assert "Quant Rigor: Signal to Validation" in full_text
    for stage_title in ("1. Signal", "2. Next-Bar Execution", "7. Validation"):
        assert stage_title in full_text


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_my_research_opt_in_claude_expander_never_auto_expanded_or_called(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("must never be called without an explicit click")

    from alpha_agent.ui import services as ui_services

    monkeypatch.setattr(ui_services, "learn_failure_autopsy_narrative", _boom, raising=False)
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "my_research"
    at.run(timeout=90)
    assert not list(at.exception)
    expanders = [e for e in at.expander if "Explain this in plain language" in (e.label or "")]
    assert expanders
    # the opt-in button exists, but merely rendering the (collapsed) expander
    # never produced any narrative output -- proving `_boom` above was never
    # reached without an explicit click.
    assert not any(m.value.strip() for m in at.info)


# ---------------------------------------------------------------------------
# Semantic hardening patch (Phase 4 review) -- UI-level regression coverage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_my_research_selector_never_offers_a_not_adjudicated_experiment():
    """The real registry has 40 real NOT_ADJUDICATED canonical rows (Phase
    15B ML trials typed-refused for insufficient training events) -- none
    of them may appear in the "My Research" experiment dropdown."""
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "my_research"
    at.run(timeout=90)
    sb = at.selectbox(key="learn_focus_experiment")
    for option_label in sb.options:
        assert "NOT_ADJUDICATED" not in option_label


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_my_research_renders_scope_caveat_and_renamed_similarity_section():
    """The default selection (CL/breakout REJECT) has real shared-reason-code
    matches (verified directly against the builder in
    test_phase4_failure_autopsy.py), so the renamed section is guaranteed to
    render here -- proving the honest wording actually reaches the page,
    not just the typed model."""
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "my_research"
    at.run(timeout=90)
    full_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "does not, by itself, invalidate the broader Factor" in full_text
    assert "Other Experiments with Shared Failure Gates" in full_text
    assert "not proven Factor, Strategy, or Mechanism similarity" in full_text
    assert "Similar Prior Failures" not in full_text


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_my_research_engineering_notes_use_system_engineering_history_label():
    """The default-selected experiment's `engineering_notes` include real
    untagged SYSTEM-scope records (verified directly against the builder in
    `test_phase4_failure_autopsy.py`), so this section always renders for
    the real registry -- proving the honest section title/caveat and the
    per-note scope tag both actually reach the page."""
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "my_research"
    at.run(timeout=90)
    full_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "System / Engineering History" in full_text
    assert "not necessarily causal or specific to this experiment" in full_text
    assert "[system-wide]" in full_text
    assert "Engineering Notes" not in full_text  # the old, overclaiming section title


@pytest.mark.skipif(not services.REGISTRY_PATH.exists(), reason="Phase 14 registry sqlite not present in this checkout")
def test_what_looked_promising_shows_descriptive_evidence_label_when_present():
    """Switches to the real CL/mean_reversion REJECT experiment (positive
    committed net PnL/Sharpe despite failing the null-hypothesis gate --
    verified directly in test_phase4_failure_autopsy.py) and proves the
    page renders the DESCRIPTIVE EVIDENCE label, distinct from the gates-
    passed label."""
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "my_research"
    at.run(timeout=90)
    options = at.selectbox(key="learn_focus_experiment").options
    target = next((o for o in options if "Mean Reversion" in o and "CL" in o), None)
    if target is None:
        pytest.skip("CL/mean_reversion REJECT experiment not present in this checkout's registry")
    at.selectbox(key="learn_focus_experiment").select(target).run(timeout=90)
    assert not list(at.exception)
    assert at.session_state["learn_focus_experiment"] == "CL__MEAN_REVERSION__CANONICAL__VALIDATION_2023_2024"
    full_text = " ".join(m.value for m in at.markdown)
    assert "DESCRIPTIVE EVIDENCE" in full_text
    assert "not validated alpha" in full_text


# ---------------------------------------------------------------------------
# Learning Paths
# ---------------------------------------------------------------------------


def test_learning_paths_tab_lists_all_paths():
    at = _learn_app()
    at.session_state["learn_landing_focus"] = "paths"
    at.run(timeout=90)
    assert not list(at.exception)
    full_text = " ".join(m.value for m in at.markdown)
    for path in services.learn_learning_paths():
        assert path["title"] in full_text


# ---------------------------------------------------------------------------
# "Learn Why" deep link from another professional surface
# ---------------------------------------------------------------------------


def test_learn_why_from_validation_sets_focus_then_learn_renders_that_concept():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    if not services.REGISTRY_PATH.exists():
        pytest.skip("Phase 14 registry sqlite not present in this checkout")

    val_at = AppTest.from_string("from alpha_agent.ui.views.validation import render\nrender()\n")
    val_at.run(timeout=90)
    assert not list(val_at.exception)
    dsr_buttons = [b for b in val_at.button if b.key and b.key.startswith("learn-why-val-dsr")]
    assert dsr_buttons, "expected a 'Learn Why' button near the Deflated Sharpe Ratio section"
    dsr_buttons[0].click().run(timeout=90)
    assert not list(val_at.exception)
    assert val_at.session_state["learn_focus_concept"] == "dsr"
    assert val_at.session_state["learn_landing_focus"] == "concepts"

    # the actual cross-page hand-off (mirrors
    # test_agent_to_strategies_handoff_renders_clean's own pattern -- AppTest
    # does not follow a real st.switch_page across instances)
    learn_at = AppTest.from_string("from alpha_agent.ui.views.learn import render\nrender()\n")
    learn_at.session_state["learn_focus_concept"] = val_at.session_state["learn_focus_concept"]
    learn_at.session_state["learn_landing_focus"] = val_at.session_state["learn_landing_focus"]
    learn_at.run(timeout=90)
    assert not list(learn_at.exception)
    full_text = " ".join(m.value for m in learn_at.markdown)
    assert "Deflated Sharpe Ratio" in full_text


def test_learn_why_renders_nothing_for_an_unknown_concept_id():
    """`render_learn_why` must render no button for a concept id that does
    not exist -- a guard against a future typo producing a dead link."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    assert services.learn_concept("not_a_real_concept_xyz") is None
    script = (
        "from alpha_agent.ui import learn_links\n"
        "learn_links.render_learn_why('not_a_real_concept_xyz', key='bogus')\n"
        "learn_links.render_learn_why('dsr', key='real')\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=90)
    assert not list(at.exception)
    keys = {b.key for b in at.button}
    assert not any(k.startswith("learn-why-bogus") for k in keys)
    assert any(k.startswith("learn-why-real") for k in keys)

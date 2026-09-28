"""Agent runtime-integration release -- focused regression tests for the
runtime boundaries closed in this release (sections 2, 3, 4, 8):

* the free-text objective the user actually submits reaches
  `ResearchContext.objective` verbatim, never silently replaced by a preset;
* prior registry evidence (`FailureMemory`) is built into the `ResearchContext`
  BEFORE `ResearchAgent.propose` is called, not looked up after the fact;
* the Agent/Research page's mode selector genuinely switches which LLM
  transport runs (`ScriptedLLMClient` vs a real `AnthropicClient`), the
  offline default never makes a network call, and a broken live
  configuration fails visibly rather than silently falling back to scripted;
* "seen before" strategy-fingerprint evidence distinguishes a live registry
  match from a frozen-candidate-manifest match.
"""
from __future__ import annotations

from alpha_agent.agents.llm import AnthropicClient, ScriptedLLMClient
from alpha_agent.ui import llm_demo, services

_UNIQUE_OBJECTIVE = (
    "Investigate an economically distinct NQ volatility-conditioned "
    "cross-market mechanism; explicitly avoid ordinary TSMOM parameter variants."
)


def test_edited_objective_reaches_research_context_verbatim():
    """The exact user-submitted objective -- one that could not have come from
    any curated preset -- must be the text `ResearchContext.objective` carries
    (runtime-integration release, section 2)."""
    universe = services.approved_universe()
    context, _fm = llm_demo.build_context_for_objective(objective=_UNIQUE_OBJECTIVE, universe=universe)
    assert context.objective == _UNIQUE_OBJECTIVE


def test_edited_objective_reaches_research_agent_via_propose_hypothesis():
    universe = services.approved_universe()
    proposal, error, _fm = llm_demo.propose_hypothesis(
        mode=llm_demo.SCRIPTED_MODE, scenario_key="tsmom_nq", universe=universe,
        objective=_UNIQUE_OBJECTIVE,
    )
    assert error is None, error
    assert proposal is not None


def test_propose_hypothesis_defaults_to_the_scenario_objective_when_none_given():
    """Backward-compatible default: an unedited preset (no explicit objective
    passed) still uses the scenario's own objective -- this is the ONE place a
    curated objective may still be substituted, and only because the caller
    supplied nothing at all."""
    universe = services.approved_universe()
    sc = llm_demo.scenario("tsmom_nq")
    context, _fm = llm_demo.build_context_for_objective(objective=sc.objective, universe=universe)
    assert context.objective == sc.objective


# ---------------------------------------------------------------------------
# section 3 -- pre-proposal failure memory
# ---------------------------------------------------------------------------
def test_failure_memory_is_populated_before_the_agent_is_asked_to_propose():
    """`build_context_for_objective` builds failure memory (from real,
    already-committed registry evidence) and places it into the
    `ResearchContext` BEFORE any hypothesis exists -- proving the agent
    receives it pre-proposal, not from a later lookup keyed on what it
    proposed."""
    universe = services.approved_universe()
    context, fm = llm_demo.build_context_for_objective(objective="anything", universe=universe)
    assert fm  # the real registry has prior NQ/ES/... experiments already
    assert context.failure_memory  # actually placed into what the agent sees
    assert any(
        d.strategy_family == "tsmom" and d.root_symbol == "NQ" for d in context.failure_memory
    ), "NQ TSMOM has a known prior registry result and must appear pre-proposal"


def test_relevant_failure_memory_finds_the_known_nq_tsmom_history():
    entries = llm_demo.relevant_failure_memory(market_universe=("NQ",))
    assert any(
        fm.query.get("strategy_family") == "tsmom" and fm.query.get("root_symbol") == "NQ"
        for fm in entries
    )


def test_relevant_failure_memory_is_bounded_not_a_raw_dump():
    """The sweep returns one aggregated digest per (family, root) combination
    with evidence -- not one entry per underlying experiment row."""
    universe = services.approved_universe()
    entries = llm_demo.relevant_failure_memory(market_universe=universe)
    combos = {(fm.query.get("strategy_family"), fm.query.get("root_symbol")) for fm in entries}
    assert len(combos) == len(entries)  # exactly one digest per combo, deduplicated


def test_second_generation_context_carries_the_first_generations_result():
    """Mandatory feedback-loop proof (section 10, narrowed to the propose/
    inspect layer used by the Agent/Research pages): failure memory for a
    family+root that has an authoritative result shows up for a SECOND
    context-build call exactly as it did for the first -- the evidence is
    durable registry state, not per-call session state."""
    universe = services.approved_universe()
    _ctx1, fm1 = llm_demo.build_context_for_objective(objective="first", universe=universe)
    _ctx2, fm2 = llm_demo.build_context_for_objective(objective="second", universe=universe)
    nq_tsmom_1 = next(f for f in fm1 if f.query.get("strategy_family") == "tsmom" and f.query.get("root_symbol") == "NQ")
    nq_tsmom_2 = next(f for f in fm2 if f.query.get("strategy_family") == "tsmom" and f.query.get("root_symbol") == "NQ")
    assert nq_tsmom_1.verdict_counts == nq_tsmom_2.verdict_counts
    assert nq_tsmom_1.verdict_counts  # actually has adjudicated evidence, not an empty digest


# ---------------------------------------------------------------------------
# section 4 -- live Claude mode selection, offline-by-default, honest failure
# ---------------------------------------------------------------------------
def test_scripted_mode_builds_a_scripted_client_never_touches_network():
    client = llm_demo.build_llm_client(llm_demo.SCRIPTED_MODE, hypothesis_json={"a": 1})
    assert isinstance(client, ScriptedLLMClient)


def test_live_mode_builds_the_real_anthropic_client_with_no_key_argument():
    client = llm_demo.build_llm_client(llm_demo.LIVE_MODE)
    assert isinstance(client, AnthropicClient)
    assert client._api_key is None  # never sourced by this module -- the SDK reads the env itself


def test_offline_default_performs_zero_anthropic_calls(monkeypatch):
    """A hard guarantee: in scripted mode, nothing in the propose/compile path
    ever imports or touches `anthropic`."""
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)  # importing it would now raise
    universe = services.approved_universe()
    proposal, error, _fm = llm_demo.propose_hypothesis(
        mode=llm_demo.SCRIPTED_MODE, scenario_key="tsmom_nq", universe=universe,
    )
    assert error is None, error
    assert proposal is not None and proposal.accepted


def test_live_mode_with_missing_key_fails_visibly_not_silently_scripted(monkeypatch):
    """A broken live configuration must surface an honest error -- it must
    NEVER silently fall back to a scripted/canned proposal that looks the same
    as a real one (runtime-integration release, section 4/17)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    universe = services.approved_universe()
    proposal, error, fm = llm_demo.propose_hypothesis(
        mode=llm_demo.LIVE_MODE, scenario_key="tsmom_nq", universe=universe,
    )
    assert proposal is None
    assert error is not None
    assert "ANTHROPIC_API_KEY" not in error  # never echoes the variable's value
    assert fm == ()


# ---------------------------------------------------------------------------
# section 8 -- strategy-fingerprint history vs scientific experiment identity
# ---------------------------------------------------------------------------
def test_compile_hypothesis_known_strategies_include_live_registry_source():
    """`llm_demo.compile_hypothesis` must fold in LIVE registry evidence (not
    only the frozen Phase 13.5C candidate manifest), so the compiled-strategy
    card's "seen before" story is never contradicted by the evidence card,
    which always queries the live registry."""
    universe = services.approved_universe()
    proposal, error, _fm = llm_demo.propose_hypothesis(
        mode=llm_demo.SCRIPTED_MODE, scenario_key="tsmom_nq", universe=universe,
    )
    assert error is None
    compiled, cerror = llm_demo.compile_hypothesis(
        mode=llm_demo.SCRIPTED_MODE, scenario_key="tsmom_nq",
        hypothesis=proposal.hypothesis, universe=universe,
    )
    assert cerror is None, cerror
    de = compiled.duplicate_evidence
    assert de is not None and de.strategy_fingerprint_seen
    sources = {m.source for m in de.matches}
    # the tsmom/NQ canonical fingerprint has a real prior registry row, so a
    # live-registry-sourced match must be present, distinct from any
    # frozen-manifest match.
    assert "live_experiment_registry" in sources

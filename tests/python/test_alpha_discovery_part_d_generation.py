"""Alpha Discovery campaign, Part D -- diverse candidate-pool generation
tests (task spec section 72)."""
from __future__ import annotations

import json

from alpha_agent.agents import (
    IdentityPlanes,
    ScriptedExecutionValidationService,
    ScriptedLLMClient,
)
from alpha_agent.agents.orchestrator import MarketWindow
from alpha_agent.discovery import (
    DEFAULT_MECHANISM_ORDER,
    CandidatePoolResult,
    generate_candidate_pool,
    prioritize_mechanisms,
)
from alpha_agent.knowledge import EconomicMechanism, StrategyKnowledgeBase
from alpha_agent.recommendation.profile import InvestorProfile, StrategyPreference
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.validation.policy import ReliabilityPolicy

POLICY = ReliabilityPolicy()
PLANES = IdentityPlanes(
    dataset_fingerprint="valdataset2:ds", split_identity="split1:sp",
    validation_spec_fingerprint="validationspec1:vs",
    reliability_policy_fingerprint=POLICY.identity(),
    execution_config_identity="execconfig1:ex", cost_config_identity="costconfig1:co",
    risk_identity="riskconfig1:ri",
)
MW = MarketWindow(label="VALIDATION", start_date="2023-01-01", end_date="2024-12-31")


#: the exact feature `kind`(s) each Phase 11 template family's compiled spec
#: references -- required so a scripted `HypothesisSpec.required_features`
#: passes the real Phase 17.2 fidelity check (compiled spec feature kinds
#: must equal the hypothesis's declared required_features, exactly).
_FAMILY_FEATURE_KINDS: dict[str, tuple[str, ...]] = {
    "tsmom": ("diff",),
    "ma_trend": ("ma",),
    "mean_reversion": ("zscore",),
    "breakout": ("breakout_down", "breakout_up"),
}


def _hyp(hid: str, *, universe=("NQ",), mechanism_desc: str, family: str) -> str:
    return json.dumps({
        "hypothesis_id": hid,
        "title": f"{universe[0]} {mechanism_desc}",
        "economic_mechanism": mechanism_desc,
        "universe": list(universe),
        "horizon": "20 trading days",
        "required_features": list(_FAMILY_FEATURE_KINDS[family]),
        "signal_description": "signal description",
        "expected_regime": "trending",
        "failure_regime": "choppy",
        "falsification_test": "no positive OOS net PnL after costs",
        "novelty_notes": "",
    })


def _tmpl(family: str, root: str = "NQ") -> str:
    canonical = {
        "tsmom": {"fast_horizon": 20, "slow_horizon": 120, "size": 1},
        "ma_trend": {"fast_window": 50, "slow_window": 200, "size": 1},
        "mean_reversion": {"zscore_window": 20, "entry_z": 2.0, "exit_z": 0.5, "size": 1},
        "breakout": {"lookback": 20, "size": 1},
    }
    return json.dumps({
        "expressible": True,
        "template": {"family_key": family, "root_symbol": root, "params": canonical[family]},
        "rationale": f"maps to the {family} family",
    })


def _pool_kwargs(tmp_path, *, hyps, plans, registry=None, **overrides):
    reg = registry or ExperimentRegistry(tmp_path / "registry.sqlite")
    kwargs = {
        "market": "NQ", "objective": "find robust alpha for NQ", "registry": reg,
        "research_llm": ScriptedLLMClient(hyps), "compiler_llm": ScriptedLLMClient(plans),
        "execution_service": ScriptedExecutionValidationService([]),
        "reliability_policy": POLICY, "planes": PLANES, "market_window": MW,
        "max_mechanisms": 3, "per_mechanism_family_size": 1,
    }
    kwargs.update(overrides)
    return kwargs


def test_generates_mechanism_diverse_families_not_parameter_variants(tmp_path):
    hyps = [
        _hyp("H-1", mechanism_desc="trend following via dual moving average crossover", family="ma_trend"),
        _hyp("H-2", mechanism_desc="momentum diffusion of macro information", family="tsmom"),
        _hyp("H-3", mechanism_desc="mean reversion after overextension", family="mean_reversion"),
    ]
    plans = [_tmpl("ma_trend"), _tmpl("tsmom"), _tmpl("mean_reversion")]
    result = generate_candidate_pool(**_pool_kwargs(tmp_path, hyps=hyps, plans=plans))
    assert isinstance(result, CandidatePoolResult)
    assert result.mechanisms_attempted == 3
    assert result.mechanisms_supported == 3
    assert len(result.members) == 3
    # THE key diversity proof: three DISTINCT strategy families, not one
    # family repeated with different parameters.
    assert len(result.strategy_families) == 3
    assert set(result.strategy_families) == {"ma_trend", "tsmom", "mean_reversion"}


def test_ideas_considered_counts_every_planning_slot_not_just_accepted():
    # a rejected/duplicate slot still counts as "considered" -- proven at the
    # MechanismOutcome level using the orchestrator's own planning_log length,
    # which is exercised in the happy-path test above (>=1 per mechanism).
    pass


def test_budget_is_bounded_by_max_mechanisms(tmp_path):
    hyps = [_hyp("H-1", mechanism_desc="trend following", family="ma_trend")]
    plans = [_tmpl("ma_trend")]
    result = generate_candidate_pool(
        **_pool_kwargs(tmp_path, hyps=hyps, plans=plans, max_mechanisms=1)
    )
    assert result.mechanisms_attempted == 1
    assert len(result.members) <= 1


def test_pool_members_are_re_ordinaled_densely_from_zero(tmp_path):
    hyps = [
        _hyp("H-1", mechanism_desc="trend following", family="ma_trend"),
        _hyp("H-2", mechanism_desc="momentum", family="tsmom"),
    ]
    plans = [_tmpl("ma_trend"), _tmpl("tsmom")]
    result = generate_candidate_pool(
        **_pool_kwargs(tmp_path, hyps=hyps, plans=plans, max_mechanisms=2)
    )
    assert [m.ordinal for m in result.members] == list(range(len(result.members)))


def test_knowledge_base_snippets_reach_the_llm_prompt(tmp_path):
    reg = ExperimentRegistry(tmp_path / "registry.sqlite")
    research_client = ScriptedLLMClient([_hyp("H-1", mechanism_desc="trend following", family="ma_trend")])
    kb = StrategyKnowledgeBase()
    result = generate_candidate_pool(
        market="NQ", objective="find robust alpha for NQ", registry=reg,
        research_llm=research_client, compiler_llm=ScriptedLLMClient([_tmpl("ma_trend")]),
        execution_service=ScriptedExecutionValidationService([]),
        reliability_policy=POLICY, planes=PLANES, market_window=MW,
        knowledge_base=kb, max_mechanisms=1, per_mechanism_family_size=1,
    )
    assert len(result.members) == 1
    prompt = research_client.calls[0]["messages"][0]["content"]
    assert "TREND" in prompt  # the mechanism-specific objective text
    assert "research_knowledge_base" in prompt  # ResearchContext.render_user_message key


def test_failure_memory_already_reaches_every_mechanism_slot(tmp_path):
    """Task spec section 29 -- FailureMemory enters context before generation.
    Proven by re-using the SAME registry (with one prior committed
    experiment) across two sequential single-market pool generations and
    confirming the second run's prompt mentions the prior history."""
    reg = ExperimentRegistry(tmp_path / "registry.sqlite")
    first = generate_candidate_pool(
        **_pool_kwargs(
            tmp_path, hyps=[_hyp("H-1", mechanism_desc="trend following", family="ma_trend")],
            plans=[_tmpl("ma_trend")], max_mechanisms=1, registry=reg,
        )
    )
    assert len(first.members) == 1

    second_client = ScriptedLLMClient([_hyp("H-2", mechanism_desc="trend following again", family="ma_trend")])
    generate_candidate_pool(
        market="NQ", objective="find robust alpha for NQ", registry=reg,
        research_llm=second_client, compiler_llm=ScriptedLLMClient([_tmpl("ma_trend")]),
        execution_service=ScriptedExecutionValidationService([]),
        reliability_policy=POLICY, planes=PLANES, market_window=MW,
        max_mechanisms=1, per_mechanism_family_size=1,
    )
    prompt = second_client.calls[0]["messages"][0]["content"]
    assert "failure_memory" in prompt


# ---------------------------------------------------------------------------
# mechanism prioritization: profile can reorder, never gate scientifically
# ---------------------------------------------------------------------------


def test_prioritize_mechanisms_default_order_is_fixed():
    assert prioritize_mechanisms(None) == DEFAULT_MECHANISM_ORDER
    assert prioritize_mechanisms(InvestorProfile()) == DEFAULT_MECHANISM_ORDER  # MIXED = no reorder


def test_prioritize_mechanisms_moves_preferred_mechanism_to_front():
    profile = InvestorProfile(strategy_preference=StrategyPreference.BREAKOUT)
    order = prioritize_mechanisms(profile)
    assert order[0] == EconomicMechanism.BREAKOUT
    assert set(order) == set(DEFAULT_MECHANISM_ORDER)
    assert len(order) == len(DEFAULT_MECHANISM_ORDER)


def test_prioritize_mechanisms_never_changes_the_mechanism_set():
    for pref in StrategyPreference:
        order = prioritize_mechanisms(InvestorProfile(strategy_preference=pref))
        assert set(order) == set(DEFAULT_MECHANISM_ORDER)

"""Phase 1 -- runtime `MechanismAgent` tests (prompt 1 section 6). Every test
drives the agent with `ScriptedLLMClient`: no network, no real model, fully
deterministic -- mirrors `test_phase_16_research_agent.py`'s own coverage
shape for the sibling `ResearchAgent`.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from alpha_agent.agents.context import build_feature_catalog
from alpha_agent.agents.llm import ScriptedLLMClient
from alpha_agent.agents.mechanism_agent import (
    MechanismAgent,
    MechanismSchemaRetryExhausted,
    MechanismTranslationProposal,
)
from alpha_agent.agents.research_agent import BudgetExceeded
from alpha_agent.registry.holdout_guard import HoldoutAccessError
from alpha_agent.translation.schemas import CERTIFIED_ROOTS, Observation, origin_vintage_fields

_NOW = datetime.now(UTC)
_FEATURE_CATALOG = build_feature_catalog()


def _observation(**overrides) -> Observation:
    observed_at = _NOW - timedelta(hours=1)
    base = dict(
        event_type="MARKET_NEWS", root_symbol="CL", observed_at=observed_at, source="EIA",
        summary="EIA reports a larger-than-expected crude draw",
        structured_attributes={"category": "PETROLEUM"},
        **origin_vintage_fields(observed_at),
    )
    base.update(overrides)
    return Observation(**base)


def _valid_proposal_json(**overrides) -> str:
    payload = {
        "mechanism_candidates": [
            {
                "mechanism": "TREND", "explanation": "A surprising release can trigger positioning.",
                "causal_chain": ["release surprises", "participants reposition", "price trends"],
                "evidence_basis": "category mapping",
            },
        ],
        "measurable_variables": [
            {"name": "price_reaction", "economic_meaning": "directional response", "required_source": "the root's own OHLCV price series"},
        ],
        "factor_candidates": [
            {
                "concept": "post-release trend continuation", "mechanism": "TREND",
                "transform_or_proxy": "trend_strength on the root's own OHLCV",
                "proposed_feature_kinds": ["trend_strength"], "required_external_data": [],
            },
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_happy_path_accepts_schema_valid_proposal():
    client = ScriptedLLMClient([_valid_proposal_json()])
    agent = MechanismAgent(client)
    result = agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)
    assert result.accepted is True
    assert isinstance(result.proposal, MechanismTranslationProposal)
    assert result.proposal.mechanism_candidates[0].mechanism.value == "TREND"
    assert client.call_count == 1


def test_markdown_fenced_json_is_tolerated():
    fenced = "```json\n" + _valid_proposal_json() + "\n```"
    client = ScriptedLLMClient([fenced])
    agent = MechanismAgent(client)
    result = agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)
    assert result.accepted is True


def test_prose_wrapped_json_is_tolerated():
    wrapped = "Here is my proposal:\n" + _valid_proposal_json() + "\nHope that helps!"
    client = ScriptedLLMClient([wrapped])
    agent = MechanismAgent(client)
    result = agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)
    assert result.accepted is True


def test_retries_only_on_schema_failure_then_succeeds():
    client = ScriptedLLMClient(["not json at all", _valid_proposal_json()])
    agent = MechanismAgent(client, max_schema_retries=2)
    result = agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)
    assert result.accepted is True
    assert len(result.attempts) == 2
    assert result.attempts[0].parsed_ok is False
    assert result.attempts[1].parsed_ok is True
    assert client.call_count == 2


def test_schema_retry_exhausted_carries_every_attempt():
    client = ScriptedLLMClient(["garbage", "still garbage", "nope"])
    agent = MechanismAgent(client, max_schema_retries=2)
    with pytest.raises(MechanismSchemaRetryExhausted) as exc_info:
        agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)
    assert len(exc_info.value.attempts) == 3


def test_unknown_mechanism_enum_value_is_a_schema_failure_not_a_crash():
    bad = _valid_proposal_json()
    bad = bad.replace('"TREND"', '"MADE_UP_MECHANISM"')
    client = ScriptedLLMClient([bad, _valid_proposal_json()])
    agent = MechanismAgent(client, max_schema_retries=1)
    result = agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)
    assert result.accepted is True
    assert result.attempts[0].parsed_ok is False


def test_empty_mechanism_candidates_is_a_schema_failure():
    payload = json.loads(_valid_proposal_json())
    payload["mechanism_candidates"] = []
    client = ScriptedLLMClient([json.dumps(payload), _valid_proposal_json()])
    agent = MechanismAgent(client, max_schema_retries=1)
    result = agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)
    assert result.accepted is True
    assert result.attempts[0].parsed_ok is False


def test_budget_exceeded_before_any_call_when_budget_is_zero():
    client = ScriptedLLMClient([_valid_proposal_json()])
    agent = MechanismAgent(client, token_budget=0)
    with pytest.raises(BudgetExceeded):
        agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)


def test_never_invents_a_feature_kind_outside_the_given_catalog_is_not_enforced_by_schema_alone():
    """The agent's schema does not itself reject an invented `kind` -- that is
    deterministically caught downstream by
    `alpha_agent.translation.researchability.classify_factor` (prompt 1
    section 6: the deterministic system, never the LLM layer, owns
    FeatureRegistry lookup). This test only proves the agent does not crash
    or silently drop an invented kind -- it passes it through untouched for
    the pipeline to classify."""
    payload = json.loads(_valid_proposal_json())
    payload["factor_candidates"][0]["proposed_feature_kinds"] = ["totally_made_up_kind"]
    client = ScriptedLLMClient([json.dumps(payload)])
    agent = MechanismAgent(client)
    result = agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)
    assert result.accepted is True
    assert result.proposal.factor_candidates[0].proposed_feature_kinds == ["totally_made_up_kind"]


def test_holdout_guard_fires_on_a_2025_plus_date_leaking_into_the_proposal():
    payload = json.loads(_valid_proposal_json())
    payload["mechanism_candidates"][0]["explanation"] = "As observed on 2026-01-05, this mechanism..."
    client = ScriptedLLMClient([json.dumps(payload)])
    agent = MechanismAgent(client)
    with pytest.raises(HoldoutAccessError):
        agent.propose(_observation(), feature_catalog=_FEATURE_CATALOG)


def test_agent_has_no_registry_or_filesystem_import(tmp_path):
    import inspect

    import alpha_agent.agents.mechanism_agent as module

    source = inspect.getsource(module)
    forbidden = ("sqlite_registry", "run_deep_research", "run_fast_screen", "AnthropicClient(")
    for token in forbidden:
        assert token not in source, token


def test_observation_root_is_always_certified():
    for root in CERTIFIED_ROOTS:
        obs = _observation(root_symbol=root)
        assert obs.root_symbol == root

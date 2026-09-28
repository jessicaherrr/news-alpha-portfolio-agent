"""Alpha Discovery campaign, Part M -- Live Claude / external research
safety tests (task spec sections 66-68)."""
from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

from alpha_agent.agents.cost_tracking import (
    PRICING_TABLE_AS_OF,
    estimate_cost_for_prompt_log,
    estimate_cost_usd,
    summarize_campaign_cost,
)


def test_estimate_cost_usd_known_model():
    est = estimate_cost_usd(model="claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert est.priced is True
    assert est.usd == 12.00  # $2 in + $10 out per 1M


def test_estimate_cost_usd_unknown_model_never_fabricates():
    est = estimate_cost_usd(model="some-future-model-nobody-priced-yet", input_tokens=1000, output_tokens=1000)
    assert est.priced is False
    assert est.usd is None


def test_estimate_cost_usd_zero_tokens_is_zero_not_none():
    est = estimate_cost_usd(model="claude-opus-5", input_tokens=0, output_tokens=0)
    assert est.priced is True
    assert est.usd == 0.0


def test_pricing_as_of_is_recorded_on_every_estimate():
    est = estimate_cost_usd(model="claude-haiku-4-5", input_tokens=100, output_tokens=100)
    assert est.pricing_as_of == PRICING_TABLE_AS_OF


@dataclass(frozen=True)
class _FakePromptLog:
    model: str
    total_input_tokens: int
    total_output_tokens: int


def test_estimate_cost_for_prompt_log_duck_types_a_real_shaped_log():
    log = _FakePromptLog(model="claude-sonnet-5", total_input_tokens=500_000, total_output_tokens=100_000)
    est = estimate_cost_for_prompt_log(log)
    assert est.priced
    assert est.usd == round(500_000 * (2.00 / 1_000_000) + 100_000 * (10.00 / 1_000_000), 6)


def test_estimate_cost_matches_real_research_agent_prompt_log_shape():
    """The real alpha_agent.agents.research_agent.PromptLog must carry the
    three attributes this module reads (model, total_input_tokens,
    total_output_tokens) -- proven against the real type, not just the fake
    above."""
    from alpha_agent.agents.research_agent import PromptLog

    log = PromptLog(
        prompt_version="v1", prompt_path="x", system_prompt_sha256="a" * 64,
        context_sha256="b" * 64, model="claude-sonnet-5", temperature=0.0,
        max_output_tokens=2048, n_attempts=1, total_input_tokens=1000, total_output_tokens=500,
    )
    est = estimate_cost_for_prompt_log(log)
    assert est.priced
    assert est.usd == round(1000 * (2.00 / 1_000_000) + 500 * (10.00 / 1_000_000), 6)


def test_summarize_campaign_cost_separates_priced_from_unpriced():
    logs = [
        _FakePromptLog(model="claude-sonnet-5", total_input_tokens=1000, total_output_tokens=500),
        _FakePromptLog(model="scripted-model", total_input_tokens=1000, total_output_tokens=500),
    ]
    summary = summarize_campaign_cost(logs)
    assert summary["priced_calls"] == 1
    assert summary["unpriced_calls"] == 1
    assert summary["total_usd"] is not None
    assert "claude-sonnet-5" in summary["per_model_usd"]


def test_summarize_campaign_cost_all_unpriced_is_none_not_zero():
    logs = [_FakePromptLog(model="scripted-model", total_input_tokens=1000, total_output_tokens=500)]
    summary = summarize_campaign_cost(logs)
    assert summary["total_usd"] is None  # never a fabricated 0.0 when nothing was priceable
    assert summary["priced_calls"] == 0


# ---------------------------------------------------------------------------
# Development-mode safety (task spec section 67): no real Anthropic call, no
# real GitHub fetch, no Databento call, no broker call, no real 2025 access
# during this test suite. Structural / static checks only -- this suite
# itself never sets ANTHROPIC_API_KEY and never constructs a real client.
# ---------------------------------------------------------------------------


def test_cost_tracking_module_is_a_pure_function_no_network():
    """No import of the Anthropic SDK, requests/httpx, or any vendor client
    anywhere in the cost-tracking module -- it only does arithmetic on
    numbers it is handed."""
    import alpha_agent.agents.cost_tracking as mod

    tree = ast.parse(inspect.getsource(mod))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    forbidden = ("anthropic", "requests", "httpx", "urllib", "databento")
    for m in imported:
        assert not any(m == f or m.startswith(f + ".") for f in forbidden), (
            f"cost_tracking.py must not import {m}"
        )


def test_knowledge_base_external_adapters_report_offline_mode_by_default():
    """Part C/M's 'OFFLINE KNOWLEDGE BASE' vs 'CONNECTED RESEARCH' modes:
    the default, no-configuration state must be OFFLINE for every external
    source, never silently CONNECTED."""
    from alpha_agent.knowledge import IngestionStatus, StrategyKnowledgeBase

    kb = StrategyKnowledgeBase()
    assert all(r.status is IngestionStatus.NOT_CONNECTED for r in kb.external_ingestion_results)


def test_llm_demo_default_mode_is_scripted_offline():
    from alpha_agent.ui import llm_demo

    assert llm_demo.SCRIPTED_MODE == "scripted"
    assert llm_demo.LIVE_MODE == "live"
    # the module's own build_llm_client never reaches the network for the
    # scripted path -- proven by using it with no ANTHROPIC_API_KEY set.
    client = llm_demo.build_llm_client(llm_demo.SCRIPTED_MODE, hypothesis_json={"a": 1})
    assert type(client).__name__ == "ScriptedLLMClient"

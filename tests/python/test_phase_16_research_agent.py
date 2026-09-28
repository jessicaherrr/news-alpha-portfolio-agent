"""Phase 16 -- runtime Research Agent (prompt 16).

Every test drives the agent with :class:`ScriptedLLMClient`: no network, no real
model, fully deterministic. What is exercised:

* structured-output validation and JSON extraction (fenced / prose-wrapped),
* retry ONLY on a schema failure, with a turn budget,
* non-retryable deterministic guardrails (universe, feature catalog),
* fail-loud locked-holdout guard on both context and output,
* prompt / version logging,
* the Phase 15B failure-memory distinction: INVALID_EXECUTION history is not a
  scientific refutation, a valid scientific refusal is.
* the agent has no filesystem / registry-write / broker surface.
"""
from __future__ import annotations

import inspect
import json

import pytest
from alpha_agent.agents import (
    BudgetExceeded,
    ResearchAgent,
    ResearchContext,
    SchemaRetryExhausted,
    ScriptedLLMClient,
    build_failure_memory_digest,
    build_research_context,
)
from alpha_agent.agents import research_agent as research_agent_module
from alpha_agent.agents.context import FailureMemoryDigest
from alpha_agent.core.instrument import AssetDomain
from alpha_agent.registry.failure_memory import FailureMemoryResponse
from alpha_agent.registry.holdout_guard import HoldoutAccessError

UNIVERSE = ("ES", "NQ", "CL", "GC", "ZN")


def _spec(**overrides) -> dict:
    base = {
        "hypothesis_id": "H-1",
        "title": "ES multi-week time-series momentum",
        "economic_mechanism": (
            "Gradual diffusion of macro information leaves ES index returns "
            "positively autocorrelated at multi-week horizons."
        ),
        "universe": ["ES"],
        "horizon": "20 trading days",
        "required_features": ["trend_strength", "realized_vol"],
        "signal_description": "Long when trend_strength(50,200) > 0 and realized_vol below median.",
        "expected_regime": "trending macro regime",
        "failure_regime": "choppy range-bound regime",
        "falsification_test": "No positive OOS net PnL after costs across ES folds, or DSR below threshold.",
        "novelty_notes": "distinct market from prior NQ rejection",
    }
    base.update(overrides)
    return base


def _context(**overrides) -> ResearchContext:
    kwargs = {
        "objective": "propose a robust trend hypothesis for index futures",
        "market_universe": UNIVERSE,
        "knowledge_base": ("TSMOM earns a cross-asset premium (Moskowitz 2012).",),
        "validation_feedback": ("NQ tsmom canonical: REJECT (deflated_sharpe_below_threshold).",),
    }
    kwargs.update(overrides)
    return build_research_context(**kwargs)


# -- happy path ---------------------------------------------------------------


def test_valid_json_becomes_hypothesis_spec():
    client = ScriptedLLMClient([json.dumps(_spec())])
    agent = ResearchAgent(client)
    result = agent.propose(_context())

    assert result.accepted
    assert result.hypothesis is not None
    assert result.hypothesis.universe == ["ES"]
    assert client.call_count == 1
    assert result.prompt_log.n_attempts == 1


def test_markdown_fenced_json_is_extracted():
    fenced = "Here is the hypothesis:\n```json\n" + json.dumps(_spec()) + "\n```\n"
    agent = ResearchAgent(ScriptedLLMClient([fenced]))
    result = agent.propose(_context())
    assert result.accepted


def test_prose_wrapped_json_is_extracted():
    wrapped = "I propose: " + json.dumps(_spec()) + " -- end."
    agent = ResearchAgent(ScriptedLLMClient([wrapped]))
    assert agent.propose(_context()).accepted


# -- retry only on schema failure ------------------------------------------


def test_schema_failure_then_success_retries_once():
    client = ScriptedLLMClient(["not json at all", json.dumps(_spec())])
    agent = ResearchAgent(client, max_schema_retries=2)
    result = agent.propose(_context())

    assert result.accepted
    assert client.call_count == 2
    assert result.attempts[0].parsed_ok is False
    assert result.attempts[1].parsed_ok is True
    # the correction turn is fed back to the model
    assert any("did not validate" in m["content"] for m in client.calls[1]["messages"])


def test_schema_failure_exhausts_retry_budget():
    client = ScriptedLLMClient(["nope", "still nope", "nope again"])
    agent = ResearchAgent(client, max_schema_retries=2)
    with pytest.raises(SchemaRetryExhausted) as exc:
        agent.propose(_context())
    assert len(exc.value.attempts) == 3
    assert client.call_count == 3


def test_invalid_spec_field_is_a_schema_retry():
    bad = _spec(universe="ES")  # must be a list -> pydantic ValidationError
    client = ScriptedLLMClient([json.dumps(bad), json.dumps(_spec())])
    agent = ResearchAgent(client)
    assert agent.propose(_context()).accepted
    assert client.call_count == 2


def test_market_outside_universe_is_rejected_not_retried():
    client = ScriptedLLMClient([json.dumps(_spec(universe=["SPY"]))])
    agent = ResearchAgent(client, max_schema_retries=3)
    result = agent.propose(_context())

    assert not result.accepted
    assert result.rejection_code == "MARKET_OUTSIDE_UNIVERSE"
    assert result.hypothesis is None
    assert client.call_count == 1  # no retry


def test_unknown_feature_is_rejected_not_retried():
    client = ScriptedLLMClient([json.dumps(_spec(required_features=["moon_phase"]))])
    agent = ResearchAgent(client, max_schema_retries=3)
    result = agent.propose(_context())

    assert not result.accepted
    assert result.rejection_code == "UNKNOWN_FEATURE"
    assert client.call_count == 1


# -- locked holdout -------------------------------------------------------


def test_holdout_reference_in_output_fails_loud():
    poisoned = _spec(
        falsification_test="Evaluate on the 2025-03-01 to 2025-06-01 window."
    )
    agent = ResearchAgent(ScriptedLLMClient([json.dumps(poisoned)]))
    with pytest.raises(HoldoutAccessError):
        agent.propose(_context())


def test_context_with_holdout_date_is_refused():
    with pytest.raises(HoldoutAccessError):
        _context(knowledge_base=("Backtest showed alpha through 2025-04-01.",))


# -- budgets ------------------------------------------------------------------


def test_token_budget_stops_the_loop():
    from alpha_agent.agents.llm import LLMResponse

    big = LLMResponse(text="garbage", model="scripted-model", output_tokens=10_000)
    client = ScriptedLLMClient([big, big, big])
    agent = ResearchAgent(client, max_schema_retries=5, token_budget=15_000)
    with pytest.raises(BudgetExceeded):
        agent.propose(_context())
    # 1st call (budget 0), 2nd call (10k), then 3rd blocked at 20k >= 15k
    assert client.call_count == 2


# -- prompt / version logging ------------------------------------------------


def test_prompt_log_is_populated_and_stable():
    seen: list = []
    client = ScriptedLLMClient([json.dumps(_spec())])
    agent = ResearchAgent(client, log_sink=seen.append)
    ctx = _context()
    result = agent.propose(ctx)

    log = result.prompt_log
    assert log.prompt_version == agent.prompt_version
    assert len(log.prompt_version) == 16
    assert log.context_sha256 == ctx.sha256()
    assert log.model == "claude-sonnet-5"
    assert log.total_output_tokens > 0
    assert seen and seen[0] == log

    # deterministic: a second agent over the same template/context logs the same hashes
    agent2 = ResearchAgent(ScriptedLLMClient([json.dumps(_spec())]))
    log2 = agent2.propose(ctx).prompt_log
    assert (log2.prompt_version, log2.system_prompt_sha256, log2.context_sha256) == (
        log.prompt_version,
        log.system_prompt_sha256,
        log.context_sha256,
    )


# -- Phase 16.1: attempts vs. outcomes vs. failure records -----------------


def _phase_15b_like_response() -> FailureMemoryResponse:
    """The real schema-v5 Phase 15B shape: 60 experiments, each with one
    INVALID_EXECUTION attempt (feature-pipeline defect) and one VALID attempt
    that was refused for insufficient event density; the scientific refusal is
    backed by two STATISTICAL_INCONCLUSIVE failure *records* per identity."""
    return FailureMemoryResponse(
        query={"strategy_family": "ml_meta_label", "root_symbol": None, "params": {}},
        execution_attempts_total=120,
        valid_execution_attempts=60,
        invalid_execution_attempts=60,
        invalidation_class_counts={"FEATURE_PIPELINE_DEFECT": 60},
        valid_scientific_outcomes=60,
        valid_scientific_refusals=60,
        scientific_verdict_counts={"NOT_ADJUDICATED": 60},
        failure_class_counts={"STATISTICAL_INCONCLUSIVE": 120},
        reason_code_counts={"INSUFFICIENT_TRAIN_EVENTS": 120, "PHASE_15_TRIAL_REFUSED": 120},
        markets_tested=("CL", "ES", "GC", "NQ", "ZN"),
        lessons=("FEATURE_PIPELINE_DEFECT: weekend gaps fragmented RESET_ON_GAP windows",),
    )


def test_digest_counts_attempts_not_failure_records():
    digest = build_failure_memory_digest(_phase_15b_like_response())

    # unique execution attempts -- not the 120 failure records
    assert digest.invalid_execution_attempts == 60
    assert digest.valid_execution_attempts == 60
    assert digest.invalidation_class_counts == {"FEATURE_PIPELINE_DEFECT": 60}

    # unique valid scientific outcomes -- one per experiment
    assert digest.valid_scientific_outcomes == 60
    assert digest.valid_scientific_refusals == 60

    # the detailed record histogram is preserved separately, and may carry
    # several records per outcome
    assert digest.scientific_failure_records == {"STATISTICAL_INCONCLUSIVE": 120}

    # guidance describes attempts / refusals, never "120 attempts" or "120 hypotheses"
    assert "60 prior execution attempt(s) were INVALID_EXECUTION" in digest.guidance
    assert "60 valid scientific refusal(s)" in digest.guidance
    assert "120" not in digest.guidance


def test_failure_memory_digest_flows_into_context_and_prompt():
    ctx = _context(failure_memory=(_phase_15b_like_response(),))
    assert isinstance(ctx.failure_memory[0], FailureMemoryDigest)

    body = ctx.render_user_message()
    assert "invalid_execution_attempts" in body
    assert "valid_scientific_refusals" in body
    assert "scientific_failure_records" in body


def test_pure_engineering_history_is_not_a_refutation():
    resp = FailureMemoryResponse(
        query={"strategy_family": "tsmom", "root_symbol": "ES"},
        invalid_execution_attempts=3,
        execution_attempts_total=3,
        invalidation_class_counts={"SOFTWARE_DEFECT": 3},
    )
    digest = build_failure_memory_digest(resp)
    assert digest.invalid_execution_attempts == 3
    assert digest.valid_scientific_outcomes == 0
    assert digest.valid_scientific_refusals == 0
    assert "not a scientific verdict" in digest.guidance


def test_real_schema_v5_phase_15b_history_conveys_60_and_60():
    """Prove against the committed schema-v5 registry that agent-facing memory
    reports 60 invalid execution attempts and 60 valid scientific refusals,
    while still preserving the underlying failure-record histogram."""
    import pathlib

    from alpha_agent.registry import ExperimentRegistry
    from alpha_agent.registry.failure_memory import FailureMemory

    repo = pathlib.Path(__file__).resolve().parents[2]
    db = repo / "data" / "registry" / "experiments.sqlite"
    if not db.exists():  # pragma: no cover - registry artifact absent
        pytest.skip("committed experiment registry not present")

    fm = FailureMemory(ExperimentRegistry(db))
    families = [
        "ml_meta_label.breakout",
        "ml_meta_label.ma_trend",
        "ml_meta_label.mean_reversion",
        "ml_meta_label.tsmom",
    ]
    total_invalid = total_valid = total_refusals = 0
    total_inconclusive_records = 0
    for fam in families:
        digest = build_failure_memory_digest(
            fm.lookup(strategy_family=fam, asset_domain=AssetDomain.FUTURES)
        )
        total_invalid += digest.invalid_execution_attempts
        total_valid += digest.valid_execution_attempts
        total_refusals += digest.valid_scientific_refusals
        total_inconclusive_records += digest.scientific_failure_records.get(
            "STATISTICAL_INCONCLUSIVE", 0
        )
        assert digest.invalidation_class_counts == {"FEATURE_PIPELINE_DEFECT": 15}

    assert total_invalid == 60
    assert total_valid == 60
    assert total_refusals == 60
    # the detail histogram still carries the 120 underlying typed records
    assert total_inconclusive_records == 120


# -- no filesystem / registry-write / broker surface ----------------------


def test_agent_module_has_no_execution_or_io_surface():
    src = inspect.getsource(research_agent_module)
    for forbidden in ("subprocess", "databento", "sqlite3", "requests", "socket", "broker"):
        assert forbidden not in src, f"research agent must not reference {forbidden!r}"


def test_agent_does_not_accept_a_registry_or_path_handle():
    params = set(inspect.signature(ResearchAgent.__init__).parameters)
    # it takes a prompt template path (documented) and an LLM client, nothing that
    # can execute a backtest, write the registry, or route an order.
    assert params == {
        "self",
        "client",
        "prompt_path",
        "model",
        "max_schema_retries",
        "max_output_tokens",
        "temperature",
        "token_budget",
        "log_sink",
    }


def test_propose_rejects_non_context_input():
    agent = ResearchAgent(ScriptedLLMClient([json.dumps(_spec())]))
    with pytest.raises(TypeError):
        agent.propose({"objective": "raw dict not allowed"})


# -- AnthropicClient response normalization (mocked SDK, no network) --------


class _FakeBlock:
    def __init__(self, type_: str, text: str = ""):
        self.type = type_
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessage:
    def __init__(self, content, model, stop_reason, usage):
        self.content = content
        self.model = model
        self.stop_reason = stop_reason
        self.usage = usage


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropicSDK:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_anthropic_client_normalizes_sdk_response_and_call_shape():
    from alpha_agent.agents import AnthropicClient
    from alpha_agent.agents.llm import LLMResponse

    sdk_response = _FakeMessage(
        content=[
            _FakeBlock("text", "part one. "),
            _FakeBlock("tool_use"),  # non-text block: ignored
            _FakeBlock("text", "part two."),
        ],
        model="claude-sonnet-5-actual",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=1234, output_tokens=56),
    )
    fake_sdk = _FakeAnthropicSDK(sdk_response)
    client = AnthropicClient(client=fake_sdk)

    out = client.complete(
        system="SYSTEM PROMPT",
        messages=[{"role": "user", "content": "hello"}],
        model="claude-sonnet-5",
        max_tokens=999,
        temperature=0.0,
    )

    assert isinstance(out, LLMResponse)
    assert out.text == "part one. part two."
    assert out.model == "claude-sonnet-5-actual"
    assert out.stop_reason == "end_turn"
    assert out.input_tokens == 1234
    assert out.output_tokens == 56
    assert out.total_tokens == 1290

    # exact call shape forwarded to messages.create -- `temperature` is
    # accepted by `complete()` (and recorded above) but deliberately NOT
    # forwarded: the installed Anthropic SDK no longer exposes a sampling
    # parameter on `messages.create()` for current-generation models.
    (call,) = fake_sdk.messages.calls
    assert call == {
        "model": "claude-sonnet-5",
        "system": "SYSTEM PROMPT",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 999,
    }


def test_anthropic_client_drives_a_full_proposal():
    from alpha_agent.agents import AnthropicClient

    sdk_response = _FakeMessage(
        content=[_FakeBlock("text", json.dumps(_spec()))],
        model="claude-sonnet-5",
        stop_reason="end_turn",
        usage=_FakeUsage(input_tokens=2000, output_tokens=300),
    )
    client = AnthropicClient(client=_FakeAnthropicSDK(sdk_response))
    result = ResearchAgent(client).propose(_context())

    assert result.accepted
    assert result.prompt_log.total_input_tokens == 2000
    assert result.prompt_log.total_output_tokens == 300

"""Phase 5 (Agentic Alpha Evolution) -- the asset-neutral research spine.

Architecture-migration-only acceptance tests. Every check here proves one of
this phase's own acceptance criteria: (1) every symbol `alpha_agent.core`
exposes is the SAME object already defined elsewhere -- zero-cost aliasing,
never a fork/copy; (2) the one genuinely new type (`InstrumentIdentity`/
`AssetDomain`) does not alter any existing identity formula; (3) the package
never writes the registry, calls a network/LLM client, or triggers backtest
execution -- mirrors `test_alpha_memory_builder.py`'s own static-guard
pattern; (4) Futures workflows (`StrategySpec`, `Observation`,
`AlphaResearchObject`, `experiment_identity`) are untouched.
"""
from __future__ import annotations

import re

import pydantic
import pytest
from alpha_agent import core as core_pkg
from alpha_agent.alpha_memory import schemas as alpha_memory_schemas
from alpha_agent.context_retrieval import schemas as context_schemas
from alpha_agent.core import (
    alpha_memory as core_alpha_memory,
)
from alpha_agent.core import (
    factor as core_factor,
)
from alpha_agent.core import (
    hypothesis as core_hypothesis,
)
from alpha_agent.core import (
    instrument as core_instrument,
)
from alpha_agent.core import (
    learn_linkage as core_learn_linkage,
)
from alpha_agent.core import (
    market_context as core_market_context,
)
from alpha_agent.core import (
    mechanism as core_mechanism,
)
from alpha_agent.core import (
    observation as core_observation,
)
from alpha_agent.core import (
    validation_evidence as core_validation_evidence,
)
from alpha_agent.core.instrument import (
    AssetDomain,
    InstrumentIdentity,
    etf_instrument_identity,
    futures_instrument_identity,
)
from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.learn import concepts as learn_concepts
from alpha_agent.market_intel import event_schemas, news_schemas
from alpha_agent.registry import enums as registry_enums
from alpha_agent.schemas import hypothesis as hypothesis_schemas
from alpha_agent.strategy.spec import StrategySpec
from alpha_agent.translation import schemas as translation_schemas

core = core_pkg

_ALL_CORE_MODULES = (
    core_pkg,
    core_alpha_memory,
    core_factor,
    core_hypothesis,
    core_instrument,
    core_learn_linkage,
    core_market_context,
    core_mechanism,
    core_observation,
    core_validation_evidence,
)


# ---------------------------------------------------------------------------
# Zero-cost aliasing: every re-export IS the original object, never a copy.
# ---------------------------------------------------------------------------


def test_mechanism_is_the_same_object_not_a_copy():
    assert core.EconomicMechanism is EconomicMechanism


def test_hypothesis_is_the_same_object_not_a_copy():
    assert core.HypothesisSpec is hypothesis_schemas.HypothesisSpec


def test_factor_types_are_the_same_objects_not_copies():
    assert core.FactorCandidate is translation_schemas.FactorCandidate
    assert core.FactorIdentity is alpha_memory_schemas.FactorIdentity


def test_observation_types_are_the_same_objects_not_copies():
    assert core.Observation is translation_schemas.Observation
    assert core.MarketNewsItem is news_schemas.MarketNewsItem
    assert core.ScheduledMarketEvent is event_schemas.ScheduledMarketEvent


def test_market_context_is_the_same_object_not_a_copy():
    assert core.MarketContextFingerprint is context_schemas.MarketContextFingerprint


def test_alpha_memory_types_are_the_same_objects_not_copies():
    assert core.AlphaResearchObject is alpha_memory_schemas.AlphaResearchObject
    assert core.EvidenceProfile is alpha_memory_schemas.EvidenceProfile
    assert core.ResearchMaturity is alpha_memory_schemas.ResearchMaturity


def test_validation_evidence_types_are_the_same_objects_not_copies():
    assert core.ExperimentEvidenceRef is alpha_memory_schemas.ExperimentEvidenceRef
    assert core.RegistryVerdict is registry_enums.RegistryVerdict
    assert core.Authority is registry_enums.Authority
    assert core.TrialRole is registry_enums.TrialRole


def test_learn_linkage_is_the_same_object_not_a_copy():
    assert core.Concept is learn_concepts.Concept
    assert core.get_concept is learn_concepts.get_concept


# ---------------------------------------------------------------------------
# InstrumentIdentity / AssetDomain: additive, does not alter existing shapes.
# ---------------------------------------------------------------------------


def test_asset_domain_has_exactly_three_members_after_phase9_1_equity_approval():
    """Phase 5 froze this at one member (FUTURES only) as a deliberate
    affordance -- adding a second (Phase 6, ETF) and third (Phase 9.1, EQUITY)
    were each reserved for an explicit approval, never a silent side effect of
    some other change. The user explicitly approved "Add: AssetDomain.ETF" as
    part of Phase 6's Registry namespacing instruction, then explicitly named
    "Add AssetDomain.EQUITY at the correct shared identity boundary" as goal 2
    of the Phase 9.1 prompt (on top of the already-approved Phase 9 proposal's
    own FROZEN RESEARCH SEMANTICS gate for this exact addition,
    docs/PHASE_9_EQUITIES_SPECIALIZED_LABS_PROPOSAL.md section 4.2); this test
    records that exact, deliberate change, not an accidental one. A fourth
    member (Options, ...) still needs the same kind of explicit approval."""
    assert list(AssetDomain) == [AssetDomain.FUTURES, AssetDomain.ETF, AssetDomain.EQUITY]


def test_etf_instrument_identity_round_trips_real_tickers():
    for ticker in ("SPY", "QQQ", "IWM", "GLD", "TLT"):
        identity = etf_instrument_identity(ticker)
        assert identity.asset_domain is AssetDomain.ETF
        assert identity.symbol == ticker
        # Same symbol, different domain -> different identity (proves the
        # domain tag is load-bearing, not decorative).
        assert identity != futures_instrument_identity(ticker)


def test_root_symbol_pattern_is_byte_identical_to_strategy_spec():
    """Proven, not merely asserted similar: Phase 5 does not loosen or
    reinterpret the Futures root-symbol shape."""
    spec_pattern = StrategySpec.model_fields["root_symbol"].metadata[0].pattern
    assert core_instrument.ROOT_SYMBOL_PATTERN == spec_pattern


def test_futures_instrument_identity_round_trips_real_roots():
    for root in ("NQ", "CL", "ES", "GC", "ZN"):
        identity = futures_instrument_identity(root)
        assert identity.asset_domain is AssetDomain.FUTURES
        assert identity.symbol == root


@pytest.mark.parametrize("bad_symbol", ["nq", "too-long-symbol-name", ""])
def test_instrument_identity_rejects_a_shape_strategy_spec_would_also_reject(bad_symbol):
    with pytest.raises(pydantic.ValidationError):
        InstrumentIdentity(asset_domain=AssetDomain.FUTURES, symbol=bad_symbol)


def test_instrument_identity_is_frozen():
    identity = futures_instrument_identity("NQ")
    with pytest.raises(pydantic.ValidationError):
        identity.symbol = "ES"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# No writes / no network / no execution -- static guards, mirroring
# test_alpha_memory_builder.py's own pattern.
# ---------------------------------------------------------------------------


def test_core_package_never_calls_a_registry_write_method():
    forbidden = [
        r"\.insert_experiment\(", r"\.record_failure\(", r"\.record_lineage\(",
        r"\.apply_bundle\(", r"\.record_attempt", r"INSERT OR REPLACE", r"\bUPDATE\s+\w+\s+SET\b",
    ]
    for module in _ALL_CORE_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for pattern in forbidden:
            assert not re.search(pattern, text), f"{module.__name__} must never call {pattern!r}"


def test_core_package_never_imports_network_or_llm_clients():
    forbidden_imports = ("AnthropicClient", "databento", "import anthropic", "requests.")
    for module in _ALL_CORE_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for token in forbidden_imports:
            assert token not in text, f"{module.__name__} must never reference {token!r}"


def test_core_package_never_triggers_backtest_or_paper_trading_execution():
    forbidden_tokens = (
        "StrategyCompilerAgent", "run_deep_research", "CppEngineRunner",
        "PaperTradingEngine", "run_targets_backtest",
    )
    for module in _ALL_CORE_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for token in forbidden_tokens:
            assert token not in text, f"{module.__name__} must never reference {token!r}"


def test_core_package_has_no_streamlit_or_ui_import():
    """Backend spine only -- mirrors alpha_agent.learn's own "no streamlit
    import" discipline (learn-and-failure-autopsy)."""
    for module in _ALL_CORE_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        assert "streamlit" not in text, f"{module.__name__} must not import streamlit"
        assert "import alpha_agent.ui" not in text and "from alpha_agent.ui" not in text, (
            f"{module.__name__} must not import the ui package"
        )


def test_core_package_never_names_an_etf_equity_options_capability():
    """No asset-class scaffolding beyond FUTURES exists yet (Phase 5
    instruction: 'no ETF capability required')."""
    forbidden_tokens = ("ETFCapability", "EquityCapability", "OptionsCapability", "class ETF", "class Equity")
    for module in _ALL_CORE_MODULES:
        with open(module.__file__, encoding="utf-8") as fh:
            text = fh.read()
        for token in forbidden_tokens:
            assert token not in text, f"{module.__name__} must not name {token!r}"

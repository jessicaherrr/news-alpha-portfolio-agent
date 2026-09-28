"""Phase 20 -- the AI Research Agent / Strategy Lab demo scenario bank.

The runtime `ResearchAgent` (Phase 16) and `StrategyCompilerAgent` (Phase 17)
are LLM-backed: they propose, they never decide. This module supplies:

* a deterministic, offline `ScriptedLLMClient` scenario per curated research
  objective, so the whole propose -> compile -> registry-lookup pipeline runs
  for a demo with zero network calls and zero cost (the default UI mode);
* an opt-in live mode that builds a real `AnthropicClient`. This app never
  reads, parses, stores, or displays an API key -- `AnthropicClient()` is
  constructed with no key argument and the Anthropic SDK reads
  `ANTHROPIC_API_KEY` from the process environment itself. If the key is
  absent the SDK raises at call time and the UI surfaces a plain "no
  credentials" message, never a stack trace containing key material.

Every canned hypothesis/plan pair below is hand-built to satisfy the same
deterministic guardrails a real model output must pass (approved universe,
registered feature kinds, HypothesisSpec<->compiled-spec fidelity) -- these are
not shortcuts around Phase 16/17, they are inputs a real model could have
produced, replayed through the real, unmodified agents.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from alpha_agent.agents import (
    AnthropicClient,
    CompiledStrategyProposal,
    CompilerContext,
    KnownStrategyRecord,
    LLMClient,
    LLMClientUnavailable,
    ResearchAgent,
    ResearchContext,
    ResearchProposal,
    ScriptedLLMClient,
    StrategyCompilerAgent,
    build_compiler_context,
    build_research_context,
)
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.failure_memory import FailureMemory, FailureMemoryResponse
from alpha_agent.registry.failure_memory import (
    relevant_failure_memory as _registry_relevant_failure_memory,
)
from alpha_agent.registry.sqlite_registry import ExperimentRegistry, RegistrySummary
from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.ui import services


@dataclass(frozen=True)
class ResearchScenario:
    key: str
    label: str
    objective: str
    hypothesis: dict
    plan: dict


SCENARIOS: tuple[ResearchScenario, ...] = (
    ResearchScenario(
        key="tsmom_nq",
        label="Multi-week time-series momentum in NQ",
        objective="Propose a robust multi-week trend-following hypothesis for an index future.",
        hypothesis={
            "hypothesis_id": "H-DEMO-TSMOM-NQ",
            "title": "NQ multi-week time-series momentum",
            "economic_mechanism": (
                "Gradual diffusion of macro / earnings information leaves index "
                "futures returns positively autocorrelated at multi-week horizons."
            ),
            "universe": ["NQ"],
            "horizon": "20-120 trading days",
            "required_features": ["diff"],
            "signal_description": "Long when both a fast and a slow price change are positive; short when both are negative.",
            "expected_regime": "trending macro regime",
            "failure_regime": "choppy, range-bound regime",
            "falsification_test": "No positive OOS net PnL after costs, or deflated Sharpe below the frozen DSR threshold.",
            "novelty_notes": "Canonical Phase 11 template; a known prior REJECT exists in the registry for this exact family/root -- proposed to demonstrate duplicate/failure-memory retrieval, not as a novel claim.",
        },
        plan={
            "expressible": True,
            "template": {
                "family_key": "tsmom",
                "root_symbol": "NQ",
                "params": {"fast_horizon": 20, "slow_horizon": 120, "size": 1},
            },
            "rationale": "Multi-horizon trend maps directly to the tsmom convenience family.",
        },
    ),
    ResearchScenario(
        key="mean_reversion_cl",
        label="Short-term mean reversion in CL after volatility spikes",
        objective="Propose a short-horizon mean-reversion hypothesis for an energy future.",
        hypothesis={
            "hypothesis_id": "H-DEMO-MEANREV-CL",
            "title": "CL short-term mean reversion",
            "economic_mechanism": (
                "Order-flow-driven overshoots in crude oil futures after a sharp "
                "move tend to partially revert within days as liquidity providers "
                "re-price the excess."
            ),
            "universe": ["CL"],
            "horizon": "5-20 trading days",
            "required_features": ["zscore"],
            "signal_description": "Long when the rolling z-score of price is below -entry_z; exit near the mean.",
            "expected_regime": "range-bound, mean-reverting regime",
            "failure_regime": "sustained directional trend regime",
            "falsification_test": "No positive OOS net PnL after costs, or gating null not rejected.",
            "novelty_notes": "Canonical Phase 11 template, proposed to exercise the mean_reversion family end to end.",
        },
        plan={
            "expressible": True,
            "template": {
                "family_key": "mean_reversion",
                "root_symbol": "CL",
                "params": {"zscore_window": 20, "entry_z": 2.0, "exit_z": 0.5, "size": 1},
            },
            "rationale": "Overshoot-and-revert maps to the mean_reversion convenience family.",
        },
    ),
    ResearchScenario(
        key="breakout_gc",
        label="Range breakout continuation in GC",
        objective="Propose a breakout-continuation hypothesis for a metals future.",
        hypothesis={
            "hypothesis_id": "H-DEMO-BREAKOUT-GC",
            "title": "GC range breakout continuation",
            "economic_mechanism": (
                "A break of a multi-month price range in gold futures attracts "
                "trend-following flow that extends the move for several weeks."
            ),
            "universe": ["GC"],
            "horizon": "10-55 trading days",
            "required_features": ["breakout_up", "breakout_down"],
            "signal_description": "Long on a new N-day high breakout, short on a new N-day low breakout.",
            "expected_regime": "regime shift / new-range formation",
            "failure_regime": "mean-reverting chop inside an established range",
            "falsification_test": "No positive OOS net PnL after costs across GC folds.",
            "novelty_notes": "Canonical Phase 11 template, proposed to exercise the breakout family end to end.",
        },
        plan={
            "expressible": True,
            "template": {
                "family_key": "breakout",
                "root_symbol": "GC",
                "params": {"lookback": 55, "size": 1},
            },
            "rationale": "Breakout continuation maps to the breakout convenience family.",
        },
    ),
    ResearchScenario(
        key="ma_trend_zn",
        label="Moving-average trend following in ZN",
        objective="Propose a classic dual moving-average trend hypothesis for a rates future.",
        hypothesis={
            "hypothesis_id": "H-DEMO-MATREND-ZN",
            "title": "ZN dual moving-average trend",
            "economic_mechanism": (
                "Slow-moving monetary-policy expectations create sustained trends "
                "in Treasury futures that a fast/slow moving-average crossover "
                "captures with a lag."
            ),
            "universe": ["ZN"],
            "horizon": "50-200 trading days",
            "required_features": ["ma"],
            "signal_description": "Long when the fast moving average is above the slow moving average, short otherwise.",
            "expected_regime": "sustained rate-cycle trend",
            "failure_regime": "choppy, range-bound rate regime",
            "falsification_test": "No positive OOS net PnL after costs, or parameter neighbourhood unstable.",
            "novelty_notes": "Canonical Phase 11 template, proposed to exercise the ma_trend family end to end.",
        },
        plan={
            "expressible": True,
            "template": {
                "family_key": "ma_trend",
                "root_symbol": "ZN",
                "params": {"fast_window": 50, "slow_window": 200, "size": 1},
            },
            "rationale": "Dual moving-average crossover maps to the ma_trend convenience family.",
        },
    ),
    ResearchScenario(
        key="trend_vol_regime_nq_novel",
        label="Trend conditioned on a volatility regime in NQ (novel blueprint)",
        objective=(
            "Propose a trend hypothesis for an index future that is conditioned on "
            "a realized-volatility regime, expressed as a full blueprint rather than "
            "a Phase 11 convenience template."
        ),
        hypothesis={
            "hypothesis_id": "H-DEMO-TRENDVOL-NQ",
            "title": "NQ trend persistence conditioned on a low realized-volatility regime",
            "economic_mechanism": (
                "Trend signals in index futures are more reliable when realized "
                "volatility is contained, because a low-vol regime indicates orderly "
                "positioning rather than a volatility-driven repricing that a naive "
                "trend rule would misread as momentum."
            ),
            "universe": ["NQ"],
            "horizon": "20-100 trading days",
            "required_features": ["ma_spread", "volatility"],
            "signal_description": (
                "Long when the 20/100 moving-average spread is positive AND 60-day "
                "realized volatility is below 2%; short when the spread is negative "
                "(regardless of the volatility regime)."
            ),
            "expected_regime": "low-volatility trending regime",
            "failure_regime": "volatility-spike regime with no genuine trend",
            "falsification_test": "No positive OOS net PnL after costs, or the volatility filter fails to raise Sharpe versus the unconditional trend rule.",
            "novelty_notes": "Not a Phase 11 template: expressed as a closed blueprint over registered ma_spread/volatility features to demonstrate the full Strategy DSL path.",
        },
        plan={
            "expressible": True,
            "blueprint": {
                "root_symbol": "NQ",
                "features": [
                    {"alias": "msp", "kind": "ma_spread", "params": {"fast": 20, "slow": 100}},
                    {"alias": "vol", "kind": "volatility", "params": {"window": 60}},
                ],
                "rules": [
                    {
                        "rule_id": "trend_long",
                        "target_units": 1,
                        "when": {
                            "type": "boolean",
                            "op": "all",
                            "nodes": [
                                {
                                    "type": "comparison", "op": "gt",
                                    "left": {"type": "feature", "feature": "msp"},
                                    "right": {"type": "const", "value": 0.0},
                                },
                                {
                                    "type": "comparison", "op": "lt",
                                    "left": {"type": "feature", "feature": "vol"},
                                    "right": {"type": "const", "value": 0.02},
                                },
                            ],
                        },
                    },
                    {
                        "rule_id": "trend_short",
                        "target_units": -1,
                        "when": {
                            "type": "comparison", "op": "lt",
                            "left": {"type": "feature", "feature": "msp"},
                            "right": {"type": "const", "value": 0.0},
                        },
                    },
                ],
                "default_action": "flat",
            },
            "rationale": "A regime-conditioned trend rule needs a volatility gate the closed templates do not expose; expressed as a blueprint over two registered features.",
        },
    ),
)

_BY_KEY: dict[str, ResearchScenario] = {s.key: s for s in SCENARIOS}


def scenario(key: str) -> ResearchScenario:
    return _BY_KEY[key]


def family_and_root_for_hypothesis(hypothesis_id: str) -> tuple[str | None, str | None]:
    """The demo scenario's own predeclared family/root for a proposed
    hypothesis (matched by `hypothesis_id`) -- used only to look up REAL
    failure-memory / registry evidence for the conversation; never fed back
    into the agent as if the agent had chosen it. `None, None` (or a root
    with no family) for the one blueprint-only scenario, which has no Phase
    11 convenience family."""
    for sc in SCENARIOS:
        if sc.hypothesis["hypothesis_id"] == hypothesis_id:
            family = sc.plan.get("template", {}).get("family_key")
            root = (
                sc.plan.get("template", {}).get("root_symbol")
                or sc.plan.get("blueprint", {}).get("root_symbol")
            )
            return family, root
    return None, None


# ---------------------------------------------------------------------------
# Root consistency -- Root is the authoritative research target, never the
# curated preset's own predeclared root.
# ---------------------------------------------------------------------------
#
# Each curated `ResearchScenario` hard-codes a canonical (family, root) pair
# (tsmom/NQ, mean_reversion/CL, breakout/GC, ma_trend/ZN, .../NQ) so the demo
# reads as a coherent story. But the Agent page's Root selector narrows
# `ResearchContext.market_universe` to exactly one user-chosen market, and
# `ResearchAgent.propose` HONESTLY rejects (`MARKET_OUTSIDE_UNIVERSE`) any
# hypothesis whose declared universe falls outside it -- correctly, since a
# real model could propose a market it was not asked about. A scripted/
# offline proposal must never trip that same guard just because a preset
# happened to be written for a different root: these two functions re-root a
# curated hypothesis/plan pair to the caller's ACTUAL selected root before it
# is ever handed to `ScriptedLLMClient`, so the deterministic proposal path
# always agrees with the approved universe it was given. Never a change to
# the guard itself.


def rooted_hypothesis(hypothesis: dict, root: str) -> dict:
    """Re-root a curated demo hypothesis to `root`. No-op if it already
    declares exactly `[root]`. Rewrites `universe` and every whole-word
    mention of the ORIGINAL predeclared root in the other free-text fields
    (title, id, mechanism, signal, novelty notes) so the returned hypothesis
    never talks about one market while declaring universe=[another]."""
    original_universe = tuple(hypothesis.get("universe") or ())
    if original_universe == (root,):
        return hypothesis
    out = dict(hypothesis)
    out["universe"] = [root]
    original_root = original_universe[0] if len(original_universe) == 1 else None
    if original_root and original_root != root:
        for field in (
            "hypothesis_id", "title", "economic_mechanism",
            "signal_description", "novelty_notes",
        ):
            value = out.get(field)
            if isinstance(value, str):
                out[field] = _retoken_root(value, original_root, root)
    return out


def rooted_plan(plan: dict, root: str) -> dict:
    """Re-root a curated demo compiled-strategy plan to `root`: whichever of
    `template`/`blueprint` is present gets its own `root_symbol` overwritten.
    No-op (returns the same dict shape) when the plan already targets `root`."""
    out = dict(plan)
    for key in ("template", "blueprint"):
        section = out.get(key)
        if section:
            rewritten = dict(section)
            rewritten["root_symbol"] = root
            out[key] = rewritten
    return out


def _retoken_root(text: str, old_root: str, new_root: str) -> str:
    """Whole-word substitution only -- a root symbol never matches inside an
    unrelated word (these are short, all-caps exchange tickers)."""
    return re.sub(rf"\b{re.escape(old_root)}\b", new_root, text)


# ---------------------------------------------------------------------------
# LLM client construction -- the ONLY place this app touches a model client
# ---------------------------------------------------------------------------

LIVE_MODE = "live"
SCRIPTED_MODE = "scripted"


def build_llm_client(mode: str, *, hypothesis_json: dict | None = None, plan_json: dict | None = None) -> LLMClient:
    """``mode == "scripted"`` (default): a `ScriptedLLMClient` pre-loaded with
    the canned JSON responses for this call, replayed through the real agent.
    ``mode == "live"``: a real `AnthropicClient()` with NO key argument -- the
    SDK reads `ANTHROPIC_API_KEY` from the environment. This module never
    inspects that variable."""
    if mode == LIVE_MODE:
        return AnthropicClient()
    responses = []
    if hypothesis_json is not None:
        responses.append(json.dumps(hypothesis_json))
    if plan_json is not None:
        responses.append(json.dumps(plan_json))
    return ScriptedLLMClient(responses)


# ---------------------------------------------------------------------------
# end-to-end calls through the REAL Phase 16/17 agents
# ---------------------------------------------------------------------------


def relevant_failure_memory(
    *, market_universe: tuple[str, ...], registry: object | None = None
) -> tuple[FailureMemoryResponse, ...]:
    """PRE-PROPOSAL failure memory: every ``(strategy_family, root_symbol)``
    combination already tested for a root in ``market_universe``, looked up
    BEFORE any hypothesis is proposed (runtime-integration release, section 3).

    This is bounded, structured research memory -- not a raw dump of every
    registry row: it is a set of already-aggregated `FailureMemoryResponse`
    digests (verdict counts, reason codes, INVALID_EXECUTION vs. scientific-
    refusal counts, superseded evidence) for exactly the (family, root)
    neighbourhoods that already have evidence for an approved market, and
    nothing else. A brand-new market/family combination with zero prior
    experiments contributes nothing here (there is nothing bounded to report),
    which is the correct, honest answer -- the agent still sees an empty
    ``failure_memory`` for that case, exactly as it would for a genuinely novel
    proposal.

    Previously this system only looked up failure memory AFTER a hypothesis
    already existed, by matching a curated demo scenario's own predeclared
    family/root -- which cannot work at all for a real free-text objective or a
    live Claude proposal (there is no scenario to match against). Sweeping the
    registry BEFORE proposing, over every root the agent is allowed to
    consider, is what lets the agent know (for example) that ordinary
    multi-week NQ TSMOM has already been scientifically rejected before it
    proposes anything.

    Thin wrapper over :func:`alpha_agent.registry.failure_memory.
    relevant_failure_memory` -- `ResearchOrchestrator._planning_context` uses
    the exact same shared sweep, so both surfaces agree.

    Scoped to :class:`AssetDomain.FUTURES` -- every demo scenario and every
    real registry row today is Futures research; this demo/UI surface has no
    ETF wiring yet (Phase 6 ETF Research Pilot's UI integration is a separate,
    later phase), so the domain is a stated fact here, not a new parameter.
    """
    if registry is not None:
        return _registry_relevant_failure_memory(
            registry, market_universe=market_universe, asset_domain=AssetDomain.FUTURES
        )
    with services.open_registry() as reg:
        return _registry_relevant_failure_memory(
            reg, market_universe=market_universe, asset_domain=AssetDomain.FUTURES
        )


def build_context_for_objective(
    *, objective: str, universe: tuple[str, ...],
    strategy_family: str | None = None, root_symbol: str | None = None,
    extra_knowledge: tuple[str, ...] = (),
) -> tuple[ResearchContext, tuple[FailureMemoryResponse, ...]]:
    """Assemble a real `ResearchContext`, including a live registry digest and
    PRE-PROPOSAL failure memory (:func:`relevant_failure_memory`) -- exactly
    what Phase 18's orchestrator hands the agent, just invoked directly for one
    interactive proposal.

    ``objective`` is the ACTUAL text the caller wants the agent to see -- it is
    never silently replaced by a curated scenario's own objective (runtime-
    integration release, section 2).

    ``strategy_family`` / ``root_symbol``, when already known (e.g. inspecting
    a specific prior demo hypothesis), ADD one more targeted lookup on top of
    the universe-wide sweep; they narrow nothing away.

    ``extra_knowledge`` appends additional free-text strings to
    ``knowledge_base`` -- the ONE sanctioned, already-frozen extension point
    for background prose the agent may read (never a feature, never scientific
    evidence). The Agent page uses this to pass an OBSERVATIONAL_CONTEXT_ONLY
    delayed-market-quote note (see `alpha_agent.marketdata`); it changes
    nothing about `ResearchContext`'s schema or the agent's authority.
    """
    with services.open_registry() as reg:
        summary: RegistrySummary = reg.summary()
        fm_list = list(relevant_failure_memory(market_universe=universe, registry=reg))
        if strategy_family is not None:
            targeted = FailureMemory(reg).lookup(
                strategy_family=strategy_family, root_symbol=root_symbol,
                asset_domain=AssetDomain.FUTURES,
            )
            if targeted.query not in (fm.query for fm in fm_list):
                fm_list.append(targeted)
    context = build_research_context(
        objective=objective,
        market_universe=universe,
        knowledge_base=(
            "TSMOM earns a documented cross-asset premium (Moskowitz, Ooi & Pedersen 2012).",
            "Reliability adjudication (BH-FDR + DSR) is frozen and cannot be relaxed by a proposal.",
            *extra_knowledge,
        ),
        registry_summary=summary,
        failure_memory=fm_list,
        validation_feedback=(),
    )
    return context, tuple(fm_list)


def propose_hypothesis(
    *, mode: str, scenario_key: str, universe: tuple[str, ...],
    objective: str | None = None, extra_knowledge: tuple[str, ...] = (),
) -> tuple[ResearchProposal | None, str | None, tuple[FailureMemoryResponse, ...]]:
    """Returns ``(proposal, error, failure_memory)``. ``error`` is a plain
    user-facing string (never a raw exception / stack trace, never key
    material). ``failure_memory`` is exactly the pre-proposal evidence that was
    placed in the `ResearchContext` handed to the agent -- callers render THIS,
    never a second, independent post-hoc lookup, so the UI can never show
    memory the agent did not actually see.

    ``objective`` is the text actually sent to the agent. It defaults to the
    scenario's own objective ONLY when the caller passes none (e.g. an
    unedited curated preset) -- a caller with a real user-edited or free-text
    objective must pass it explicitly (runtime-integration release, section 2).
    ``scenario_key`` in scripted mode still selects which canned JSON response
    the `ScriptedLLMClient` replays; in live mode it is unused.

    ``universe`` is the caller's ACTUAL approved market universe for this
    call. When it names exactly one market -- the Agent page's Root selector
    always narrows to exactly one -- the scripted scenario's own predeclared
    hypothesis is re-rooted to it (:func:`rooted_hypothesis`) before it is
    ever handed to `ScriptedLLMClient`, so a curated preset written for a
    different root (e.g. the `tsmom_nq` preset selected while Root=CL) can
    never trip `MARKET_OUTSIDE_UNIVERSE` -- Root is the authoritative research
    target, not the preset. A multi-market universe (e.g. the unnarrowed
    approved universe) is left exactly as the preset declared it.
    """
    sc = scenario(scenario_key)
    resolved_objective = objective if objective is not None else sc.objective
    target_root = universe[0] if len(universe) == 1 else None
    hypothesis_json = (
        sc.hypothesis if target_root is None else rooted_hypothesis(sc.hypothesis, target_root)
    )
    try:
        client = build_llm_client(mode, hypothesis_json=hypothesis_json)
        agent = ResearchAgent(client)
        context, fm = build_context_for_objective(
            objective=resolved_objective, universe=universe, extra_knowledge=extra_knowledge,
        )
        proposal = agent.propose(context)
        return proposal, None, fm
    except LLMClientUnavailable as exc:
        return None, f"Live LLM unavailable: {exc}", ()
    except Exception as exc:  # noqa: BLE001 -- defensive UI boundary, never a raw stack trace or key material
        return None, f"{type(exc).__name__}: {exc}", ()


def _live_known_strategies(registry: ExperimentRegistry) -> tuple[KnownStrategyRecord, ...]:
    """Strategy-fingerprint evidence from the LIVE registry, tagged
    ``source="live_experiment_registry"`` -- mirrors
    `ResearchOrchestrator._build_compiler_context` exactly (runtime-integration
    release, section 8), so the compiled-strategy card's "seen before" evidence
    and the evidence card's registry lookup are never contradictory: both are
    live-registry-sourced. `build_compiler_context` additionally merges in the
    frozen Phase 13.5C candidate-manifest fingerprints
    (``source="frozen_candidate_manifest"``) -- the UI must label those
    precisely as "present in the frozen candidate manifest", never as a live
    scientific duplicate."""
    known: list[KnownStrategyRecord] = []
    for view in registry.experiments(authoritative_only=False):
        fp = view.experiment.strategy_fingerprint
        if not fp:
            continue
        known.append(
            KnownStrategyRecord(
                source="live_experiment_registry",
                strategy_fingerprint=fp,
                experiment_id=view.experiment_id,
                experiment_identity=view.experiment_identity,
                has_valid_authoritative_result=view.has_valid_authoritative_result,
                valid_execution_attempts=view.n_valid_attempts,
                invalid_execution_attempts=view.n_invalid_attempts,
            )
        )
    return tuple(known)


def compile_hypothesis(
    *, mode: str, scenario_key: str, hypothesis: HypothesisSpec, universe: tuple[str, ...]
) -> tuple[CompiledStrategyProposal | None, str | None]:
    """``hypothesis`` is the ACCEPTED `HypothesisSpec` from the proposal step
    above -- its own `universe` (already root-consistent, see
    :func:`propose_hypothesis`) is what the scripted compiled-strategy plan is
    re-rooted to (:func:`rooted_plan`), so the compiled `StrategySpec.root_symbol`
    can never disagree with the hypothesis it was compiled from."""
    sc = scenario(scenario_key)
    target_root = hypothesis.universe[0] if len(hypothesis.universe) == 1 else None
    plan_json = sc.plan if target_root is None else rooted_plan(sc.plan, target_root)
    try:
        client = build_llm_client(mode, plan_json=plan_json)
        agent = StrategyCompilerAgent(client)
        with services.open_registry() as reg:
            known = _live_known_strategies(reg)
        context: CompilerContext = build_compiler_context(
            approved_universe=universe, known_strategies=known
        )
        proposal = agent.compile_hypothesis(hypothesis, context)
        return proposal, None
    except LLMClientUnavailable as exc:
        return None, f"Live LLM unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 -- defensive UI boundary, never a raw stack trace or key material
        return None, f"{type(exc).__name__}: {exc}"

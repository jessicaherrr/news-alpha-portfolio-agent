"""Evidence-bound response composition for the conversational Agent
(Release UX Part A/I). This is the ONE place a free-text turn is routed
(`alpha_agent.agents.conversation.classify_intent`) and answered using the
SAME read-only boundaries every other page already uses --
`alpha_agent.ui.services` (registry evidence), `alpha_agent.ui.
databento_context` (current/delayed market observation), and
`alpha_agent.recommendation` (profile/fit) -- never a second, parallel
evidence path.

READ-ONLY, STRUCTURALLY (Part A section 4): this module never imports a
scientific EXECUTION entry point -- no `run_fast_screen`, no `freeze_top_k` /
`adopt_frozen_candidate_set`, no `run_strict_validation`, no
`run_deep_research`, no C++ engine runner. A turn classified as one of the
three action intents (`RUN_FAST_SCREEN` / `FREEZE_CANDIDATES` /
`RUN_STRICT_VALIDATION`) always produces a message asking the user to click
the corresponding explicit action -- it is architecturally impossible for
this module to run one itself (`tests/python/test_conversation_engine.py`'s
static import check enforces this, not just a docstring promise).

DIFFERENT QUESTIONS USE DIFFERENT EVIDENCE (task spec section 3): a market
question is answered from `databento_context` snapshots, never invented
macro commentary; a validation question is answered from the registry's own
`plain_language_reason` / `gate_evidence_text`, never a new judgment; a
source question is answered from the hypothesis's own recorded
`source_inspirations`, never a fabricated citation.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from alpha_agent.agents.conversation import (
    ACTION_INTENTS,
    ConversationSignalContext,
    ResearchConversationIntent,
    classify_intent,
)
from alpha_agent.news_alpha import DOMAIN_LABELS, UserDescribedEvent
from alpha_agent.recommendation.fit import score_user_fit
from alpha_agent.recommendation.profile import (
    MaxDrawdown,
    OvernightPreference,
    TradingFrequency,
)
from alpha_agent.translation.pipeline import render_summary_text
from alpha_agent.ui import (
    asset_expression_view,
    candidate_signal_view,
    databento_context,
    mechanism_graph_view,
    news_alpha_context,
    services,
    signal_path_view,
    signal_ranking_view,
    translation_context,
)

__all__ = ["ConversationBoundContext", "ConversationResponse", "handle_conversation_turn"]


class ConversationBoundContext(BaseModel):
    """Whatever the current page/session already has assembled about ONE
    candidate -- the exact same shape `agent.py`'s `_build_research_details_
    target` already carries to Research Details. Passing this (rather than
    re-deriving from an experiment id) guarantees the conversation answers
    about EXACTLY the object the user is looking at, never a fresh, possibly
    different lookup."""

    model_config = {"frozen": True, "extra": "forbid"}

    root: str | None = None
    hypothesis: dict | None = None
    compiled: dict | None = None
    evidence: dict | None = None  # services.find_registry_evidence_for_compiled(...) shape


class ConversationResponse(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    intent: ResearchConversationIntent
    text: str
    evidence: dict = Field(default_factory=dict)
    suggested_action: str | None = None
    citations: tuple[str, ...] = ()
    #: Release UX product-consolidation acceptance pass, task spec section 5:
    #: the SMALLEST typed signal that "a response proposed a concrete,
    #: falsifiable research hypothesis" -- set ONLY by a structured extraction
    #: (see `claude_conversation._extract_proposed_hypothesis`), never by a
    #: substring search over `text` (e.g. "NOT VALIDATED" appearing anywhere).
    #: This means only "Claude proposed a research hypothesis" -- never
    #: validated, accepted, PASS, or scientifically supported; see
    #: `ConversationResponse`'s callers for the explicit-action-only contract.
    proposed_hypothesis: str | None = None


def handle_conversation_turn(
    text: str,
    *,
    context: ConversationBoundContext | None = None,
    has_history: bool = False,
) -> ConversationResponse:
    context = context or ConversationBoundContext()
    roots = tuple(services.approved_universe())
    sig_ctx = ConversationSignalContext(
        approved_roots=roots,
        has_bound_strategy=bool(context.hypothesis or context.compiled),
        has_history=has_history,
    )
    classification = classify_intent(text, context=sig_ctx)
    handler = _HANDLERS.get(classification.intent, _handle_general_quant_question)
    return handler(text, classification, context)


# ---------------------------------------------------------------------------
# market plane -- OBSERVATION ONLY, never a scientific claim
# ---------------------------------------------------------------------------


def _resolve_root(classification, context: ConversationBoundContext) -> str:
    if classification.mentioned_roots:
        return classification.mentioned_roots[0]
    return context.root or "NQ"


def _describe_snapshot(root: str) -> tuple[str, dict]:
    snap = databento_context.market_snapshot(root)
    if snap is None or snap.last is None:
        return (
            (
                f"I don't have current market data for {root} right now "
                f"(provider not connected, or no capability proven for this key)."
            ),
            {"root": root, "available": False},
        )
    parts = [f"{root} last {snap.last:g}, source Databento, capability {snap.capability.value}, as of {snap.as_of.isoformat()}."]
    if snap.change_pct is not None:
        parts.append(f"Change vs. prior close: {snap.change_pct:+.2f}%.")
    if snap.session is not None:
        parts.append(
            f"Session high/low: {snap.session.session_high:g} / {snap.session.session_low:g} "
            f"({snap.session.session_label})."
        )
    if snap.freshness_seconds is not None:
        parts.append(f"Latest observation is {snap.freshness_seconds / 3600:.1f} hour(s) old.")
    return " ".join(parts), {"root": root, "available": True, "snapshot": snap.model_dump(mode="json")}


def _handle_market_question(text, classification, context) -> ConversationResponse:
    root = _resolve_root(classification, context)
    narrative, evidence = _describe_snapshot(root)
    return ConversationResponse(intent=classification.intent, text=narrative, evidence=evidence)


def _handle_causal_current_event(text, classification, context) -> ConversationResponse:
    root = _resolve_root(classification, context)
    narrative, evidence = _describe_snapshot(root)
    narrative += (
        " I do not currently have enough sourced evidence to attribute this move causally -- "
        "no trusted current-news source is connected, so I will not guess at a macro cause."
    )
    return ConversationResponse(intent=classification.intent, text=narrative, evidence=evidence)


def _handle_market_comparison(text, classification, context) -> ConversationResponse:
    roots = classification.mentioned_roots[:2]
    if len(roots) < 2:
        return ConversationResponse(
            intent=classification.intent,
            text="Tell me which two markets to compare (e.g. \"compare ES and NQ\").",
            evidence={},
        )
    lines = []
    snaps = {}
    for root in roots:
        narrative, ev = _describe_snapshot(root)
        lines.append(narrative)
        snaps[root] = ev
    return ConversationResponse(intent=classification.intent, text=" ".join(lines), evidence={"snapshots": snaps})


# ---------------------------------------------------------------------------
# scientific evidence plane -- registry-grounded, never a new judgment
# ---------------------------------------------------------------------------


def _handle_validation_explanation(text, classification, context) -> ConversationResponse:
    evidence = context.evidence or {}
    result = evidence.get("result")
    exp = evidence.get("experiment") or {}
    if not result:
        return ConversationResponse(
            intent=classification.intent,
            text="I don't have a bound experiment with committed validation evidence to explain yet -- "
                 "open a candidate's Research Details, or ask me to research one first.",
            evidence={},
        )
    trial_role = exp.get("trial_role", "CANONICAL")
    verdict = result.get("headline_verdict")
    reason = services.plain_language_reason(result=result, trial_role=trial_role, verdict=verdict)
    stats = services.gate_evidence_text(result)
    stat_line = "; ".join(f"{k}: {v}" for k, v in stats.items())
    narrative = f"Verdict: {verdict or 'NOT_ADJUDICATED'}. {reason}"
    if stat_line:
        narrative += f" Supporting evidence -- {stat_line}."
    return ConversationResponse(
        intent=classification.intent, text=narrative,
        evidence={"gate_table": services.gate_table(reason_codes=tuple(result.get("reason_codes") or ()), verdict=verdict, trial_role=trial_role)},
    )


def _handle_strategy_explanation(text, classification, context) -> ConversationResponse:
    hyp = context.hypothesis
    if not hyp:
        return ConversationResponse(
            intent=classification.intent,
            text="I don't have a bound hypothesis/strategy to explain yet -- propose or open one first.",
            evidence={},
        )
    parts = [f"Economic mechanism: {hyp.get('economic_mechanism', 'unspecified')}."]
    if hyp.get("novelty_notes"):
        parts.append(f"How it differs from prior work: {hyp['novelty_notes']}")
    parts.append(f"What would falsify it: {hyp.get('falsification_test', 'not stated')}")
    result = (context.evidence or {}).get("result")
    if result:
        verdict = result.get("headline_verdict")
        parts.append(
            f"Historical result: net PnL ${result.get('net_pnl_usd', 0):,.0f}, "
            f"Sharpe {result.get('annualized_sharpe', 0):.2f}, verdict {verdict}."
        )
    return ConversationResponse(intent=classification.intent, text=" ".join(parts), evidence={"hypothesis": hyp})


def _handle_source_question(text, classification, context) -> ConversationResponse:
    hyp = context.hypothesis or {}
    citations = tuple(hyp.get("source_inspirations") or ())
    if not citations:
        return ConversationResponse(
            intent=classification.intent,
            text="No sourced inspirations were recorded for this proposal -- it may have been generated "
                 "without external knowledge-base context, or the source lineage was not carried into this session.",
            evidence={},
        )
    narrative = "This proposal cited: " + "; ".join(citations) + ". These are lineage only, never our own evidence."
    return ConversationResponse(intent=classification.intent, text=narrative, evidence={}, citations=citations)


def _handle_candidate_comparison(text, classification, context) -> ConversationResponse:
    evidence = context.evidence or {}
    exp = evidence.get("experiment") or {}
    family, root = exp.get("strategy_family"), exp.get("root_symbol") or context.root
    if not family or not root:
        return ConversationResponse(
            intent=classification.intent,
            text="I need a bound candidate to compare against its siblings -- open one from Strategies first.",
            evidence={},
        )
    siblings = [r for r in services.list_experiments(strategy_family=family, root_symbol=root, trial_role=None)
                if r.get("experiment_id") != exp.get("experiment_id")]
    if not siblings:
        return ConversationResponse(
            intent=classification.intent,
            text=f"No other {services.strategy_name(family)}/{root} variants exist in the registry to compare against.",
            evidence={},
        )
    other = siblings[0]
    this_verdict = (evidence.get("result") or {}).get("headline_verdict", "NOT_ADJUDICATED")
    narrative = (
        f"Compared with `{other['experiment_id']}` ({other.get('trial_role')}): this candidate is "
        f"{this_verdict}; the other is {other.get('headline_verdict', 'NOT_ADJUDICATED')}."
    )
    return ConversationResponse(intent=classification.intent, text=narrative, evidence={"other": other})


# ---------------------------------------------------------------------------
# profile plane -- User Fit only, never a scientific verdict change
# ---------------------------------------------------------------------------


def _handle_profile_question(text, classification, context) -> ConversationResponse:
    profile = services.load_investor_profile()
    narrative = (
        f"Your saved profile: {profile.risk_style.value} risk, {profile.holding_period.value} holding period, "
        f"max drawdown tolerance {profile.max_drawdown.value}, strategy preference {profile.strategy_preference.value}."
    )
    return ConversationResponse(intent=classification.intent, text=narrative, evidence={"profile": profile.model_dump(mode="json")})


_DRAWDOWN_RE = re.compile(r"(\d+)\s*%")
_DRAWDOWN_BUCKETS = (
    (7.5, MaxDrawdown.PCT_5), (12.5, MaxDrawdown.PCT_10), (17.5, MaxDrawdown.PCT_15),
)


def _nearest_drawdown_bucket(pct: float) -> MaxDrawdown:
    for ceiling, bucket in _DRAWDOWN_BUCKETS:
        if pct <= ceiling:
            return bucket
    return MaxDrawdown.PCT_20_PLUS


def _handle_what_if_profile(text, classification, context) -> ConversationResponse:
    """Recomputes User Fit against a HYPOTHETICAL profile derived from simple,
    explicit keyword clues in the question -- never persisted
    (`services.save_investor_profile` is never called here), and never a
    change to the candidate's scientific verdict (mission section 50)."""
    base = services.load_investor_profile()
    lowered = text.lower()
    overrides: dict = {}
    m = _DRAWDOWN_RE.search(lowered)
    if m and "drawdown" in lowered:
        overrides["max_drawdown"] = _nearest_drawdown_bucket(float(m.group(1)))
    if "overnight" in lowered:
        overrides["overnight"] = OvernightPreference.ALLOWED if "allow" in lowered else OvernightPreference.AVOID
    if "trading frequency" in lowered or "trade more" in lowered or "trade less" in lowered:
        overrides["trading_frequency"] = TradingFrequency.HIGH if "more" in lowered else TradingFrequency.LOW
    if not overrides:
        return ConversationResponse(
            intent=classification.intent,
            text="I couldn't identify a specific constraint to vary in that question -- try naming a concrete "
                 "value, e.g. \"what if I tolerate 20% drawdown\".",
            evidence={},
        )
    hypothetical = base.model_copy(update=overrides)
    result = (context.evidence or {}).get("result")
    family = ((context.evidence or {}).get("experiment") or {}).get("strategy_family")
    fit = score_user_fit(strategy_family=family, result=result, profile=hypothetical)
    narrative = (
        f"Under that hypothetical profile ({', '.join(f'{k}={v.value}' for k, v in overrides.items())}), "
        f"this candidate's User Fit would be {fit.label} ({fit.personalization_state.value}). "
        "This never changes the scientific verdict -- only presentation/priority."
    )
    return ConversationResponse(intent=classification.intent, text=narrative, evidence={"hypothetical_fit": fit.model_dump(mode="json")})


def _handle_what_if_strategy(text, classification, context) -> ConversationResponse:
    return ConversationResponse(
        intent=classification.intent,
        text="Changing a strategy's own logic (e.g. allowing overnight positions) requires compiling a new "
             "StrategySpec and running a new backtest -- I can propose this as a new research hypothesis, but "
             "I can't estimate its performance without actually running it. Would you like me to research it?",
        evidence={}, suggested_action=None,
    )


# ---------------------------------------------------------------------------
# discovery / follow-up -- suggestion only, never an execution
# ---------------------------------------------------------------------------


def _handle_research_discovery(text, classification, context) -> ConversationResponse:
    root = context.root
    family = ((context.evidence or {}).get("experiment") or {}).get("strategy_family")
    guidance = None
    if family and root:
        fm = services.failure_memory_lookup(strategy_family=family, root_symbol=root)
        guidance = (fm.get("lessons") or [None])[0]
    narrative = (
        guidance
        or "I can propose a new, mechanism-diverse research direction -- use Discover Strategies (or Generate "
           "Hypothesis) to actually run it; I won't create a new experiment from this chat message alone."
    )
    return ConversationResponse(intent=classification.intent, text=narrative, evidence={})


def _handle_research_followup(text, classification, context) -> ConversationResponse:
    resp = _handle_research_discovery(text, classification, context)
    return resp.model_copy(update={"intent": classification.intent})


# ---------------------------------------------------------------------------
# Phase 1 -- Observation -> Mechanism -> Factor translation (prompt 1 section
# 12). ALWAYS offline/deterministic here (`translation_context.SCRIPTED_MODE`)
# -- this module never calls Claude (see the module docstring); the optional
# live-Claude mechanism/factor reasoning is reachable only from the Agent
# page's own explicit Research Translation panel, never from a chat message
# alone (prompt 1 section 14: no automatic paid network calls).
# ---------------------------------------------------------------------------


#: News Alpha Phase A/B/C chat integration. A message that DESCRIBES an event
#: ("OPEC+ agreed a production cut -- what could I research?") is routed
#: through the SAME Phase A path the "Describe a market event" box uses
#: (`news_alpha_context.add_user_event` -> `scan_user_event`), and the SAME
#: scan feeds the Phase B mechanism graph and its Phase C signal paths and
#: mechanism-adjusted impact -- never translated as whatever
#: unrelated news item happens to be cached. A message that points at cached
#: news ("the latest FOMC news", "this release") keeps the Phase 1
#: cached-item path below, unchanged.
_CACHED_NEWS_RE = re.compile(r"\b(news|headlines?|releases?|cached|from this|about this)\b")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")


def _event_text(text: str) -> str:
    """The user's event description without a trailing research question
    ("OPEC+ cut output. What could I research?" -> "OPEC+ cut output.").
    A one-sentence message is kept whole."""
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text or "") if m.group(0).strip()]
    statements = [s for s in sentences if not s.endswith("?")]
    return " ".join(statements) if statements and len(statements) < len(sentences) else " ".join(sentences)


def _described_event_scan(text: str):
    """``(event, scan)`` when ``text`` describes an event the deterministic
    Phase A scan recognizes (at least one economic channel) and does not point
    at cached news; ``None`` otherwise. Registers the event in the SAME session
    list the triage cards render, so chat and card share one event_id."""
    if not text or _CACHED_NEWS_RE.search(text.lower()):
        return None
    event_text = _event_text(text)
    if len(event_text) < 3:  # nothing describable (e.g. "..."): not an event
        return None
    mandate = news_alpha_context.current_mandate()
    probe = news_alpha_context.scan_user_event(UserDescribedEvent.create(event_text), mandate)
    if not probe.channels:
        return None
    event = news_alpha_context.add_user_event(event_text)
    if event is None:
        return None
    return event, news_alpha_context.scan_user_event(event, mandate)


def _level_text(assessment) -> str:
    if assessment.impact_level is None:
        return "excluded" if assessment.availability.value == "EXCLUDED_BY_MANDATE" else "not supported"
    return assessment.impact_level.value.replace("_", " ")


def _handle_described_event(event, scan, classification) -> ConversationResponse:
    graph = news_alpha_context.mechanism_graph(scan)  # the SAME scan -- no parallel pipeline
    paths = news_alpha_context.signal_paths(graph)
    plan = news_alpha_context.handoff_plan(scan)
    lines = [
        f"EVENT: {event.text} (user-described, {event.event_id}).",
        "IMPACT TRIAGE (initial scan, research triage only -- never a return, probability, or trade "
        f"instruction; mandate: {scan.mandate_summary}): "
        + " · ".join(f"{DOMAIN_LABELS[a.asset_domain]} {_level_text(a)}" for a in scan.assessments) + ".",
        "CHANNELS: " + " · ".join(
            f"{c.channel_label} ({c.strength.value.lower()}{', ' + c.shift.replace('_', ' ').lower() if c.shift else ''})"
            for c in scan.channels
        ) + ".",
    ]
    summary = mechanism_graph_view.summary_line(graph)
    if summary:
        lines.append(summary)
    path_summary = signal_path_view.summary_line(paths)
    if path_summary:
        lines.append(path_summary)

    evidence: dict = {
        "user_event_id": event.event_id,
        "impact_scan": scan.model_dump(mode="json"),
        "mechanism_graph": {"event_id": graph.event_id, "fingerprint": graph.fingerprint()},
        "signal_paths": {"event_id": paths.event_id, "fingerprint": paths.fingerprint(), "n_paths": len(paths.paths)},
    }
    if not graph.is_empty:
        adjusted = news_alpha_context.adjusted_impact(scan, paths)
        lines.append(
            "MECHANISM-ADJUSTED IMPACT (second pass; the initial scan above is kept): "
            + signal_path_view.adjusted_summary_text(adjusted) + "."
        )
        evidence["mechanism_adjusted_impact"] = adjusted.model_dump(mode="json")
        expressions = news_alpha_context.asset_expressions(paths, news_alpha_context.current_mandate())
        expression_text = asset_expression_view.chat_text(expressions)
        if expression_text:
            lines.append(expression_text)
        evidence["asset_expression"] = {
            "event_id": expressions.event_id, "fingerprint": expressions.fingerprint(),
            "n_expressions": len(expressions.expressions), "n_measurements": len(expressions.measurements),
            "n_measurable_expressions": len(expressions.measurable_expressions()),
            "gaps": [b.gap for b in expressions.blockers()],
        }
        candidates = news_alpha_context.candidate_signals(expressions, paths)
        candidate_text = candidate_signal_view.chat_text(candidates)
        if candidate_text:
            lines.append(candidate_text)
        evidence["candidate_signals"] = {
            "event_id": candidates.event_id, "fingerprint": candidates.fingerprint(),
            "candidates": [
                {"candidate_signal_id": c.candidate_signal_id, "factor_identity": c.factor_identity,
                 "expression": c.expression, "instrument": c.spec.instrument,
                 "formation_lookback": c.spec.formation_lookback,
                 "prediction_horizon": c.spec.prediction_horizon.value}
                for c in candidates.candidates
            ],
            "n_refusals": len(candidates.refusals),
        }
        ranked = signal_ranking_view.rank_for([candidates], news_alpha_context.current_mandate())
        ranking_text = signal_ranking_view.chat_text(ranked)
        if ranking_text:
            lines.append(ranking_text)
        evidence["signal_ranking"] = {
            "fingerprint": ranked.fingerprint(), "policy_fingerprint": ranked.policy_fingerprint,
            "independent_exposures": ranked.independent_exposures,
            "signals": [
                {"candidate_signal_id": s.candidate_signal_id, "tier": s.tier.value, "rank": s.rank,
                 "role": s.role.value if s.role else None, "exposure_group_id": s.exposure_group_id,
                 "not_rankable_reason": s.not_rankable_reason.value if s.not_rankable_reason else None}
                for s in ranked.signals
            ],
        }
    handoff = plan.handoffs[0] if plan.handoffs else None
    if handoff is not None:
        translation, error = translation_context.translate_observation(
            handoff.observation, mode=translation_context.SCRIPTED_MODE,
        )
        if translation is not None and not error:
            lines.append(render_summary_text(translation))
            evidence["observation_translation"] = translation.model_dump(mode="json")
            evidence["translation_handoff_key"] = handoff.key
    elif plan.gaps:
        lines.append(f"TRANSLATION: no certified Futures root to translate yet -- {plan.gaps[0].reason}")
    lines.append("This event is now on News too -- open it there as a research thread to follow it step by step.")
    return ConversationResponse(intent=classification.intent, text="\n".join(lines), evidence=evidence)


def _handle_observation_to_factor(text, classification, context) -> ConversationResponse:
    described = _described_event_scan(text)
    if described is not None:
        return _handle_described_event(*described, classification)

    root = classification.mentioned_roots[0] if classification.mentioned_roots else context.root
    candidates = translation_context.candidate_observations()
    if root:
        scoped = tuple(c for c in candidates if c.root_symbol == root)
        candidates = scoped or candidates

    if not candidates:
        return ConversationResponse(
            intent=classification.intent,
            text=(
                "I don't have a cached market observation to translate yet -- this only uses news/events "
                "already fetched via an explicit Market refresh, never a live fetch triggered by this chat "
                "message. Refresh Market news/events first, or open the Research Translation panel on this "
                "page once observations are cached."
            ),
            evidence={},
        )

    candidate = candidates[0]
    observation = translation_context.build_observation(candidate)
    translation, error = translation_context.translate_observation(observation, mode=translation_context.SCRIPTED_MODE)
    if error or translation is None:
        return ConversationResponse(
            intent=classification.intent,
            text=error or "I could not translate that observation into a research direction.",
            evidence={},
        )
    return ConversationResponse(
        intent=classification.intent,
        text=render_summary_text(translation),
        evidence={"observation_translation": translation.model_dump(mode="json")},
    )


# ---------------------------------------------------------------------------
# actions -- recognised, never executed
# ---------------------------------------------------------------------------

_ACTION_LABELS = {
    ResearchConversationIntent.RUN_FAST_SCREEN: "Run Fast Screen",
    ResearchConversationIntent.FREEZE_CANDIDATES: "Freeze Candidates",
    ResearchConversationIntent.RUN_STRICT_VALIDATION: "Run Strict Validation",
}


def _handle_action_intent(text, classification, context) -> ConversationResponse:
    label = _ACTION_LABELS[classification.intent]
    return ConversationResponse(
        intent=classification.intent,
        text=f"I can help with that -- click [{label}] to actually run it. I never trigger a scientific "
             "action from a chat message alone.",
        evidence={}, suggested_action=classification.intent.value,
    )


# ---------------------------------------------------------------------------
# general fallback -- a small, honest, closed glossary
# ---------------------------------------------------------------------------

_GLOSSARY: tuple[tuple[tuple[str, ...], str], ...] = (
    (("deflated sharpe", "dsr"), (
        "The Deflated Sharpe Ratio (DSR) adjusts an observed Sharpe ratio for the "
        "number of trials/configurations tested and the length of the track record, estimating the probability "
        "the true Sharpe ratio is positive after accounting for selection bias."
    )),
    (("bh-fdr", "benjamini", "multiple testing", "multiple-testing"), (
        "Benjamini-Hochberg FDR control adjusts "
        "p-values across every hypothesis in a predeclared family so the expected false-discovery rate stays "
        "bounded, rather than evaluating each hypothesis's significance in isolation."
    )),
    (("look-ahead", "lookahead"), (
        "Look-ahead bias means a signal used information that would not actually "
        "have been available at decision time -- this platform enforces no-look-ahead by construction in the "
        "C++ execution engine."
    )),
    (("walk-forward", "walk forward"), (
        "Walk-forward validation evaluates a strategy on data strictly after "
        "the window it was discovered/tuned on, repeated across multiple folds, to check the result is not an "
        "artifact of one lucky period."
    )),
    (("holdout",), (
        "The 2025 holdout is a locked, never-accessed period reserved for final, one-time forward "
        "evaluation -- no code path in this platform may query, cost-fetch, or feature it before that point."
    )),
)


def _handle_general_quant_question(text, classification, context) -> ConversationResponse:
    lowered = text.lower()
    for keywords, explanation in _GLOSSARY:
        if any(k in lowered for k in keywords):
            return ConversationResponse(intent=classification.intent, text=explanation, evidence={})
    return ConversationResponse(
        intent=classification.intent,
        text="I don't have a canned explanation for that yet -- try rephrasing, or ask about a specific "
             "market, strategy, or validation result.",
        evidence={},
    )


_HANDLERS = {
    ResearchConversationIntent.MARKET_QUESTION: _handle_market_question,
    ResearchConversationIntent.UNSUPPORTED_CURRENT_EVENT_CAUSAL_QUESTION: _handle_causal_current_event,
    ResearchConversationIntent.MARKET_COMPARISON: _handle_market_comparison,
    ResearchConversationIntent.VALIDATION_EXPLANATION: _handle_validation_explanation,
    ResearchConversationIntent.STRATEGY_EXPLANATION: _handle_strategy_explanation,
    ResearchConversationIntent.SOURCE_QUESTION: _handle_source_question,
    ResearchConversationIntent.CANDIDATE_COMPARISON: _handle_candidate_comparison,
    ResearchConversationIntent.PROFILE_QUESTION: _handle_profile_question,
    ResearchConversationIntent.WHAT_IF_PROFILE: _handle_what_if_profile,
    ResearchConversationIntent.WHAT_IF_STRATEGY: _handle_what_if_strategy,
    ResearchConversationIntent.RESEARCH_DISCOVERY: _handle_research_discovery,
    ResearchConversationIntent.RESEARCH_FOLLOWUP: _handle_research_followup,
    ResearchConversationIntent.OBSERVATION_TO_FACTOR: _handle_observation_to_factor,
    ResearchConversationIntent.RUN_FAST_SCREEN: _handle_action_intent,
    ResearchConversationIntent.FREEZE_CANDIDATES: _handle_action_intent,
    ResearchConversationIntent.RUN_STRICT_VALIDATION: _handle_action_intent,
    ResearchConversationIntent.GENERAL_QUANT_QUESTION: _handle_general_quant_question,
}

assert set(_HANDLERS) == set(ResearchConversationIntent), "every intent must have a handler"
assert all(i in _HANDLERS for i in ACTION_INTENTS)

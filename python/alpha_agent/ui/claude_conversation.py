"""Live Claude conversational path for the Agent page's "Ask Agentic Alpha"
surface (CLAUDE LIVE INTEGRATION V0).

This is ADDITIVE to `alpha_agent.ui.conversation_engine`, never a
replacement -- that module remains the always-available deterministic
fallback (no `ANTHROPIC_API_KEY`, an action-shaped question, or a live-call
failure all route there; see :func:`handle_turn`). This module reuses the
ONE existing Anthropic transport (`alpha_agent.agents.llm.AnthropicClient`)
and the SAME intent router (`alpha_agent.agents.conversation.classify_intent`)
the deterministic engine already uses, so an action-shaped question ("run
fast screen", "freeze candidates", "run strict validation") never reaches
the model at all -- it is answered by the exact same non-executing
suggestion handler regardless of whether Claude is connected.

CONTEXT DISCIPLINE: the grounded context handed to Claude is built ONLY from
evidence already available to this process -- the caller's already-cached
Opportunity snapshot (never re-fetched here), local registry/failure-memory
reads (SQLite, no network), and the saved investor profile. This module
never imports `alpha_agent.ui.databento_context` and never triggers a
Databento refresh merely because the user asked a question; if no current
Opportunity snapshot has been loaded yet, the context says so explicitly
instead of fetching one.

EVIDENCE SEPARATION (Agent Evidence Pack acceptance pass): the system prompt
keeps CURRENT OBSERVATION, CATALYSTS / NEWS, RESEARCH EVIDENCE, USER FIT, and
EVIDENCE GAPS in five clearly labelled, separate sections, and the system
instruction states outright that one must never be used to infer a
conclusion about another (a live market observation is not a scientific
claim; a historical REJECT does not mean the current market is
uninteresting). CATALYSTS / NEWS and the per-root "Available testing" lines
inside RESEARCH EVIDENCE are built ONLY from evidence this process has
already cached via an explicit Market/Opportunity refresh -- never fetched
live merely because the user asked a question (see `market_intel_context
.cached_recent_news`/`cached_upcoming_events`).

QUESTION-SCOPED EVIDENCE (Agent Experience Consolidation campaign): the Agent
answers "what deserves attention across markets?" -- a GLOBAL question -- so
this module never lets a Market/Research/Paper page's own LOCAL selection
(Market's own local selected product, `st.session_state["market_selected_root"]`) narrow
what evidence a question sees. `alpha_agent.agents.evidence_scope` resolves a
small, typed, deterministic scope (GLOBAL / ROOT_SET / CATEGORY /
RESEARCH_OBJECT) from the question text, the intent router's own signals, the
already-cached Opportunity snapshot, and the currently BOUND research object
(`context.root` -- the local research thread's own root, set only by an
explicit user action on Agent itself, e.g. Generate Hypothesis or Research
This). A global question ("what's interesting right now?") always sees every
cached root regardless of any page's local selection; an explicit
cross-market question ("why CL instead of NQ?") always sees every root it
names, never silently narrowed to one.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

from alpha_agent.agents import DEFAULT_MODEL, AnthropicClient, LLMClientUnavailable
from alpha_agent.agents.conversation import (
    ACTION_INTENTS,
    ConversationSignalContext,
    classify_intent,
)
from alpha_agent.agents.evidence_scope import (
    EvidenceScope,
    EvidenceScopeKind,
    EvidenceScopeSignals,
    resolve_evidence_scope,
)
from alpha_agent.marketdata.product_catalog import catalog_entry
from alpha_agent.ui import llm_demo, market_intel_context, services
from alpha_agent.ui.conversation_engine import (
    ConversationBoundContext,
    ConversationResponse,
    handle_conversation_turn,
)

__all__ = ["LIVE_MODE", "SCRIPTED_MODE", "handle_turn", "is_connected"]

#: Mirrors `llm_demo.LIVE_MODE` / `llm_demo.SCRIPTED_MODE` exactly -- the same
#: two engine labels the rest of the Agent page already renders.
LIVE_MODE = "live"
SCRIPTED_MODE = "scripted"

_MAX_TOKENS = 1024
#: Prior conversational exchanges replayed to Claude -- cost discipline
#: (section 11): a compact rolling window, never the whole transcript.
_MAX_HISTORY_TURNS = 6

SYSTEM_INSTRUCTION = """You are the conversational research interface for Agentic Alpha, an
Agentic Quant Research Environment (today's implemented universe is Futures).

Use the supplied project evidence as the source of truth for current market
observations, opportunities, experiments, validation, risk, and research
history. Do not rely on outside/background knowledge for any of those -- if
the evidence below does not cover something, say so explicitly rather than
guessing.

The evidence below is organised into five separate sections that answer
different questions and must never be conflated:
- CURRENT OBSERVATION is what the market looks like right now (or was last
  observed) -- it is not a scientific claim about whether any strategy works.
- CATALYSTS / NEWS is cached official news/economic-event evidence -- never
  fetched live during this conversation. If it says evidence is not loaded,
  say so plainly; never guess at a catalyst that is not listed there.
- RESEARCH EVIDENCE is historical, registry-adjudicated results -- it is not
  a statement about current market conditions. A past REJECT does not mean
  the current market is uninteresting, and a live observation does not
  imply a scientific verdict. Only ever cite a verdict/PnL/Sharpe/trade count
  that appears verbatim in this section -- it already excludes superseded or
  contaminated records, but you must still never invent one that is absent.
- USER FIT is the user's saved risk/research preferences -- it never changes
  a scientific verdict, only how a result should be presented to this user.
  Use it meaningfully (e.g. an event a few days out matters more to a 1-3 day
  horizon than one over a month away), but never let it change a verdict.
- EVIDENCE GAPS explicitly lists what is NOT currently available -- treat
  these as settled facts about this turn's evidence, not things to guess
  around.

Never invent quantitative results: prices, PnL, Sharpe ratios, fills, risk
values, validation statistics, or a PASS/REJECT/INCONCLUSIVE verdict. Only
state a number or verdict that appears in the evidence below.

You may explain, compare, summarize evidence, identify contradictions, and
propose a falsifiable research hypothesis. Any hypothesis you propose is NOT
VALIDATED -- say so plainly every time. You never claim to have run, or
offer to autonomously run, a backtest, a paper-trading session, or a market
data refresh -- those stay explicit, user-controlled actions on other
pages; you may only point the user at them.

For a substantive market/comparison question (e.g. "Why CL instead of NQ?",
"Compare ES and NQ"), synthesize the evidence with a concise structure when
it helps readability -- for example CURRENT READ / CATALYST-NEWS / HISTORICAL
RESEARCH / FIT TO YOUR HORIZON / WHAT WE DO NOT KNOW / NEXT RESEARCH STEP.
Do not force this structure onto a short/simple question -- keep short
questions short. When comparing markets, never declare one objectively
"better" -- explain what differs, what is stronger/weaker on the actual
evidence, what better fits the user's stated horizon/preferences, and what
evidence is missing. If RESEARCH EVIDENCE has a directly relevant
authoritative result, you may say "We tested a related mechanism..." and
summarize it; if it does not, say plainly "We have not validated this exact
combination" and suggest Research This / Generate Hypothesis as the next
step -- never call a loosely related backtest proof of the current
opportunity.

Whenever -- and ONLY whenever -- you are proposing a concrete, falsifiable
research hypothesis (a specific mechanism/market/condition worth testing,
not just discussing one that already exists), end your reply with a line of
the exact form:
PROPOSED_HYPOTHESIS: <the one-sentence hypothesis, self-contained>
This exact marker line is parsed by the application to offer an explicit
"Research This Hypothesis" action -- it is never validated, accepted, or run
automatically. Omit this line entirely for every other kind of reply
(explaining, comparing, answering a market/validation/profile question).

If a market named in the question has no entry in CURRENT OBSERVATION below,
that only means it is not present in the currently cached scan -- never
assume or state that it "failed a threshold" or "was not interesting" unless
that is explicitly represented in the evidence given to you.

If an earlier reply in this conversation said current Opportunity evidence
was not loaded, and CURRENT OBSERVATION below now includes data, that earlier
reply was correct at the time -- never describe it as a mistake or apologize
for it. Simply note that Opportunity data is now available and answer from
the current snapshot (e.g. "Opportunity data is now available. Based on the
latest snapshot...")."""


#: Structured extraction, not a substring search over prose (task spec
#: section 5A): matches ONLY a line of the exact `PROPOSED_HYPOTHESIS: ...`
#: form the system instruction asks Claude to emit, anchored at line start so
#: an incidental mention of the phrase inside ordinary prose never matches.
_PROPOSED_HYPOTHESIS_RE = re.compile(r"^PROPOSED_HYPOTHESIS:\s*(.+)$", re.MULTILINE)


def _extract_proposed_hypothesis(text: str) -> tuple[str, str | None]:
    """Splits Claude's raw reply into `(display_text, proposed_hypothesis)`.
    The marker line itself is stripped from what is shown to the user (it is
    application-internal structure, not part of the readable answer);
    `proposed_hypothesis` is `None` whenever the marker is absent -- the
    typed default this platform never infers from prose alone."""
    match = _PROPOSED_HYPOTHESIS_RE.search(text)
    if match is None:
        return text, None
    hypothesis = match.group(1).strip()
    display_text = (text[: match.start()] + text[match.end() :]).strip()
    return (display_text or text.strip()), (hypothesis or None)


def is_connected() -> bool:
    """Boolean presence check ONLY -- mirrors `alpha_agent.ui.views.
    discover._anthropic_configured` / `alpha_agent.ui.panels.
    render_agent_runtime_status` exactly. Never reads, parses, or displays
    the key value itself (CLAUDE.md: never store/echo API key material)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# grounded context -- CURRENT OBSERVATION / RESEARCH EVIDENCE / USER FIT
# ---------------------------------------------------------------------------


def _evidence_ref_value(refs: tuple, key: str) -> str | None:
    prefix = f"{key}="
    for ref in refs:
        if ref.startswith(prefix):
            return ref[len(prefix):]
    return None


_SCOPE_LABEL = {
    EvidenceScopeKind.GLOBAL: "GLOBAL -- every currently cached opportunity observation",
    EvidenceScopeKind.ROOT_SET: "ROOT_SET -- market(s) explicitly named in the question",
    EvidenceScopeKind.CATEGORY: "CATEGORY -- an asset-class question",
    EvidenceScopeKind.RESEARCH_OBJECT: "RESEARCH_OBJECT -- the currently bound research object's own market",
}


def _fmt_current_observation(opportunity_snapshot: dict | None, *, scope: EvidenceScope) -> str:
    """Uses ONLY the already-cached Opportunity snapshot the caller passes in
    (`agent.py`'s own `_OPPORTUNITIES_SNAPSHOT_KEY` session state) -- never
    fetched here. Absent/empty is reported honestly, never silently
    backfilled with a fresh Databento call.

    `scope` (`alpha_agent.agents.evidence_scope`) decides WHICH cached roots
    are shown -- GLOBAL shows every cached root regardless of any page's
    local selection (task spec sections 5/35 E-F); ROOT_SET/CATEGORY show
    exactly the resolved roots, and a root named but genuinely absent from
    the cache gets an explicit "not available in this snapshot" line, never
    an invented "failed the threshold" claim (task spec sections 7/36)."""
    if not opportunity_snapshot or not opportunity_snapshot.get("opportunities"):
        return (
            "No current Opportunity evidence is loaded for this session -- the user has not clicked "
            "'Refresh Opportunities' on the Agent page yet. Treat current market conditions as UNKNOWN; "
            "do not guess at price, trend, or volatility."
        )
    snaps = list(opportunity_snapshot["opportunities"])
    by_product = {s.product: s for s in snaps}
    lines: list[str] = []
    observed_at = opportunity_snapshot.get("observed_at")
    if observed_at is not None:
        lines.append(f"Snapshot fetched at: {observed_at.isoformat()}.")
    lines.append(f"Evidence scope for this question: {_SCOPE_LABEL[scope.kind]}.")

    if scope.kind == EvidenceScopeKind.GLOBAL:
        selected = snaps
    else:
        selected = [by_product[r] for r in scope.roots if r in by_product]
        for r in scope.roots:
            if r not in by_product:
                lines.append(
                    f"- {r}: current evidence is not available in this snapshot (simply absent from the "
                    "cached scan -- this is not evidence that it failed any threshold)."
                )
        if not scope.roots and scope.kind == EvidenceScopeKind.RESEARCH_OBJECT:
            lines.append("No specific market is currently bound to this research thread.")

    for s in selected:
        lines.append(
            f"- {s.product} ({s.display_name}), observed {s.observed_at.isoformat()}: {s.setup} "
            f"[opportunity state: {s.opportunity_state.value}]. Why now: {'; '.join(s.why_now)}. "
            f"Catalyst: {s.catalyst}. Risks: {'; '.join(s.risks)}. Next action: {s.next_action}."
        )
    return "\n".join(lines)


#: Cost discipline (task spec section 32): a per-root failure-memory sweep is
#: a real SQLite read -- bounded even for a CATEGORY scope naming many roots.
#: A comparison in practice names 2 roots; this is deliberately generous
#: without being "send the entire registry."
_MAX_ROOTS_FOR_FAILURE_MEMORY = 4


def _fmt_research_evidence(
    *, context: ConversationBoundContext, opportunity_snapshot: dict | None, scope: EvidenceScope,
) -> str:
    """Registry-grounded only: a live local SQLite summary/failure-memory
    sweep (no network) plus whatever candidate evidence is already bound to
    this conversation -- never a second, independent scientific judgment.

    Cost-scoped (task spec section 32): the per-root failure-memory sweep
    below only runs for the roots `scope` actually resolved to (ROOT_SET /
    CATEGORY / RESEARCH_OBJECT) -- a GLOBAL question gets the cheap registry
    headline count plus each cached Opportunity's own already-computed
    research-verdict tag, never a full per-root sweep across the whole
    universe. The currently BOUND candidate/hypothesis (`context.evidence` /
    `context.hypothesis`) is always included regardless of scope -- it is the
    read-only research artifact this conversation is actually about (task
    spec sections 9/23), never something a market-scope choice should hide."""
    lines: list[str] = []
    try:
        summary = services.registry_summary()
        lines.append(
            f"Registry: {summary.get('authoritative_statistical_hypotheses', '?')} authoritative "
            f"hypotheses committed, identity schema `{summary.get('identity_schema', '?')}`."
        )
    except Exception:  # noqa: BLE001, S110 -- evidence assembly must never crash the conversation
        pass

    if scope.kind != EvidenceScopeKind.GLOBAL:
        for root in scope.roots[:_MAX_ROOTS_FOR_FAILURE_MEMORY]:
            try:
                fm_list = llm_demo.relevant_failure_memory(market_universe=(root,))
            except Exception:  # noqa: BLE001
                fm_list = ()
            for fm in fm_list:
                counts = fm.verdict_counts
                if not counts:
                    continue
                lines.append(
                    f"Prior research for `{fm.query.get('strategy_family')}`/{root}: "
                    + ", ".join(f"{k}={n}" for k, n in counts.items())
                )
                if fm.lessons:
                    lines.append(f"  Lesson: {fm.lessons[0]}")

            # AVAILABLE TESTING (task spec section 4F): the single best
            # existing authoritative candidate for this root, or an explicit
            # "not tested yet" -- `services.research_candidate_summaries` is
            # the SAME CANONICAL-only, non-superseded source
            # `opportunity_context.py` already uses for a card's own
            # research-verdict tag, so this can never disagree with it, and
            # it never surfaces a superseded/contaminated record as clean
            # evidence (task spec section 16).
            try:
                candidates = [c for c in services.research_candidate_summaries() if c.root_symbol == root]
            except Exception:  # noqa: BLE001
                candidates = []
            if not candidates:
                lines.append(
                    f"Available testing for {root}: no authoritative registry experiment exists yet -- this "
                    "would be a genuinely new test. Suggest Research This / Generate Hypothesis."
                )
            else:
                best = max(candidates, key=lambda c: (c.scientific_verdict == "PASS", c.research_promise_score))
                pnl = f"${best.net_pnl_usd:,.0f}" if best.net_pnl_usd is not None else "N/A"
                sharpe = f"{best.annualized_sharpe:.2f}" if best.annualized_sharpe is not None else "N/A"
                trades = best.trade_count if best.trade_count is not None else "N/A"
                lines.append(
                    f"Available testing for {root}: `{best.friendly_strategy_name}` "
                    f"(`{best.experiment_id}`) -- verdict {best.scientific_verdict}, net PnL {pnl}, "
                    f"Sharpe {sharpe}, {trades} trades."
                )

    if context.evidence and context.evidence.get("result"):
        result = context.evidence["result"]
        exp = context.evidence.get("experiment") or {}
        pnl = result.get("net_pnl_usd")
        sharpe = result.get("annualized_sharpe")
        pnl_text = f"${pnl:,.0f}" if pnl is not None else "N/A"
        sharpe_text = f"{sharpe:.2f}" if sharpe is not None else "N/A"
        lines.append(
            f"Bound candidate `{exp.get('experiment_id', '?')}` "
            f"({exp.get('strategy_family', '?')}/{exp.get('root_symbol', '?')}): "
            f"verdict {result.get('headline_verdict', 'NOT_ADJUDICATED')}, "
            f"net PnL {pnl_text}, Sharpe {sharpe_text}."
        )
    if context.hypothesis:
        h = context.hypothesis
        lines.append(f"Currently bound hypothesis: \"{h.get('title')}\" -- {h.get('economic_mechanism', '')}")

    if opportunity_snapshot:
        allowed = set(scope.roots) if scope.kind != EvidenceScopeKind.GLOBAL else None
        for s in opportunity_snapshot.get("opportunities") or []:
            if allowed is not None and s.product not in allowed:
                continue
            verdict = _evidence_ref_value(s.evidence_refs, "research_verdict")
            if verdict:
                lines.append(f"Opportunity evidence tags `{s.product}` research status: {verdict}.")

    return "\n".join(lines) if lines else "No committed registry evidence relevant to this turn was found."


#: Cost discipline (task spec section 11): bounded like
#: `_MAX_ROOTS_FOR_FAILURE_MEMORY`, and the item count is the spec's own
#: "3-5 most relevant items maximum" (section 4C).
_MAX_ROOTS_FOR_CATALYST_NEWS = 4
_MAX_CATALYST_NEWS_ITEMS = 5


def _catalyst_news_roots(scope: EvidenceScope, opportunity_snapshot: dict | None) -> tuple[str, ...]:
    if scope.kind == EvidenceScopeKind.GLOBAL:
        return _cached_roots(opportunity_snapshot)[:_MAX_ROOTS_FOR_CATALYST_NEWS]
    return scope.roots[:_MAX_ROOTS_FOR_CATALYST_NEWS]


def _fmt_catalysts_news(*, scope: EvidenceScope, opportunity_snapshot: dict | None) -> str:
    """Bounded, CACHE-ONLY catalyst/news digest (task spec sections 4C/12/16:
    "NO AUTO NETWORK FROM CHAT") -- reads whatever
    `market_intel_context`'s in-process store already holds, populated only
    by an explicit Market/Opportunity refresh elsewhere on this page. Never
    calls `market_intel_context.recent_news`/`upcoming_events` (those may
    trigger a real connector refresh when the in-process TTL has expired);
    only the non-refreshing `cached_*` reads."""
    if market_intel_context.last_refresh_at() is None:
        return (
            "Current news/event evidence is not loaded for this session -- no Market or Opportunity refresh "
            "has run yet in this process. This does not mean no catalysts exist; say the evidence is "
            "unavailable rather than guessing."
        )
    roots = _catalyst_news_roots(scope, opportunity_snapshot)
    if not roots:
        return "No specific market is in scope for catalyst/news evidence on this question."

    dated_items: list[tuple[datetime, str]] = []
    for root in roots:
        for n in market_intel_context.cached_recent_news(related_product=root, limit=5):
            news_text = (
                f"[NEWS] {root}: \"{n.headline}\" -- {n.source_name}, {n.published_at.isoformat()} "
                f"({n.mapping_reason})."
            )
            dated_items.append((n.published_at, news_text))
        for e in market_intel_context.cached_upcoming_events():
            if root.upper() in e.affected_products:
                event_text = (
                    f"[EVENT] {root}: {e.name} ({e.importance.value} impact) -- {e.source_name}, "
                    f"scheduled {e.scheduled_at.isoformat()}."
                )
                dated_items.append((e.scheduled_at, event_text))
    if not dated_items:
        return f"No cached news/event items are currently relevant to {', '.join(roots)}."
    dated_items.sort(key=lambda pair: pair[0], reverse=True)
    lines = [text for _, text in dated_items[:_MAX_CATALYST_NEWS_ITEMS]]
    lines.append(f"(Catalyst/news cache last refreshed: {market_intel_context.last_refresh_at().isoformat()}.)")
    return "\n".join(lines)


def _fmt_evidence_gaps(
    *, opportunity_snapshot: dict | None, scope: EvidenceScope,
) -> str:
    """Explicit gap list (task spec section 4A/15) -- settled facts about
    what THIS turn's evidence does not cover, so Claude states them rather
    than guessing around them."""
    gaps: list[str] = []
    if not opportunity_snapshot or not opportunity_snapshot.get("opportunities"):
        gaps.append("No current Opportunity snapshot has been loaded this session.")
    if market_intel_context.last_refresh_at() is None:
        gaps.append("No news/event refresh has run this session.")
    if scope.kind != EvidenceScopeKind.GLOBAL:
        for root in scope.roots:
            try:
                has_candidate = any(
                    c.root_symbol == root for c in services.research_candidate_summaries()
                )
            except Exception:  # noqa: BLE001
                has_candidate = False
            if not has_candidate:
                gaps.append(f"No committed, authoritative registry experiment exists yet for {root}.")
    gaps.append("2025 is a sealed research holdout and is never accessed, queried, or reported by this system.")
    return "\n".join(f"- {g}" for g in gaps)


def _fmt_user_fit(profile) -> str:
    return (
        f"risk style: {profile.risk_style.value}; holding period: {profile.holding_period.value}; "
        f"max drawdown tolerance: {profile.max_drawdown.value}; trading frequency: "
        f"{profile.trading_frequency.value}; overnight preference: {profile.overnight.value}; "
        f"strategy preference: {profile.strategy_preference.value}."
    )


def _build_grounded_context(
    *, context: ConversationBoundContext, opportunity_snapshot: dict | None, scope: EvidenceScope,
) -> str:
    profile = services.load_investor_profile()
    sections = [
        "CURRENT OBSERVATION (live/delayed market state -- NOT a scientific claim):",
        _fmt_current_observation(opportunity_snapshot, scope=scope),
        "",
        "CATALYSTS / NEWS (cached official sources only -- never fetched live from this conversation):",
        _fmt_catalysts_news(scope=scope, opportunity_snapshot=opportunity_snapshot),
        "",
        "RESEARCH EVIDENCE (historical, registry-adjudicated -- NOT a statement about current market conditions):",
        _fmt_research_evidence(context=context, opportunity_snapshot=opportunity_snapshot, scope=scope),
        "",
        "USER FIT (saved investor profile/preferences -- never changes a scientific verdict):",
        _fmt_user_fit(profile),
        "",
        "EVIDENCE GAPS (state these explicitly rather than guessing around them):",
        _fmt_evidence_gaps(opportunity_snapshot=opportunity_snapshot, scope=scope),
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# evidence scope -- deterministic, no I/O (see `alpha_agent.agents.
# evidence_scope` module docstring for the full rationale)
# ---------------------------------------------------------------------------


def _cached_roots(opportunity_snapshot: dict | None) -> tuple[str, ...]:
    if not opportunity_snapshot or not opportunity_snapshot.get("opportunities"):
        return ()
    return tuple(s.product for s in opportunity_snapshot["opportunities"])


def _root_category_map(roots: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in roots:
        entry = catalog_entry(r)
        if entry is not None:
            out[r] = entry.asset_class.value
    return out


def _resolve_scope(
    text: str, *, classification, context: ConversationBoundContext, opportunity_snapshot: dict | None,
) -> EvidenceScope:
    cached_roots = _cached_roots(opportunity_snapshot)
    known_roots = set(cached_roots) | set(classification.mentioned_roots)
    if context.root:
        known_roots.add(context.root)
    signals = EvidenceScopeSignals(
        text=text, mentioned_roots=classification.mentioned_roots, intent=classification.intent,
        bound_root=context.root, cached_roots=cached_roots, root_category=_root_category_map(known_roots),
    )
    return resolve_evidence_scope(signals)


# ---------------------------------------------------------------------------
# message history -- a compact rolling window of the actual chat sub-thread
# ---------------------------------------------------------------------------


def _build_messages(transcript: list[dict]) -> list[dict]:
    """The chat sub-thread only (`user` / `conversation` turns) -- the
    composer's `hypothesis`/`compiled`/`evidence`/`suggestion` turns are a
    separate pipeline and are not replayed as conversation history here
    (their content, where relevant, already reaches Claude through the
    grounded context above)."""
    relevant = [t for t in transcript if t.get("type") in ("user", "conversation")]
    relevant = relevant[-(2 * _MAX_HISTORY_TURNS):]
    return [
        {"role": "user" if t["type"] == "user" else "assistant", "content": t["text"]}
        for t in relevant
    ]


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def _ask_claude(
    text: str,
    *,
    classification,
    context: ConversationBoundContext,
    transcript: list[dict],
    opportunity_snapshot: dict | None,
) -> ConversationResponse:
    scope = _resolve_scope(text, classification=classification, context=context, opportunity_snapshot=opportunity_snapshot)
    grounded = _build_grounded_context(context=context, opportunity_snapshot=opportunity_snapshot, scope=scope)
    messages = _build_messages(transcript)
    client = AnthropicClient()
    resp = client.complete(
        system=SYSTEM_INSTRUCTION + "\n\n---\nPROJECT EVIDENCE FOR THIS TURN\n---\n\n" + grounded,
        messages=messages,
        model=DEFAULT_MODEL,
        max_tokens=_MAX_TOKENS,
        temperature=0.0,
    )
    display_text, proposed_hypothesis = _extract_proposed_hypothesis(resp.text)
    return ConversationResponse(
        intent=classification.intent,
        text=display_text,
        proposed_hypothesis=proposed_hypothesis,
        evidence={
            "engine": "claude",
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
        },
    )


def handle_turn(
    text: str,
    *,
    context: ConversationBoundContext,
    has_history: bool,
    transcript: list[dict],
    opportunity_snapshot: dict | None,
) -> tuple[ConversationResponse | None, str, str | None]:
    """Returns ``(response, engine, error)``. ``engine`` is ``LIVE_MODE`` or
    ``SCRIPTED_MODE`` (never both) -- always set, even on error, so the
    caller can log which path was attempted. ``error`` is a plain,
    user-facing string (never a raw key value) and, when set, ``response``
    is ``None``: the caller should render an honest error turn rather than a
    fabricated answer.

    No `root` parameter: the only root this module ever treats as "bound" is
    `context.root` -- the currently bound research object's own root, set
    only by an explicit local action on Agent itself (Generate Hypothesis,
    Research This). A Market/Research/Paper page's own local selection
    (e.g. Market's `st.session_state["market_selected_root"]`) never reaches
    this function at all -- see `alpha_agent.agents.evidence_scope` for how a question's actual
    evidence scope is decided instead.

    Falls back to the deterministic `conversation_engine.handle_conversation_turn`
    (unchanged) whenever Claude is not configured, OR the question classifies
    as an action intent (`RUN_FAST_SCREEN` / `FREEZE_CANDIDATES` /
    `RUN_STRICT_VALIDATION`) -- an action-shaped question never reaches the
    model, so this guarantee holds regardless of what Claude might say."""
    roots = tuple(services.approved_universe())
    sig_ctx = ConversationSignalContext(
        approved_roots=roots,
        has_bound_strategy=bool(context.hypothesis or context.compiled),
        has_history=has_history,
    )
    classification = classify_intent(text, context=sig_ctx)

    if not is_connected() or classification.intent in ACTION_INTENTS:
        response = handle_conversation_turn(text, context=context, has_history=has_history)
        return response, SCRIPTED_MODE, None

    try:
        response = _ask_claude(
            text, classification=classification, context=context, transcript=transcript,
            opportunity_snapshot=opportunity_snapshot,
        )
        return response, LIVE_MODE, None
    except LLMClientUnavailable as exc:
        return None, LIVE_MODE, f"Claude is unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 -- defensive UI boundary, never a raw stack trace or key material
        return None, LIVE_MODE, f"{type(exc).__name__}: {exc}"

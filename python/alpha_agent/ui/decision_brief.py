"""Decision Brief (Research Golden Path, section 11/11A/11B/11C): Claude as
the SYNTHESIS layer over one completed research thread's already-committed
evidence -- never a new scientific verdict, never a new quantitative claim.

Two-phase, matching every other Claude integration in this app
(`llm_demo`/`claude_conversation`):

1. :func:`build_decision_brief_evidence` assembles a small, BOUNDED, typed
   evidence dict from real sources only -- `services.get_experiment` (C++
   result + registry provenance), `services.gate_table`/`plain_language_reason`
   (validation), `services.failure_memory_lookup` (research memory),
   `market_intel_context` cached-only news/events, `services.load_investor_profile`
   (user fit), and an optional already-cached Opportunity snapshot (current
   market). Nothing here calls an LLM or fetches anything live.
2. :func:`generate_decision_brief` turns that evidence into readable text.
   Offline / Deterministic mode (the default, and the one automated tests
   use) is a pure Python template -- zero network calls, explicitly labelled
   as a deterministic synthesis rather than a Claude-generated one. Claude
   Research mode makes one real, bounded `AnthropicClient` call with a system
   instruction that forbids inventing numbers, forbids overriding the
   registry's verdict, and restricts "Next Action" to the approved vocabulary.
"""
from __future__ import annotations

from typing import Any

from alpha_agent.agents import DEFAULT_MODEL, AnthropicClient, LLMClientUnavailable
from alpha_agent.ui import market_intel_context, services

OFFLINE_MODE = "offline"
CLAUDE_MODE = "claude"

#: The only vocabulary a Next Action may use (section 11B) -- deliberately
#: excludes any buy/sell/guaranteed-alpha language regardless of verdict.
NEXT_ACTION_VOCABULARY = ("INVESTIGATE", "REFINE", "RETEST", "PAPER TEST", "WAIT", "NOT SUPPORTED")

_MAX_NEWS_ITEMS = 5


def _current_market_section(root: str, opportunity_snapshot: dict | None) -> str:
    if not opportunity_snapshot or not opportunity_snapshot.get("opportunities"):
        return f"No current Opportunity evidence is loaded for {root} this session -- treat current conditions as UNKNOWN."
    by_product = {s.product: s for s in opportunity_snapshot["opportunities"]}
    s = by_product.get(root)
    if s is None:
        return f"{root} is not present in the currently cached Opportunity scan -- this is not evidence it failed any threshold."
    return (
        f"{s.product} ({s.display_name}), observed {s.observed_at.isoformat()}: {s.setup} "
        f"[state: {s.opportunity_state.value}]. Catalyst: {s.catalyst}. Risks: {'; '.join(s.risks)}."
    )


def _news_catalyst_section(root: str) -> str:
    if market_intel_context.last_refresh_at() is None:
        return "No news/event refresh has run this session -- catalyst evidence is not loaded."
    items = [
        f"[NEWS] \"{n.headline}\" -- {n.source_name}, {n.published_at.isoformat()}"
        for n in market_intel_context.cached_recent_news(related_product=root, limit=_MAX_NEWS_ITEMS)
    ]
    if not items:
        return f"No cached news/event items are currently relevant to {root}."
    return "\n".join(items)


def _validation_section(result: dict | None, exp: dict | None) -> dict[str, Any]:
    if not result or not exp:
        return {"verdict": None, "gates": [], "reason_codes": [], "plain_reason": "No committed result yet."}
    gates = services.gate_table(
        reason_codes=result.get("reason_codes") or (), verdict=result.get("headline_verdict"),
        trial_role=exp.get("trial_role", "CANONICAL"),
    )
    plain = services.plain_language_reason(
        result=result, trial_role=exp.get("trial_role", "CANONICAL"), verdict=result.get("headline_verdict"),
    )
    return {
        "verdict": result.get("headline_verdict"),
        "gates": gates,
        "reason_codes": list(result.get("reason_codes") or []),
        "plain_reason": plain,
    }


def build_decision_brief_evidence(
    experiment_id: str | None,
    *,
    hypothesis: dict | None = None,
    opportunity_snapshot: dict | None = None,
) -> dict[str, Any]:
    """Bounded, typed evidence for one research thread. `experiment_id` is
    `None` for a hypothesis that has not been compiled/run yet -- the brief
    still renders, honestly stating that no C++ result or validation
    evidence exists."""
    detail = services.get_experiment(experiment_id) if experiment_id else None
    exp = (detail or {}).get("experiment")
    result = (detail or {}).get("result")
    root = (exp or {}).get("root_symbol") or (hypothesis or {}).get("universe", [None])[0]
    family = (exp or {}).get("strategy_family")

    fm = None
    if family and root:
        fm = services.failure_memory_lookup(strategy_family=family, root_symbol=root)

    gaps: list[str] = []
    if result is None:
        gaps.append("No committed C++ execution result exists yet for this hypothesis.")
    if not opportunity_snapshot:
        gaps.append("No current Opportunity snapshot has been loaded this session.")
    if market_intel_context.last_refresh_at() is None:
        gaps.append("No news/event refresh has run this session.")
    gaps.append("2025 is a sealed research holdout and is never accessed by this brief.")

    return {
        "experiment_id": experiment_id,
        "root_symbol": root,
        "strategy_family": family,
        "strategy_name": services.strategy_name(family) if family else None,
        "hypothesis": hypothesis,
        "current_market": _current_market_section(root, opportunity_snapshot) if root else "No market bound to this thread.",
        "mechanism": (hypothesis or {}).get("economic_mechanism") or "No HypothesisSpec attached to this thread.",
        "cpp_historical_test": {
            "gross_pnl_usd": (result or {}).get("gross_pnl_usd"),
            "net_pnl_usd": (result or {}).get("net_pnl_usd"),
            "costs_usd": (result or {}).get("costs_usd"),
            "annualized_sharpe": (result or {}).get("annualized_sharpe"),
            "n_trades": (result or {}).get("n_trades"),
            "n_fills": (result or {}).get("n_fills"),
            "source_artifact": (result or {}).get("source_artifact"),
        },
        "validation": _validation_section(result, exp),
        "research_memory": fm,
        "news_catalyst": _news_catalyst_section(root) if root else "No market bound to this thread.",
        "user_fit": services.load_investor_profile().model_dump(mode="json"),
        "evidence_gaps": gaps,
        "paper_eligible": services.paper_eligible_experiment(experiment_id) if experiment_id else False,
        "research_classification": _research_classification_section(detail),
    }


def _research_classification_section(detail: dict | None) -> dict[str, Any]:
    """Section 1C: the honest, presentation-only distinction between a
    frozen, predeclared research program and a later standalone research-
    session append -- see `services.research_classification`'s own
    docstring for exactly what each field means and does not mean. `None`
    fields (no experiment yet) render as an honest "not yet known" in the
    brief, never guessed."""
    exp = (detail or {}).get("experiment")
    result = (detail or {}).get("result")
    if not exp:
        return {"classification": None, "label": None, "origin": None, "execution_evidence": None, "holdout_eligible": None}
    rc = services.research_classification(exp)
    return {
        "classification": rc["classification"],
        "label": rc["label"],
        "origin": rc["origin"],
        "execution_evidence": (
            f"{detail.get('authority', 'UNKNOWN')} real C++ execution" if result else "No committed execution yet"
        ),
        "holdout_eligible": (result or {}).get("holdout_eligible", False),
    }


def _deterministic_next_action(evidence: dict[str, Any]) -> str:
    """A fixed, documented mapping from the committed verdict to the approved
    vocabulary -- used only in Offline / Deterministic mode, never overriding
    what Claude Research mode's own bounded judgment picks."""
    verdict = evidence["validation"]["verdict"]
    if verdict == "PASS":
        return "PAPER TEST" if evidence["paper_eligible"] else "INVESTIGATE"
    if verdict == "REJECT":
        return "NOT SUPPORTED"
    if verdict == "INCONCLUSIVE":
        return "RETEST"
    return "INVESTIGATE"


def render_deterministic_brief(evidence: dict[str, Any]) -> str:
    """Zero-network, template-only synthesis -- explicitly labelled as such,
    never presented as if Claude wrote it."""
    v = evidence["validation"]
    cpp = evidence["cpp_historical_test"]
    fm = evidence.get("research_memory") or {}
    rc = evidence.get("research_classification") or {}
    lines = [
        "DECISION BRIEF (Deterministic synthesis -- Offline / Deterministic mode, not Claude-generated)",
        "",
        "RESEARCH CLASSIFICATION",
        (
            f"{rc['label']} -- {rc['origin']}. Execution evidence: {rc['execution_evidence']}. "
            f"2025 holdout eligibility: {'ELIGIBLE' if rc['holdout_eligible'] else 'NOT ELIGIBLE'}."
            if rc.get("classification") else "Not yet known -- no experiment attached to this thread."
        ),
        "",
        "CURRENT MARKET", evidence["current_market"], "",
        "MECHANISM", evidence["mechanism"], "",
        "C++ HISTORICAL TEST",
        (
            f"Net PnL: {cpp['net_pnl_usd']:,.0f} USD; Sharpe: {cpp['annualized_sharpe']:.2f}; "
            f"Trades: {cpp['n_trades']}. Raw historical performance is not the same as statistically "
            f"supported alpha -- see VALIDATION below."
            if cpp["net_pnl_usd"] is not None else "No committed C++ execution result yet."
        ),
        "",
        "VALIDATION",
        f"Scientific Verdict: {v['verdict'] or 'NOT_ADJUDICATED'}. {v['plain_reason']}",
        "",
        "RESEARCH MEMORY",
        (
            f"{fm.get('valid_execution_attempts', 0)} valid prior attempt(s), "
            f"verdict counts {fm.get('scientific_verdict_counts', fm.get('verdict_counts', {}))}."
            if fm else "No prior registry history for this family/root."
        ),
        "",
        "NEWS / CATALYST", evidence["news_catalyst"], "",
        "USER FIT", f"Saved investor profile: {evidence['user_fit']}", "",
        "EVIDENCE GAPS",
        "\n".join(f"- {g}" for g in evidence["evidence_gaps"]),
        "",
        "RESEARCH RECOMMENDATION", _deterministic_next_action(evidence),
    ]
    return "\n".join(lines)


_SYSTEM_INSTRUCTION = """You are writing a Decision Brief for Agentic Alpha, an Agentic Quant
Research Environment (today's implemented universe is Futures) -- a synthesis
over ALREADY-COMMITTED evidence for one research thread, never a new
scientific judgment.

Rules, no exceptions:
- Never invent a number. Every PnL, Sharpe, trade count, or verdict you state
  must appear verbatim in the evidence below.
- The Scientific Verdict is fixed by the VALIDATION section. You may explain
  it; you may never change it, soften it, or imply a different one.
- Positive historical PnL / Sharpe is NOT the same claim as statistically
  supported alpha. If the C++ historical test shows a positive net PnL or
  Sharpe but VALIDATION's verdict is REJECT or INCONCLUSIVE, you must say so
  explicitly and explain WHY the statistical evidence rejected or could not
  confirm the hypothesis despite the raw historical performance -- never
  write or imply "profitable therefore good" or "positive Sharpe therefore
  validated".
- RESEARCH CLASSIFICATION distinguishes a frozen, predeclared research
  program from a later standalone research-session append. It is real,
  already-committed provenance, not a judgment you make -- restate it, never
  reinterpret it. An EXPLORATORY classification or a NOT ELIGIBLE 2025
  holdout-eligibility value must never be described as if it were part of
  the frozen scientific program's own multiple-testing family.
- USER FIT describes the saved investor profile (holding period, risk style,
  strategy preference). If the tested mechanism's own horizon/style
  materially mismatches the user's stated profile, note the mismatch
  plainly. User Fit may inform your RESEARCH RECOMMENDATION's relevance and
  urgency; it may never change the Scientific Verdict.
- Structure your reply with these exact section headings, in this order:
  RESEARCH CLASSIFICATION, CURRENT MARKET, NEWS / CATALYST, MECHANISM,
  C++ HISTORICAL TEST, VALIDATION, RESEARCH MEMORY, USER FIT, EVIDENCE GAPS,
  RESEARCH RECOMMENDATION. Keep the Scientific Verdict, the current
  Opportunity/market read, User Fit, and your RESEARCH RECOMMENDATION as
  visibly separate judgments -- never collapse them into one combined score.
- RESEARCH RECOMMENDATION must be exactly one of: INVESTIGATE, REFINE, RETEST,
  PAPER TEST, WAIT, NOT SUPPORTED. Never write BUY, SELL, or any language
  implying guaranteed or expected future returns. Only recommend PAPER TEST
  when the evidence's own `paper_eligible` value is true.
- If a section's evidence is missing, say so plainly in that section rather
  than guessing."""


def generate_decision_brief(evidence: dict[str, Any], *, mode: str) -> tuple[str | None, str | None]:
    """Returns `(brief_text, error)`. `mode` is `OFFLINE_MODE` (default, zero
    network calls) or `CLAUDE_MODE` (one bounded live call)."""
    if mode != CLAUDE_MODE:
        return render_deterministic_brief(evidence), None
    try:
        client = AnthropicClient()
        resp = client.complete(
            system=_SYSTEM_INSTRUCTION + "\n\n---\nEVIDENCE\n---\n\n" + _evidence_as_text(evidence),
            messages=[{"role": "user", "content": "Write the Decision Brief."}],
            model=DEFAULT_MODEL,
            max_tokens=1024,
            temperature=0.0,
        )
        return resp.text, None
    except LLMClientUnavailable as exc:
        return None, f"Claude is unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 -- defensive UI boundary, never a raw stack trace or key material
        return None, f"{type(exc).__name__}: {exc}"


def _evidence_as_text(evidence: dict[str, Any]) -> str:
    v = evidence["validation"]
    cpp = evidence["cpp_historical_test"]
    fm = evidence.get("research_memory") or {}
    rc = evidence.get("research_classification") or {}
    return "\n".join(
        [
            f"ROOT: {evidence['root_symbol']}; STRATEGY: {evidence['strategy_name']}",
            f"RESEARCH CLASSIFICATION: {rc}",
            f"CURRENT MARKET: {evidence['current_market']}",
            f"MECHANISM: {evidence['mechanism']}",
            f"C++ HISTORICAL TEST: {cpp}",
            f"VALIDATION: verdict={v['verdict']}; reason_codes={v['reason_codes']}; {v['plain_reason']}",
            f"RESEARCH MEMORY: {fm}",
            f"NEWS / CATALYST: {evidence['news_catalyst']}",
            f"USER FIT: {evidence['user_fit']}",
            f"EVIDENCE GAPS: {evidence['evidence_gaps']}",
            f"PAPER ELIGIBLE: {evidence['paper_eligible']}",
        ]
    )

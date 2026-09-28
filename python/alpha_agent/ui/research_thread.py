"""Research Thread -- the ONE persistent research object the News Alpha
workspace enriches step by step (News -> Reasoning -> Market -> Signals ->
Portfolio -> Backtest -> Learn), with the Research Setup as persistent context.

A thread stores REFERENCES and the user's navigation/focus -- never a copy of
a scientific result. Every stage value (impact scan, mechanism graph, signal
paths, asset expressions, candidate signals, ranking, portfolio plan,
validation gate, route memory) is resolved from the canonical backend on
render through `ThreadPipeline`, which only calls the existing UI boundaries
(`news_alpha_context`, `candidate_signal_view`, `signal_ranking_view`,
`portfolio_plan_view`, `research_loop_view`, `services`). Nothing here scores,
ranks, sizes, validates or recalls anything itself.

Selections are FOCUS, never a change to the tested family: the candidate set,
its ranking and the portfolio are always the event's full canonical ones (a
user deselecting markets after seeing screen results would otherwise shrink
the multiple-testing family post hoc).

Research Setup is the user's ONE `ResearchMandate` (access half via
`MandateStore`, risk half via the saved `InvestorProfile`) -- not a per-thread
copy, because `MandateStore` exists precisely so the two halves can never
drift into disagreeing copies. Each thread records the setup fingerprint its
Market-onward choices were made under; `reconcile_setup` invalidates those
steps when the setup changes (Reasoning is mandate-independent by
construction -- the graph is built from the event alone -- and stays).

Persistence: `data/user_prefs/research_threads.json` (gitignored UI workspace
state, like the mandate and profile files; never the registry).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from alpha_agent.market_intel.event_schemas import ScheduledMarketEvent
from alpha_agent.market_intel.news_schemas import MarketNewsItem
from alpha_agent.news_alpha import (
    AssetExpressionPlan,
    CandidateSignalSet,
    EconomicMechanismGraph,
    ExpressionStatus,
    ImpactLevel,
    InitialImpactScan,
    MechanismAdjustedImpact,
    ResearchMandate,
    SignalPathDiscovery,
    UserDescribedEvent,
)
from alpha_agent.news_alpha.mandate import DEFAULT_MANDATE_PATH

__all__ = [
    "MANDATE_DEPENDENT_STEPS",
    "STEPS",
    "STEP_LABELS",
    "STEP_QUESTIONS",
    "THREAD_STORE",
    "MarketGroup",
    "ResearchThread",
    "ResearchThreadStore",
    "ThreadEventRef",
    "ThreadPipeline",
    "ThreadStep",
    "active_thread",
    "advance",
    "create_thread",
    "deep_link",
    "go_to",
    "open_thread_for_scan",
    "pipeline_for",
    "reconcile_setup",
    "set_active",
    "thread_id_for_event",
    "update_thread",
]


class ThreadStep(str, Enum):
    NEWS = "news"
    REASONING = "reasoning"
    MARKET = "market"
    SIGNALS = "signals"
    PORTFOLIO = "portfolio"
    BACKTEST = "backtest"
    LEARN = "learn"


#: Declaration order IS the research order.
STEPS: tuple[ThreadStep, ...] = tuple(ThreadStep)

STEP_LABELS: dict[ThreadStep, str] = {
    ThreadStep.NEWS: "News", ThreadStep.REASONING: "Reasoning", ThreadStep.MARKET: "Market",
    ThreadStep.SIGNALS: "Signals", ThreadStep.PORTFOLIO: "Portfolio", ThreadStep.BACKTEST: "Backtest",
    ThreadStep.LEARN: "Learn",
}

#: The question each step answers -- the user's mental journey.
STEP_QUESTIONS: dict[ThreadStep, str] = {
    ThreadStep.NEWS: "What happened?",
    ThreadStep.REASONING: "Why could this matter economically?",
    ThreadStep.MARKET: "Where is this effect showing up in markets -- and can we measure it?",
    ThreadStep.SIGNALS: "Is there a quantitative signal?",
    ThreadStep.PORTFOLIO: "How would these signals form a portfolio?",
    ThreadStep.BACKTEST: "Does it survive a real backtest and scientific validation?",
    ThreadStep.LEARN: "What did we learn?",
}

#: Steps whose content depends on the Research Setup. Reasoning is built from
#: the event alone (mandate-independent by test in Phase B), so a setup change
#: never invalidates it.
MANDATE_DEPENDENT_STEPS: tuple[ThreadStep, ...] = (
    ThreadStep.MARKET, ThreadStep.SIGNALS, ThreadStep.PORTFOLIO, ThreadStep.BACKTEST, ThreadStep.LEARN,
)

SourceType = Literal["news", "scheduled", "user"]


class ThreadEventRef(BaseModel):
    """The event a thread is about. ``source`` is the canonical source object
    itself (its own JSON form), kept so the initial impact scan stays
    reproducible after the news cache rotates or the session that described
    the event ends. ``headline`` is display-only, for thread lists."""

    model_config = {"frozen": True, "extra": "forbid"}

    event_id: str
    kind: str
    headline: str
    source_type: SourceType
    source: dict[str, Any]

    def source_object(self) -> MarketNewsItem | ScheduledMarketEvent | UserDescribedEvent:
        model = {"news": MarketNewsItem, "scheduled": ScheduledMarketEvent, "user": UserDescribedEvent}[self.source_type]
        return model.model_validate(self.source)

    @classmethod
    def from_scan(cls, scan: InitialImpactScan) -> ThreadEventRef:
        src = scan.event.source
        if isinstance(src, UserDescribedEvent):
            source_type: SourceType = "user"
        elif isinstance(src, ScheduledMarketEvent):
            source_type = "scheduled"
        else:
            source_type = "news"
        return cls(
            event_id=scan.event.event_id, kind=scan.event.kind.value, headline=scan.event.headline,
            source_type=source_type, source=src.model_dump(mode="json"),
        )


class ResearchThread(BaseModel):
    """One persistent research thread -- references and focus only."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: Literal["research-thread/1"] = "research-thread/1"
    thread_id: str
    name: str = Field(min_length=1, max_length=120)
    event: ThreadEventRef
    created_at: datetime
    updated_at: datetime
    current_step: ThreadStep = ThreadStep.NEWS
    completed_steps: tuple[ThreadStep, ...] = ()
    #: `ResearchMandate.fingerprint()` the Market-onward choices were made under.
    setup_fingerprint: str | None = None
    #: Reasoning focus -- `SignalPath.path_id` values the user follows.
    followed_path_ids: tuple[str, ...] = ()
    #: Market focus -- the `AssetExpression.expression_id` being inspected.
    focus_expression_id: str | None = None
    #: Last seen `EconomicMechanismGraph.fingerprint()` (a drift notice only).
    mechanism_graph_id: str | None = None
    #: The furthest step ever opened -- stepping back never makes it unreachable.
    reached_step: ThreadStep = ThreadStep.NEWS

    @model_validator(mode="after")
    def _utc(self) -> ResearchThread:
        for ts in (self.created_at, self.updated_at):
            if ts.tzinfo is None:
                raise ValueError("thread timestamps must be UTC-aware")
        return self

    def status(self, step: ThreadStep) -> Literal["done", "current", "todo"]:
        if step is self.current_step:
            return "current"
        return "done" if step in self.completed_steps else "todo"

    def furthest_step(self) -> ThreadStep:
        reached = {self.current_step, self.reached_step, *self.completed_steps}
        return max(reached, key=STEPS.index)

    def is_reachable(self, step: ThreadStep) -> bool:
        """Completed steps, the current one, and the step right after the
        furthest one reached -- never further, because every later step reads
        what the earlier ones resolved. Not a rigid wizard: any completed step
        can be revisited at any time."""
        return STEPS.index(step) <= STEPS.index(self.furthest_step()) + 1


def thread_id_for_event(event_id: str) -> str:
    """Deterministic: one thread per event, never a random UUID."""
    return "thread-" + hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:12]


def _now() -> datetime:
    return datetime.now(UTC)


class ResearchThreadStore:
    """Local JSON persistence for research threads (gitignored
    `data/user_prefs/`, beside the mandate and profile files; never the
    registry). A corrupt file is not fatal -- it reads as no threads, and is
    only replaced on the next save."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_MANDATE_PATH.parent / "research_threads.json"

    def list(self) -> tuple[ResearchThread, ...]:
        if not self.path.exists():
            return ()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            threads = [ResearchThread.model_validate(t) for t in raw.get("threads", [])]
        except Exception:  # noqa: BLE001 -- a corrupt local workspace file is not fatal
            return ()
        return tuple(sorted(threads, key=lambda t: t.updated_at, reverse=True))

    def get(self, thread_id: str) -> ResearchThread | None:
        return next((t for t in self.list() if t.thread_id == thread_id), None)

    def save(self, thread: ResearchThread) -> None:
        others = [t for t in self.list() if t.thread_id != thread.thread_id]
        payload = {
            "schema_version": "research-threads/1",
            "threads": [t.model_dump(mode="json") for t in (thread, *others)],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".research_threads.", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)


#: Module-level so tests (`tests/python/conftest.py`) point it at a temporary
#: path -- never the user's real workspace file.
THREAD_STORE = ResearchThreadStore()

_ACTIVE_KEY = "research_thread_active_id"


# ---------------------------------------------------------------------------
# thread lifecycle (Streamlit session aware)
# ---------------------------------------------------------------------------


def _session():
    import streamlit as st

    return st.session_state


def active_thread() -> ResearchThread | None:
    thread_id = _session().get(_ACTIVE_KEY)
    return THREAD_STORE.get(thread_id) if thread_id else None


def set_active(thread_id: str | None) -> None:
    if thread_id is None:
        _session().pop(_ACTIVE_KEY, None)
    else:
        _session()[_ACTIVE_KEY] = thread_id


def update_thread(thread: ResearchThread, **changes: Any) -> ResearchThread:
    new = thread.model_copy(update={**changes, "updated_at": _now()})
    new = ResearchThread.model_validate(new.model_dump())  # re-validate the copy
    THREAD_STORE.save(new)
    return new


def _default_name(scan: InitialImpactScan, graph: EconomicMechanismGraph) -> str:
    if graph.anchors:
        label = graph.anchors[0].label
        return label[:1].upper() + label[1:]
    headline = scan.event.headline
    return headline if len(headline) <= 80 else headline[:77].rstrip() + "…"


def open_thread_for_scan(scan: InitialImpactScan, mandate: ResearchMandate) -> ResearchThread:
    """Find-or-create the thread for this scan's event and make it active.
    Re-opening an event returns the SAME thread (its progress intact)."""
    thread = create_thread(scan, mandate)
    set_active(thread.thread_id)
    return thread


def create_thread(scan: InitialImpactScan, mandate: ResearchMandate) -> ResearchThread:
    """Find-or-create (no session state): the saved thread for this scan's
    event, or a new one at the News step."""
    from alpha_agent.ui import news_alpha_context

    thread_id = thread_id_for_event(scan.event.event_id)
    existing = THREAD_STORE.get(thread_id)
    if existing is not None:
        return existing
    graph = news_alpha_context.mechanism_graph(scan)
    now = _now()
    thread = ResearchThread(
        thread_id=thread_id, name=_default_name(scan, graph), event=ThreadEventRef.from_scan(scan),
        created_at=now, updated_at=now, setup_fingerprint=mandate.fingerprint(),
        mechanism_graph_id=None if graph.is_empty else graph.fingerprint(),
    )
    THREAD_STORE.save(thread)
    return thread


def _later(a: ThreadStep, b: ThreadStep) -> ThreadStep:
    return a if STEPS.index(a) >= STEPS.index(b) else b


def go_to(thread: ResearchThread, step: ThreadStep) -> ResearchThread:
    """Move to a reachable step; completion marks are untouched."""
    if not thread.is_reachable(step):
        raise ValueError(f"{step.value} is not reachable yet from {thread.current_step.value}")
    return update_thread(thread, current_step=step, reached_step=_later(thread.furthest_step(), step))


def advance(thread: ResearchThread) -> ResearchThread:
    """The explicit "Next" action: completes the current step, moves on."""
    i = STEPS.index(thread.current_step)
    completed = tuple(s for s in STEPS if s in {*thread.completed_steps, thread.current_step})
    nxt = STEPS[min(i + 1, len(STEPS) - 1)]
    return update_thread(thread, completed_steps=completed, current_step=nxt,
                         reached_step=_later(thread.furthest_step(), nxt))


def deep_link(thread: ResearchThread, step: ThreadStep) -> ResearchThread:
    """Open a thread directly at ``step`` (from My Alpha, a signal, a past
    experiment, a URL): the earlier lineage counts as completed -- it is
    inspectable, never replayed."""
    before = STEPS[: STEPS.index(step)]
    completed = tuple(s for s in STEPS if s in {*thread.completed_steps, *before})
    return update_thread(thread, completed_steps=completed, current_step=step,
                         reached_step=_later(thread.furthest_step(), step))


def reconcile_setup(
    thread: ResearchThread, mandate: ResearchMandate, expressions: AssetExpressionPlan,
) -> tuple[ResearchThread, str | None]:
    """When the Research Setup changed since this thread's Market-onward
    choices were made: those steps need review again (their completion marks
    are cleared), and a market focus that no longer continues under the new
    setup is dropped. Returns the (saved) thread and a notice, or the thread
    unchanged and ``None``."""
    fp = mandate.fingerprint()
    if thread.setup_fingerprint == fp:
        return thread, None
    progressed = any(s in MANDATE_DEPENDENT_STEPS for s in (*thread.completed_steps, thread.current_step))
    changes: dict[str, Any] = {"setup_fingerprint": fp}
    notice = None
    if progressed and thread.setup_fingerprint is not None:
        changes["completed_steps"] = tuple(s for s in thread.completed_steps if s not in MANDATE_DEPENDENT_STEPS)
        focus = thread.focus_expression_id
        still = next((e for e in expressions.expressions if e.expression_id == focus), None)
        if focus is not None and (still is None or still.status is not ExpressionStatus.CONTINUES):
            changes["focus_expression_id"] = None
        if thread.current_step in MANDATE_DEPENDENT_STEPS:
            changes["current_step"] = ThreadStep.MARKET
        changes["reached_step"] = ThreadStep.MARKET  # later steps need the new setup's review first
        notice = (
            "Your Research Setup changed since you last worked on this thread. Markets, signals and the "
            "portfolio were rebuilt under the new setup, so those steps are marked for review again. "
            "Reasoning is unaffected -- it depends on the event alone."
        )
    return update_thread(thread, **changes), notice


# ---------------------------------------------------------------------------
# pipeline -- canonical backend objects, resolved lazily per render
# ---------------------------------------------------------------------------


@dataclass
class ThreadPipeline:
    """The thread's canonical backend objects, each computed at most once per
    render and only when a step actually needs it. Every property delegates
    to an existing UI boundary; none computes a research value itself."""

    thread: ResearchThread
    mandate: ResearchMandate
    _memo: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def scan(self) -> InitialImpactScan:
        from alpha_agent.ui import news_alpha_context

        return news_alpha_context.scan_event_source(self.thread.event.source_object(), self.mandate)

    @cached_property
    def graph(self) -> EconomicMechanismGraph:
        from alpha_agent.ui import news_alpha_context

        return news_alpha_context.mechanism_graph(self.scan)

    @cached_property
    def paths(self) -> SignalPathDiscovery:
        from alpha_agent.ui import news_alpha_context

        return news_alpha_context.signal_paths(self.graph)

    @cached_property
    def adjusted(self) -> MechanismAdjustedImpact | None:
        from alpha_agent.ui import news_alpha_context

        return None if self.graph.is_empty else news_alpha_context.adjusted_impact(self.scan, self.paths)

    @cached_property
    def expressions(self) -> AssetExpressionPlan:
        from alpha_agent.ui import news_alpha_context

        return news_alpha_context.asset_expressions(self.paths, self.mandate)

    @cached_property
    def candidates(self) -> CandidateSignalSet:
        from alpha_agent.ui import news_alpha_context

        return news_alpha_context.candidate_signals(self.expressions, self.paths)

    @cached_property
    def screens(self) -> dict:
        from alpha_agent.ui import candidate_signal_view

        return candidate_signal_view.stored_screens(self.candidates)

    @cached_property
    def ranked(self):
        from alpha_agent.ui import signal_ranking_view

        return signal_ranking_view.rank_for([self.candidates], self.mandate)

    def plan(self, mode):
        """``(plan | None, missing_snapshots)`` for one eligibility mode."""
        key = f"plan:{mode.value}"
        if key not in self._memo:
            from alpha_agent.ui import portfolio_plan_view

            self._memo[key] = portfolio_plan_view.build_plan(self.ranked, [self.candidates], self.mandate, mode)
        return self._memo[key]

    @cached_property
    def gate(self):
        """The Phase H validation gate on the QUALIFIED plan -- ``None`` while
        its market data is not loaded, or when the plan is eligible (the gate
        reports only what blocks validation)."""
        from alpha_agent.portfolio import EligibilityMode
        from alpha_agent.ui import research_loop_view

        plan, _missing = self.plan(EligibilityMode.QUALIFIED)
        return None if plan is None else research_loop_view.gate_for(plan, self.ranked, self.screens)

    # -- focus (display filters over canonical objects; never a new family) --

    def followed_paths(self):
        ids = set(self.thread.followed_path_ids)
        return tuple(p for p in self.paths.paths if p.path_id in ids)

    def focus_expression(self):
        fid = self.thread.focus_expression_id
        return next((e for e in self.expressions.expressions if e.expression_id == fid), None)

    def market_groups(self) -> tuple[MarketGroup, ...]:
        """The plan's asset expressions grouped by the market they name (same
        asset class, concept and instruments), in the backend's own order.
        One instrument is often reached through several consequences (the AI
        slice reaches NQ through six); a group keeps every one of them."""
        groups: dict[tuple, list] = {}
        for e in self.expressions.expressions:
            groups.setdefault((e.domain, e.concept, e.symbols), []).append(e)
        return tuple(MarketGroup(expressions=tuple(v)) for v in groups.values())

    def focus_group(self) -> MarketGroup | None:
        fid = self.thread.focus_expression_id
        return next((g for g in self.market_groups() if fid in g.expression_ids), None)

    def initial_exposure(self, group: MarketGroup) -> tuple[str, ImpactLevel] | None:
        """The Initial Impact Scan's OWN exposure for this market: the scan
        assigns each candidate market to one exposure and each exposure a
        level; this only looks that pair up (first match, in the scan's
        order). ``None`` when the scan did not name the market -- it was then
        reached only through the mechanism graph."""
        try:
            assessment = self.scan.assessment(group.domain)
        except StopIteration:
            return None
        symbols = set(group.symbols)
        market = next((m for m in assessment.candidate_markets if m.symbol in symbols), None)
        if market is None:
            return None
        contribution = next((c for c in assessment.contributions if c.exposure_label == market.exposure_label), None)
        if contribution is None or contribution.impact_level is None:
            return None
        return contribution.exposure_label, contribution.impact_level


@dataclass(frozen=True)
class MarketGroup:
    """Every asset expression naming one market. A presentation grouping --
    each member stays the canonical `AssetExpression` it was."""

    expressions: tuple

    @property
    def first(self):
        return self.expressions[0]

    @property
    def domain(self):
        return self.first.domain

    @property
    def concept(self) -> str:
        return self.first.concept

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.first.symbols

    @property
    def status(self) -> ExpressionStatus:
        return self.first.status

    @property
    def expression_ids(self) -> tuple[str, ...]:
        return tuple(e.expression_id for e in self.expressions)

    @property
    def path_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p for e in self.expressions for p in e.path_ids))

    @property
    def consequences(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.consequence_label for e in self.expressions))

    @property
    def measurement_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(m for e in self.expressions for m in e.measurement_ids))

    @property
    def key(self) -> str:
        return hashlib.sha256(
            f"{self.domain.value}|{self.concept}|{','.join(self.symbols)}".encode()
        ).hexdigest()[:12]


def pipeline_for(thread: ResearchThread, mandate: ResearchMandate) -> ThreadPipeline:
    return ThreadPipeline(thread=thread, mandate=mandate)

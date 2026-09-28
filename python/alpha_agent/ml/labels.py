"""The typed :class:`MetaLabelEvent` and its frozen :class:`MetaLabelSpec`.

The single causal invariant of Phase 15
---------------------------------------
::

    feature_timestamp  <=  decision_timestamp  <  label_start  <=  label_end

The LABEL may look forward -- that is what a supervised target *is*. No FEATURE
may. :func:`assert_event_causality` enforces exactly that, per event, and is
called on every event the builder emits, so a leaky event cannot reach a
training set even by mistake.

``label_start_timestamp`` is the episode's entry decision and
``label_end_timestamp`` is its C++ information horizon (the last
``ts_close_ns`` of an attributed Fill-derived trade). Purge and embargo are
computed from that horizon, not from the exit decision, because that is
genuinely the last instant the label depends on.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from alpha_agent.ml.enums import EpisodeExclusionReason, LabelKind
from alpha_agent.ml.episodes import EpisodeEconomics, PrimaryEpisode
from alpha_agent.ml.errors import LeakageError
from alpha_agent.ml.guards import assert_within_development_corpus
from alpha_agent.validation.fingerprint import fingerprint

METALABEL_SPEC_SCHEMA = "meta-label-spec/1"
METALABEL_EVENT_SCHEMA = "meta-label-event/1"


class MetaLabelSpec(BaseModel):
    """How the supervised target is defined. Frozen before any training."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = METALABEL_SPEC_SCHEMA
    kind: LabelKind = LabelKind.PRIMARY_EPISODE_NET_PNL_SIGN
    #: label 1 ("the primary signal was worth taking") iff net_pnl_usd > this.
    #: Strictly greater: a zero-PnL episode labels 0, because taking it spent
    #: risk and cost for nothing.
    positive_threshold_usd: float = 0.0
    #: An episode with no attributed fill is UNLABELABLE, never labelled 0.
    require_attributed_fills: bool = True
    #: An episode still open at the corpus end is UNLABELABLE, never labelled 0.
    require_terminated_episode: bool = True
    #: Features are read at the entry decision bar itself; the Feature Engine is
    #: point-in-time so a value at T uses only information at or before T.
    feature_offset_bars: int = Field(default=0, ge=0)
    label_definition_text: str = (
        "label = 1 iff the SUM of C++ Fill-derived net_pnl_usd over the closed "
        "trades attributed to the primary episode is strictly greater than "
        "positive_threshold_usd; else 0. Attribution: a trade belongs to the "
        "episode that was open when the trade opened. Episodes that are "
        "unterminated at the corpus end, or that produced no attributed fill, "
        "are EXCLUDED with a typed reason and never labelled 0."
    )

    def identity(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("label_definition_text", None)   # prose, never semantics
        return fingerprint("mlmetalabelspec1", payload)


class MetaLabelEvent(BaseModel):
    """One decision point: a primary signal, its causal features, its label."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = METALABEL_EVENT_SCHEMA
    event_id: str
    root_symbol: str
    strategy_family: str
    strategy_fingerprint: str

    # -- the primary signal being judged --------------------------------------
    primary_signal_timestamp: int = Field(gt=0)
    primary_side: int                                   # +1 long, -1 short
    primary_target: int                                 # the primary's own target_units
    episode_index: int = Field(ge=0)

    # -- the causal decision boundary -----------------------------------------
    decision_timestamp: int = Field(gt=0)
    feature_timestamp: int = Field(gt=0)

    # -- the supervised target (may look forward) -----------------------------
    label_start_timestamp: int = Field(gt=0)
    label_end_timestamp: int = Field(gt=0)
    label: int = Field(ge=0, le=1)
    label_definition: str
    #: audit only -- NEVER a model feature, and never used for selection
    episode_net_pnl_usd: float
    episode_gross_pnl_usd: float
    episode_costs_usd: float
    n_attributed_trades: int = Field(ge=0)

    # -- fold assignment (filled by the split planner) ------------------------
    development_fold: str | None = None

    # -- source fingerprints ---------------------------------------------------
    meta_label_spec_fingerprint: str
    feature_set_fingerprint: str
    dataset_fingerprint: str
    target_schedule_hash: str

    @model_validator(mode="after")
    def _causal(self) -> MetaLabelEvent:
        if self.primary_side not in (-1, 1):
            raise ValueError("primary_side must be +1 or -1")
        assert_event_causality(self)
        return self


def assert_event_causality(event: MetaLabelEvent) -> None:
    """The Phase 15 causal invariant, enforced per event.

    Also refuses any locked-holdout timestamp: a 2025 label horizon is not a
    "slightly optimistic" event, it is holdout access.
    """
    if event.feature_timestamp > event.decision_timestamp:
        raise LeakageError(
            f"event {event.event_id}: feature_timestamp {event.feature_timestamp} is AFTER "
            f"decision_timestamp {event.decision_timestamp}; a model feature may never use "
            "information the decision did not have"
        )
    if event.primary_signal_timestamp != event.decision_timestamp:
        raise LeakageError(
            f"event {event.event_id}: the meta-label decision must be taken at the primary "
            f"signal timestamp ({event.primary_signal_timestamp}), got "
            f"{event.decision_timestamp}"
        )
    if event.label_start_timestamp < event.decision_timestamp:
        raise LeakageError(
            f"event {event.event_id}: label_start_timestamp {event.label_start_timestamp} "
            f"precedes the decision {event.decision_timestamp}; the label would then partly "
            "describe the past the decision already knew"
        )
    if event.label_end_timestamp <= event.decision_timestamp:
        raise LeakageError(
            f"event {event.event_id}: label_end_timestamp {event.label_end_timestamp} does not "
            f"extend past the decision {event.decision_timestamp}; a label that resolves at or "
            "before the decision carries no future information and is a definition bug"
        )
    if event.label_end_timestamp < event.label_start_timestamp:
        raise LeakageError(f"event {event.event_id}: label window ends before it starts")
    assert_within_development_corpus(
        [
            event.feature_timestamp,
            event.decision_timestamp,
            event.label_start_timestamp,
            event.label_end_timestamp,
        ],
        what=f"MetaLabelEvent[{event.event_id}]",
    )


class ExcludedEpisode(BaseModel):
    """An episode that produced no labelled event. First-class, never dropped."""

    model_config = {"frozen": True, "extra": "forbid"}

    episode_index: int
    root_symbol: str
    strategy_family: str
    entry_decision_ts_ns: int
    reason: EpisodeExclusionReason
    detail: str = ""


class MetaLabelEventSet(BaseModel):
    """Every labelled event for one primary, plus every exclusion and why."""

    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    strategy_family: str
    strategy_fingerprint: str
    meta_label_spec_fingerprint: str
    events: tuple[MetaLabelEvent, ...]
    excluded: tuple[ExcludedEpisode, ...]
    unattributed_trade_count: int = 0

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def positive_rate(self) -> float:
        return (sum(e.label for e in self.events) / len(self.events)) if self.events else 0.0

    def label_horizons_ns(self) -> tuple[tuple[int, int], ...]:
        return tuple((e.label_start_timestamp, e.label_end_timestamp) for e in self.events)


def build_meta_label_events(
    *,
    episodes: tuple[PrimaryEpisode, ...],
    economics: dict[int, EpisodeEconomics],
    spec: MetaLabelSpec,
    strategy_family: str,
    dataset_fingerprint: str,
    feature_set_fingerprint: str,
    target_schedule_hash: str,
    feature_availability: dict[int, bool] | None = None,
) -> MetaLabelEventSet:
    """Turn primary episodes + their C++ Fill-derived economics into labelled events.

    ``feature_availability`` maps episode index -> whether every declared ML
    feature was actually present at that episode's decision bar. A missing
    feature EXCLUDES the event; it is never imputed at build time (imputation is
    a *fitted* transform and belongs strictly inside a training fold).
    """
    if not episodes:
        return MetaLabelEventSet(
            root_symbol="",
            strategy_family=strategy_family,
            strategy_fingerprint="",
            meta_label_spec_fingerprint=spec.identity(),
            events=(),
            excluded=(),
        )

    # Guard FIRST, at the boundary, so a locked-holdout value raises the typed
    # HoldoutAccessError. The per-event check inside MetaLabelEvent stays as
    # defence in depth, but there pydantic wraps it (HoldoutAccessError is an
    # AssertionError so the Phase 14 registry can keep catching it), and a typed
    # error at the real entry point is what a caller should actually see.
    assert_within_development_corpus(
        [e.entry_decision_ts_ns for e in episodes], what="primary episode entry decisions"
    )
    assert_within_development_corpus(
        [
            e.information_horizon_ts_ns
            for e in economics.values()
            if e.information_horizon_ts_ns is not None
        ],
        what="primary episode information horizons",
    )

    root = episodes[0].root_symbol
    strat_fp = episodes[0].strategy_fingerprint
    events: list[MetaLabelEvent] = []
    excluded: list[ExcludedEpisode] = []

    def drop(ep: PrimaryEpisode, reason: EpisodeExclusionReason, detail: str = "") -> None:
        excluded.append(
            ExcludedEpisode(
                episode_index=ep.episode_index,
                root_symbol=ep.root_symbol,
                strategy_family=strategy_family,
                entry_decision_ts_ns=ep.entry_decision_ts_ns,
                reason=reason,
                detail=detail,
            )
        )

    for ep in episodes:
        if spec.require_terminated_episode and not ep.is_terminated:
            drop(ep, EpisodeExclusionReason.UNTERMINATED_AT_CORPUS_END,
                 "still open at the end of the schedule; its outcome is not observable")
            continue
        if feature_availability is not None and not feature_availability.get(ep.episode_index, True):
            drop(ep, EpisodeExclusionReason.FEATURES_MISSING_AT_DECISION,
                 "at least one declared ML feature was missing at the decision bar")
            continue
        econ = economics.get(ep.episode_index)
        if econ is None or (spec.require_attributed_fills and econ.n_attributed_trades == 0):
            drop(ep, EpisodeExclusionReason.NO_ATTRIBUTED_FILLS,
                 "the primary decision produced no C++ fill, so it has no Fill-derived outcome")
            continue

        events.append(
            MetaLabelEvent(
                event_id=f"{root}__{strategy_family}__EP{ep.episode_index:05d}",
                root_symbol=root,
                strategy_family=strategy_family,
                strategy_fingerprint=strat_fp,
                primary_signal_timestamp=ep.entry_decision_ts_ns,
                primary_side=ep.side,
                primary_target=ep.entry_target_units,
                episode_index=ep.episode_index,
                decision_timestamp=ep.entry_decision_ts_ns,
                feature_timestamp=ep.entry_decision_ts_ns,
                label_start_timestamp=ep.entry_decision_ts_ns,
                label_end_timestamp=econ.information_horizon_ts_ns,
                label=int(econ.net_pnl_usd > spec.positive_threshold_usd),
                label_definition=spec.label_definition_text,
                episode_net_pnl_usd=econ.net_pnl_usd,
                episode_gross_pnl_usd=econ.gross_pnl_usd,
                episode_costs_usd=econ.costs_usd,
                n_attributed_trades=econ.n_attributed_trades,
                meta_label_spec_fingerprint=spec.identity(),
                feature_set_fingerprint=feature_set_fingerprint,
                dataset_fingerprint=dataset_fingerprint,
                target_schedule_hash=target_schedule_hash,
            )
        )

    return MetaLabelEventSet(
        root_symbol=root,
        strategy_family=strategy_family,
        strategy_fingerprint=strat_fp,
        meta_label_spec_fingerprint=spec.identity(),
        events=tuple(events),
        excluded=tuple(excluded),
    )

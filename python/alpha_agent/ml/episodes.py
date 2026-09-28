"""The PRIMARY TRADE EPISODE -- the unit a Phase 15 meta-label decides about.

Why an episode and not a fixed future-return horizon
----------------------------------------------------
Prompt 15A section 5 asks for a label aligned to the primary strategy's own
closed trade episode rather than an arbitrary horizon. An episode is definable
with **zero** change to primary strategy semantics and **zero** change to frozen
C++ economics, because both halves already exist:

* the *boundaries* are pure target intent, already in the Phase 11
  :class:`~alpha_agent.backtest.targets.TargetSchedule`;
* the *economics* are already in ``BacktestResult::trades``, and Phase 15A adds
  only a READ-ONLY CSV export of that audit trail.

Episode definition (FROZEN)
---------------------------
Let ``sign(u) in {-1, 0, +1}`` over the schedule's rows, which are strictly
time-ordered. Under the reference ``no_decision`` policy an absent row means "no
new decision", so the target between rows is the last row's target.

* An episode OPENS at the first row whose sign is non-zero and differs from the
  sign in force immediately before it.
* It CLOSES at the first later row whose sign differs from the episode's sign
  (a flat, or a reversal). That closing row is *not* part of the episode; a
  reversal simultaneously opens the next episode.
* A same-sign size change does not open a new episode.
* An episode still open at the end of the schedule is UNTERMINATED and yields no
  labelled event -- its outcome is not observable inside the corpus.

Trade attribution (FROZEN)
--------------------------
A closed trade belongs to the episode that was open when the trade OPENED:
trade ``j`` is attributed to the episode with the greatest
``entry_decision_ts_ns <= j.ts_open_ns``, provided that episode has not already
closed at its own ``exit_decision_ts_ns``. This is total, deterministic and
order-independent, and it puts a roll's close-leg and re-open-leg (``close_reason
== "roll"``) inside the episode that spans the roll -- which is what "this
episode's economics" means.

The episode's information horizon is therefore ``max(ts_close_ns)`` over its
attributed trades -- NOT the exit decision timestamp. Purge/embargo uses that
horizon, because that is genuinely the last instant the label depends on.

Costs: the full round turn, attributed PER CONTRACT
---------------------------------------------------
``ClosedTrade.costs_usd`` is the CLOSING fill's commission only (see
``cpp/src/position_ledger.cpp``); the opening commission is real, is in the
engine's headline total, and belongs to no ``ClosedTrade``. So an episode's cost
is summed from the FILL audit trail instead. Gross PnL still comes from the
engine's own ``gross_pnl_usd``; Python sums, it never prices.

Commission is a flat USD-per-contract charge, so a fill's commission is split by
CONTRACT, not handed whole to one episode:

* a fill's per-contract rate is ``commission_usd / quantity``;
* the contracts that CLOSED exposure are exactly the ``quantity`` of the
  ``ClosedTrade`` the engine emitted for that fill, and their commission belongs
  to the episode that owns that trade;
* every remaining contract of the fill OPENED exposure, and its commission
  belongs to the episode open at the fill instant.

That split is what makes a direct LONG -> SHORT flip attribute correctly. The
engine flips through zero with ONE fill of two contracts: one closes the long,
one opens the short. Charging the whole commission to the outgoing episode would
overstate its cost and understate the incoming episode's -- biasing two labels in
opposite directions at every reversal. A roll needs no special case and gets
none: the close leg carries a ``ClosedTrade`` on the outgoing instrument and the
re-open leg carries none on the incoming one, and both resolve to the SAME
episode because the primary target direction never changed.

Nothing is dropped. A fill whose contracts resolve to no episode is reported as
an unattributed residual, and :func:`reconcile_episode_economics` proves the
episode economics add back up to the engine's own headline net PnL.
"""
from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

from pydantic import BaseModel, Field, model_validator

from alpha_agent.backtest.targets import TargetSchedule
from alpha_agent.ml.trade_export import ClosedTradeRecord, FillRecord

EPISODE_SCHEMA_VERSION = "primary-episode/1"


def _sign(u: int) -> int:
    return (u > 0) - (u < 0)


class PrimaryEpisode(BaseModel):
    """One directional exposure run of the PRIMARY strategy, in target-intent space."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = EPISODE_SCHEMA_VERSION
    episode_index: int = Field(ge=0)
    root_symbol: str
    strategy_fingerprint: str

    entry_row_index: int = Field(ge=0)
    #: exclusive; None when the episode is still open at the end of the schedule
    exit_row_index: int | None = None

    entry_decision_ts_ns: int = Field(gt=0)
    exit_decision_ts_ns: int | None = None

    side: int                                   # +1 long, -1 short
    entry_target_units: int

    @model_validator(mode="after")
    def _check(self) -> PrimaryEpisode:
        if self.side not in (-1, 1):
            raise ValueError("episode side must be +1 or -1")
        if _sign(self.entry_target_units) != self.side:
            raise ValueError("entry_target_units sign disagrees with the episode side")
        if (self.exit_row_index is None) != (self.exit_decision_ts_ns is None):
            raise ValueError("exit_row_index and exit_decision_ts_ns must be set together")
        if self.exit_row_index is not None:
            if self.exit_row_index <= self.entry_row_index:
                raise ValueError("episode exit row must come after its entry row")
            if self.exit_decision_ts_ns <= self.entry_decision_ts_ns:
                raise ValueError("episode exit ts must come after its entry ts")
        return self

    @property
    def is_terminated(self) -> bool:
        return self.exit_row_index is not None


class EpisodeEconomics(BaseModel):
    """The C++ Fill-derived outcome of one episode. Python sums; it never prices.

    Every field is an aggregation of values the engine produced. There is no
    Python-side mark, no proxy price and no back-adjusted return anywhere in it.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    episode_index: int = Field(ge=0)
    n_attributed_trades: int = Field(ge=0)
    gross_pnl_usd: float
    costs_usd: float
    net_pnl_usd: float
    #: the LAST instant the label depends on -- max ts_close_ns of attributed trades
    information_horizon_ts_ns: int | None = None
    n_roll_closes: int = Field(ge=0, default=0)
    n_attributed_fills: int = Field(ge=0, default=0)
    contracts_traded: tuple[str, ...] = ()
    official_source: str = "cpp_closed_trade_and_fill_audit_trails"
    #: True when costs were summed from the FILL trail (the full round turn).
    #: False means they came from ClosedTrade.costs_usd, which omits the entry
    #: commission and therefore understates the cost of every episode.
    costs_are_full_round_turn: bool = False


def extract_primary_episodes(schedule: TargetSchedule) -> tuple[PrimaryEpisode, ...]:
    """Split a primary :class:`TargetSchedule` into episodes. Pure target intent.

    Reads no market data, no fills and no prices; it is a function of the
    schedule alone, so it is computable before any backtest runs.
    """
    episodes: list[PrimaryEpisode] = []
    prev_sign = 0
    open_ep: dict | None = None

    for i, row in enumerate(schedule.rows):
        s = _sign(row.target_units)
        if open_ep is not None and s != open_ep["side"]:
            episodes.append(
                PrimaryEpisode(
                    episode_index=len(episodes),
                    root_symbol=schedule.root_symbol,
                    strategy_fingerprint=schedule.strategy_fingerprint,
                    entry_row_index=open_ep["entry_row_index"],
                    exit_row_index=i,
                    entry_decision_ts_ns=open_ep["entry_decision_ts_ns"],
                    exit_decision_ts_ns=row.ts_event_ns,
                    side=open_ep["side"],
                    entry_target_units=open_ep["entry_target_units"],
                )
            )
            open_ep = None
        if open_ep is None and s != 0 and s != prev_sign:
            open_ep = {
                "entry_row_index": i,
                "entry_decision_ts_ns": row.ts_event_ns,
                "side": s,
                "entry_target_units": row.target_units,
            }
        prev_sign = s

    if open_ep is not None:
        episodes.append(
            PrimaryEpisode(
                episode_index=len(episodes),
                root_symbol=schedule.root_symbol,
                strategy_fingerprint=schedule.strategy_fingerprint,
                entry_row_index=open_ep["entry_row_index"],
                exit_row_index=None,
                entry_decision_ts_ns=open_ep["entry_decision_ts_ns"],
                exit_decision_ts_ns=None,
                side=open_ep["side"],
                entry_target_units=open_ep["entry_target_units"],
            )
        )
    return tuple(episodes)


class EpisodeAttribution(BaseModel):
    """The complete, reconcilable attribution of one backtest to its episodes."""

    model_config = {"frozen": True, "extra": "forbid"}

    economics: dict[int, EpisodeEconomics]
    #: trades whose open instant belongs to no episode. Surfaced, never dropped:
    #: a non-empty tail means the schedule and the backtest disagree.
    unattributed_trades: tuple[ClosedTradeRecord, ...] = ()
    #: commission of fill contracts that resolved to no episode, and which fills
    #: they came from. Also surfaced, for the same reason.
    unattributed_fill_costs_usd: float = 0.0
    unattributed_fill_indices: tuple[int, ...] = ()
    #: closed trades the fill trail could not be matched to (an integrity signal)
    unmatched_trade_indices: tuple[int, ...] = ()
    costs_are_full_round_turn: bool = False

    @property
    def gross_pnl_usd(self) -> float:
        return float(sum(e.gross_pnl_usd for e in self.economics.values()))

    @property
    def costs_usd(self) -> float:
        return float(sum(e.costs_usd for e in self.economics.values()))

    @property
    def net_pnl_usd(self) -> float:
        return float(sum(e.net_pnl_usd for e in self.economics.values()))


def _owning_episode(
    episodes: tuple[PrimaryEpisode, ...], ts_ns: int
) -> PrimaryEpisode | None:
    """The episode that was open at ``ts_ns``, or ``None``.

    Deterministic and order-independent: the latest episode whose entry decision
    is at or before ``ts_ns`` and which had not already closed by then.
    """
    owner: PrimaryEpisode | None = None
    for e in episodes:
        if e.entry_decision_ts_ns > ts_ns:
            break
        if e.exit_decision_ts_ns is None or ts_ns < e.exit_decision_ts_ns:
            owner = e
    return owner


def attribute_episode_economics(
    episodes: tuple[PrimaryEpisode, ...],
    trades: tuple[ClosedTradeRecord, ...],
    fills: tuple[FillRecord, ...] | None = None,
) -> EpisodeAttribution:
    """Attribute every closed trade and every commission contract to an episode.

    Trades: a closed trade belongs to the episode that was open when the trade
    OPENED. That puts a roll's close leg inside the episode that spans the roll.

    Costs: per contract, as described in the module docstring. Without a fill
    trail the cost falls back to ``ClosedTrade.costs_usd`` -- the closing fill's
    commission only, which understates every episode by its entry commission.
    That fallback exists so the function stays usable in unit fixtures, and
    ``costs_are_full_round_turn`` records which of the two was used.
    """
    if any(a.entry_decision_ts_ns >= b.entry_decision_ts_ns for a, b in pairwise(episodes)):
        raise ValueError("episodes must be strictly time-ordered by entry_decision_ts_ns")

    buckets: dict[int, list[ClosedTradeRecord]] = {e.episode_index: [] for e in episodes}
    trade_owner: dict[int, int] = {}
    unattributed_trades: list[ClosedTradeRecord] = []
    for t in trades:
        owner = _owning_episode(episodes, t.ts_open_ns)
        if owner is None:
            unattributed_trades.append(t)
        else:
            buckets[owner.episode_index].append(t)
            trade_owner[t.trade_index] = owner.episode_index

    fill_costs: dict[int, float] = {e.episode_index: 0.0 for e in episodes}
    fill_counts: dict[int, int] = {e.episode_index: 0 for e in episodes}
    residual_costs = 0.0
    residual_fills: list[int] = []
    unmatched_trades: list[int] = []

    if fills is not None:
        # One ClosedTrade per REDUCING fill (position_ledger.cpp), keyed by the
        # instant and instrument it closed on. Matched greedily in index order so
        # the pairing is deterministic.
        pending: dict[tuple[int, int], list[ClosedTradeRecord]] = {}
        for t in trades:
            pending.setdefault((t.ts_close_ns, t.instrument_id), []).append(t)
        matched: set[int] = set()

        for f in fills:
            rate = f.commission_usd / f.quantity
            queue = pending.get((f.ts_fill_ns, f.instrument_id))
            closed_units = 0
            if queue:
                t = queue.pop(0)
                matched.add(t.trade_index)
                closed_units = min(t.quantity, f.quantity)
                owner_index = trade_owner.get(t.trade_index)
                if owner_index is None:
                    residual_costs += rate * closed_units
                    residual_fills.append(f.fill_index)
                else:
                    fill_costs[owner_index] += rate * closed_units
                    fill_counts[owner_index] += 1

            opening_units = f.quantity - closed_units
            if opening_units > 0:
                owner = _owning_episode(episodes, f.ts_fill_ns)
                if owner is None:
                    residual_costs += rate * opening_units
                    residual_fills.append(f.fill_index)
                else:
                    fill_costs[owner.episode_index] += rate * opening_units
                    if closed_units == 0:
                        fill_counts[owner.episode_index] += 1
        unmatched_trades = sorted(t.trade_index for t in trades if t.trade_index not in matched)

    econ: dict[int, EpisodeEconomics] = {}
    for e in episodes:
        ts = buckets[e.episode_index]
        gross = float(sum(t.gross_pnl_usd for t in ts))
        costs = (
            float(fill_costs[e.episode_index]) if fills is not None
            else float(sum(t.costs_usd for t in ts))
        )
        econ[e.episode_index] = EpisodeEconomics(
            episode_index=e.episode_index,
            n_attributed_trades=len(ts),
            gross_pnl_usd=gross,
            costs_usd=costs,
            net_pnl_usd=gross - costs,
            information_horizon_ts_ns=(max(t.ts_close_ns for t in ts) if ts else None),
            n_roll_closes=sum(1 for t in ts if t.close_reason == "roll"),
            n_attributed_fills=fill_counts[e.episode_index],
            contracts_traded=tuple(sorted({t.raw_symbol for t in ts})),
            costs_are_full_round_turn=fills is not None,
        )

    return EpisodeAttribution(
        economics=econ,
        unattributed_trades=tuple(unattributed_trades),
        unattributed_fill_costs_usd=residual_costs,
        unattributed_fill_indices=tuple(sorted(set(residual_fills))),
        unmatched_trade_indices=tuple(unmatched_trades),
        costs_are_full_round_turn=fills is not None,
    )


def attribute_trades_to_episodes(
    episodes: tuple[PrimaryEpisode, ...],
    trades: tuple[ClosedTradeRecord, ...],
    fills: tuple[FillRecord, ...] | None = None,
) -> tuple[dict[int, EpisodeEconomics], tuple[ClosedTradeRecord, ...]]:
    """``(economics_by_episode_index, unattributed_trades)``.

    The narrow view of :func:`attribute_episode_economics`, kept because most
    callers only need the labelling inputs. Use the full attribution when you
    need the residuals -- reconciliation does.
    """
    a = attribute_episode_economics(episodes, trades, fills)
    return a.economics, a.unattributed_trades


class EconomicReconciliation(BaseModel):
    """Proof that episode economics add back up to the engine's own net PnL.

    The identity that must hold for a fully closed run::

        sum(episode gross) - sum(episode-attributed commissions)
            == BacktestResult.net_pnl_usd

    When the final episode is deliberately left unterminated, the CLOSED
    labelled economics and the EXCLUDED/open economics are reported separately
    and their difference from the engine total must be exactly the open
    episode's own economics. Nothing is ever silently dropped: unattributed
    trades and unattributed commission appear as their own terms.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    engine_gross_pnl_usd: float
    engine_costs_usd: float
    engine_net_pnl_usd: float

    n_episodes: int
    n_labelled_episodes: int
    n_excluded_episodes: int

    labelled_gross_pnl_usd: float
    labelled_costs_usd: float
    labelled_net_pnl_usd: float

    excluded_gross_pnl_usd: float
    excluded_costs_usd: float
    excluded_net_pnl_usd: float

    unattributed_trade_gross_pnl_usd: float
    unattributed_fill_costs_usd: float

    gross_residual_usd: float
    costs_residual_usd: float
    net_residual_usd: float
    reconciles: bool
    tolerance_usd: float

    @property
    def open_episode_net_pnl_usd(self) -> float:
        """The part of the engine total the labelled set deliberately excludes."""
        return self.excluded_net_pnl_usd


def reconcile_episode_economics(
    attribution: EpisodeAttribution,
    *,
    engine_gross_pnl_usd: float,
    engine_costs_usd: float,
    engine_net_pnl_usd: float,
    labelled_episode_indices: Iterable[int] | None = None,
    tolerance_usd: float = 1e-6,
) -> EconomicReconciliation:
    """Reconcile attributed episode economics against the C++ headline result.

    ``labelled_episode_indices`` names the episodes that actually produced a
    labelled training event; everything else is EXCLUDED (unterminated at the
    corpus end, no attributed fill, a missing feature). Both halves are reported,
    and their sum plus the unattributed residuals must equal the engine's own
    numbers to within ``tolerance_usd``.
    """
    econ = attribution.economics
    labelled = set(econ) if labelled_episode_indices is None else set(labelled_episode_indices)
    unknown = labelled - set(econ)
    if unknown:
        raise ValueError(f"labelled_episode_indices names unknown episodes: {sorted(unknown)}")

    def _sum(indices: set[int], field: str) -> float:
        return float(sum(getattr(econ[i], field) for i in indices))

    excluded = set(econ) - labelled
    lab_gross, lab_costs = _sum(labelled, "gross_pnl_usd"), _sum(labelled, "costs_usd")
    exc_gross, exc_costs = _sum(excluded, "gross_pnl_usd"), _sum(excluded, "costs_usd")
    resid_gross = float(sum(t.gross_pnl_usd for t in attribution.unattributed_trades))
    resid_costs = float(attribution.unattributed_fill_costs_usd)

    gross_residual = engine_gross_pnl_usd - (lab_gross + exc_gross + resid_gross)
    costs_residual = engine_costs_usd - (lab_costs + exc_costs + resid_costs)
    net_residual = engine_net_pnl_usd - (
        (lab_gross + exc_gross + resid_gross) - (lab_costs + exc_costs + resid_costs)
    )
    return EconomicReconciliation(
        engine_gross_pnl_usd=engine_gross_pnl_usd,
        engine_costs_usd=engine_costs_usd,
        engine_net_pnl_usd=engine_net_pnl_usd,
        n_episodes=len(econ),
        n_labelled_episodes=len(labelled),
        n_excluded_episodes=len(excluded),
        labelled_gross_pnl_usd=lab_gross,
        labelled_costs_usd=lab_costs,
        labelled_net_pnl_usd=lab_gross - lab_costs,
        excluded_gross_pnl_usd=exc_gross,
        excluded_costs_usd=exc_costs,
        excluded_net_pnl_usd=exc_gross - exc_costs,
        unattributed_trade_gross_pnl_usd=resid_gross,
        unattributed_fill_costs_usd=resid_costs,
        gross_residual_usd=gross_residual,
        costs_residual_usd=costs_residual,
        net_residual_usd=net_residual,
        reconciles=(
            abs(gross_residual) <= tolerance_usd
            and abs(costs_residual) <= tolerance_usd
            and abs(net_residual) <= tolerance_usd
        ),
        tolerance_usd=tolerance_usd,
    )

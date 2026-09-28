"""Phase 21 -- deterministic bars/contracts/schedule window builder for one
paper-trading step.

Reuses the EXACT Phase 09/10/11/13.5C machinery the research path uses
(``compile_strategy``, ``compute_features``, ``build_target_schedule``) --
this module builds target-position INTENT only, never a fill, a price, or a
PnL number. The schedule-remap logic (synthetic contiguous daily index ->
real timestamps) is the same one
``alpha_agent.validation.phase_13_5c_matrix._Phase135cAdapter._baseline_schedule``
uses, factored out here so it does not depend on a
:class:`~alpha_agent.data.real_market_dataset.ReconstitutedRoot` -- a
:class:`MarketWindowProvider` is the only market-data dependency, so tests can
supply a deterministic synthetic fixture instead of decoding real DBN bytes.

Scope (Phase 21 MVP): the four Phase 13.5C daily-signal baseline families
(``tsmom``, ``ma_trend``, ``breakout``, ``mean_reversion`` --
:data:`alpha_agent.strategy.candidates_phase_13_5c.BASELINE_FAMILIES``), which
is also everything :mod:`alpha_agent.paper.eligibility` currently admits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from alpha_agent.backtest.targets import TargetSchedule, build_target_schedule
from alpha_agent.features import compute_features
from alpha_agent.features.source import SourceSeries
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.strategy import StrategySpec, compile_strategy
from alpha_agent.strategy.candidates_phase_13_5c import BASELINE_FAMILIES

DAY_NS = 86_400_000_000_000


@runtime_checkable
class MarketWindowProvider(Protocol):
    """Supplies the raw-contract execution bars, contract metadata, and the
    causal daily SIGNAL series for one root over ``[start_ns, end_ns]``
    (inclusive of any bar/day whose timestamp falls in the range).

    The SIGNAL series is deliberately a separate method from the execution
    bars: CLAUDE.md keeps the signal and execution planes distinct (back-
    adjusted prices may drive a decision, never a fill)."""

    def execution_bars(self, root_symbol: str, start_ns: int, end_ns: int) -> pd.DataFrame: ...

    def contracts_frame(self, root_symbol: str) -> pd.DataFrame: ...

    def daily_signal_bars(self, root_symbol: str, start_ns: int, end_ns: int) -> pd.DataFrame:
        """One row per trading day: ``ts_event_ns, open, high, low, close,
        volume``, sorted ascending. ``ts_event_ns`` is real (not yet
        remapped to the synthetic contiguous index this module builds)."""
        ...


@dataclass(frozen=True)
class ScheduleWindow:
    bars: pd.DataFrame
    contracts: pd.DataFrame
    #: None when the signal window has fewer than 3 rows -- too short for the
    #: reference evaluator to produce any decision (matches
    #: ``_Phase135cAdapter``'s own ``len(w) < 3`` guard).
    schedule: TargetSchedule | None


def build_schedule_window(
    *,
    strategy_family: str,
    root_symbol: str,
    strategy_spec: StrategySpec,
    provider: MarketWindowProvider,
    window_start_ns: int,
    watermark_ns: int,
    emit_from_ts_ns: int,
) -> ScheduleWindow:
    """Build the ``[window_start_ns, watermark_ns]`` bars/contracts/schedule
    bundle for one paper-trading step. ``emit_from_ts_ns`` drops any schedule
    row before that timestamp (used to keep the schedule's OWN watermark
    semantics -- e.g. never re-emit a decision from before the run started --
    separate from the replay window, which always starts at
    ``window_start_ns`` so the C++ engine sees the strategy's full warm-up)."""
    if strategy_family not in BASELINE_FAMILIES:
        raise ValueError(
            f"strategy_family {strategy_family!r} is not one of the Phase 21 MVP's "
            f"deterministically-reconstructable daily baseline families "
            f"{sorted(BASELINE_FAMILIES)}; a native-1m / silver_bullet schedule window "
            "needs its own builder (not implemented in this phase)"
        )
    bars = provider.execution_bars(root_symbol, window_start_ns, watermark_ns)
    contracts = provider.contracts_frame(root_symbol)
    daily = provider.daily_signal_bars(root_symbol, window_start_ns, watermark_ns)
    if len(daily) < 3 or bars.empty:
        return ScheduleWindow(bars=bars, contracts=contracts, schedule=None)

    daily = daily.sort_values("ts_event_ns", kind="stable").reset_index(drop=True)
    real_ts = daily["ts_event_ns"].to_numpy("int64")
    synth = (np.arange(len(daily), dtype="int64") + 1) * DAY_NS
    frame = daily.assign(ts_event_ns=synth)
    src = SourceSeries(
        frame=frame[["ts_event_ns", "open", "high", "low", "close", "volume"]],
        price_domain=PriceDomain.BACK_ADJUSTED,
        adjustment_mode="forward_adjusted",
        identity={"root_symbol": root_symbol},
        interval_ns=DAY_NS,
    )
    plan = compile_strategy(strategy_spec)
    feats = compute_features(
        src, [b.spec for b in plan.feature_bindings], require_point_in_time=False
    )
    sched = build_target_schedule(plan, feats)
    by_synth = {int(s): int(r) for s, r in zip(synth, real_ts)}
    rows = tuple(
        row.model_copy(update={"ts_event_ns": by_synth[int(row.ts_event_ns)]})
        for row in sched.rows
        if by_synth[int(row.ts_event_ns)] >= emit_from_ts_ns
    )
    return ScheduleWindow(
        bars=bars, contracts=contracts, schedule=sched.model_copy(update={"rows": rows})
    )


class RealMarketWindowProvider:
    """The real-data :class:`MarketWindowProvider`: already-acquired 2018-2024
    CME data only (:mod:`alpha_agent.data.real_market_dataset`), reconstituted
    OFFLINE from immutable raw bytes. Never reaches 2025
    (``assert_no_holdout_ts`` inside ``reconstitute_root`` /
    ``execution_bars`` refuses any window touching the locked holdout).

    Reconstitution is real work (decoding a root's whole 2018-2024 raw byte
    store); this provider caches one :class:`ReconstitutedRoot` per root for
    its lifetime so repeated paper-trading steps against the same run do not
    re-decode the archive every call.
    """

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def _recon(self, root_symbol: str):
        from alpha_agent.data.real_market_dataset import reconstitute_root

        cached = self._cache.get(root_symbol)
        if cached is None:
            cached = reconstitute_root(root_symbol)
            self._cache[root_symbol] = cached
        return cached

    def execution_bars(self, root_symbol: str, start_ns: int, end_ns: int) -> pd.DataFrame:
        from alpha_agent.data.real_market_dataset import execution_bars

        return execution_bars(self._recon(root_symbol), start_ns, end_ns)

    def contracts_frame(self, root_symbol: str) -> pd.DataFrame:
        return self._recon(root_symbol).contracts

    def daily_signal_bars(self, root_symbol: str, start_ns: int, end_ns: int) -> pd.DataFrame:
        from alpha_agent.data.calendars import default_calendar
        from alpha_agent.data.real_market_dataset import daily_signal_series

        recon = self._recon(root_symbol)
        daily = daily_signal_series(recon.forward_adjusted, root_symbol, calendar=default_calendar())
        return daily[(daily["ts_event_ns"] >= start_ns) & (daily["ts_event_ns"] <= end_ns)].reset_index(
            drop=True
        )

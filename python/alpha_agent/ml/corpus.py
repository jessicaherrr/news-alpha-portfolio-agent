"""Build the labelled :class:`MetaLabelEvent` corpus for one primary scope.

frozen primary ``TargetSchedule``
    -> C++ trades/fills export (:mod:`alpha_agent.ml.engine_io`)
    -> primary episodes (:func:`extract_primary_episodes`)
    -> per-episode C++ Fill-derived economics + exact reconciliation
    -> :class:`MetaLabelEvent`s (label = 1 iff episode net PnL > 0)

Episodes that cannot be labelled keep a typed exclusion reason
(``UNTERMINATED_AT_CORPUS_END``, ``NO_ATTRIBUTED_FILLS``,
``FEATURES_MISSING_AT_DECISION``); nothing is silently dropped, and the episode
economics must reconcile to the engine's own headline net PnL with any open /
excluded remainder reported explicitly.

Two providers supply the primary schedules:

* :class:`SyntheticPrimaryProvider` -- deterministic schedules over 2018-2024,
  for ``--synthetic-integration``. No market data.
* :class:`Phase135cPrimaryProvider` -- the real frozen Phase 13.5C schedules,
  compiled through the existing Phase 13.5C adapter. ``--run-real`` only.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel

from alpha_agent.backtest.targets import TargetSchedule, TargetScheduleRow
from alpha_agent.ml.engine_io import MetaEngineRunner
from alpha_agent.ml.enums import EpisodeExclusionReason
from alpha_agent.ml.episodes import (
    EconomicReconciliation,
    PrimaryEpisode,
    attribute_episode_economics,
    extract_primary_episodes,
    reconcile_episode_economics,
)
from alpha_agent.ml.guards import (
    DEVELOPMENT_CORPUS_END_NS,
    DEVELOPMENT_CORPUS_START_NS,
    assert_within_development_corpus,
)
from alpha_agent.ml.labels import (
    MetaLabelEventSet,
    MetaLabelSpec,
    build_meta_label_events,
)
from alpha_agent.ml.manifest import PRIMARY_FAMILIES, ROOTS
from alpha_agent.strategy.candidates_phase_13_5c import baseline_spec
from alpha_agent.strategy.fingerprint import strategy_fingerprint

_NS_PER_DAY = 86_400_000_000_000
#: 2018-01-02 .. deep inside the corpus; the synthetic schedules start here.
_SYNTH_START_NS = DEVELOPMENT_CORPUS_START_NS + _NS_PER_DAY


class PrimaryScope(BaseModel):
    """One (primary family, economics root) scope of the Phase 15 matrix."""

    model_config = {"frozen": True, "extra": "forbid"}

    primary_family: str
    root_symbol: str
    strategy_fingerprint: str
    strategy_id: str


class CorpusResult(BaseModel):
    """The labelled event set for one primary scope, plus its reconciliation."""

    model_config = {"frozen": True, "extra": "forbid"}

    scope: PrimaryScope
    primary_schedule_hash: str
    events: MetaLabelEventSet
    reconciliation: EconomicReconciliation
    n_episodes: int
    n_labelled: int
    n_excluded: int
    exclusion_reason_counts: dict[str, int]
    engine: str

    @property
    def n_events(self) -> int:
        return self.events.n_events


class PrimaryScheduleProvider(Protocol):
    """Supplies the frozen primary target schedule for one scope."""

    provider_tag: str

    def scopes(self) -> tuple[PrimaryScope, ...]: ...

    def schedule_for(self, scope: PrimaryScope) -> TargetSchedule: ...

    def feature_availability(
        self, scope: PrimaryScope, episodes: tuple[PrimaryEpisode, ...]
    ) -> dict[int, bool]: ...


# --------------------------------------------------------------------------
# synthetic provider
# --------------------------------------------------------------------------
def _synthetic_scope(primary_family: str, root: str) -> PrimaryScope:
    spec = baseline_spec(primary_family, root)
    return PrimaryScope(
        primary_family=primary_family,
        root_symbol=root,
        strategy_fingerprint=strategy_fingerprint(spec),
        strategy_id=spec.strategy_id,
    )


class SyntheticPrimaryProvider:
    """Deterministic primary schedules over the development corpus. TESTS ONLY.

    One target row per synthetic trading day. The target path is a seeded
    mean-reverting integer process so that it opens a realistic number of
    directional episodes (order 30-120 over the corpus), with a deliberate
    unterminated tail and a deliberate missing-feature episode so every typed
    exclusion path is exercised.
    """

    provider_tag = "synthetic_primary_provider__software_fixture_only"

    def __init__(
        self,
        *,
        n_days: int = 410,
        spacing_days: int = 6,
        families: tuple[str, ...] = PRIMARY_FAMILIES,
        roots: tuple[str, ...] = ROOTS,
    ):
        self._n_days = int(n_days)
        self._spacing = int(spacing_days)
        self._families = tuple(families)
        self._roots = tuple(roots)
        self._scopes = tuple(
            _synthetic_scope(f, r) for f in self._families for r in self._roots
        )

    def scopes(self) -> tuple[PrimaryScope, ...]:
        return self._scopes

    def schedule_for(self, scope: PrimaryScope) -> TargetSchedule:
        rng = np.random.default_rng(
            int.from_bytes(
                __import__("hashlib").sha256(
                    f"{scope.primary_family}|{scope.root_symbol}".encode()
                ).digest()[:8],
                "big",
            )
        )
        ts = [
            _SYNTH_START_NS + i * self._spacing * _NS_PER_DAY for i in range(self._n_days)
        ]
        assert_within_development_corpus(ts, what="synthetic primary schedule")
        target = 0
        units: list[int] = []
        for _ in range(self._n_days):
            step = rng.integers(-1, 2)          # {-1, 0, +1}
            if rng.random() < 0.18:             # occasional flatten
                target = 0
            else:
                target = int(np.clip(target + step, -1, 1))
            units.append(target)
        # ensure an unterminated tail: force the final run to a fixed direction
        for i in range(self._n_days - 8, self._n_days):
            units[i] = 1 if units[self._n_days - 9] <= 0 else -1
        rows = tuple(
            TargetScheduleRow(
                ts_event_ns=t,
                root_symbol=scope.root_symbol,
                target_units=u,
                strategy_fingerprint=scope.strategy_fingerprint,
                matched_rule_id=None,
            )
            for t, u in zip(ts, units)
        )
        return TargetSchedule(
            root_symbol=scope.root_symbol,
            strategy_fingerprint=scope.strategy_fingerprint,
            strategy_id=scope.strategy_id,
            strategy_dsl_version="strategy-dsl/1",
            feature_engine_version="synthetic",
            warmup_bars=0,
            rows=rows,
        )

    def feature_availability(
        self, scope: PrimaryScope, episodes: tuple[PrimaryEpisode, ...]
    ) -> dict[int, bool]:
        """Every episode has features except a deterministic one, so the typed
        ``FEATURES_MISSING_AT_DECISION`` exclusion is always exercised."""
        avail = {ep.episode_index: True for ep in episodes}
        if episodes:
            rng = np.random.default_rng(
                int.from_bytes(
                    __import__("hashlib").sha256(
                        f"{scope.root_symbol}|feat".encode()
                    ).digest()[:8],
                    "big",
                )
            )
            drop = int(rng.integers(0, max(1, len(episodes))))
            avail[episodes[drop].episode_index] = False
        return avail


# --------------------------------------------------------------------------
# the corpus builder
# --------------------------------------------------------------------------
def build_corpus_for_scope(
    scope: PrimaryScope,
    *,
    provider: PrimaryScheduleProvider,
    engine: MetaEngineRunner,
    meta_label_spec: MetaLabelSpec,
    dataset_fingerprint: str,
    feature_set_fingerprint: str,
    corpus_start_ts_ns: int = DEVELOPMENT_CORPUS_START_NS,
    corpus_end_ts_ns: int = DEVELOPMENT_CORPUS_END_NS,
    reconcile_tolerance_usd: float = 1e-3,
) -> CorpusResult:
    """One primary scope -> its labelled :class:`MetaLabelEvent` corpus.

    Runs the economics engine ONCE, at baseline cost, with the audit trails on.
    Every episode's economics are C++ Fill-derived; Python only sums and labels.
    """
    schedule = provider.schedule_for(scope)
    if schedule.strategy_fingerprint != scope.strategy_fingerprint:
        raise ValueError("provider returned a schedule for a different strategy")

    run = engine.run(
        schedule,
        cost_scenario_label="baseline_1_0x",
        cost_multiplier=1.0,
        want_audit_trails=True,
    )
    if not run.audit_trails_present:
        raise ValueError("corpus building needs the trades/fills audit trails")

    episodes = extract_primary_episodes(schedule)
    # keep only episodes that decide inside the development corpus
    episodes = tuple(
        ep for ep in episodes
        if corpus_start_ts_ns <= ep.entry_decision_ts_ns < corpus_end_ts_ns
    )
    attribution = attribute_episode_economics(episodes, run.closed_trades, run.fills)

    feature_avail = provider.feature_availability(scope, episodes)

    events = build_meta_label_events(
        episodes=episodes,
        economics=attribution.economics,
        spec=meta_label_spec,
        strategy_family=scope.primary_family,
        dataset_fingerprint=dataset_fingerprint,
        feature_set_fingerprint=feature_set_fingerprint,
        target_schedule_hash=schedule.schedule_hash(),
        feature_availability=feature_avail,
    )

    labelled_indices = {e.episode_index for e in events.events}
    reconciliation = reconcile_episode_economics(
        attribution,
        engine_gross_pnl_usd=run.gross_pnl_usd,
        engine_costs_usd=run.costs_usd,
        engine_net_pnl_usd=run.net_pnl_usd,
        labelled_episode_indices=labelled_indices,
        tolerance_usd=reconcile_tolerance_usd,
    )

    reason_counts: dict[str, int] = {r.value: 0 for r in EpisodeExclusionReason}
    for ex in events.excluded:
        reason_counts[ex.reason.value] += 1

    return CorpusResult(
        scope=scope,
        primary_schedule_hash=schedule.schedule_hash(),
        events=events,
        reconciliation=reconciliation,
        n_episodes=len(episodes),
        n_labelled=len(events.events),
        n_excluded=len(events.excluded),
        exclusion_reason_counts={k: v for k, v in reason_counts.items() if v},
        engine=run.engine,
    )

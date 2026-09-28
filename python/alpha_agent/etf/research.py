"""Phase 6 -- runs the real ETF tsmom hypothesis end-to-end through the REAL,
UNMODIFIED validation stack: the same :class:`~alpha_agent.validation.engine.
ValidationEngine`, the same frozen :class:`~alpha_agent.validation.policy.
ReliabilityPolicy` gate thresholds
(:func:`alpha_agent.validation.phase_13_5c_matrix.frozen_policy` -- Validation
PHILOSOPHY is explicitly shared across domains per Phase 6's own opening
principle), and the same real
:class:`~alpha_agent.validation.runner.CliBacktestRunner` subprocess boundary
every Futures family in this repository uses.

REAL data throughout: the acquired primary-listing-venue bars
(``alpha_agent.etf.data_source``, labelled honestly per instruction 1, never
"consolidated"), a synthetic-but-correctly-typed :class:`ContractSpecModel`
(the Phase 6 instrument-economics adapter, ``etf_contract_spec``). This
function does not touch corporate actions -- see the module docstring on
``alpha_agent.etf.strategy_family`` and the Phase 6 session report for why
(the CLI/engine corporate-action wiring and a genuinely complete dividend
history are each a separate, disclosed piece of work) and pick a ticker/
window pair for which that omission is an honest, disclosed limitation
rather than a silent one.

Own split/validation windows and own trial_family_id (Phase 6 kickoff
instruction 3: "never mix with old Futures families") -- entirely inside the
real acquired 2018-05-01..2024-12-30 pre-holdout window.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from alpha_agent.core.instrument import AssetDomain
from alpha_agent.etf.calendar import etf_calendar
from alpha_agent.etf.data_source import etf_contract_spec, load_primary_listing_bars
from alpha_agent.etf.schemas import EtfDataSourceProvenance
from alpha_agent.etf.strategy_family import (
    CANONICAL_SLOW_HORIZON,
    ETF_TSMOM_FAMILY,
    etf_tsmom_adapter,
    etf_tsmom_spec,
)
from alpha_agent.strategy import strategy_fingerprint
from alpha_agent.validation.dataset import DatasetIdentity, frame_content_hash
from alpha_agent.validation.engine import ValidationEngine
from alpha_agent.validation.enums import SplitRole
from alpha_agent.validation.phase_13_5c_matrix import (
    frozen_bootstrap_config,
    frozen_cost_plan,
    frozen_null_config,
    frozen_policy,
)
from alpha_agent.validation.report import ValidationReport
from alpha_agent.validation.runner import CliBacktestRunner
from alpha_agent.validation.spec import ValidationSpec
from alpha_agent.validation.splits import SplitPlan, SplitWindow
from alpha_agent.validation.walkforward import WalkForwardConfig

#: NOT "build/cpp/cpp/..." -- that nested path is a stale, orphaned build
#: artifact (confirmed 2026-09-22: a plain default `cmake --build build`
#: regenerates ONLY "build/cpp/quant_backtest_targets_csv"; the doubly-nested
#: copy is untouched and was last built 2026-09-09, predating this session's
#: corporate-action engine work entirely). Many existing modules across this
#: repo still default to the stale nested path -- see the Phase 6 session
#: report for the full list; not fixed here, out of this phase's scope.
DEFAULT_CLI = Path("build/cpp/quant_backtest_targets_csv")

_DAY_NS = 86_400_000_000_000


def _ts(iso_date: str) -> int:
    return int(pd.Timestamp(iso_date, tz="UTC").value)


def etf_split_plan(root_symbol: str) -> SplitPlan:
    """Entirely inside the real acquired pre-holdout window
    (2018-05-01..2024-12-30). LOCKED_HOLDOUT's own window (2024-07-01
    onward) deliberately coincides with EQUS.SUMMARY's real coverage start,
    so a future consolidated-tape cross-check of the holdout period is
    possible without any new acquisition -- never used to influence this
    run's own verdict."""
    del root_symbol  # identical window for every ticker in this pilot; kept for a future per-ticker override point
    return SplitPlan(
        windows=(
            SplitWindow(role=SplitRole.TRAIN, start_ts_ns=_ts("2018-05-01"), end_ts_ns=_ts("2023-01-01")),
            SplitWindow(role=SplitRole.VALIDATION, start_ts_ns=_ts("2023-01-01"), end_ts_ns=_ts("2024-06-01")),
            SplitWindow(role=SplitRole.LOCKED_HOLDOUT, start_ts_ns=_ts("2024-07-01"), end_ts_ns=_ts("2024-12-31")),
        ),
        embargo_days=5,
    )


def etf_walk_forward() -> WalkForwardConfig:
    """Scaled for the real ~4.6-year TRAIN window this pilot actually has
    (2018-05-01..2023-01-01) -- NOT the production
    ``phase_13_5c_matrix.frozen_walk_forward()``, which is calibrated for
    minute-bar warmup over the Futures dataset; own window per Phase 6
    kickoff instruction 3, same documented-override precedent
    ``alpha_agent.crypto.research.crypto_walk_forward`` already established
    for a different new asset domain."""
    return WalkForwardConfig(
        n_folds=5, scheme="expanding", min_train_days=250, test_days=90,
        embargo_days=5, warmup_bars=CANONICAL_SLOW_HORIZON + 5,
    )


class EtfResearchArtifact(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    asset_domain: AssetDomain = AssetDomain.ETF
    strategy_family: str
    root_symbol: str
    params: dict
    data_provenance: EtfDataSourceProvenance
    dataset_identity: DatasetIdentity
    strategy_fingerprint: str
    validation_fingerprint: str
    corporate_action_disclosure: str
    report: ValidationReport
    #: exposed so a caller can independently reconstruct the same pre-run
    #: experiment_identity (registry.identity.experiment_identity) without
    #: needing to re-derive any sub-fingerprint by hand.
    validation_spec: ValidationSpec


def run_etf_research(
    root_symbol: str,
    *,
    executable: str | Path = DEFAULT_CLI,
    work_dir: str | Path = "outputs/phase_6/_scratch",
) -> EtfResearchArtifact:
    """Runs the canonical Phase 6 ETF tsmom hypothesis on `root_symbol`
    through the real, unmodified validation stack over real acquired data."""
    frame, provenance = load_primary_listing_bars(root_symbol)
    bars = frame.rename(columns={"ts_event": "ts_event_ns"})[
        ["ts_event_ns", "instrument_id", "open", "high", "low", "close", "volume"]
    ].reset_index(drop=True)
    instrument_id = int(bars["instrument_id"].iloc[0])
    activation_ns = int(bars["ts_event_ns"].iloc[0])
    contract = etf_contract_spec(instrument_id, root_symbol, activation_ns=activation_ns)
    contracts = pd.DataFrame(
        [
            {
                "instrument_id": contract.instrument_id, "raw_symbol": contract.raw_symbol,
                "root_symbol": contract.root_symbol, "exchange": contract.exchange,
                "tick_size": contract.tick_size, "multiplier": contract.multiplier,
                "activation_ns": contract.activation_ns, "expiration_ns": contract.expiration_ns,
                "first_notice_ns": "", "last_trade_ns": "",
            }
        ]
    )

    adapter = etf_tsmom_adapter(root_symbol)
    canonical_spec = etf_tsmom_spec(root_symbol)
    canon_fp = strategy_fingerprint(canonical_spec)

    policy = frozen_policy()
    dataset_identity = DatasetIdentity(
        root_symbol=root_symbol, price_domain="raw",
        source_fingerprint=f"{provenance.dataset}:{provenance.role.value}",
        bars_content_hash=frame_content_hash(bars), n_bars=len(bars),
        first_ts_ns=int(bars["ts_event_ns"].min()), last_ts_ns=int(bars["ts_event_ns"].max()),
    )
    spec = ValidationSpec(
        label=f"phase6_etf_tsmom_{root_symbol}",
        strategy_fingerprint=canon_fp,
        strategy_key=ETF_TSMOM_FAMILY,
        dataset=dataset_identity,
        split_plan=etf_split_plan(root_symbol),
        walk_forward=etf_walk_forward(),
        capital_base_usd=100_000.0,
        cost_stress=frozen_cost_plan(),
        null_test=frozen_null_config(),
        bootstrap=frozen_bootstrap_config(),
        trial_family_id=f"phase6.etf.{root_symbol}.{ETF_TSMOM_FAMILY}",
        reliability_policy_fingerprint=policy.identity(),
        random_seed=6,
    )
    runner = CliBacktestRunner(executable, work_dir)
    engine = ValidationEngine(spec, policy, runner, adapter, bars, contracts, calendar=etf_calendar())
    report = engine.run_research()

    return EtfResearchArtifact(
        strategy_family=ETF_TSMOM_FAMILY,
        root_symbol=root_symbol,
        params=dict(adapter.canonical_params),
        data_provenance=provenance,
        dataset_identity=dataset_identity,
        strategy_fingerprint=canon_fp,
        validation_fingerprint=spec.validation_fingerprint(),
        corporate_action_disclosure=(
            "This run's C++ execution does NOT apply any split or cash-distribution "
            "action -- the engine-level corporate-action CSV wiring and a complete "
            "dividend history for this ticker are each tracked as separate, disclosed "
            "work (see alpha_agent.etf.corporate_actions.coverage() for this ticker's "
            "real sourcing status). Fills/PnL in this report are RAW-price economics "
            "only; total return during the window is understated to the extent this "
            "ticker paid distributions, and any undisclosed split would corrupt the "
            "series -- alpha_agent.etf.corporate_actions confirms the known-split "
            "status for this specific ticker."
        ),
        report=report,
        validation_spec=spec,
    )

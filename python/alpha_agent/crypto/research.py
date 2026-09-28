"""Wires the Phase 22 crypto scaffold into the REAL, unmodified validation
stack: the same :class:`~alpha_agent.validation.engine.ValidationEngine`, the
same frozen :class:`~alpha_agent.validation.policy.ReliabilityPolicy` gate
thresholds (:func:`alpha_agent.validation.phase_13_5c_matrix.frozen_policy`),
the same null/bootstrap/cost-stress configuration, and -- when the compiled
C++ core is present -- the same real
:class:`~alpha_agent.validation.runner.CliBacktestRunner` subprocess boundary
every other family in this repository uses.

REAL: the C++ Quant Core, the Phase 10 DSL/compiler, and this validation
implementation. SYNTHETIC: the market data, the alternative data, the contract
economics fixture, and the calendar -- so the resulting research evidence is
SYNTHETIC too. Precisely: this function drives a SYNTHETIC CME-futures-shaped
execution fixture (:mod:`alpha_agent.crypto.synthetic_fixtures`) through the
real, unmodified C++ Quant Core -- never a network call, never real CME
BTC/ETH history. :func:`run_crypto_research` refuses to run unless every input
series' provenance is verifiably ``SYNTHETIC`` (CLAUDE.md "no LLM/no shortcut
overrides a typed guard"; here neither does a caller).

Returns a :class:`~alpha_agent.crypto.envelope.SyntheticCryptoValidationArtifact`
-- never a bare :class:`~alpha_agent.validation.report.ValidationReport` --
so complete per-series provenance and a stable, wall-clock-independent content
identity travel with the result everywhere it goes (dataset -> feature/
research bundle -> validation artifact -> isolated registry -> UI).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_agent.crypto.calendar import crypto_calendar
from alpha_agent.crypto.envelope import SyntheticCryptoValidationArtifact
from alpha_agent.crypto.strategy_family import (
    CRYPTO_FUNDING_CONTRARIAN_FAMILY,
    FundingContrarianParams,
    funding_contrarian_adapter,
    make_funding_contrarian_spec,
)
from alpha_agent.crypto.synthetic_fixtures import (
    DAY_NS,
    SYNTHETIC_BASE_NS,
    SyntheticCryptoBundle,
    synthetic_contract_row,
    synthetic_crypto_bars,
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
from alpha_agent.validation.runner import CliBacktestRunner
from alpha_agent.validation.spec import ValidationSpec
from alpha_agent.validation.splits import SplitPlan, SplitWindow
from alpha_agent.validation.walkforward import WalkForwardConfig

#: Reference CLI path, same convention as tests/python/test_reliability_validation.py.
DEFAULT_CLI = Path("build/cpp/cpp/quant_backtest_targets_csv")

#: Synthetic-fixture-scale split (calendar days, DAY_NS cadence) -- proportioned
#: like the Phase 13 daily-bar test fixture (`_tsmom_spec_and_engine`), just
#: larger, to clear `MinimumSampleRequirements` defaults (>=60 OOS days, >=3
#: walk-forward folds) with margin. Entirely inside the synthetic 2016-era
#: fixture calendar (`SYNTHETIC_BASE_NS`) -- nowhere near the real 2025 locked
#: holdout.
TRAIN_DAYS = 480
VALIDATION_DAYS = 240
EMBARGO_DAYS = 30
HOLDOUT_DAYS = 150
N_DAYS = TRAIN_DAYS + VALIDATION_DAYS + EMBARGO_DAYS + HOLDOUT_DAYS


def crypto_split_plan(*, start_ns: int = SYNTHETIC_BASE_NS) -> SplitPlan:
    return SplitPlan(
        windows=(
            SplitWindow(
                role=SplitRole.TRAIN, start_ts_ns=start_ns,
                end_ts_ns=start_ns + TRAIN_DAYS * DAY_NS,
            ),
            SplitWindow(
                role=SplitRole.VALIDATION, start_ts_ns=start_ns + TRAIN_DAYS * DAY_NS,
                end_ts_ns=start_ns + (TRAIN_DAYS + VALIDATION_DAYS) * DAY_NS,
            ),
            SplitWindow(
                role=SplitRole.LOCKED_HOLDOUT,
                start_ts_ns=start_ns + (TRAIN_DAYS + VALIDATION_DAYS + EMBARGO_DAYS) * DAY_NS,
                end_ts_ns=start_ns + N_DAYS * DAY_NS,
            ),
        ),
        embargo_days=5,
    )


def crypto_walk_forward() -> WalkForwardConfig:
    """Scaled for a daily synthetic fixture -- NOT the frozen production
    ``phase_13_5c_matrix.frozen_walk_forward()``, which is calibrated for
    minute-bar warmup over a multi-year real dataset. Existing precedent:
    ``tests/python/test_reliability_validation.py``'s own
    ``_tsmom_spec_and_engine`` test helper likewise defines its own
    fixture-scaled ``WalkForwardConfig`` rather than reusing the production one."""
    return WalkForwardConfig(n_folds=4, min_train_days=120, test_days=75, embargo_days=5, warmup_bars=60)


def build_crypto_dataset(
    *, root_symbol: str = "BTC", instrument_id: int = 990001, seed: int = 22,
) -> tuple[SyntheticCryptoBundle, pd.DataFrame]:
    bundle = synthetic_crypto_bars(
        root_symbol=root_symbol, instrument_id=instrument_id, n_days=N_DAYS, seed=seed,
    )
    for series_name, prov in bundle.provenance_by_series.items():
        try:
            prov.assert_synthetic()
        except Exception as exc:
            raise type(exc)(f"series {series_name!r}: {exc}") from exc
    expiration_ns = SYNTHETIC_BASE_NS + (N_DAYS + 4000) * DAY_NS
    contracts = synthetic_contract_row(
        instrument_id=instrument_id, raw_symbol=f"{root_symbol}SYNTH26",
        root_symbol=root_symbol, expiration_ns=expiration_ns,
    )
    return bundle, contracts


def run_crypto_research(
    *,
    root_symbol: str = "BTC",
    window: int = 14,
    threshold: float = 1.5,
    size: int = 1,
    seed: int = 22,
    executable: str | Path = DEFAULT_CLI,
    work_dir: str | Path = "outputs/phase_22/_scratch",
) -> SyntheticCryptoValidationArtifact:
    """Run the funding-crowding-contrarian crypto hypothesis end-to-end through
    the REAL C++ engine + REAL validation stack, over a SYNTHETIC
    CME-futures-shaped execution fixture only.

    Raises :class:`~alpha_agent.crypto.provenance.RealVendorNotConnectedError`
    if any input series' provenance is not verifiably ``SYNTHETIC`` -- the one
    hard gate that keeps this function from ever being pointed at real data
    without a deliberate, separate, future change.
    """
    bundle, contracts = build_crypto_dataset(root_symbol=root_symbol, seed=seed)
    bars = bundle.bars

    adapter = funding_contrarian_adapter(
        root_symbol=root_symbol, window=window, threshold=threshold, size=size,
    )
    canonical_spec = make_funding_contrarian_spec(FundingContrarianParams(**adapter.canonical_params))
    canon_fp = strategy_fingerprint(canonical_spec)

    policy = frozen_policy()
    dataset_identity = DatasetIdentity(
        root_symbol=root_symbol, price_domain="raw_contract",
        # Phase 22.1: a stable, content-based fingerprint over the generator's
        # own schema/seed/shape/delay parameters -- NOT a bare descriptive
        # seed string, and NOT wall-clock dependent -- see
        # `synthetic_crypto_generator_identity`.
        source_fingerprint=bundle.generator_identity,
        bars_content_hash=frame_content_hash(bars), n_bars=len(bars),
    )
    spec = ValidationSpec(
        label="phase22_crypto_funding_contrarian_synthetic",
        strategy_fingerprint=canon_fp,
        strategy_key=CRYPTO_FUNDING_CONTRARIAN_FAMILY,
        dataset=dataset_identity,
        split_plan=crypto_split_plan(),
        walk_forward=crypto_walk_forward(),
        capital_base_usd=100_000.0,
        cost_stress=frozen_cost_plan(),
        null_test=frozen_null_config(),
        bootstrap=frozen_bootstrap_config(),
        trial_family_id=f"phase22.crypto_synthetic.{root_symbol}.funding_contrarian",
        reliability_policy_fingerprint=policy.identity(),
        random_seed=seed,
    )
    runner = CliBacktestRunner(executable, work_dir)
    engine = ValidationEngine(
        spec, policy, runner, adapter, bars, contracts, calendar=crypto_calendar(),
    )
    report = engine.run_research()

    return SyntheticCryptoValidationArtifact(
        strategy_family=CRYPTO_FUNDING_CONTRARIAN_FAMILY,
        params=dict(adapter.canonical_params),
        fixture_seed=seed,
        provenance_by_series=dict(bundle.provenance_by_series),
        dataset_identity=dataset_identity,
        generator_identity=bundle.generator_identity,
        strategy_fingerprint=canon_fp,
        validation_fingerprint=spec.validation_fingerprint(),
        report=report,
    )

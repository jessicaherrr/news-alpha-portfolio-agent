"""A Phase 22 crypto strategy family, built from the SAME frozen Phase 10 DSL
types used by every futures baseline (:mod:`alpha_agent.strategy.spec`) and
driven through the SAME :class:`~alpha_agent.validation.engine.ValidationEngine`
/ :class:`~alpha_agent.validation.engine.DslFamilyAdapter` used for every other
family (:mod:`alpha_agent.validation.engine`).

The only new thing is *where the feature reads its column from*:
``FeatureSpec(kind="zscore", price_field="funding_rate", ...)`` asks the
existing, unmodified Phase 09 feature registry for the existing, unmodified
``zscore`` kind, over a column that happens to hold a synthetic perpetual
funding rate rather than a price. No new feature kind, no new registry, no
change to the real futures family's feature identity.

``strategy_key`` is deliberately **not** one of
:data:`alpha_agent.strategy.candidates_phase_13_5c.BASELINE_FAMILIES` --
:mod:`alpha_agent.paper.eligibility` rebuilds a strategy from a committed
``(strategy_family, params)`` pair using exactly that frozen mapping, so a
family key outside it is structurally unrebuildable there and therefore never
paper-trading eligible, independent of any registry-side guard.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from alpha_agent.crypto.synthetic_fixtures import CRYPTO_FEATURE_COLUMNS
from alpha_agent.features.compute import compute_features
from alpha_agent.features.source import SourceSeries
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.strategy.enums import Comparator, DefaultAction
from alpha_agent.strategy.spec import (
    ComparisonNode,
    ConstOperand,
    FeatureDeclaration,
    FeatureOperand,
    Rule,
    StrategySpec,
    TargetAction,
)
from alpha_agent.validation.engine import DslFamilyAdapter

#: Never one of the real BASELINE_FAMILIES keys (tsmom / ma_trend / breakout /
#: mean_reversion) -- see module docstring.
CRYPTO_FUNDING_CONTRARIAN_FAMILY = "crypto_funding_contrarian"

#: EVERY causal crypto feature column the Phase 22 alternative-data surface
#: can produce (`alpha_agent.crypto.synthetic_fixtures.CRYPTO_FEATURE_COLUMNS`
#: -- funding, open interest, on-chain MVRV/SOPR/active addresses, CME basis,
#: liquidation aggregates, exchange flows), exposed to the DSL feature engine
#: via the existing `zscore` (or any other registered) kind's `price_field`.
#: The demonstration strategy below only reads `funding_rate` -- prompt 22
#: does not require one strategy per feature, only that the full typed
#: surface CAN feed the same FeatureSpec/StrategySpec pipeline (proven by
#: test_E2b in tests/python/test_phase_22_crypto_extension.py, which declares
#: a FeatureSpec against every column here).
CRYPTO_FEATURE_PRICE_FIELDS = CRYPTO_FEATURE_COLUMNS


class FundingContrarianParams(BaseModel):
    """Typed, validated parameters (mirrors the discipline of
    :mod:`alpha_agent.strategy.baselines.params`, kept package-local rather than
    importing that module's private `_BaseParams`)."""

    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str = Field(pattern=r"^[A-Z0-9]{1,12}$")
    price_field: str = "funding_rate"
    window: int = Field(default=14, ge=2, le=100_000)
    threshold: float = Field(default=1.5, gt=0.0, le=10.0)
    size: int = Field(default=1, ge=1, le=5)


def make_funding_contrarian_spec(params: FundingContrarianParams) -> StrategySpec:
    """Economic mechanism: see
    :data:`alpha_agent.crypto.hypotheses.FUNDING_CROWDING_CONTRARIAN`.

    Formula / timing (bar ``T``, backward-looking only):
      ``funding_z = zscore(funding_rate, window)``
      ``funding_z > threshold``  -> target ``-size`` (fade crowded longs)
      ``funding_z < -threshold`` -> target ``+size`` (fade crowded shorts)
      otherwise                  -> flat
    """
    funding_z = FeatureSpec(
        kind="zscore", params={"window": params.window}, price_field=params.price_field,
        required_price_domain=(PriceDomain.RAW_CONTRACT,),
    )
    return StrategySpec(
        strategy_name=f"crypto funding-contrarian {params.root_symbol} w{params.window} t{params.threshold}",
        strategy_id=(
            f"CRYPTO-FUNDING-CONTRARIAN-{params.root_symbol}-{params.window}-"
            f"{params.threshold}-{params.size}"
        ),
        root_symbol=params.root_symbol,
        rationale=(
            "Phase 22 synthetic scaffold -- perpetual funding-rate crowding "
            "contrarian, executed on a synthetic CME-futures-shaped execution "
            "fixture processed through the real, unmodified C++ Quant Core. "
            "SYNTHETIC evidence only; not a claim about real funding-rate data, "
            "real CME BTC/ETH history, or real profitability."
        ),
        features=[FeatureDeclaration(alias="funding_z", spec=funding_z)],
        rules=[
            Rule(
                rule_id="fade_crowded_longs",
                when=ComparisonNode(
                    op=Comparator.GT,
                    left=FeatureOperand(feature="funding_z"),
                    right=ConstOperand(value=params.threshold),
                ),
                action=TargetAction(target_units=-params.size),
                rationale="funding z-score extremely positive: crowded longs paying "
                "shorts -- fade",
            ),
            Rule(
                rule_id="fade_crowded_shorts",
                when=ComparisonNode(
                    op=Comparator.LT,
                    left=FeatureOperand(feature="funding_z"),
                    right=ConstOperand(value=-params.threshold),
                ),
                action=TargetAction(target_units=params.size),
                rationale="funding z-score extremely negative: crowded shorts paying "
                "longs -- fade",
            ),
        ],
        default_action=DefaultAction.FLAT,
    )


def _compute_frame(plan, bars: pd.DataFrame):
    """``StrategyFamilyAdapter.schedule_for``'s ``compute_frame``: builds the
    :class:`SourceSeries` the compiled plan's declared features read from.
    Only the crypto feature columns are exposed here -- OHLCV stays reserved
    for the C++ engine boundary and is never fed to a feature (Phase 09
    `assert_no_execution_identity_leak` guards the reverse leak; this is the
    domain-separation half of the same discipline)."""
    cols = ["ts_event_ns", *CRYPTO_FEATURE_PRICE_FIELDS]
    present = [c for c in cols if c in bars.columns]
    df = bars.loc[:, present].reset_index(drop=True)
    src = SourceSeries(
        frame=df,
        price_domain=PriceDomain.RAW_CONTRACT,
        identity={"root_symbol": plan.root_symbol},
        interval_ns=86_400_000_000_000,
        causal=True,
    )
    return compute_features(src, [b.spec for b in plan.feature_bindings])


def funding_contrarian_adapter(
    *, root_symbol: str = "BTC", window: int = 14, threshold: float = 1.5, size: int = 1,
) -> DslFamilyAdapter:
    """A :class:`~alpha_agent.validation.engine.DslFamilyAdapter` for the
    funding-crowding-contrarian family -- the exact same adapter shape every
    other family in this repository uses with
    :class:`~alpha_agent.validation.engine.ValidationEngine`."""
    canonical = {
        "root_symbol": root_symbol, "window": window, "threshold": threshold, "size": size,
    }

    def spec_factory(params: dict) -> StrategySpec:
        return make_funding_contrarian_spec(FundingContrarianParams(**params))

    return DslFamilyAdapter(
        CRYPTO_FUNDING_CONTRARIAN_FAMILY, canonical, spec_factory, _compute_frame,
    )

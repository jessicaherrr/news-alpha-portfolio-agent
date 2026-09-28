"""The Phase 6 ETF strategy family: reuses the existing, asset-neutral
Phase 11 tsmom baseline factory UNCHANGED (``make_tsmom_spec``/``TsmomParams``
take a bare ``root_symbol`` string with no futures-specific field anywhere),
via the existing, asset-neutral :class:`~alpha_agent.validation.engine.
DslFamilyAdapter` UNCHANGED.

The only genuinely new code here is ``compute_frame``: how a window of ETF
bars turns into a point-in-time :class:`~alpha_agent.features.frame.
FeatureFrame`. It sets ``price_domain=PriceDomain.RAW`` (Phase 6's ETF
execution-safe price domain, instruction 6) and
``session_policy=SessionPolicy.CONTINUOUS`` -- ETFs have no CME-style trading
session to reset on, and a weekend/holiday gap in a genuinely daily series is
normal calendar structure, not a data defect. This is a DELIBERATE echo of a
real, documented Phase 15B incident (weekend gaps fragmenting
RESET_ON_GAP-policy windows and starving every feature to NaN, fixed in
cc206de) -- CONTINUOUS avoids that failure mode structurally rather than
needing a synthetic contiguous-index workaround.

Family label: ``"etf_tsmom"`` -- deliberately distinct from Futures'
``"tsmom"`` string (Phase 6 instruction: ETF and Futures must not share
statistical families "merely because labels match").
"""
from __future__ import annotations

import pandas as pd

from alpha_agent.etf.universe import etf_dsl_root_symbol
from alpha_agent.features.enums import SessionPolicy
from alpha_agent.features.source import SourceSeries
from alpha_agent.schemas.market_data import PriceDomain
from alpha_agent.strategy.baselines.factories import make_tsmom_spec
from alpha_agent.strategy.baselines.params import TsmomParams
from alpha_agent.strategy.spec import StrategySpec
from alpha_agent.validation.engine import DslFamilyAdapter

ETF_TSMOM_FAMILY = "etf_tsmom"

#: Same canonical horizons the frozen Futures tsmom trials use
#: (candidates_phase_13_5c.py) -- reused for consistency/comparability, not
#: re-derived or tuned against ETF data.
CANONICAL_FAST_HORIZON = 21
CANONICAL_SLOW_HORIZON = 120

_DAY_NS = 86_400_000_000_000


def _compute_frame(plan, bars: pd.DataFrame, *, root_symbol: str):
    from alpha_agent.features import compute_features

    # identity["root_symbol"] must match StrategySpec.root_symbol (the
    # synthetic DSL virtual-root label) -- the evaluator gates on this
    # (StrategyEvaluationError: "FeatureFrame root(s) do not match strategy
    # root_symbol"), so this is the DSL root, not the real ticker.
    src = SourceSeries(
        frame=bars[["ts_event_ns", "open", "high", "low", "close", "volume"]],
        price_domain=PriceDomain.RAW,
        identity={"root_symbol": etf_dsl_root_symbol(root_symbol)},
        interval_ns=_DAY_NS,
    )
    return compute_features(src, [b.spec for b in plan.feature_bindings], require_point_in_time=False)


def etf_tsmom_spec(root_symbol: str) -> StrategySpec:
    """The canonical Phase 6 pilot ETF hypothesis's StrategySpec, with
    CONTINUOUS session policy baked into its features so a weekend/holiday
    gap in the real daily bars never trips RESET_ON_GAP.

    Uses the synthetic DSL virtual-root label (`etf_dsl_root_symbol`), not
    the bare ticker, as `StrategySpec.root_symbol` -- see that function's
    docstring for why. `root_symbol` here is still the real ticker;
    `etf_tsmom_spec` does the substitution."""
    spec = make_tsmom_spec(
        TsmomParams(
            root_symbol=etf_dsl_root_symbol(root_symbol), fast_horizon=CANONICAL_FAST_HORIZON,
            slow_horizon=CANONICAL_SLOW_HORIZON, size=1, price_field="close",
        )
    )
    return spec.model_copy(
        update={
            "features": [
                decl.model_copy(update={"spec": decl.spec.model_copy(update={"session_policy": SessionPolicy.CONTINUOUS})})
                for decl in spec.features
            ]
        }
    )


def etf_tsmom_adapter(root_symbol: str) -> DslFamilyAdapter:
    return DslFamilyAdapter(
        strategy_key=ETF_TSMOM_FAMILY,
        canonical_params={
            "root_symbol": root_symbol, "fast_horizon": CANONICAL_FAST_HORIZON,
            "slow_horizon": CANONICAL_SLOW_HORIZON, "size": 1, "price_field": "close",
        },
        spec_factory=lambda params: etf_tsmom_spec(params["root_symbol"]),
        compute_frame=lambda plan, bars: _compute_frame(plan, bars, root_symbol=root_symbol),
    )

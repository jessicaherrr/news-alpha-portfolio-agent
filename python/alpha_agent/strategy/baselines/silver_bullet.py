"""Phase 12 -- ICT "Silver Bullet" / market-structure benchmark factory.

An informal trader-folklore idea, formalised as a closed Phase 10
:class:`StrategySpec`:

    make_silver_bullet_spec(SilverBulletParams)  ->  StrategySpec
        ->  StrategyCompiler  ->  ReferenceEvaluator  ->  TargetSchedule
        ->  C++ ScheduledTargetStrategy  ->  official Fill-derived PnL

All market-structure state (sweep -> displacement -> FVG -> retracement) lives in
the closed :mod:`alpha_agent.features.market_structure` detector as a registered,
causal, point-in-time feature (``silver_bullet_intent``). The DSL here only maps
that signed intent to a small integer target position -- no hidden execution
logic, no arbitrary code, exactly the Phase 11 bridge.

**Phase 12 makes no claim of profitability, alpha, robustness or statistical
significance.** Phase 13 reliability validation decides whether there is any
evidence. This is a formalised hypothesis and a benchmark, nothing more.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from alpha_agent.features.market_structure import (
    DISTANCE_UNITS,
    LIQUIDITY_REFERENCES,
    distance_unit_error,
)
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.strategy.baselines.families import BaselineFamilyDoc
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

MAX_SILVER_BULLET_SIZE = 5


class SilverBulletParams(BaseModel):
    """Typed parameter contract for the Silver Bullet benchmark.

    Every folklore concept is a number here; nothing is left to visual judgement.
    Invalid combinations are rejected at construction.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str = Field(pattern=r"^[A-Z0-9]{1,12}$")
    size: int = Field(default=1, ge=1, le=MAX_SILVER_BULLET_SIZE)

    # -- liquidity reference (section 4)
    liquidity_lookback: int = Field(default=20, ge=2, le=100_000)
    liquidity_reference: str = "rolling_nbar_extreme"

    # -- liquidity sweep (section 5) + explicit distance-unit semantics (Phase 12.1)
    sweep_penetration: float = Field(default=0.0, ge=0.0, le=1.0e12)
    # "price_units" (default) -- sweep_penetration / fvg_min_width are already in
    # normalized price units. "ticks" -- they are multiples of tick_size, which
    # MUST then be a finite value > 0 (from typed contract/tick metadata; never
    # assumed). No silent 1.0 fallback.
    distance_unit: str = "price_units"
    tick_size: float = Field(default=0.0, ge=0.0, le=1.0e9)

    # -- displacement (section 6)
    displacement_atr_window: int = Field(default=14, ge=2, le=100_000)
    displacement_atr_multiple: float = Field(default=1.5, gt=0.0, le=1.0e6)
    displacement_close_loc: float = Field(default=0.6, ge=0.0, le=1.0)

    # -- fair value gap (section 7)
    fvg_min_width: float = Field(default=0.0, ge=0.0, le=1.0e12)

    # -- retracement (section 8): 0.0 = near edge, 0.5 = midpoint, 1.0 = far edge
    retracement_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    # -- setup lifecycle / exits (sections 9, 15)
    setup_expiry_bars: int = Field(default=12, ge=1, le=100_000)
    max_holding_bars: int = Field(default=24, ge=1, le=100_000)

    # -- session window (section 11); exchange-local wall clock, DST-safe
    window_start_local: str = ""
    window_end_local: str = ""
    window_calendar: str = ""

    @model_validator(mode="after")
    def _check(self) -> SilverBulletParams:
        if self.liquidity_reference not in LIQUIDITY_REFERENCES:
            raise ValueError(
                f"liquidity_reference {self.liquidity_reference!r} must be one of "
                f"{list(LIQUIDITY_REFERENCES)}"
            )
        if bool(self.window_start_local) != bool(self.window_end_local):
            raise ValueError(
                "window_start_local and window_end_local must be set together (or both empty)"
            )
        unit_msg = distance_unit_error(self.distance_unit, self.tick_size)
        if unit_msg:
            raise ValueError(unit_msg)
        return self

    def feature_params(self) -> dict:
        return {
            "liquidity_lookback": self.liquidity_lookback,
            "liquidity_reference": self.liquidity_reference,
            "sweep_penetration": float(self.sweep_penetration),
            "distance_unit": self.distance_unit,
            "tick_size": float(self.tick_size),
            "displacement_atr_window": self.displacement_atr_window,
            "displacement_atr_multiple": float(self.displacement_atr_multiple),
            "displacement_close_loc": float(self.displacement_close_loc),
            "fvg_min_width": float(self.fvg_min_width),
            "retracement_fraction": float(self.retracement_fraction),
            "setup_expiry_bars": self.setup_expiry_bars,
            "max_holding_bars": self.max_holding_bars,
            "window_start_local": self.window_start_local,
            "window_end_local": self.window_end_local,
            "window_calendar": self.window_calendar or (
                self.root_symbol if self.window_start_local else ""
            ),
        }


def make_silver_bullet_spec(params: SilverBulletParams) -> StrategySpec:
    """Return the authoritative Phase 10 :class:`StrategySpec` for the Silver
    Bullet benchmark. It compiles through the standard compiler with no special
    handling: the only feature is the closed ``silver_bullet_intent`` detector
    column and the rules map its sign to ``+size`` / ``-size`` / flat.
    """
    intent = FeatureSpec(kind="silver_bullet_intent", params=params.feature_params())
    win = (
        f" {params.window_start_local}-{params.window_end_local}"
        if params.window_start_local
        else ""
    )
    return StrategySpec(
        strategy_name=f"silver_bullet {params.root_symbol}{win}",
        strategy_id=(
            f"SB-{params.root_symbol}-L{params.liquidity_lookback}-"
            f"D{params.displacement_atr_multiple:g}-R{params.retracement_fraction:g}-{params.size}"
        ),
        root_symbol=params.root_symbol,
        rationale=(
            "Phase 12 formalised ICT Silver Bullet market-structure benchmark. "
            "Formalised trader folklore, NOT claimed alpha: no profitability, robustness "
            "or statistical-significance claim -- Phase 13 decides. Every concept "
            "(liquidity reference / sweep / displacement / FVG / retracement) has a "
            "deterministic causal definition in the closed feature detector."
        ),
        features=[FeatureDeclaration(alias="sb", spec=intent)],
        rules=[
            Rule(
                rule_id="silver_bullet_long",
                when=ComparisonNode(
                    op=Comparator.GT,
                    left=FeatureOperand(feature="sb"),
                    right=ConstOperand(value=0.5),
                ),
                action=TargetAction(target_units=params.size),
                rationale="bullish setup: sell-side sweep -> displacement -> FVG -> retracement",
            ),
            Rule(
                rule_id="silver_bullet_short",
                when=ComparisonNode(
                    op=Comparator.LT,
                    left=FeatureOperand(feature="sb"),
                    right=ConstOperand(value=-0.5),
                ),
                action=TargetAction(target_units=-params.size),
                rationale="bearish setup: buy-side sweep -> displacement -> FVG -> retracement",
            ),
        ],
        # intent already encodes the full held state every bar (incl. typed
        # exits -> 0); no keep-previous needed.
        default_action=DefaultAction.FLAT,
    )


# ==========================================================================
# One canonical "as taught" benchmark + a PRE-DECLARED research grid (section 23)
# ==========================================================================
SILVER_BULLET_BENCHMARK = SilverBulletParams(
    root_symbol="NQ",
    size=1,
    liquidity_lookback=20,
    liquidity_reference="rolling_nbar_extreme",
    sweep_penetration=0.0,
    displacement_atr_window=14,
    displacement_atr_multiple=1.5,
    displacement_close_loc=0.6,
    fvg_min_width=0.0,
    retracement_fraction=0.5,
    setup_expiry_bars=12,
    max_holding_bars=24,
    window_start_local="09:00",
    window_end_local="10:00",
    window_calendar="NQ",
)

# Ranges only -- declared before any results are inspected. NOT a grid to search
# in Phase 12; Phase 13 owns multiple-testing-aware validation (section 23).
SILVER_BULLET_GRID_RANGES: dict[str, str] = {
    "liquidity_lookback": "10 .. 60 bars",
    "liquidity_reference": f"one of {list(LIQUIDITY_REFERENCES)}",
    "distance_unit": f"one of {list(DISTANCE_UNITS)} (fixed per run; part of the fingerprint)",
    "sweep_penetration": "0 .. 4 (in distance_unit; 'ticks' needs a real tick_size)",
    "displacement_atr_window": "10 .. 30 bars",
    "displacement_atr_multiple": "1.0 .. 2.5",
    "displacement_close_loc": "0.5 .. 0.8",
    "fvg_min_width": "0 .. 3 (in distance_unit; 'ticks' needs a real tick_size)",
    "retracement_fraction": "0.0 (near edge) .. 1.0 (far edge)",
    "setup_expiry_bars": "6 .. 24 bars",
    "max_holding_bars": "12 .. 48 bars",
    "window": "London / NY-AM / NY-PM local windows (start<end, no midnight wrap)",
}

SILVER_BULLET_FAMILY = BaselineFamilyDoc(
    key="silver_bullet",
    name="ICT Silver Bullet market-structure benchmark",
    factory="make_silver_bullet_spec",
    economic_mechanism=(
        "Trader folklore: a stop-run (liquidity sweep) beyond a recent extreme, followed "
        "by an impulsive displacement move that leaves a fair value gap, offers a "
        "favourable entry on the retracement back into that gap, inside a specific "
        "intraday window. Phase 12 treats this as a HYPOTHESIS to be measured, not a "
        "known edge."
    ),
    formula_and_timing=(
        "T0 sweep (low<prior_low-pen & close>prior_low, or mirror); Td displacement "
        "STRICTLY after T0 (body/ATR>=mult & close-location>=thr); Tf=Td+1 the 3-bar FVG "
        "(Td-1,Td,Td+1) first observable; Tr>Tf retracement to depth "
        "retracement_fraction into the FVG, inside the window -> StrategyDecision at Tr; "
        "C++ executes next eligible bar. Exit: max_holding_bars | window close | session "
        "end | close through the swept level."
    ),
    required_data=["one OHLC price series per root (needs open/high/low/close)"],
    parameters=sorted(SILVER_BULLET_GRID_RANGES),
    param_grid_ranges=SILVER_BULLET_GRID_RANGES,
    failure_regimes=[
        "no folklore edge exists -> costs dominate a low-frequency benchmark",
        "trending regimes: the swept level keeps failing -> repeated invalidation exits",
        "parameter sensitivity of displacement / retracement thresholds",
        "window mis-specification (wrong session, DST) changes the setup population",
        "few setups on a short sample -> descriptive counts only, no significance",
    ],
    default_action="flat",
    is_claimed_alpha=False,
)


def silver_bullet_family_doc() -> BaselineFamilyDoc:
    return SILVER_BULLET_FAMILY

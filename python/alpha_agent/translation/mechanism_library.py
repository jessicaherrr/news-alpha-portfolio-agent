"""The deterministic, no-network mechanism/factor library (prompt 1 sections
5/6/8) -- the ALWAYS-AVAILABLE fallback the translation pipeline uses when no
live Claude proposal is requested, and the fixture data the three acceptance
examples and every offline test are built on.

Keyed by ``alpha_agent.market_intel.news_schemas.NewsCategory`` -- the SAME
closed, deterministic category vocabulary
``alpha_agent.market_intel.mapping`` already uses to decide which products a
news/event item affects. Never keyed by root symbol (CLAUDE.md: "never
special-case a root" -- the economic reasoning here is about WHAT KIND of
official release happened, not WHICH certified market it happens to map to
today).

Every ``FactorTemplate`` below is a hand-reviewed, documented judgment about
whether a concept is structurally expressible in today's StrategySpec /
FeatureRegistry model (``structurally_expressible``) and which registered
feature kinds / external data it needs -- never a claim this module expects
the reader to trust blindly: ``alpha_agent.translation.researchability
.classify_factor`` re-derives the actual researchability status from live
``alpha_agent.features.REGISTRY`` state, so a stale template can only ever
under-claim, never silently over-claim, availability.
"""
from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.knowledge.models import EconomicMechanism
from alpha_agent.market_intel.news_schemas import NewsCategory
from alpha_agent.translation.schemas import MeasurableVariable

__all__ = ["CATEGORY_MECHANISM_TEMPLATES", "FactorTemplate", "MechanismTemplate"]


@dataclass(frozen=True)
class FactorTemplate:
    concept: str
    mechanism: EconomicMechanism
    transform_or_proxy: str
    proposed_feature_kinds: tuple[str, ...] = ()
    required_external_data: tuple[str, ...] = ()
    #: See module docstring -- False only for a concept this library already
    #: knows today's single-instrument FeatureRegistry model cannot express
    #: at all, regardless of data (e.g. a cross-contract/cross-instrument
    #: join).
    structurally_expressible: bool = True


@dataclass(frozen=True)
class MechanismTemplate:
    mechanism: EconomicMechanism
    explanation: str
    causal_chain: tuple[str, ...]
    evidence_basis: str
    measurable_variables: tuple[MeasurableVariable, ...]
    factors: tuple[FactorTemplate, ...]


# ---------------------------------------------------------------------------
# measurable variables -- reused across the mechanism templates that need them
# ---------------------------------------------------------------------------

_RELEASE_SURPRISE_VAR = MeasurableVariable(
    name="realized_release_surprise",
    economic_meaning=(
        "The actual reported value of the official release relative to its recent trend "
        "(e.g. the week-over-week inventory/storage change, or the policy/inflation print itself)."
    ),
    required_source="the official release value (currently ingested as a news headline item only)",
    point_in_time_available=False,
    point_in_time_note=(
        "This platform ingests the CURRENT release as a MarketNewsItem headline only -- it does not "
        "yet store a point-in-time historical numeric series for this release usable for backtesting."
    ),
)

_CONSENSUS_SURPRISE_VAR = MeasurableVariable(
    name="consensus_surprise",
    economic_meaning="Actual reported value minus the pre-release consensus/analyst expectation.",
    required_source="a historical, point-in-time consensus-expectation series for this release",
    point_in_time_available=False,
    point_in_time_note="No consensus-expectation data source is currently connected or ingested by this platform.",
)

_PRICE_REACTION_VAR = MeasurableVariable(
    name="price_reaction_window",
    economic_meaning=(
        "Directional and volatility response of the affected root's own price in the bars "
        "following the release."
    ),
    required_source="the root's own OHLCV price series (already acquired, point-in-time safe)",
    point_in_time_available=True,
    point_in_time_note="Backed by the same raw contract OHLCV bars every registered price/volatility feature already uses.",
)

_CURVE_STATE_VAR = MeasurableVariable(
    name="term_structure_state",
    economic_meaning="Shape of the futures curve (contango/backwardation) across nearby contract months.",
    required_source="multiple simultaneous contract months (a cross-contract join)",
    point_in_time_available=True,
    point_in_time_note=(
        "Computed today for DISPLAY only (`alpha_agent.ui.market_contracts`) from already-acquired raw "
        "contract data; it is not a registered FeatureRegistry kind, so no StrategySpec can condition "
        "on it yet."
    ),
)

_CROSS_MARKET_VAR = MeasurableVariable(
    name="cross_market_confirmation",
    economic_meaning="Whether related markets (rates, FX, equities, gold) move consistently with the same macro read.",
    required_source="a joint, multi-instrument feature (more than one root's series at once)",
    point_in_time_available=True,
    point_in_time_note=(
        "Computed today for DISPLAY only (`alpha_agent.ui.market_relative`) from already-acquired data; "
        "the FeatureRegistry only computes features over a single instrument's own series, so no "
        "StrategySpec can condition on it yet."
    ),
)


# ---------------------------------------------------------------------------
# periodic official supply/demand report (EIA petroleum/natural gas, USDA
# WASDE/crop reports) -- the same underlying economic structure in every case:
# a scheduled official release of a quantitative supply/demand fact.
# ---------------------------------------------------------------------------

_SUPPLY_DEMAND_REPORT_TEMPLATES: tuple[MechanismTemplate, ...] = (
    MechanismTemplate(
        mechanism=EconomicMechanism.TERM_STRUCTURE,
        explanation=(
            "A larger-than-expected change in a reported supply/demand aggregate (e.g. a crude/product "
            "inventory draw, or a USDA stocks/production revision) signals tighter or looser near-term "
            "physical balance, which is typically expressed first in the futures curve (a shift toward "
            "backwardation or contango) before it necessarily shows up in the outright price."
        ),
        causal_chain=(
            "Official supply/demand report surprises relative to the recent trend",
            "Market reprices near-term physical scarcity or surplus",
            "Near-dated contracts move relative to deferred contracts (curve shape shifts)",
        ),
        evidence_basis="Category mapping in alpha_agent.market_intel.mapping (PETROLEUM/NATURAL_GAS/USDA_GRAIN_OILSEED).",
        measurable_variables=(_RELEASE_SURPRISE_VAR, _CONSENSUS_SURPRISE_VAR, _CURVE_STATE_VAR),
        factors=(
            FactorTemplate(
                concept="Curve-shape confirmation following a supply/demand surprise",
                mechanism=EconomicMechanism.TERM_STRUCTURE,
                transform_or_proxy="Term-structure slope (near vs. deferred contract) sign/state around the release.",
                structurally_expressible=False,
            ),
        ),
    ),
    MechanismTemplate(
        mechanism=EconomicMechanism.TREND,
        explanation=(
            "A surprising supply/demand release can trigger directional positioning as participants adjust "
            "to the new balance; if the surprise is large enough this can start a short-horizon trend that "
            "persists beyond the release bar."
        ),
        causal_chain=(
            "Official report surprises relative to the recent trend",
            "Participants adjust directional positioning",
            "Price trends in the surprise's direction over the following sessions",
        ),
        evidence_basis="Category mapping in alpha_agent.market_intel.mapping (PETROLEUM/NATURAL_GAS/USDA_GRAIN_OILSEED).",
        measurable_variables=(_RELEASE_SURPRISE_VAR, _PRICE_REACTION_VAR),
        factors=(
            FactorTemplate(
                concept="Post-release trend continuation in the reacting root",
                mechanism=EconomicMechanism.TREND,
                transform_or_proxy=(
                    "trend_strength(fast, slow) and realized_vol on the root's own OHLCV in the bars "
                    "following the release."
                ),
                proposed_feature_kinds=("trend_strength", "realized_vol"),
            ),
            FactorTemplate(
                concept="Release-surprise magnitude as a conditioning variable",
                mechanism=EconomicMechanism.TREND,
                transform_or_proxy="Bucket the realized surprise magnitude and condition the trend factor on it.",
                required_external_data=("historical point-in-time consensus/expectation series for the release",),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# scheduled macro release (FOMC policy decision, CPI/PPI/employment prints)
# ---------------------------------------------------------------------------

_MACRO_SURPRISE_VAR = MeasurableVariable(
    name="policy_or_inflation_surprise",
    economic_meaning="Actual policy decision / inflation-employment print relative to the pre-release consensus expectation.",
    required_source="a historical, point-in-time consensus-expectation series for this release",
    point_in_time_available=False,
    point_in_time_note="No consensus-expectation data source is currently connected or ingested by this platform.",
)

_MACRO_TEMPLATES: tuple[MechanismTemplate, ...] = (
    MechanismTemplate(
        mechanism=EconomicMechanism.VOLATILITY_BREAKOUT,
        explanation=(
            "A macro policy or inflation/employment release resolves real uncertainty about the near-term "
            "rate path in a single instant, which typically produces a step change in realized volatility "
            "around the release regardless of which direction price ultimately moves."
        ),
        causal_chain=(
            "Scheduled macro release resolves uncertainty about the rate path",
            "Market repositions rapidly across rate-sensitive instruments",
            "Realized volatility expands sharply around the release, then decays",
        ),
        evidence_basis="Category mapping in alpha_agent.market_intel.mapping (FOMC_POLICY/CPI_PPI_EMPLOYMENT).",
        measurable_variables=(_MACRO_SURPRISE_VAR, _PRICE_REACTION_VAR),
        factors=(
            FactorTemplate(
                concept="Volatility-regime shift around a high-impact macro release",
                mechanism=EconomicMechanism.VOLATILITY_BREAKOUT,
                transform_or_proxy="realized_vol / vol_percentile on the root's own OHLCV before vs. after the release.",
                proposed_feature_kinds=("realized_vol", "vol_percentile"),
            ),
            FactorTemplate(
                concept="Surprise-magnitude-conditioned volatility response",
                mechanism=EconomicMechanism.VOLATILITY_BREAKOUT,
                transform_or_proxy="Bucket the realized surprise magnitude and condition the volatility factor on it.",
                required_external_data=("historical point-in-time consensus/expectation series for the release",),
            ),
        ),
    ),
    MechanismTemplate(
        mechanism=EconomicMechanism.CROSS_MARKET_LEAD_LAG,
        explanation=(
            "A genuine macro repricing should show up jointly across rate-sensitive markets (rates, equity "
            "index, gold, FX); the breadth of confirmation across related markets indicates how broadly the "
            "market believes the new information, versus an isolated, single-market move."
        ),
        causal_chain=(
            "Scheduled macro release changes the expected policy path",
            "Rates, equity index, gold, and FX markets reprice together",
            "Breadth of joint confirmation signals conviction behind the move",
        ),
        evidence_basis="Category -> multi-asset product mapping in alpha_agent.market_intel.mapping (FOMC_POLICY/CPI_PPI_EMPLOYMENT).",
        measurable_variables=(_CROSS_MARKET_VAR,),
        factors=(
            FactorTemplate(
                concept="Cross-market confirmation breadth following a macro release",
                mechanism=EconomicMechanism.CROSS_MARKET_LEAD_LAG,
                transform_or_proxy="Count of related markets moving in the macro-consistent direction.",
                structurally_expressible=False,
            ),
        ),
    ),
)


CATEGORY_MECHANISM_TEMPLATES: dict[NewsCategory, tuple[MechanismTemplate, ...]] = {
    NewsCategory.PETROLEUM: _SUPPLY_DEMAND_REPORT_TEMPLATES,
    NewsCategory.NATURAL_GAS: _SUPPLY_DEMAND_REPORT_TEMPLATES,
    NewsCategory.USDA_GRAIN_OILSEED: _SUPPLY_DEMAND_REPORT_TEMPLATES,
    NewsCategory.FOMC_POLICY: _MACRO_TEMPLATES,
    NewsCategory.CPI_PPI_EMPLOYMENT: _MACRO_TEMPLATES,
    NewsCategory.OTHER: (),
}

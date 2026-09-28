"""News Alpha Phase A -- the deterministic ECONOMIC CHANNEL table the Initial
Impact Scan reads.

A channel is the first-order economic route by which a class of news reaches
markets (the rate path, the crude supply balance, an investment cycle, ...).
Each `ChannelDefinition` is a hand-reviewed, inspectable judgment -- the same
kind of artifact as `alpha_agent.translation.mechanism_library` -- recording:

* how the channel is DETECTED: an authoritative market_intel source category
  (`alpha_agent.market_intel.mapping`'s own deterministic vocabulary), an
  issuer cue, or a keyword lexicon over the headline/summary;
* which DOMAINS / market GROUPS it exposes, at what strength, and why --
  expressed against each domain's own existing classification (Futures asset
  class or root, Equity sector, ETF ticker), never a new taxonomy;
* a DIRECTION only where the first-order price pressure is textbook (e.g.
  Treasury futures prices fall when a tightening surprise lifts yields).
  Everything else deliberately carries no sign -- the Mechanism Graph stage
  resolves spender-vs-supplier and other second-order paths;
* a typical reaction HORIZON with its rationale;
* the `NewsCategory` whose Phase 1 translation templates model the SAME
  channel (``None`` when no deterministic template exists yet).

Never keyed by an example headline and never special-cases a root
(CLAUDE.md). `tests/python/test_news_alpha_phase_a.py` checks every exposure
against the live domain universes, so a stale entry fails loudly instead of
silently matching nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from alpha_agent.market_intel.news_schemas import NewsCategory
from alpha_agent.news_alpha.mandate import MandateDomain

__all__ = [
    "CHANNELS",
    "SALIENCE_TERMS",
    "ChannelDefinition",
    "DirectionCue",
    "EconomicChannel",
    "ExposureStrength",
    "ImpactHorizon",
    "MarketExposure",
    "PressureSign",
    "channel_definition",
    "normalize_text",
]


class EconomicChannel(str, Enum):
    MONETARY_POLICY = "MONETARY_POLICY"
    INFLATION_AND_EMPLOYMENT = "INFLATION_AND_EMPLOYMENT"
    CRUDE_OIL_SUPPLY = "CRUDE_OIL_SUPPLY"
    NATURAL_GAS_SUPPLY = "NATURAL_GAS_SUPPLY"
    AGRICULTURAL_SUPPLY = "AGRICULTURAL_SUPPLY"
    TECHNOLOGY_INVESTMENT = "TECHNOLOGY_INVESTMENT"
    BANKING_AND_CREDIT = "BANKING_AND_CREDIT"
    GEOPOLITICAL_RISK = "GEOPOLITICAL_RISK"
    TRADE_POLICY = "TRADE_POLICY"
    DIGITAL_ASSETS = "DIGITAL_ASSETS"


class ExposureStrength(str, Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    TERTIARY = "TERTIARY"


class ImpactHorizon(str, Enum):
    INTRADAY_TO_DAYS = "INTRADAY_TO_DAYS"
    DAYS_TO_WEEKS = "DAYS_TO_WEEKS"
    WEEKS_TO_MONTHS = "WEEKS_TO_MONTHS"


class PressureSign(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


def normalize_text(text: str) -> str:
    """Lowercase, hyphens/dashes -> spaces, collapsed whitespace -- applied
    identically to headlines and to every lexicon term, so
    "larger-than-expected" and "data-center" match their spaced forms."""
    return " ".join(re.sub(r"[-\u2010-\u2015/]", " ", text.lower()).split())


@dataclass(frozen=True)
class DirectionCue:
    """Phrases that establish WHICH WAY the channel moved (e.g. a TIGHTENING
    vs. EASING policy shift). Evaluated only inside an already-detected
    channel."""

    shift: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class MarketExposure:
    domain: MandateDomain
    strength: ExposureStrength
    label: str
    rationale: str
    #: Domain-native groups (Futures `AssetClass` value / Equity sector) whose
    #: members are candidates.
    groups: tuple[str, ...] = ()
    #: Explicit domain symbols (Futures roots / ETF tickers).
    symbols: tuple[str, ...] = ()
    #: ``(shift, sign)``: textbook first-order price pressure on THIS group
    #: under that shift. Empty = no direction is justified at this stage.
    pressure: tuple[tuple[str, PressureSign], ...] = ()


@dataclass(frozen=True)
class ChannelDefinition:
    channel: EconomicChannel
    rule_id: str
    label: str
    description: str
    source_categories: tuple[str, ...]
    issuer_cues: tuple[str, ...]
    lexicon: tuple[str, ...]
    direction_cues: tuple[DirectionCue, ...]
    typical_horizon: ImpactHorizon
    horizon_rationale: str
    exposures: tuple[MarketExposure, ...]
    translation_category: NewsCategory | None


#: Global SURPRISE cues. A channel detected with at least MODERATE strength
#: AND one of these (or a scheduled event the market_intel importance rule
#: marks HIGH) is raised one impact level. Surprise relative to expectations
#: is what moves prices; a routine or explanatory item on the same topic is
#: not. Superlatives ("record", "largest") are deliberately NOT cues -- real
#: EIA explainers ("...second-largest LNG facility", "record production")
#: use them without any expectation surprise.
SALIENCE_TERMS: tuple[str, ...] = (
    "surprise", "surprising", "surprisingly", "unexpected", "unexpectedly", "shock", "unscheduled", "emergency",
    "larger than expected", "smaller than expected", "hotter than expected", "cooler than expected",
    "better than expected", "worse than expected", "stronger than expected", "weaker than expected",
)

_F, _E, _Q, _C = MandateDomain.FUTURES, MandateDomain.ETF, MandateDomain.EQUITY, MandateDomain.CRYPTO
_P, _S, _T = ExposureStrength.PRIMARY, ExposureStrength.SECONDARY, ExposureStrength.TERTIARY
_UP, _DOWN = PressureSign.UP, PressureSign.DOWN

_TREASURY_FUTURES_PRESSURE_NOTE = "Treasury futures prices move inversely to the yields a rate-path repricing moves."

CHANNELS: tuple[ChannelDefinition, ...] = (
    ChannelDefinition(
        channel=EconomicChannel.MONETARY_POLICY,
        rule_id="channel/monetary-policy/1",
        label="Monetary policy / rate path",
        description="A change in the expected policy-rate path reprices every rate-sensitive market at once.",
        source_categories=(NewsCategory.FOMC_POLICY.value,),
        issuer_cues=("federal reserve",),
        lexicon=(
            "fomc", "federal reserve", "the fed", "fed chair", "fed officials", "interest rate", "interest rates",
            "rate decision", "policy rate", "federal funds", "fed funds", "rate hike", "rate hikes", "rate cut",
            "rate cuts", "raises rates", "raised rates", "hikes rates", "hiked rates", "cuts rates", "cut rates",
            "basis points", "hawkish", "dovish", "monetary policy", "central bank", "quantitative tightening",
        ),
        direction_cues=(
            DirectionCue("TIGHTENING", (
                "rate hike", "rate hikes", "raises rates", "raised rates", "raise rates", "hikes rates", "hiked rates",
                "hawkish", "tightening",
            )),
            DirectionCue("EASING", (
                "rate cut", "rate cuts", "cuts rates", "cut rates", "lowers rates", "lowered rates", "dovish", "easing",
            )),
        ),
        typical_horizon=ImpactHorizon.INTRADAY_TO_DAYS,
        horizon_rationale="Policy decisions reprice rate-sensitive markets within the release session; follow-through "
                          "is typically measured over the following days.",
        exposures=(
            MarketExposure(_F, _P, "Treasury futures", _TREASURY_FUTURES_PRESSURE_NOTE, groups=("RATES",),
                           pressure=(("TIGHTENING", _DOWN), ("EASING", _UP))),
            MarketExposure(_F, _P, "Equity index futures",
                           "Discount-rate and growth expectations reprice broad equity index futures.",
                           groups=("EQUITY_INDEX",)),
            MarketExposure(_F, _S, "FX futures", "Relative policy-rate expectations drive G10 currency pricing.",
                           groups=("FX",)),
            MarketExposure(_F, _S, "Gold futures", "Real-rate expectations are a first-order driver of gold.",
                           symbols=("GC", "MGC")),
            MarketExposure(_E, _P, "Treasury ETFs", "Bond-fund prices move inversely to yields.",
                           symbols=("SHY", "IEF", "TLT"), pressure=(("TIGHTENING", _DOWN), ("EASING", _UP))),
            MarketExposure(_E, _S, "Broad & rate-sensitive equity ETFs",
                           "Broad valuations, bank margins (XLF) and bond-proxy utilities (XLU) are rate-sensitive.",
                           symbols=("SPY", "QQQ", "IWM", "XLF", "XLU")),
            MarketExposure(_E, _S, "Credit ETFs", "Corporate-bond funds carry the same duration exposure.",
                           symbols=("LQD", "HYG")),
            MarketExposure(_Q, _S, "Financials", "Bank net-interest margins depend directly on the policy-rate path.",
                           groups=("Financials",)),
            MarketExposure(_Q, _T, "Long-duration growth equities",
                           "Growth valuations are the most discount-rate sensitive.", groups=("Technology",)),
        ),
        translation_category=NewsCategory.FOMC_POLICY,
    ),
    ChannelDefinition(
        channel=EconomicChannel.INFLATION_AND_EMPLOYMENT,
        rule_id="channel/inflation-employment/1",
        label="Inflation & labor data",
        description="Inflation and employment prints are the inputs to the expected policy path.",
        source_categories=(NewsCategory.CPI_PPI_EMPLOYMENT.value,),
        issuer_cues=(),
        lexicon=(
            "cpi", "consumer price index", "inflation", "ppi", "producer price index", "producer prices", "pce",
            "core pce", "payrolls", "nonfarm", "jobs report", "unemployment rate", "employment report",
            "wage growth", "average hourly earnings", "jobless claims",
        ),
        direction_cues=(
            DirectionCue("HOTTER", (
                "hotter than expected", "inflation accelerated", "inflation accelerates", "accelerating inflation",
                "prices rose faster",
            )),
            DirectionCue("COOLER", (
                "cooler than expected", "inflation eased", "inflation slowed", "disinflation", "prices rose slower",
            )),
        ),
        typical_horizon=ImpactHorizon.INTRADAY_TO_DAYS,
        horizon_rationale="Scheduled macro prints are absorbed within the release session, with follow-through over days.",
        exposures=(
            MarketExposure(_F, _P, "Treasury futures", _TREASURY_FUTURES_PRESSURE_NOTE, groups=("RATES",),
                           pressure=(("HOTTER", _DOWN), ("COOLER", _UP))),
            MarketExposure(_F, _P, "Equity index futures", "Macro prints reset growth and discount-rate expectations.",
                           groups=("EQUITY_INDEX",)),
            MarketExposure(_F, _S, "FX futures", "Macro surprises shift relative rate expectations.", groups=("FX",)),
            MarketExposure(_F, _S, "Gold futures", "Inflation and real-rate expectations drive gold.",
                           symbols=("GC", "MGC")),
            MarketExposure(_E, _P, "Treasury ETFs", "Bond-fund prices move inversely to yields.",
                           symbols=("SHY", "IEF", "TLT"), pressure=(("HOTTER", _DOWN), ("COOLER", _UP))),
            MarketExposure(_E, _S, "Broad equity ETFs", "Macro prints reset broad equity valuations.",
                           symbols=("SPY", "QQQ", "IWM")),
            MarketExposure(_Q, _S, "Consumer sectors", "Pricing power and real consumer demand are inflation-sensitive.",
                           groups=("Consumer Staples", "Consumer Discretionary")),
            MarketExposure(_Q, _T, "Financials", "Via the policy path the print implies.", groups=("Financials",)),
        ),
        translation_category=NewsCategory.CPI_PPI_EMPLOYMENT,
    ),
    ChannelDefinition(
        channel=EconomicChannel.CRUDE_OIL_SUPPLY,
        rule_id="channel/crude-oil-supply/1",
        label="Crude oil supply/demand balance",
        description="Producer decisions, disruptions and inventory surprises shift the physical crude balance.",
        source_categories=(NewsCategory.PETROLEUM.value,),
        issuer_cues=(),
        lexicon=(
            "opec", "opec+", "crude", "crude oil", "oil price", "oil prices", "barrels per day", "barrels a day",
            "oil output", "oil production", "production cut", "output cut", "refinery", "refineries", "gasoline",
            "diesel", "distillate", "petroleum", "brent", "wti", "strategic petroleum reserve",
        ),
        direction_cues=(
            DirectionCue("TIGHTER_BALANCE", (
                "production cut", "production cuts", "output cut", "output cuts", "cut output", "cuts output",
                "cut production", "cuts production", "supply disruption", "supply disruptions", "crude draw",
                "inventory draw", "draw in crude",
            )),
            DirectionCue("LOOSER_BALANCE", (
                "production increase", "output increase", "raise output", "raises output", "boost output",
                "boosts output", "increase production", "increases production", "crude build", "inventory build",
            )),
        ),
        typical_horizon=ImpactHorizon.DAYS_TO_WEEKS,
        horizon_rationale="Physical supply shocks are absorbed over sessions to weeks as the curve and outright price adjust.",
        exposures=(
            MarketExposure(_F, _P, "Crude & refined-product futures",
                           "Directly priced off the physical crude balance.", symbols=("CL", "MCL", "RB", "HO"),
                           pressure=(("TIGHTER_BALANCE", _UP), ("LOOSER_BALANCE", _DOWN))),
            MarketExposure(_E, _P, "Crude oil ETF", "Holds crude futures directly.", symbols=("USO",),
                           pressure=(("TIGHTER_BALANCE", _UP), ("LOOSER_BALANCE", _DOWN))),
            MarketExposure(_E, _S, "Energy sector ETF", "Producer revenues track realized crude prices.",
                           symbols=("XLE",), pressure=(("TIGHTER_BALANCE", _UP), ("LOOSER_BALANCE", _DOWN))),
            MarketExposure(_Q, _S, "Energy sector", "Integrated producers' revenues track realized crude prices.",
                           groups=("Energy",), pressure=(("TIGHTER_BALANCE", _UP), ("LOOSER_BALANCE", _DOWN))),
            MarketExposure(_Q, _T, "Industrials", "Fuel is an input cost for industrial and transport activity.",
                           groups=("Industrials",)),
        ),
        translation_category=NewsCategory.PETROLEUM,
    ),
    ChannelDefinition(
        channel=EconomicChannel.NATURAL_GAS_SUPPLY,
        rule_id="channel/natural-gas-supply/1",
        label="Natural gas supply/demand balance",
        description="Storage surprises, LNG flows and weather-driven demand shift the gas balance.",
        source_categories=(NewsCategory.NATURAL_GAS.value,),
        issuer_cues=(),
        lexicon=(
            "natural gas", "henry hub", "lng", "liquefied natural gas", "gas storage", "storage injection",
            "storage withdrawal", "gas prices", "gas production",
        ),
        direction_cues=(
            DirectionCue("TIGHTER_BALANCE", (
                "storage draw", "larger than expected withdrawal", "smaller than expected injection",
                "supply disruption",
            )),
            DirectionCue("LOOSER_BALANCE", (
                "storage build", "larger than expected injection", "smaller than expected withdrawal",
            )),
        ),
        typical_horizon=ImpactHorizon.DAYS_TO_WEEKS,
        horizon_rationale="Storage and supply surprises reprice the front of the gas curve over days to weeks.",
        exposures=(
            MarketExposure(_F, _P, "Natural gas futures", "Directly priced off the gas balance.", symbols=("NG",),
                           pressure=(("TIGHTER_BALANCE", _UP), ("LOOSER_BALANCE", _DOWN))),
            MarketExposure(_E, _T, "Utilities sector ETF", "Gas is a generation fuel cost for utilities.",
                           symbols=("XLU",)),
            MarketExposure(_Q, _T, "Energy sector", "Integrated producers also produce gas.", groups=("Energy",)),
        ),
        translation_category=NewsCategory.NATURAL_GAS,
    ),
    ChannelDefinition(
        channel=EconomicChannel.AGRICULTURAL_SUPPLY,
        rule_id="channel/agricultural-supply/1",
        label="Grain & oilseed supply/demand",
        description="Crop reports, weather and export policy shift the grain/oilseed balance.",
        source_categories=(NewsCategory.USDA_GRAIN_OILSEED.value,),
        issuer_cues=(),
        lexicon=(
            "usda", "wasde", "crop", "crops", "harvest", "corn", "wheat", "soybean", "soybeans", "grain", "grains",
            "oilseed", "drought", "planting", "acreage",
        ),
        direction_cues=(
            DirectionCue("TIGHTER_BALANCE", ("drought", "crop failure", "lower yields", "smaller harvest", "export ban")),
            DirectionCue("LOOSER_BALANCE", ("record harvest", "bumper crop", "higher yields", "larger harvest")),
        ),
        typical_horizon=ImpactHorizon.DAYS_TO_WEEKS,
        horizon_rationale="Crop-balance revisions are repriced over sessions to weeks around each report.",
        exposures=(
            MarketExposure(_F, _P, "Grain & oilseed futures", "Directly priced off the crop balance.",
                           groups=("AGRICULTURE",), pressure=(("TIGHTER_BALANCE", _UP), ("LOOSER_BALANCE", _DOWN))),
            MarketExposure(_Q, _T, "Consumer staples", "Grains are an input cost for packaged food.",
                           groups=("Consumer Staples",)),
        ),
        translation_category=NewsCategory.USDA_GRAIN_OILSEED,
    ),
    ChannelDefinition(
        channel=EconomicChannel.TECHNOLOGY_INVESTMENT,
        rule_id="channel/technology-investment/1",
        label="Technology / AI investment cycle",
        description="Changes in technology capital spending move the spenders, their suppliers, and the build-out "
                    "inputs (power, construction, metals).",
        source_categories=(),
        issuer_cues=(),
        lexicon=(
            "ai infrastructure", "artificial intelligence", "ai spending", "ai investment", "ai chips", "data center",
            "data centers", "datacenter", "datacenters", "hyperscaler", "hyperscalers", "capex",
            "capital expenditure", "capital expenditures", "capital spending", "semiconductor", "semiconductors",
            "chipmaker", "chipmakers", "gpu", "gpus", "cloud computing",
        ),
        direction_cues=(
            DirectionCue("SPENDING_INCREASE", (
                "spending increase", "increase spending", "increases spending", "increased spending", "raise spending",
                "raises spending", "raised spending", "boost spending", "boosts spending", "higher capex",
                "raises capex", "raised capex", "capex increase", "spending surge", "ramps up spending",
            )),
            DirectionCue("SPENDING_DECREASE", (
                "cuts capex", "capex cut", "spending cut", "cut spending", "cuts spending", "reduces spending",
                "slows spending", "spending pullback",
            )),
        ),
        typical_horizon=ImpactHorizon.WEEKS_TO_MONTHS,
        horizon_rationale="Capital-spending cycles play out over quarters of guidance and orders, not single sessions.",
        exposures=(
            # No price sign at this stage: the SAME capex increase is revenue
            # for a supplier and a cost for a spender, and both sit in these
            # sectors -- the Mechanism Graph resolves those paths.
            MarketExposure(_Q, _P, "Technology & communication services",
                           "Both the hyperscaler spenders and their semiconductor/hardware suppliers sit here.",
                           groups=("Technology", "Communication Services")),
            MarketExposure(_Q, _S, "Industrials", "Power, cooling and construction equipment for the build-out.",
                           groups=("Industrials",)),
            MarketExposure(_E, _S, "Technology & growth ETFs", "Concentrated in the spenders and suppliers.",
                           symbols=("XLK", "QQQ")),
            MarketExposure(_E, _T, "Utilities sector ETF", "Data-center power demand.", symbols=("XLU",)),
            MarketExposure(_F, _S, "Nasdaq-100 index futures",
                           "Index weight is concentrated in the spenders and suppliers.", symbols=("NQ", "MNQ")),
            MarketExposure(_F, _T, "Copper futures", "Data-center and grid build-out is copper-intensive.",
                           symbols=("HG",)),
            MarketExposure(_F, _T, "Natural gas futures", "Incremental power demand for data centers.",
                           symbols=("NG",)),
        ),
        translation_category=None,
    ),
    ChannelDefinition(
        channel=EconomicChannel.BANKING_AND_CREDIT,
        rule_id="channel/banking-credit/1",
        label="Banking & credit conditions",
        description="Bank stress, credit-spread moves and bank regulation change funding conditions and risk appetite.",
        source_categories=(),
        issuer_cues=(),
        lexicon=(
            "bank failure", "bank failures", "bank run", "bank collapse", "regional bank", "regional banks",
            "deposit outflows", "deposit flight", "credit spreads", "credit spread", "debt default", "bankruptcy",
            "liquidity crisis", "credit crunch", "silicon valley bank", "stress test", "stress testing",
            "capital requirements", "discount window", "banking stress", "contagion",
        ),
        direction_cues=(
            DirectionCue("STRESS_RISING", (
                "bank failure", "bank failures", "bank run", "bank collapse", "deposit outflows", "deposit flight",
                "spreads widen", "spreads widened", "contagion", "liquidity crisis", "credit crunch",
            )),
            DirectionCue("STRESS_EASING", ("spreads tighten", "spreads narrowed", "backstop", "rescue deal")),
        ),
        typical_horizon=ImpactHorizon.DAYS_TO_WEEKS,
        horizon_rationale="Credit stress propagates over days to weeks through funding markets and risk appetite.",
        exposures=(
            MarketExposure(_Q, _P, "Financials", "Bank equity is first-loss capital in a banking-stress episode.",
                           groups=("Financials",), pressure=(("STRESS_RISING", _DOWN), ("STRESS_EASING", _UP))),
            MarketExposure(_E, _P, "Financials & credit ETFs", "Direct bank-equity and corporate-credit exposure.",
                           symbols=("XLF", "HYG", "LQD"), pressure=(("STRESS_RISING", _DOWN), ("STRESS_EASING", _UP))),
            MarketExposure(_F, _S, "Treasury futures", "Flight-to-quality demand and an expected policy response.",
                           groups=("RATES",), pressure=(("STRESS_RISING", _UP),)),
            MarketExposure(_F, _S, "Equity index futures", "Risk-appetite channel.", groups=("EQUITY_INDEX",)),
            MarketExposure(_F, _T, "Gold futures", "Safe-haven demand.", symbols=("GC", "MGC")),
        ),
        translation_category=None,
    ),
    ChannelDefinition(
        channel=EconomicChannel.GEOPOLITICAL_RISK,
        rule_id="channel/geopolitical-risk/1",
        label="Geopolitical risk",
        description="Conflict, sanctions and blockades raise safe-haven demand and supply-risk premia.",
        source_categories=(),
        issuer_cues=(),
        lexicon=(
            "geopolitical", "military conflict", "armed conflict", "invasion", "invades", "missile", "missiles",
            "airstrike", "airstrikes", "sanctions", "escalation", "ceasefire", "blockade", "strait of hormuz", "troops",
        ),
        direction_cues=(
            DirectionCue("ESCALATION", (
                "escalation", "escalates", "invasion", "invades", "airstrike", "airstrikes", "missile strike",
                "blockade", "new sanctions",
            )),
            DirectionCue("DE_ESCALATION", ("ceasefire", "peace deal", "de escalation", "truce", "sanctions lifted")),
        ),
        typical_horizon=ImpactHorizon.DAYS_TO_WEEKS,
        horizon_rationale="Risk-off repricing is immediate; the supply-risk premium persists while the risk does.",
        exposures=(
            MarketExposure(_F, _P, "Gold futures", "Safe-haven demand rises with geopolitical risk.",
                           symbols=("GC", "MGC"), pressure=(("ESCALATION", _UP), ("DE_ESCALATION", _DOWN))),
            MarketExposure(_F, _S, "Crude & refined-product futures",
                           "Supply-risk premium, when producing regions or routes are involved.",
                           symbols=("CL", "MCL", "RB", "HO")),
            MarketExposure(_F, _S, "Treasury futures", "Flight-to-quality demand.", groups=("RATES",)),
            MarketExposure(_F, _S, "Equity index futures", "Risk-off repricing.", groups=("EQUITY_INDEX",)),
            MarketExposure(_E, _P, "Gold ETF", "Safe-haven demand.", symbols=("GLD",),
                           pressure=(("ESCALATION", _UP), ("DE_ESCALATION", _DOWN))),
            MarketExposure(_E, _S, "Treasury, broad equity & oil ETFs", "Flight to quality, risk-off, supply risk.",
                           symbols=("TLT", "SPY", "USO")),
            MarketExposure(_Q, _S, "Energy sector", "Supply-risk premium in realized prices.", groups=("Energy",)),
            MarketExposure(_Q, _T, "Industrials", "Supply-chain and defense-adjacent exposure.", groups=("Industrials",)),
        ),
        translation_category=None,
    ),
    ChannelDefinition(
        channel=EconomicChannel.TRADE_POLICY,
        rule_id="channel/trade-policy/1",
        label="Trade policy",
        description="Tariffs and export controls reprice globally exposed sectors, exporters' commodities and FX.",
        source_categories=(),
        issuer_cues=(),
        lexicon=(
            "tariff", "tariffs", "trade war", "export controls", "export restrictions", "import duties", "trade deal",
            "trade agreement", "retaliatory", "customs duties", "trade talks",
        ),
        direction_cues=(
            DirectionCue("RESTRICTION", (
                "new tariffs", "raises tariffs", "raised tariffs", "tariff increase", "export controls",
                "export restrictions", "retaliatory", "trade war",
            )),
            DirectionCue("LIBERALIZATION", ("trade deal", "tariff cut", "tariffs lifted", "tariff relief", "trade agreement")),
        ),
        typical_horizon=ImpactHorizon.DAYS_TO_WEEKS,
        horizon_rationale="Announcements reprice immediately; implementation and retaliation unfold over weeks.",
        exposures=(
            MarketExposure(_Q, _P, "Globally exposed sectors",
                           "Industrials, technology hardware and consumer discretionary carry the most cross-border "
                           "supply-chain and revenue exposure.",
                           groups=("Industrials", "Technology", "Consumer Discretionary")),
            MarketExposure(_F, _S, "Grain & oilseed futures", "Export-dependent crops are frequent retaliation targets.",
                           groups=("AGRICULTURE",)),
            MarketExposure(_F, _S, "FX futures", "Trade balances and policy retaliation move currencies.", groups=("FX",)),
            MarketExposure(_F, _S, "Equity index futures", "Broad earnings and risk-appetite repricing.",
                           groups=("EQUITY_INDEX",)),
            MarketExposure(_E, _S, "Broad & technology equity ETFs", "Broad earnings exposure.",
                           symbols=("SPY", "QQQ", "IWM", "XLK")),
        ),
        translation_category=None,
    ),
    ChannelDefinition(
        channel=EconomicChannel.DIGITAL_ASSETS,
        rule_id="channel/digital-assets/1",
        label="Digital assets",
        description="Crypto-specific regulation, flows and market-structure news.",
        source_categories=(),
        issuer_cues=(),
        lexicon=(
            "bitcoin", "btc", "ether", "ethereum", "crypto", "cryptocurrency", "cryptocurrencies", "stablecoin",
            "stablecoins", "digital asset", "digital assets",
        ),
        direction_cues=(),
        typical_horizon=ImpactHorizon.INTRADAY_TO_DAYS,
        horizon_rationale="Crypto markets trade continuously and reprice news within hours.",
        exposures=(
            MarketExposure(_C, _P, "Spot crypto assets", "Direct exposure."),
            MarketExposure(_F, _P, "CME crypto futures", "Cash-settled on the same underlying.", groups=("CRYPTO",)),
        ),
        translation_category=None,
    ),
)


def _validate_channels(channels: tuple[ChannelDefinition, ...]) -> None:
    """Self-contained structural checks at import (mirrors
    `product_catalog._validate_catalog`). Universe membership of every
    exposure is checked by the test suite against the live universes."""
    rule_ids = [c.rule_id for c in channels]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("duplicate channel rule_id")
    if {c.channel for c in channels} != set(EconomicChannel):
        raise ValueError("every EconomicChannel needs exactly one ChannelDefinition")
    valid_categories = {c.value for c in NewsCategory}
    for c in channels:
        if not set(c.source_categories) <= valid_categories:
            raise ValueError(f"{c.rule_id}: unknown source category")
        shifts = {d.shift for d in c.direction_cues}
        for e in c.exposures:
            for shift, _sign in e.pressure:
                if shift not in shifts:
                    raise ValueError(f"{c.rule_id}: pressure names undeclared shift {shift!r}")


_validate_channels(CHANNELS)

_BY_CHANNEL = {c.channel: c for c in CHANNELS}


def channel_definition(channel: EconomicChannel) -> ChannelDefinition:
    return _BY_CHANNEL[channel]

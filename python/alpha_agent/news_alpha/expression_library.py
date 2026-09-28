"""News Alpha Phase D -- the reviewed tables Asset Expression and measurement
resolution read.

The same kind of artifact as `news_alpha.channels`,
`news_alpha.transmission_library` and `news_alpha.signal_path_library`:
hand-reviewed, inspectable, deterministic, validated at import. Two tables:

* `EXPRESSION_RULES` -- for every economic state a signal path can end at,
  the ways that consequence could be EXPRESSED in a market: a domain, an
  expression concept ("AI accelerator designers", "copper futures"), its
  form (the instrument prices the quantity itself, is one exposed company,
  or a basket/index holding exposed names among others), the sign of the
  relation (the instrument's value moves WITH or AGAINST the consequence),
  and the domain symbols/groups that carry it. A rule may name NO symbol:
  the concept is tradable in principle (copper miners, HBM makers) but this
  platform's universe has no member -- reported, never filled with a
  nearest-looking ticker. Economic relevance only: the mandate is applied
  later, per expression, by `news_alpha.asset_expression`.
* `MEASUREMENT_TEMPLATES` -- the candidate measurement language of each
  domain (momentum, term structure, open interest, COT, revenue growth,
  flows, implied volatility, funding ...). A template names the IDEAL
  measurement and the data it requires; it never claims the data exists.
  Whether it does, from which dataset, field and window, and whether it was
  knowable at the time is RESOLVED against live repository state by
  `news_alpha.measurement` -- a template can only under-claim.

Options and Crypto templates exist so the architecture can describe those
expressions; the default rules name no Options/Crypto expression, because
the platform has no options universe and only synthetic crypto data, and no
seeded consequence is forced into them for symmetry.

The forms and signs are transmission judgments (like `EconomicSector`),
not an index methodology: "SECTOR_BASKET" says exposed names sit inside a
broader fund, not what weight they carry. No rule carries a weight, size,
return or trade.

FIDELITY (library /2, Phase D acceptance). Every rule also states HOW
DIRECTLY its instrument carries the consequence (`ExpressionFidelity`:
direct underlying, direct company, one segment of a company, constituent
of a basket, ecosystem proxy, macro proxy), so a copper future and a
basket that merely sits near the actors are never semantically the same
expression. It is a descriptive category -- never a score, rank or
expected-return strength, and nothing orders or filters by it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from alpha_agent.equities.universe import EQUITY_UNIVERSE
from alpha_agent.etf.universe import PILOT_UNIVERSE
from alpha_agent.marketdata.product_catalog import PRODUCT_CATALOG, AssetClass
from alpha_agent.news_alpha.channels import CHANNELS
from alpha_agent.news_alpha.mandate import MandateDomain
from alpha_agent.news_alpha.signal_path_library import EXPOSURE_BASIS
from alpha_agent.news_alpha.transmission import Polarity
from alpha_agent.news_alpha.transmission_library import SEEDED_CLAIMS, STATE_CATALOG

__all__ = [
    "DEFAULT_EXPRESSION_LIBRARY",
    "EXPRESSION_LIBRARY_VERSION",
    "EXPRESSION_RULES",
    "MEASUREMENT_TEMPLATES",
    "DataRequirement",
    "ExpressionFidelity",
    "ExpressionForm",
    "ExpressionLibrary",
    "ExpressionRule",
    "MeasurementRole",
    "MeasurementTemplate",
]

EXPRESSION_LIBRARY_VERSION = "asset-expression-library/2"


class ExpressionForm(str, Enum):
    #: The instrument prices the consequence's own quantity (copper futures
    #: for copper demand, Treasury futures for yields).
    UNDERLYING = "UNDERLYING"
    #: One company whose own economics the consequence moves.
    SINGLE_NAME = "SINGLE_NAME"
    #: A sector fund holding the exposed names among others.
    SECTOR_BASKET = "SECTOR_BASKET"
    #: A broad index or index fund where exposed names are a fraction.
    BROAD_INDEX = "BROAD_INDEX"


class ExpressionFidelity(str, Enum):
    """HOW DIRECTLY an instrument carries a consequence -- a categorical,
    descriptive label of the mapping, never a score, rank, weight or
    expected-return strength (nothing orders, filters or sizes by it).
    Distinct from `ExpressionForm` (what the instrument IS): a broad index
    can hold the exposed names as constituents, or only their ecosystem."""

    #: The instrument prices the consequence's own quantity (copper futures
    #: for copper demand; bond futures for yields).
    DIRECT_UNDERLYING = "DIRECT_UNDERLYING"
    #: A company whose own economics ARE the concept (NVDA for accelerator
    #: demand; the hyperscalers for their own free cash flow).
    DIRECT_COMPANY = "DIRECT_COMPANY"
    #: A company for which the consequence moves ONE segment among several
    #: (Caterpillar's power-generation equipment).
    SEGMENT_EXPOSURE = "SEGMENT_EXPOSURE"
    #: A fund or index that holds the directly exposed companies among
    #: other constituents (XLK holds the equipment makers).
    CONSTITUENT_EXPOSURE = "CONSTITUENT_EXPOSURE"
    #: The exposed actors are NOT inside; the instrument holds adjacent
    #: businesses whose fortunes are related (XLK/QQQ/NQ for leading-edge
    #: foundry utilization; a home-improvement retailer for housing activity).
    ECOSYSTEM_PROXY = "ECOSYSTEM_PROXY"
    #: The instrument tracks the consequence through a shared macro
    #: sensitivity, not through the actors (long-duration tech names or a
    #: bond-like sector for equity valuation multiples).
    MACRO_PROXY = "MACRO_PROXY"


#: Forms each company-level fidelity requires; the others fit any form.
_FIDELITY_FORMS: dict[ExpressionFidelity, tuple[ExpressionForm, ...]] = {
    ExpressionFidelity.DIRECT_COMPANY: (ExpressionForm.SINGLE_NAME,),
    ExpressionFidelity.SEGMENT_EXPOSURE: (ExpressionForm.SINGLE_NAME,),
    ExpressionFidelity.CONSTITUENT_EXPOSURE: (
        ExpressionForm.SECTOR_BASKET, ExpressionForm.BROAD_INDEX, ExpressionForm.UNDERLYING,
    ),
}


class MeasurementRole(str, Enum):
    """What aspect a measurement looks at -- how the market prices the
    expression, or whether the economics behind the consequence confirm it."""

    PRICE_RESPONSE = "PRICE_RESPONSE"
    ACTIVITY = "ACTIVITY"
    POSITIONING = "POSITIONING"
    CURVE_STRUCTURE = "CURVE_STRUCTURE"
    PHYSICAL_BALANCE = "PHYSICAL_BALANCE"
    FUNDAMENTAL = "FUNDAMENTAL"
    EXPECTATIONS = "EXPECTATIONS"
    BREADTH = "BREADTH"
    FACTOR_EXPOSURE = "FACTOR_EXPOSURE"
    IMPLIED_VOLATILITY = "IMPLIED_VOLATILITY"
    DERIVATIVES_POSITIONING = "DERIVATIVES_POSITIONING"
    ON_CHAIN = "ON_CHAIN"


class DataRequirement(str, Enum):
    """The data a measurement needs. `news_alpha.measurement` resolves each
    one against repository state -- one resolver per requirement."""

    MARKET_BARS = "MARKET_BARS"
    BENCHMARK_BARS = "BENCHMARK_BARS"
    CURVE_CONTRACTS = "CURVE_CONTRACTS"
    SPOT_REFERENCE = "SPOT_REFERENCE"
    OPEN_INTEREST = "OPEN_INTEREST"
    COT_REPORTS = "COT_REPORTS"
    PHYSICAL_INVENTORY = "PHYSICAL_INVENTORY"
    PIT_FUNDAMENTALS = "PIT_FUNDAMENTALS"
    CONSENSUS_HISTORY = "CONSENSUS_HISTORY"
    FUND_FLOWS = "FUND_FLOWS"
    CONSTITUENT_MEMBERSHIP = "CONSTITUENT_MEMBERSHIP"
    OPTION_CHAIN = "OPTION_CHAIN"
    CRYPTO_DERIVATIVES = "CRYPTO_DERIVATIVES"
    ON_CHAIN_METRICS = "ON_CHAIN_METRICS"


_ALL_FORMS = tuple(ExpressionForm)


@dataclass(frozen=True)
class MeasurementTemplate:
    measurement_id: str
    domain: MandateDomain
    label: str
    role: MeasurementRole
    requirement: DataRequirement
    #: The ideal economic measurement, with ``{instrument}`` filled in.
    meaning: str
    transform: str
    #: The source field the measurement reads, when it reads market bars.
    field: str | None = None
    #: Registered `FeatureRegistry` kinds the transform needs (checked live).
    feature_kinds: tuple[str, ...] = ()
    #: Needs more than one instrument's (or contract's) series at once --
    #: every registered feature kind computes over exactly one series.
    cross_instrument: bool = False
    forms: tuple[ExpressionForm, ...] = _ALL_FORMS
    #: True: applies to every expression of its domain and form. False: only
    #: where an `ExpressionRule` names it (fundamentals, inventories).
    generic: bool = True
    #: What the actually-available quantity stands in for, when it is not
    #: the ideal one (applied only when data is present).
    proxy: str | None = None


_F, _E, _Q = MandateDomain.FUTURES, MandateDomain.ETF, MandateDomain.EQUITY
_O, _C = MandateDomain.OPTIONS, MandateDomain.CRYPTO
_R, _D = MeasurementRole, DataRequirement
_UNDER, _NAME = ExpressionForm.UNDERLYING, ExpressionForm.SINGLE_NAME
_BASKET, _INDEX = ExpressionForm.SECTOR_BASKET, ExpressionForm.BROAD_INDEX

_MOMENTUM_KINDS = ("return", "trend_strength")
_ACTIVITY_KINDS = ("rel_volume", "volume_zscore")
_PRIMARY_VENUE_VOLUME = (
    "one primary-listing venue's volume stands in for whole-market volume (a single venue's prints; consolidated "
    "EQUS.SUMMARY volume covers only 2024-07-01 onward)"
)

MEASUREMENT_TEMPLATES: tuple[MeasurementTemplate, ...] = (
    # -- Futures ---------------------------------------------------------------
    MeasurementTemplate(
        "futures.momentum", _F, "Price momentum", _R.PRICE_RESPONSE, _D.MARKET_BARS,
        "How far {instrument}'s own price trend already reflects the consequence.",
        "Trailing return and trend strength on the roll-resolved research bars (signals may use back-adjusted "
        "prices; fills always resolve to the raw contract).",
        field="close", feature_kinds=_MOMENTUM_KINDS,
    ),
    MeasurementTemplate(
        "futures.activity", _F, "Trading activity", _R.ACTIVITY, _D.MARKET_BARS,
        "Whether trading in {instrument} is unusually active around the consequence.",
        "Volume relative to its trailing average, and its z-score.",
        field="volume", feature_kinds=_ACTIVITY_KINDS,
        proxy="the front contract's volume stands in for volume across all listed months",
    ),
    MeasurementTemplate(
        "futures.term_structure", _F, "Term structure & carry", _R.CURVE_STRUCTURE, _D.CURVE_CONTRACTS,
        "The slope of {instrument}'s futures curve (front vs next month) and the carry it implies.",
        "Front-minus-next price spread over simultaneous raw contracts, annualized by time between expiries "
        "(the typed math is `features.carry`; it is not a registered feature kind).",
        field="close", cross_instrument=True,
    ),
    MeasurementTemplate(
        "futures.open_interest", _F, "Open interest", _R.POSITIONING, _D.OPEN_INTEREST,
        "Whether positions in {instrument} are building or unwinding.",
        "Change in exchange-reported open interest.",
    ),
    MeasurementTemplate(
        "futures.cot", _F, "Trader positioning (COT)", _R.POSITIONING, _D.COT_REPORTS,
        "How commercial and speculative traders are positioned in {instrument}.",
        "Net positioning by trader category from the CFTC Commitments of Traders report.",
    ),
    MeasurementTemplate(
        "futures.basis", _F, "Spot-futures basis", _R.CURVE_STRUCTURE, _D.SPOT_REFERENCE,
        "{instrument}'s futures price against the physical spot price -- physical tightness.",
        "Futures minus a spot assessment of the same quantity.",
        forms=(_UNDER,), generic=False,
    ),
    MeasurementTemplate(
        "futures.physical_inventory", _F, "Physical inventories", _R.PHYSICAL_BALANCE, _D.PHYSICAL_INVENTORY,
        "Reported physical stocks of the commodity {instrument} prices -- the supply/demand balance itself.",
        "Change in reported inventories against their seasonal norm.",
        forms=(_UNDER,), generic=False,
    ),
    # -- ETF ------------------------------------------------------------------
    MeasurementTemplate(
        "etf.momentum", _E, "Price momentum", _R.PRICE_RESPONSE, _D.MARKET_BARS,
        "How far {instrument}'s own price trend already reflects the consequence.",
        "Trailing return and trend strength on daily primary-listing bars.",
        field="close", feature_kinds=_MOMENTUM_KINDS,
        proxy="price return stands in for total return: no distribution record is sourced for 2018-2024 "
              "(etf.corporate_actions)",
    ),
    MeasurementTemplate(
        "etf.relative_strength", _E, "Relative strength", _R.PRICE_RESPONSE, _D.BENCHMARK_BARS,
        "{instrument}'s return relative to the broad market (SPY): the sector's own repricing.",
        "Trailing return of the fund minus that of SPY, aligned backward in time.",
        field="close", cross_instrument=True, forms=(_BASKET,),
    ),
    MeasurementTemplate(
        "etf.activity", _E, "Trading activity", _R.ACTIVITY, _D.MARKET_BARS,
        "Whether trading in {instrument} is unusually active around the consequence.",
        "Volume relative to its trailing average, and its z-score.",
        field="volume", feature_kinds=_ACTIVITY_KINDS, proxy=_PRIMARY_VENUE_VOLUME,
    ),
    MeasurementTemplate(
        "etf.flows", _E, "Fund flows", _R.POSITIONING, _D.FUND_FLOWS,
        "Money entering or leaving {instrument} (creations and redemptions).",
        "Change in shares outstanding times net asset value.",
    ),
    MeasurementTemplate(
        "etf.breadth", _E, "Breadth", _R.BREADTH, _D.CONSTITUENT_MEMBERSHIP,
        "How many of {instrument}'s constituents participate in its move.",
        "Share of the fund's constituents above their own trailing average.",
        forms=(_BASKET, _INDEX),
    ),
    MeasurementTemplate(
        "etf.factor_exposure", _E, "Factor exposure", _R.FACTOR_EXPOSURE, _D.BENCHMARK_BARS,
        "How much of {instrument}'s move is broad-market beta rather than the sector's own story.",
        "Rolling return beta of the fund to SPY.",
        field="close", cross_instrument=True, forms=(_BASKET,),
    ),
    # -- Equity ---------------------------------------------------------------
    MeasurementTemplate(
        "equity.momentum", _Q, "Price momentum", _R.PRICE_RESPONSE, _D.MARKET_BARS,
        "How far {instrument}'s own price trend already reflects the consequence.",
        "Trailing return and trend strength on daily primary-listing bars.",
        field="close", feature_kinds=_MOMENTUM_KINDS,
        proxy="unadjusted price return stands in for total return until splits and distributions are sourced "
              "(equities.corporate_actions)",
    ),
    MeasurementTemplate(
        "equity.relative_strength", _Q, "Relative strength", _R.PRICE_RESPONSE, _D.BENCHMARK_BARS,
        "{instrument}'s return relative to the broad market (SPY).",
        "Trailing return of the stock minus that of SPY, aligned backward in time.",
        field="close", cross_instrument=True,
    ),
    MeasurementTemplate(
        "equity.activity", _Q, "Trading activity", _R.ACTIVITY, _D.MARKET_BARS,
        "Whether trading in {instrument} is unusually active around the consequence.",
        "Volume relative to its trailing average, and its z-score.",
        field="volume", feature_kinds=_ACTIVITY_KINDS, proxy=_PRIMARY_VENUE_VOLUME,
    ),
    MeasurementTemplate(
        "equity.estimate_revisions", _Q, "Estimate revisions", _R.EXPECTATIONS, _D.CONSENSUS_HISTORY,
        "Whether analysts are raising or cutting {instrument}'s expected earnings.",
        "Change in consensus forward estimates, as each stood at the time.",
    ),
    MeasurementTemplate(
        "equity.revenue_growth", _Q, "Revenue growth", _R.FUNDAMENTAL, _D.PIT_FUNDAMENTALS,
        "Whether {instrument}'s reported revenue confirms the consequence.",
        "Year-over-year growth of reported quarterly revenue.",
        generic=False,
        proxy="company-wide revenue stands in for the segment the consequence moves (segment disclosures are "
              "not standardized)",
    ),
    MeasurementTemplate(
        "equity.gross_margin", _Q, "Gross margin", _R.FUNDAMENTAL, _D.PIT_FUNDAMENTALS,
        "Whether {instrument} is gaining pricing power from the consequence.",
        "Reported gross profit over revenue, quarter by quarter.",
        generic=False,
    ),
    MeasurementTemplate(
        "equity.inventory_growth", _Q, "Inventory growth", _R.FUNDAMENTAL, _D.PIT_FUNDAMENTALS,
        "Whether {instrument}'s inventories are building faster than sales (a demand-slowdown warning).",
        "Year-over-year growth of reported inventories against revenue growth.",
        generic=False,
    ),
    MeasurementTemplate(
        "equity.capex", _Q, "Capital expenditure", _R.FUNDAMENTAL, _D.PIT_FUNDAMENTALS,
        "How much {instrument} is actually spending on capacity.",
        "Reported capital expenditure, year over year.",
        generic=False,
    ),
    MeasurementTemplate(
        "equity.free_cash_flow", _Q, "Free cash flow", _R.FUNDAMENTAL, _D.PIT_FUNDAMENTALS,
        "Whether {instrument}'s spending is eating into its cash generation.",
        "Operating cash flow minus capital expenditure, year over year.",
        generic=False,
    ),
    MeasurementTemplate(
        "equity.net_interest_margin", _Q, "Net interest margin", _R.FUNDAMENTAL, _D.PIT_FUNDAMENTALS,
        "Whether {instrument}'s lending margin moves with the rate path.",
        "Reported net interest income over average earning assets.",
        generic=False,
    ),
    # -- Options (no options universe on this platform) -------------------------
    MeasurementTemplate(
        "options.atm_iv", _O, "At-the-money implied volatility", _R.IMPLIED_VOLATILITY, _D.OPTION_CHAIN,
        "How much movement the options market prices for {instrument}.",
        "Implied volatility of at-the-money options at a fixed tenor.",
    ),
    MeasurementTemplate(
        "options.skew", _O, "Skew", _R.IMPLIED_VOLATILITY, _D.OPTION_CHAIN,
        "Whether the options market prices downside or upside moves in {instrument} more richly.",
        "Implied volatility of out-of-the-money puts minus calls at a fixed delta.",
    ),
    MeasurementTemplate(
        "options.vol_term_structure", _O, "Volatility term structure", _R.IMPLIED_VOLATILITY, _D.OPTION_CHAIN,
        "When the options market expects {instrument}'s movement to arrive.",
        "At-the-money implied volatility across expiries.",
    ),
    MeasurementTemplate(
        "options.expected_move", _O, "Expected move", _R.IMPLIED_VOLATILITY, _D.OPTION_CHAIN,
        "The size of move priced into {instrument} over a window.",
        "At-the-money straddle price over the underlying price.",
    ),
    # -- Crypto (synthetic scaffold only) -------------------------------------------
    MeasurementTemplate(
        "crypto.funding", _C, "Perpetual funding", _R.DERIVATIVES_POSITIONING, _D.CRYPTO_DERIVATIVES,
        "Whether leveraged traders pay to hold long or short {instrument}.",
        "Perpetual-swap funding rate.",
    ),
    MeasurementTemplate(
        "crypto.basis", _C, "Futures basis", _R.CURVE_STRUCTURE, _D.CRYPTO_DERIVATIVES,
        "{instrument}'s futures premium over spot.",
        "Annualized futures-minus-spot premium.",
    ),
    MeasurementTemplate(
        "crypto.open_interest", _C, "Open interest", _R.DERIVATIVES_POSITIONING, _D.CRYPTO_DERIVATIVES,
        "Whether leveraged positions in {instrument} are building or unwinding.",
        "Change in derivatives open interest.",
    ),
    MeasurementTemplate(
        "crypto.liquidations", _C, "Liquidations", _R.DERIVATIVES_POSITIONING, _D.CRYPTO_DERIVATIVES,
        "Forced deleveraging in {instrument}.",
        "Liquidated notional by side.",
    ),
    MeasurementTemplate(
        "crypto.on_chain", _C, "On-chain activity", _R.ON_CHAIN, _D.ON_CHAIN_METRICS,
        "Network usage and holder behaviour for {instrument}.",
        "Active addresses, exchange flows and realized-value ratios.",
    ),
)


@dataclass(frozen=True)
class ExpressionRule:
    """One way a consequence could be expressed in a domain. `relation` is
    the sign of the instrument's value against the consequence (POSITIVE:
    it rises when the consequence rises); AMBIGUOUS when the same move helps
    and hurts the concept (capacity spending is a supplier's revenue and the
    spender's cost)."""

    state_id: str
    domain: MandateDomain
    concept: str
    form: ExpressionForm
    relation: Polarity
    rationale: str
    #: Explicit domain symbols (Futures roots, ETF tickers, Equity tickers).
    symbols: tuple[str, ...] = ()
    #: Domain-native groups -- Futures `AssetClass` values only; single names
    #: are always listed explicitly.
    groups: tuple[str, ...] = ()
    #: Concept-specific (non-generic) measurement template ids.
    measures: tuple[str, ...] = ()
    #: How directly the instrument carries the consequence -- required
    #: (checked at import); descriptive only.
    fidelity: ExpressionFidelity | None = None

    @property
    def rule_id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.concept.lower()).strip("-")
        return f"expression/{self.state_id}/{self.domain.value.lower()}/{slug}"


_POS, _NEG, _AMB = Polarity.POSITIVE, Polarity.NEGATIVE, Polarity.AMBIGUOUS
_REV, _MARGIN, _INV = "equity.revenue_growth", "equity.gross_margin", "equity.inventory_growth"
_CAPEX, _FCF, _NIM = "equity.capex", "equity.free_cash_flow", "equity.net_interest_margin"
_BASIS, _STOCKS = "futures.basis", "futures.physical_inventory"
_HYPERSCALERS = ("MSFT", "GOOGL", "AMZN", "META")
_UNDERLYING_F, _COMPANY = ExpressionFidelity.DIRECT_UNDERLYING, ExpressionFidelity.DIRECT_COMPANY
_SEGMENT, _CONSTITUENT = ExpressionFidelity.SEGMENT_EXPOSURE, ExpressionFidelity.CONSTITUENT_EXPOSURE
_ECOSYSTEM, _MACRO = ExpressionFidelity.ECOSYSTEM_PROXY, ExpressionFidelity.MACRO_PROXY


def _tech_baskets(state_id: str, why: str, fidelity: ExpressionFidelity) -> tuple[ExpressionRule, ...]:
    """The fund/index expressions of an AI spender or supplier state -- the
    same instruments Phase C's `EXPOSURE_BASIS` rests the Technology &
    growth ETF and Nasdaq-100 exposures on. `fidelity` says whether the
    exposed companies are inside (CONSTITUENT_EXPOSURE) or only their
    ecosystem is (ECOSYSTEM_PROXY)."""
    return (
        ExpressionRule(state_id, _E, "Technology-sector fund", _BASKET, _POS,
                       f"{why} The fund holds the sector's hardware, software and semiconductor names.",
                       symbols=("XLK",), fidelity=fidelity),
        ExpressionRule(state_id, _E, "Nasdaq-100 fund", _INDEX, _POS,
                       f"{why} A broad growth index where these names are a large share.", symbols=("QQQ",),
                       fidelity=fidelity),
        ExpressionRule(state_id, _F, "Nasdaq-100 index futures", _INDEX, _POS,
                       f"{why} A broad growth index where these names are a large share.", symbols=("NQ", "MNQ"),
                       fidelity=fidelity),
    )


EXPRESSION_RULES: tuple[ExpressionRule, ...] = (
    # -- AI infrastructure -------------------------------------------------------
    ExpressionRule("compute_demand", _Q, "AI accelerator designers", _NAME, _POS,
                   "Accelerator designers sell the compute hardware this demand buys.",
                   symbols=("NVDA",), measures=(_REV,), fidelity=_COMPANY),
    ExpressionRule("compute_demand", _Q, "Cloud platforms", _NAME, _POS,
                   "Cloud platforms rent out the compute this demand buys -- one segment of each company beside "
                   "software, advertising or retail.", symbols=("MSFT", "AMZN", "GOOGL"), measures=(_REV,),
                   fidelity=ExpressionFidelity.SEGMENT_EXPOSURE),
    ExpressionRule("accelerator_demand", _Q, "AI accelerator designers", _NAME, _POS,
                   "Accelerator demand is these designers' own unit demand.",
                   symbols=("NVDA",), measures=(_REV, _MARGIN, _INV), fidelity=_COMPANY),
    *_tech_baskets("accelerator_demand", "Accelerator designers sit inside.", _CONSTITUENT),
    ExpressionRule("hbm_demand", _Q, "High-bandwidth-memory makers", _NAME, _POS,
                   "HBM demand is the memory makers' own demand; none is in the declared equity universe.",
                   measures=(_REV, _MARGIN, _INV), fidelity=_COMPANY),
    *_tech_baskets("hbm_demand", "Memory makers sit inside some of these baskets.", _CONSTITUENT),
    ExpressionRule("advanced_foundry_utilization", _Q, "Leading-edge foundries and packagers", _NAME, _POS,
                   "Utilization drives the foundries' own margins; none is in the declared equity universe.",
                   measures=(_REV, _MARGIN), fidelity=_COMPANY),
    *_tech_baskets("advanced_foundry_utilization",
                   "The foundries themselves are not inside; the semiconductor ecosystem around them is.", _ECOSYSTEM),
    ExpressionRule("foundry_capacity_expansion", _Q, "Foundry and memory capacity spenders", _NAME, _AMB,
                   "Capacity spending is the spender's cost now and its capacity later; none is in the declared "
                   "equity universe.", measures=(_CAPEX, _FCF), fidelity=_COMPANY),
    ExpressionRule("semiconductor_equipment_demand", _Q, "Semiconductor-equipment makers", _NAME, _POS,
                   "Equipment demand is these makers' own order book; none is in the declared equity universe.",
                   measures=(_REV, _MARGIN), fidelity=_COMPANY),
    *_tech_baskets("semiconductor_equipment_demand", "Equipment makers sit inside.", _CONSTITUENT),
    ExpressionRule("data_center_construction", _Q, "Data-center developers and REITs", _NAME, _POS,
                   "Construction volume is these developers' own growth; none is in the declared equity universe.",
                   measures=(_REV, _CAPEX), fidelity=_COMPANY),
    ExpressionRule("electricity_demand", _E, "Utilities-sector fund", _BASKET, _POS,
                   "Load growth raises generators' and regulated utilities' earnings base.", symbols=("XLU",), fidelity=_CONSTITUENT),
    ExpressionRule("electricity_demand", _Q, "Power generators and utilities", _NAME, _POS,
                   "Load growth is these companies' own volume; none is in the declared equity universe.",
                   measures=(_REV,), fidelity=_COMPANY),
    ExpressionRule("grid_investment", _E, "Utilities-sector fund", _BASKET, _POS,
                   "Regulated utilities earn a return on the grid assets they build.", symbols=("XLU",), fidelity=_CONSTITUENT),
    ExpressionRule("grid_investment", _Q, "Grid utilities", _NAME, _POS,
                   "Grid build-out grows these utilities' rate base; none is in the declared equity universe.",
                   measures=(_CAPEX,), fidelity=_COMPANY),
    ExpressionRule("power_equipment_demand", _Q, "Power-generation equipment makers", _NAME, _POS,
                   "Caterpillar's power-generation equipment (engines, generator sets, turbines) is one segment; "
                   "construction and mining machinery dominate the rest.", symbols=("CAT",), measures=(_REV,), fidelity=_SEGMENT),
    ExpressionRule("gas_power_generation_demand", _F, "Natural gas futures", _UNDER, _POS,
                   "Power-burn demand is part of the gas balance these futures price.", symbols=("NG",),
                   measures=(_BASIS, _STOCKS), fidelity=_UNDERLYING_F),
    ExpressionRule("gas_power_generation_demand", _Q, "Integrated oil and gas producers", _NAME, _POS,
                   "Both produce gas, a minority of their output.", symbols=("XOM", "CVX"), measures=(_REV,), fidelity=_SEGMENT),
    ExpressionRule("copper_demand", _F, "Copper futures", _UNDER, _POS,
                   "Copper demand is part of the balance these futures price.", symbols=("HG",),
                   measures=(_BASIS, _STOCKS), fidelity=_UNDERLYING_F),
    ExpressionRule("copper_demand", _Q, "Copper miners", _NAME, _POS,
                   "Copper demand is these miners' own revenue; none is in the declared equity universe.",
                   measures=(_REV,), fidelity=_COMPANY),
    ExpressionRule("copper_demand", _E, "Copper-miner funds", _BASKET, _POS,
                   "Funds of copper miners; none is in the ETF pilot universe.", fidelity=_CONSTITUENT),
    ExpressionRule("ai_spender_free_cash_flow", _Q, "AI infrastructure spenders", _NAME, _POS,
                   "The hyperscalers whose own cash generation this is.", symbols=_HYPERSCALERS,
                   measures=(_CAPEX, _FCF), fidelity=_COMPANY),
    *_tech_baskets("ai_spender_free_cash_flow", "The spenders sit inside (Alphabet and Meta only in the index).", _CONSTITUENT),
    ExpressionRule("ai_services_revenue", _Q, "AI infrastructure spenders", _NAME, _POS,
                   "The hyperscalers whose own services revenue this is.", symbols=_HYPERSCALERS, measures=(_REV,), fidelity=_SEGMENT),
    *_tech_baskets("ai_services_revenue", "The spenders sit inside (Alphabet and Meta only in the index).", _CONSTITUENT),
    # -- crude oil --------------------------------------------------------------------
    ExpressionRule("crude_oil_supply", _F, "Crude oil futures", _UNDER, _NEG,
                   "More supply loosens the balance these futures price.", symbols=("CL", "MCL"),
                   measures=(_STOCKS,), fidelity=_UNDERLYING_F),
    ExpressionRule("crude_oil_supply", _E, "Crude oil fund", _UNDER, _NEG,
                   "The fund holds near-dated crude futures.", symbols=("USO",), fidelity=_UNDERLYING_F),
    ExpressionRule("crude_oil_price", _F, "Crude oil futures", _UNDER, _POS,
                   "These futures price crude directly.", symbols=("CL", "MCL"), measures=(_BASIS, _STOCKS), fidelity=_UNDERLYING_F),
    ExpressionRule("crude_oil_price", _E, "Crude oil fund", _UNDER, _POS,
                   "The fund holds near-dated crude futures.", symbols=("USO",), fidelity=_UNDERLYING_F),
    ExpressionRule("refined_product_prices", _F, "Gasoline and diesel futures", _UNDER, _POS,
                   "These futures price the refined products directly.", symbols=("RB", "HO"),
                   measures=(_STOCKS,), fidelity=_UNDERLYING_F),
    ExpressionRule("oil_producer_revenue", _Q, "Integrated oil producers", _NAME, _POS,
                   "Their revenue is realized crude and product prices times volume.", symbols=("XOM", "CVX"),
                   measures=(_REV,), fidelity=_COMPANY),
    ExpressionRule("oil_producer_revenue", _E, "Energy-sector fund", _BASKET, _POS,
                   "The fund holds the producers and their service companies.", symbols=("XLE",), fidelity=_CONSTITUENT),
    ExpressionRule("transport_fuel_costs", _Q, "Fuel-intensive transport companies", _NAME, _NEG,
                   "Fuel is a large cost for airlines and freight carriers; none is in the declared equity "
                   "universe.", measures=(_MARGIN,), fidelity=_COMPANY),
    ExpressionRule("headline_inflation", _E, "Inflation-protected Treasury funds", _UNDER, _POS,
                   "TIPS principal is indexed to consumer prices; no TIPS fund is in the ETF pilot universe.", fidelity=_UNDERLYING_F),
    ExpressionRule("shale_drilling_activity", _E, "Energy-sector fund", _BASKET, _POS,
                   "The fund holds shale producers and oilfield-service companies.", symbols=("XLE",), fidelity=_CONSTITUENT),
    ExpressionRule("shale_drilling_activity", _Q, "Oilfield-service companies", _NAME, _POS,
                   "Drilling activity is their order book; none is in the declared equity universe.",
                   measures=(_REV,), fidelity=_COMPANY),
    # -- monetary policy ----------------------------------------------------------------
    ExpressionRule("policy_rate_path", _F, "2-year Treasury note futures", _UNDER, _NEG,
                   "Short-dated note prices fall as the expected policy path rises.", symbols=("ZT",), fidelity=_UNDERLYING_F),
    ExpressionRule("policy_rate_path", _E, "Short-term Treasury fund", _UNDER, _NEG,
                   "Short-dated Treasury prices fall as the expected policy path rises.", symbols=("SHY",), fidelity=_UNDERLYING_F),
    ExpressionRule("treasury_yields", _F, "Treasury futures", _UNDER, _NEG,
                   "Bond futures prices move inversely to yields.", groups=(AssetClass.RATES.value,), fidelity=_UNDERLYING_F),
    ExpressionRule("treasury_yields", _E, "Treasury funds", _UNDER, _NEG,
                   "Bond-fund prices move inversely to yields.", symbols=("SHY", "IEF", "TLT"), fidelity=_UNDERLYING_F),
    ExpressionRule("equity_valuation_multiples", _F, "Equity index futures", _INDEX, _POS,
                   "Index prices are earnings times the multiple.", groups=(AssetClass.EQUITY_INDEX.value,), fidelity=_UNDERLYING_F),
    ExpressionRule("equity_valuation_multiples", _E, "Broad equity funds", _INDEX, _POS,
                   "Index prices are earnings times the multiple.", symbols=("SPY", "QQQ", "IWM"), fidelity=_UNDERLYING_F),
    ExpressionRule("equity_valuation_multiples", _E, "Utilities-sector fund", _BASKET, _POS,
                   "A bond-like, rate-sensitive sector whose valuation tracks the discount rate.", symbols=("XLU",), fidelity=_MACRO),
    ExpressionRule("equity_valuation_multiples", _Q, "Long-duration technology companies", _NAME, _POS,
                   "Earnings far in the future make their valuations the most rate-sensitive; these are the "
                   "universe's Technology names.", symbols=("AAPL", "MSFT", "NVDA"), fidelity=_MACRO),
    ExpressionRule("usd_exchange_value", _F, "Currency futures", _UNDER, _NEG,
                   "Quoted in US dollars per unit of foreign currency: a stronger dollar lowers these prices.",
                   groups=(AssetClass.FX.value,), fidelity=_UNDERLYING_F),
    ExpressionRule("gold_investment_demand", _F, "Gold futures", _UNDER, _POS,
                   "Investment demand is part of the balance these futures price.", symbols=("GC", "MGC"),
                   measures=(_STOCKS,), fidelity=_UNDERLYING_F),
    ExpressionRule("gold_investment_demand", _E, "Gold fund", _UNDER, _POS,
                   "The fund holds physical gold.", symbols=("GLD",), fidelity=_UNDERLYING_F),
    ExpressionRule("borrowing_costs", _E, "Corporate-bond funds", _UNDER, _NEG,
                   "Bond prices fall as borrowers' yields rise.", symbols=("LQD", "HYG"), fidelity=_UNDERLYING_F),
    ExpressionRule("housing_activity", _Q, "Home-improvement retailer", _NAME, _POS,
                   "Home-improvement spending follows housing turnover.", symbols=("HD",), measures=(_REV,), fidelity=_ECOSYSTEM),
    ExpressionRule("housing_activity", _Q, "Homebuilders", _NAME, _POS,
                   "Housing activity is their own volume; none is in the declared equity universe.",
                   measures=(_REV,), fidelity=_COMPANY),
    ExpressionRule("bank_net_interest_margin", _Q, "Large banks", _NAME, _POS,
                   "Net interest margin is a core driver of their earnings.", symbols=("JPM", "BAC"),
                   measures=(_NIM,), fidelity=_COMPANY),
    ExpressionRule("bank_net_interest_margin", _E, "Financials-sector fund", _BASKET, _POS,
                   "Banks are a large share of the fund.", symbols=("XLF",), fidelity=_CONSTITUENT),
    ExpressionRule("credit_losses", _Q, "Large banks", _NAME, _NEG,
                   "Credit losses come straight out of their earnings.", symbols=("JPM", "BAC"), fidelity=_COMPANY),
    ExpressionRule("credit_losses", _E, "Financials-sector fund", _BASKET, _NEG,
                   "Banks are a large share of the fund.", symbols=("XLF",), fidelity=_CONSTITUENT),
    ExpressionRule("credit_losses", _E, "High-yield bond fund", _UNDER, _NEG,
                   "High-yield bond prices fall as expected defaults rise.", symbols=("HYG",), fidelity=_CONSTITUENT),
    ExpressionRule("bank_earnings", _Q, "Large banks", _NAME, _POS,
                   "Their own earnings.", symbols=("JPM", "BAC"), measures=(_NIM,), fidelity=_COMPANY),
    ExpressionRule("bank_earnings", _E, "Financials-sector fund", _BASKET, _POS,
                   "Banks are a large share of the fund.", symbols=("XLF",), fidelity=_CONSTITUENT),
)


@dataclass(frozen=True)
class ExpressionLibrary:
    rules: tuple[ExpressionRule, ...]
    templates: tuple[MeasurementTemplate, ...]

    def rules_for(self, state_id: str) -> tuple[ExpressionRule, ...]:
        return tuple(r for r in self.rules if r.state_id == state_id)

    def template(self, measurement_id: str) -> MeasurementTemplate:
        return next(t for t in self.templates if t.measurement_id == measurement_id)

    def templates_for(self, rule: ExpressionRule) -> tuple[MeasurementTemplate, ...]:
        """The generic templates of the rule's domain and form, then the
        rule's own concept-specific ones, in declaration order."""
        named = set(rule.measures)
        return tuple(
            t for t in self.templates
            if t.domain is rule.domain and rule.form in t.forms and (t.generic or t.measurement_id in named)
        )


_DOMAIN_SYMBOLS: dict[MandateDomain, set[str]] = {
    _F: {e.root_symbol for e in PRODUCT_CATALOG},
    _E: set(PILOT_UNIVERSE),
    _Q: set(EQUITY_UNIVERSE),
}


def _validate(library: ExpressionLibrary) -> None:
    """Structural checks at import: every state a seeded claim can reach has
    at least one rule; every symbol exists in its domain's own universe;
    rules name only real, applicable templates; and Phase C's exposure
    bridge and these rules agree (each ETF/Futures exposure's instruments are
    expressed by at least one state its basis rests on)."""
    ids = [t.measurement_id for t in library.templates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate measurement template id")
    for t in library.templates:
        if not t.measurement_id.startswith(t.domain.value.lower() + "."):
            raise ValueError(f"{t.measurement_id}: template id must start with its domain")
    catalog = {s.state_id for s in STATE_CATALOG}
    reachable = {c.target for c in SEEDED_CLAIMS}
    ruled = {r.state_id for r in library.rules}
    if missing := reachable - ruled:
        raise ValueError(f"states a signal path can end at have no expression rule: {sorted(missing)}")
    if extra := ruled - catalog:
        raise ValueError(f"expression rules name states outside the catalog: {sorted(extra)}")
    by_id = {t.measurement_id: t for t in library.templates}
    groups = {c.value for c in AssetClass}
    for r in library.rules:
        if r.fidelity is None:
            raise ValueError(f"{r.rule_id}: every rule states how directly it carries the consequence (fidelity)")
        if r.form not in _FIDELITY_FORMS.get(r.fidelity, tuple(ExpressionForm)):
            raise ValueError(f"{r.rule_id}: fidelity {r.fidelity.value} does not fit form {r.form.value}")
        if r.relation not in (_POS, _NEG, _AMB):
            raise ValueError(f"{r.rule_id}: relation must be POSITIVE, NEGATIVE or AMBIGUOUS")
        if r.symbols and r.domain in _DOMAIN_SYMBOLS and not set(r.symbols) <= _DOMAIN_SYMBOLS[r.domain]:
            raise ValueError(f"{r.rule_id}: symbols outside the {r.domain.value} universe")
        if r.groups and (r.domain is not _F or not set(r.groups) <= groups):
            raise ValueError(f"{r.rule_id}: groups are Futures asset classes only")
        for m in r.measures:
            t = by_id.get(m)
            if t is None or t.generic or t.domain is not r.domain or r.form not in t.forms:
                raise ValueError(f"{r.rule_id}: {m} is not a concept-specific template for this domain and form")
    exposures = {(c.channel, e.label): e for c in CHANNELS for e in c.exposures}
    for b in EXPOSURE_BASIS:
        e = exposures[(b.channel, b.exposure_label)]
        if e.domain not in (_F, _E):
            continue
        rules = [r for r in library.rules if r.state_id in b.state_ids and r.domain is e.domain]
        if not set(e.symbols) <= {s for r in rules for s in r.symbols} or not set(e.groups) <= {
            g for r in rules for g in r.groups
        }:
            raise ValueError(f"exposure {e.label!r} is not expressed by any state its basis rests on")


DEFAULT_EXPRESSION_LIBRARY = ExpressionLibrary(rules=EXPRESSION_RULES, templates=MEASUREMENT_TEMPLATES)
_validate(DEFAULT_EXPRESSION_LIBRARY)

"""News Alpha Phase C -- the reviewed tables Signal Path Discovery reads.

The same kind of artifact as `news_alpha.channels` and
`news_alpha.transmission_library`: hand-reviewed, inspectable, deterministic,
validated at import. Three tables, each a DATA change when a new seed lands --
never a code path:

* `STATE_PROFILES` -- for every economic state in the Phase B catalog, the
  `EconomicSector` it belongs to and its CANDIDATE TARGET CONCEPT: the
  economic actor or quantity whose fundamentals the state moves ("HBM memory
  makers", "copper as an industrial metal"). A concept, never an instrument:
  no ticker, root, contract or trade -- mapping a concept onto tradable
  assets is Asset Expression, a later stage.
* `CHANNEL_ROLES` -- whether a `TransmissionChannel` moves the effect to a new
  COUNTERPARTY along a value chain (SUPPLIER: the target supplies an input or
  capacity to the source; CUSTOMER: the target buys the source's output and
  pays its price) or stays with the same party / the same market (no role).
  ``DEMAND_PULL`` has no role: it is the event's own spending re-expressed as
  demand, not a step to a new counterparty.
* `EXPOSURE_BASIS` -- for every Phase A `MarketExposure` of a SEEDED channel,
  the economic states the exposure's own rationale rests on (the copper
  futures exposure rests on copper demand). This is the bridge the
  mechanism-adjusted impact pass uses to revise the initial triage; it never
  maps a state onto an instrument and never carries a direction or a trade.

`EconomicSector` IS AN INTERNAL TRANSMISSION TAXONOMY. It exists only to
answer one question -- does an effect leave the economic sphere where it
first landed? -- and is NOT GICS, NAICS, SIC or any industry standard, NOT the
Equity universe's sector groups, and NOT an Asset Expression mapping: no
sector here selects, weights or names a tradable instrument. Its boundaries
are transmission judgments (semiconductors, hyperscalers and data-center
construction are one technology complex, which is what makes accelerator ->
HBM -> foundry a supply chain rather than a sector crossing; the rate path
and yields are a "monetary and rates" sphere that no classification standard
has). Change it only as a reviewed data edit, with the path-type tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from alpha_agent.news_alpha.channels import CHANNELS, EconomicChannel
from alpha_agent.news_alpha.transmission import TransmissionChannel
from alpha_agent.news_alpha.transmission_library import ANCHOR_RULES, STATE_CATALOG

__all__ = [
    "CHANNEL_ROLES",
    "DEFAULT_PATH_LIBRARY",
    "EXPOSURE_BASIS",
    "SIGNAL_PATH_LIBRARY_VERSION",
    "STATE_PROFILES",
    "EconomicSector",
    "ExposureBasis",
    "SignalPathLibrary",
    "StateProfile",
    "ValueChainRole",
]

SIGNAL_PATH_LIBRARY_VERSION = "signal-path-library/1"


class EconomicSector(str, Enum):
    TECHNOLOGY = "TECHNOLOGY"
    UTILITIES = "UTILITIES"
    ENERGY = "ENERGY"
    MATERIALS = "MATERIALS"
    INDUSTRIALS = "INDUSTRIALS"
    FINANCIALS = "FINANCIALS"
    REAL_ESTATE = "REAL_ESTATE"
    MONETARY_AND_RATES = "MONETARY_AND_RATES"
    CURRENCIES = "CURRENCIES"
    EQUITY_MARKETS = "EQUITY_MARKETS"
    MACROECONOMY = "MACROECONOMY"


class ValueChainRole(str, Enum):
    #: The target supplies an input or capacity to the source.
    SUPPLIER = "SUPPLIER"
    #: The target buys the source's output and pays its price.
    CUSTOMER = "CUSTOMER"


@dataclass(frozen=True)
class StateProfile:
    state_id: str
    sector: EconomicSector
    target_concept: str


@dataclass(frozen=True)
class ExposureBasis:
    """The economic states one Phase A exposure rests on. Keyed by the
    exposure's channel and label, exactly as `news_alpha.channels` declares
    them (checked at import)."""

    channel: EconomicChannel
    exposure_label: str
    state_ids: tuple[str, ...]


_T, _U, _E = EconomicSector.TECHNOLOGY, EconomicSector.UTILITIES, EconomicSector.ENERGY
_M, _I, _F = EconomicSector.MATERIALS, EconomicSector.INDUSTRIALS, EconomicSector.FINANCIALS
_R = EconomicSector.MONETARY_AND_RATES

STATE_PROFILES: tuple[StateProfile, ...] = (
    # -- AI infrastructure ---------------------------------------------------
    StateProfile("ai_infrastructure_investment", _T, "AI infrastructure spenders (their capital budgets)"),
    StateProfile("compute_demand", _T, "AI compute capacity (training and inference)"),
    StateProfile("accelerator_demand", _T, "AI accelerator designers (GPUs and custom ASICs)"),
    StateProfile("hbm_demand", _T, "High-bandwidth-memory makers"),
    StateProfile("advanced_foundry_utilization", _T, "Leading-edge foundries and advanced packagers"),
    StateProfile("foundry_capacity_expansion", _T, "Foundries' and memory makers' capacity spending"),
    StateProfile("semiconductor_equipment_demand", _T, "Semiconductor-equipment makers"),
    StateProfile("data_center_construction", _T, "Data-center developers and builders"),
    StateProfile("electricity_demand", _U, "Power generators and utilities (load growth)"),
    StateProfile("grid_investment", _U, "Grid utilities (transmission and distribution build-out)"),
    StateProfile("power_equipment_demand", _I, "Electrical-equipment makers"),
    StateProfile("gas_power_generation_demand", _E, "Natural gas (power-burn demand)"),
    StateProfile("copper_demand", _M, "Copper (industrial-metal demand)"),
    StateProfile("ai_spender_free_cash_flow", _T, "AI spenders (cash generation)"),
    StateProfile("ai_services_revenue", _T, "AI spenders (AI and cloud services revenue)"),
    # -- crude oil -------------------------------------------------------------
    StateProfile("crude_oil_supply", _E, "Crude oil (physical supply balance)"),
    StateProfile("crude_oil_price", _E, "Crude oil (price level)"),
    StateProfile("refined_product_prices", _E, "Refined products (gasoline, diesel, jet fuel)"),
    StateProfile("oil_producer_revenue", _E, "Oil producers (realized revenue)"),
    StateProfile("transport_fuel_costs", _I, "Fuel-intensive transport and industrial users"),
    StateProfile("headline_inflation", EconomicSector.MACROECONOMY, "Headline consumer prices"),
    StateProfile("shale_drilling_activity", _E, "Shale producers and oilfield services"),
    # -- monetary policy -------------------------------------------------------
    StateProfile("policy_rate_path", _R, "Short-rate expectations"),
    StateProfile("treasury_yields", _R, "Government bond yields"),
    StateProfile("equity_valuation_multiples", EconomicSector.EQUITY_MARKETS,
                 "Equity valuations (long-duration earnings most)"),
    StateProfile("usd_exchange_value", EconomicSector.CURRENCIES, "US dollar exchange value"),
    StateProfile("gold_investment_demand", _M, "Gold (investment demand)"),
    StateProfile("borrowing_costs", _R, "Borrowers (mortgage, corporate and consumer credit)"),
    StateProfile("housing_activity", EconomicSector.REAL_ESTATE, "Homebuilders and housing turnover"),
    StateProfile("bank_net_interest_margin", _F, "Banks' lending margins"),
    StateProfile("credit_losses", _F, "Lenders' credit quality"),
    StateProfile("bank_earnings", _F, "Banks' earnings"),
)

_SUP, _CUS = ValueChainRole.SUPPLIER, ValueChainRole.CUSTOMER

CHANNEL_ROLES: dict[TransmissionChannel, ValueChainRole | None] = {
    TransmissionChannel.DEMAND_PULL: None,
    TransmissionChannel.INPUT_DEMAND: _SUP,
    TransmissionChannel.CAPACITY_UTILIZATION: _SUP,
    TransmissionChannel.INVESTMENT_RESPONSE: None,
    TransmissionChannel.SUPPLY_RESPONSE: None,
    TransmissionChannel.SUPPLY_BALANCE: None,
    TransmissionChannel.COST_PASS_THROUGH: _CUS,
    TransmissionChannel.COST_BURDEN: None,
    TransmissionChannel.REVENUE: None,
    TransmissionChannel.POLICY_TRANSMISSION: None,
    TransmissionChannel.POLICY_REACTION: None,
    TransmissionChannel.DISCOUNT_RATE: None,
    TransmissionChannel.OPPORTUNITY_COST: None,
    TransmissionChannel.FINANCING_CONDITIONS: None,
    TransmissionChannel.OTHER: None,
}

_TECH, _OIL, _MP = (
    EconomicChannel.TECHNOLOGY_INVESTMENT, EconomicChannel.CRUDE_OIL_SUPPLY, EconomicChannel.MONETARY_POLICY,
)
_AI_SPENDERS_AND_SUPPLIERS = (
    "ai_spender_free_cash_flow", "ai_services_revenue", "accelerator_demand", "hbm_demand",
    "advanced_foundry_utilization", "semiconductor_equipment_demand",
)

EXPOSURE_BASIS: tuple[ExposureBasis, ...] = (
    # -- technology / AI investment --------------------------------------------
    ExposureBasis(_TECH, "Technology & communication services", _AI_SPENDERS_AND_SUPPLIERS),
    ExposureBasis(_TECH, "Industrials", ("power_equipment_demand",)),
    ExposureBasis(_TECH, "Technology & growth ETFs", _AI_SPENDERS_AND_SUPPLIERS),
    ExposureBasis(_TECH, "Utilities sector ETF", ("electricity_demand", "grid_investment")),
    ExposureBasis(_TECH, "Nasdaq-100 index futures", _AI_SPENDERS_AND_SUPPLIERS),
    ExposureBasis(_TECH, "Copper futures", ("copper_demand",)),
    ExposureBasis(_TECH, "Natural gas futures", ("gas_power_generation_demand",)),
    # -- crude oil ---------------------------------------------------------------
    ExposureBasis(_OIL, "Crude & refined-product futures", ("crude_oil_price", "refined_product_prices")),
    ExposureBasis(_OIL, "Crude oil ETF", ("crude_oil_price",)),
    ExposureBasis(_OIL, "Energy sector ETF", ("oil_producer_revenue",)),
    ExposureBasis(_OIL, "Energy sector", ("oil_producer_revenue",)),
    ExposureBasis(_OIL, "Industrials", ("transport_fuel_costs",)),
    # -- monetary policy -----------------------------------------------------------
    ExposureBasis(_MP, "Treasury futures", ("treasury_yields",)),
    ExposureBasis(_MP, "Equity index futures", ("equity_valuation_multiples",)),
    ExposureBasis(_MP, "FX futures", ("usd_exchange_value",)),
    ExposureBasis(_MP, "Gold futures", ("gold_investment_demand",)),
    ExposureBasis(_MP, "Treasury ETFs", ("treasury_yields",)),
    ExposureBasis(_MP, "Broad & rate-sensitive equity ETFs", ("equity_valuation_multiples", "bank_earnings")),
    ExposureBasis(_MP, "Credit ETFs", ("treasury_yields", "borrowing_costs")),
    ExposureBasis(_MP, "Financials", ("bank_net_interest_margin", "bank_earnings")),
    ExposureBasis(_MP, "Long-duration growth equities", ("equity_valuation_multiples",)),
)


@dataclass(frozen=True)
class SignalPathLibrary:
    profiles: tuple[StateProfile, ...]
    channel_roles: dict[TransmissionChannel, ValueChainRole | None]
    exposure_basis: tuple[ExposureBasis, ...]

    def profile(self, state_id: str) -> StateProfile | None:
        return next((p for p in self.profiles if p.state_id == state_id), None)

    def basis(self, channel: EconomicChannel, exposure_label: str) -> ExposureBasis | None:
        return next(
            (b for b in self.exposure_basis if b.channel is channel and b.exposure_label == exposure_label), None,
        )


def _validate(library: SignalPathLibrary) -> None:
    """Structural checks at import: every catalog state is profiled exactly
    once, every transmission channel has a role, and every exposure of a
    SEEDED channel has a basis whose states exist -- so a new seed cannot
    silently skip its classification data."""
    catalog = [s.state_id for s in STATE_CATALOG]
    profiled = [p.state_id for p in library.profiles]
    if sorted(profiled) != sorted(catalog):
        missing, extra = set(catalog) - set(profiled), set(profiled) - set(catalog)
        raise ValueError(f"state profiles drifted from the catalog: missing {sorted(missing)}, extra {sorted(extra)}")
    for p in library.profiles:
        if not p.target_concept.strip():
            raise ValueError(f"{p.state_id}: empty target concept")
    if set(library.channel_roles) != set(TransmissionChannel):
        raise ValueError("every TransmissionChannel needs exactly one value-chain role entry")
    exposures = {(c.channel, e.label) for c in CHANNELS for e in c.exposures}
    keys = [(b.channel, b.exposure_label) for b in library.exposure_basis]
    if len(set(keys)) != len(keys):
        raise ValueError("an exposure has more than one basis entry")
    for b in library.exposure_basis:
        if (b.channel, b.exposure_label) not in exposures:
            raise ValueError(f"basis names an exposure that does not exist: {b.channel.value} / {b.exposure_label}")
        if not b.state_ids or not set(b.state_ids) <= set(catalog):
            raise ValueError(f"basis for {b.exposure_label} must name catalogued states")
    seeded = {a.channel for a in ANCHOR_RULES}
    for channel, label in sorted(exposures, key=lambda k: (k[0].value, k[1])):
        if channel in seeded and (channel, label) not in set(keys):
            raise ValueError(f"seeded channel {channel.value} exposure {label!r} has no economic basis")


DEFAULT_PATH_LIBRARY = SignalPathLibrary(
    profiles=STATE_PROFILES, channel_roles=CHANNEL_ROLES, exposure_basis=EXPOSURE_BASIS,
)
_validate(DEFAULT_PATH_LIBRARY)

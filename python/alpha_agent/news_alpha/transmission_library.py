"""News Alpha Phase B -- the reviewed SEED library for the economic mechanism
graph: a state catalog, seeded transmission claims, and the anchor rules that
attach a Phase A `EconomicChannel` to its root state.

The same kind of artifact as `news_alpha.channels` and
`translation.mechanism_library`: hand-reviewed, inspectable, deterministic --
and explicitly NOT the only possible graph. The library is one pool of
claims; an event's graph is whatever part of the pool (plus any extra claims,
e.g. a model proposal) is reachable from the event's anchors. A new channel,
state or link is a data change here, never a code path.

PROVENANCE DISCIPLINE. Every seeded claim is MANUALLY_SEEDED and carries its
seed's PLATFORM_RULE id -- a reasoned judgment, which alone makes a link
PROPOSED, not SUPPORTED. A link is SUPPORTED only where a source below was
opened and the recorded ``claim`` checked against it (``verified_on``). Those
sources were verified on 2026-09-25 against the documents named; where the
fetched text was a secondary transcript that is stated in ``note``. Links
without a verified source (e.g. electricity demand -> grid investment) stay
PROPOSED on purpose -- none was verified, so none is cited.

Seeded today: TECHNOLOGY_INVESTMENT (the AI-infrastructure vertical slice),
CRUDE_OIL_SUPPLY and MONETARY_POLICY. Every other Phase A channel is reported
as unseeded by the graph stage -- an honest gap, never an empty graph
presented as a finding.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from alpha_agent.news_alpha.channels import CHANNELS, EconomicChannel
from alpha_agent.news_alpha.transmission import (
    EconomicState,
    EdgeOrigin,
    LinkConfidence,
    Movement,
    Polarity,
    ProvenanceKind,
    ProvenanceSource,
    StateKind,
    TransmissionChannel,
    TransmissionClaim,
    TransmissionLag,
    Verification,
    canonical_state_id,
)

__all__ = [
    "AI_INFRASTRUCTURE_SEED",
    "ANCHOR_RULES",
    "CRUDE_OIL_SEED",
    "DEFAULT_LIBRARY",
    "MONETARY_POLICY_SEED",
    "SEEDED_CLAIMS",
    "STATE_CATALOG",
    "AnchorRule",
    "TransmissionLibrary",
]

AI_INFRASTRUCTURE_SEED = "transmission-seed/ai-infrastructure/1"
CRUDE_OIL_SEED = "transmission-seed/crude-oil/1"
MONETARY_POLICY_SEED = "transmission-seed/monetary-policy/1"

_SEED_TITLES = {
    AI_INFRASTRUCTURE_SEED: "Reviewed seed: AI infrastructure transmission",
    CRUDE_OIL_SEED: "Reviewed seed: crude oil supply transmission",
    MONETARY_POLICY_SEED: "Reviewed seed: monetary policy transmission",
}

_VERIFIED_ON = date(2026, 9, 25)

_S = StateKind


def _state(state_id: str, label: str, kind: StateKind, description: str = "", *aliases: str) -> EconomicState:
    return EconomicState(state_id=state_id, label=label, kind=kind, description=description, aliases=aliases)


STATE_CATALOG: tuple[EconomicState, ...] = (
    # -- AI infrastructure ---------------------------------------------------
    _state("ai_infrastructure_investment", "AI infrastructure investment", _S.INVESTMENT,
           "Capital spending on AI compute, data centers and the infrastructure that supports them.",
           "ai capex", "ai capital expenditure", "ai infrastructure spending", "ai spending"),
    _state("compute_demand", "AI compute demand", _S.DEMAND, "Demand for AI training and inference capacity.",
           "compute demand", "ai compute"),
    _state("accelerator_demand", "AI accelerator demand", _S.DEMAND, "GPUs and custom AI ASICs.",
           "gpu demand", "ai chip demand", "ai accelerators"),
    _state("hbm_demand", "HBM demand", _S.DEMAND, "High-bandwidth memory packaged alongside AI accelerators.",
           "high bandwidth memory demand", "hbm"),
    _state("advanced_foundry_utilization", "Leading-edge foundry utilization", _S.CAPACITY,
           "Utilization of leading-edge wafer and advanced-packaging capacity.",
           "foundry utilization", "foundry demand", "advanced packaging utilization"),
    _state("foundry_capacity_expansion", "Foundry & memory capacity expansion", _S.INVESTMENT,
           "New leading-edge fab, advanced-packaging and HBM capacity.",
           "fab capacity expansion", "semiconductor capacity expansion"),
    _state("semiconductor_equipment_demand", "Semiconductor equipment demand", _S.DEMAND,
           "Wafer-fab, packaging and test equipment.", "wafer fab equipment demand", "chip equipment demand"),
    _state("data_center_construction", "Data-center construction", _S.INVESTMENT, "",
           "data center buildout", "data centre construction"),
    _state("electricity_demand", "Electricity demand", _S.DEMAND, "", "power demand", "electric load"),
    _state("grid_investment", "Grid investment", _S.INVESTMENT,
           "Transmission, distribution and interconnection spending.", "transmission investment", "grid capex"),
    _state("power_equipment_demand", "Power-equipment demand", _S.DEMAND,
           "Transformers, switchgear, turbines and backup generation.", "electrical equipment demand",
           "transformer demand"),
    _state("gas_power_generation_demand", "Natural-gas demand for power", _S.DEMAND,
           "Gas burned to generate electricity.", "natural gas demand for power", "power burn"),
    _state("copper_demand", "Copper demand", _S.DEMAND),
    _state("ai_spender_free_cash_flow", "AI spenders' free cash flow", _S.FINANCIAL,
           "Operating cash flow minus capital expenditure of the companies doing the spending.",
           "hyperscaler free cash flow"),
    _state("ai_services_revenue", "AI services revenue", _S.FINANCIAL,
           "Revenue the spenders earn from the AI/cloud services the new capacity sells.",
           "cloud ai revenue"),
    # -- crude oil -------------------------------------------------------------
    _state("crude_oil_supply", "Crude oil supply", _S.SUPPLY,
           "Available crude: production plus inventories.", "oil supply", "crude supply", "oil production"),
    _state("crude_oil_price", "Crude oil price", _S.PRICE, "", "oil price", "oil prices"),
    _state("refined_product_prices", "Refined-product prices", _S.PRICE, "Gasoline, diesel and jet fuel.",
           "fuel prices", "gasoline prices"),
    _state("oil_producer_revenue", "Oil producers' revenue", _S.FINANCIAL),
    _state("transport_fuel_costs", "Transport & industrial fuel costs", _S.PRICE),
    _state("headline_inflation", "Headline inflation", _S.PRICE, "", "inflation", "cpi"),
    _state("shale_drilling_activity", "Shale drilling activity", _S.INVESTMENT, "", "drilling activity", "rig count"),
    # -- monetary policy -------------------------------------------------------
    _state("policy_rate_path", "Expected policy-rate path", _S.RATE, "", "policy rate", "fed funds rate",
           "interest rates"),
    _state("treasury_yields", "Treasury yields", _S.RATE, "", "bond yields"),
    _state("equity_valuation_multiples", "Equity valuation multiples", _S.PRICE, "Price-to-earnings multiples."),
    _state("usd_exchange_value", "US dollar exchange value", _S.RATE, "", "us dollar", "usd"),
    _state("gold_investment_demand", "Gold investment demand", _S.DEMAND),
    _state("borrowing_costs", "Borrowing costs", _S.RATE, "Mortgage, corporate and consumer credit rates.",
           "mortgage rates", "credit costs"),
    _state("housing_activity", "Housing activity", _S.ACTIVITY),
    _state("bank_net_interest_margin", "Bank net interest margin", _S.FINANCIAL),
    _state("credit_losses", "Credit losses", _S.FINANCIAL),
    _state("bank_earnings", "Bank earnings", _S.FINANCIAL),
)


# ---------------------------------------------------------------------------
# verified external / definitional sources (opened and checked 2026-09-25)
# ---------------------------------------------------------------------------


def _verified(kind: ProvenanceKind, reference: str, title: str, claim: str, *, publisher: str | None = None,
              published: date | None = None, note: str | None = None) -> ProvenanceSource:
    return ProvenanceSource(
        kind=kind, reference=reference, title=title, publisher=publisher, published=published, claim=claim,
        verification=Verification.VERIFIED, verified_on=_VERIFIED_ON, note=note,
    )


_DOC = ProvenanceKind.EXTERNAL_DOCUMENT

_IEA_ENERGY_AND_AI = _verified(
    _DOC,
    "https://www.iea.org/news/ai-is-set-to-drive-surging-electricity-demand-from-data-centres-while-offering-the-"
    "potential-to-transform-how-the-energy-sector-works",
    "Energy and AI (special report) -- press release", publisher="International Energy Agency",
    published=date(2025, 4, 10),
    claim="Electricity demand from data centres worldwide is set to more than double by 2030 to around 945 TWh; "
          "AI will be the most significant driver of this increase.",
)
_DOE_LBNL_2024 = _verified(
    _DOC, "https://www.energy.gov/articles/doe-releases-new-report-evaluating-increase-electricity-demand-data-centers",
    "DOE releases new report evaluating increase in electricity demand from data centers (LBNL 2024 United States "
    "Data Center Energy Usage Report)", publisher="U.S. Department of Energy", published=date(2024, 12, 20),
    claim="Data centers consumed about 4.4% of total U.S. electricity in 2023 and are expected to consume "
          "approximately 6.7 to 12% of total U.S. electricity by 2028.",
)
_SKHYNIX_4Q24 = _verified(
    _DOC, "https://news.skhynix.com/sk-hynix-announces-4q24-financial-results/",
    "SK hynix Announces 4Q24 Financial Results", publisher="SK hynix", published=date(2025, 1, 23),
    claim="HBM marked over 40% of total DRAM revenue in 4Q24; demand for HBM will continue to increase as global "
          "big tech companies' investment in AI servers grows.",
)
_TSMC_4Q24_CALL = _verified(
    _DOC, "https://www.fool.com/earnings/call-transcripts/2025/01/16/taiwan-semiconductor-manufacturing-tsm-q4-2024-ear/",
    "TSMC Q4 2024 earnings call", publisher="TSMC (transcript published by The Motley Fool)",
    published=date(2025, 1, 16),
    claim="Revenue from AI accelerators more than tripled in 2024 and is forecast to double in 2025; Q4 gross margin "
          "rose to 59%, mainly reflecting a higher capacity utilization rate and productivity gains.",
    note="Checked against a secondary transcript; TSMC's own PDF transcript was not machine-readable here.",
)
_SEMI_2024_BILLINGS = _verified(
    _DOC,
    "https://www.prnewswire.com/news-releases/global-semiconductor-equipment-billings-surged-to-117-billion-in-2024-"
    "semi-reports-302423925.html",
    "Global Semiconductor Equipment Billings Surged to $117 Billion in 2024", publisher="SEMI",
    published=date(2025, 4, 9),
    claim="Worldwide equipment sales rose 10% to $117.1 billion in 2024, fueled by investments in expanding capacity "
          "for leading-edge and mature logic, advanced packaging, and high-bandwidth memory (HBM).",
)
_FCF_IDENTITY = _verified(
    ProvenanceKind.DEFINITIONAL_IDENTITY, "identity/free-cash-flow",
    "Free cash flow = operating cash flow - capital expenditures",
    claim="Capital expenditure is subtracted in full from free cash flow in the period it is paid.",
)


# ---------------------------------------------------------------------------
# seeded claims
# ---------------------------------------------------------------------------

_P, _N = Polarity.POSITIVE, Polarity.NEGATIVE
_C = TransmissionChannel
_L = TransmissionLag
_HI, _MED, _LO = LinkConfidence.HIGH, LinkConfidence.MEDIUM, LinkConfidence.LOW


def _seed(rule: str, source: str, target: str, channel: TransmissionChannel, polarity: Polarity,
          lag: TransmissionLag, confidence: LinkConfidence, rationale: str,
          *evidence: ProvenanceSource) -> TransmissionClaim:
    rule_source = ProvenanceSource(kind=ProvenanceKind.PLATFORM_RULE, reference=rule, title=_SEED_TITLES[rule])
    return TransmissionClaim(
        source=source, target=target, channel=channel, polarity=polarity, lag=lag, confidence=confidence,
        origin=EdgeOrigin.MANUALLY_SEEDED, rationale=rationale, provenance=(rule_source, *evidence),
    )


def _ai(*args, **kwargs) -> TransmissionClaim:
    return _seed(AI_INFRASTRUCTURE_SEED, *args, **kwargs)


def _oil(*args, **kwargs) -> TransmissionClaim:
    return _seed(CRUDE_OIL_SEED, *args, **kwargs)


def _mp(*args, **kwargs) -> TransmissionClaim:
    return _seed(MONETARY_POLICY_SEED, *args, **kwargs)


SEEDED_CLAIMS: tuple[TransmissionClaim, ...] = (
    # -- AI infrastructure: the semiconductor chain ---------------------------
    _ai("ai_infrastructure_investment", "compute_demand", _C.DEMAND_PULL, _P, _L.MONTHS, _HI,
        "AI infrastructure spending is largely the purchase of AI compute capacity."),
    _ai("compute_demand", "accelerator_demand", _C.INPUT_DEMAND, _P, _L.MONTHS, _HI,
        "AI compute is delivered by accelerators (GPUs and custom ASICs)."),
    _ai("accelerator_demand", "hbm_demand", _C.INPUT_DEMAND, _P, _L.MONTHS, _HI,
        "Each AI accelerator package carries high-bandwidth memory stacks.", _SKHYNIX_4Q24),
    _ai("accelerator_demand", "advanced_foundry_utilization", _C.CAPACITY_UTILIZATION, _P, _L.MONTHS, _MED,
        "Accelerators are made on leading-edge nodes and advanced packaging with limited capacity.", _TSMC_4Q24_CALL),
    _ai("advanced_foundry_utilization", "foundry_capacity_expansion", _C.INVESTMENT_RESPONSE, _P, _L.QUARTERS, _MED,
        "Sustained high utilization justifies adding capacity."),
    _ai("hbm_demand", "foundry_capacity_expansion", _C.INVESTMENT_RESPONSE, _P, _L.QUARTERS, _MED,
        "Memory makers add HBM and packaging capacity to meet demand."),
    _ai("foundry_capacity_expansion", "semiconductor_equipment_demand", _C.INPUT_DEMAND, _P, _L.QUARTERS, _HI,
        "New capacity is built with wafer-fab, packaging and test equipment.", _SEMI_2024_BILLINGS),
    _ai("foundry_capacity_expansion", "advanced_foundry_utilization", _C.SUPPLY_RESPONSE, _N, _L.YEARS, _MED,
        "Once new capacity comes online it relieves utilization -- the cycle's balancing force."),
    # -- AI infrastructure: the power chain -------------------------------------
    _ai("ai_infrastructure_investment", "data_center_construction", _C.DEMAND_PULL, _P, _L.MONTHS, _HI,
        "AI capacity has to be housed in new or expanded data centers."),
    _ai("data_center_construction", "electricity_demand", _C.INPUT_DEMAND, _P, _L.QUARTERS, _HI,
        "Operating data centers are large, continuous electricity loads.", _IEA_ENERGY_AND_AI, _DOE_LBNL_2024),
    _ai("electricity_demand", "grid_investment", _C.INVESTMENT_RESPONSE, _P, _L.YEARS, _MED,
        "New large loads need transmission, distribution and interconnection upgrades."),
    _ai("grid_investment", "power_equipment_demand", _C.INPUT_DEMAND, _P, _L.QUARTERS, _HI,
        "Grid build-out consumes transformers, switchgear and cabling."),
    _ai("data_center_construction", "power_equipment_demand", _C.INPUT_DEMAND, _P, _L.MONTHS, _MED,
        "Each data center needs its own substations, switchgear and backup generation."),
    _ai("electricity_demand", "gas_power_generation_demand", _C.INPUT_DEMAND, _P, _L.MONTHS, _MED,
        "Gas-fired plants are often the marginal generation that meets new load."),
    _ai("grid_investment", "copper_demand", _C.INPUT_DEMAND, _P, _L.QUARTERS, _MED,
        "Transmission and distribution build-out is copper-intensive."),
    _ai("data_center_construction", "copper_demand", _C.INPUT_DEMAND, _P, _L.QUARTERS, _LO,
        "Data-center electrical and cooling systems use copper; a small share of global demand."),
    # -- AI infrastructure: the spender side ------------------------------------
    _ai("ai_infrastructure_investment", "ai_spender_free_cash_flow", _C.COST_BURDEN, _N, _L.MONTHS, _HI,
        "Capex is paid out of the spenders' cash flow as it is incurred.", _FCF_IDENTITY),
    _ai("ai_infrastructure_investment", "ai_services_revenue", _C.REVENUE, _P, _L.YEARS, _LO,
        "The capacity is built to sell AI services; whether and when that revenue arrives is the open question."),
    _ai("ai_services_revenue", "ai_spender_free_cash_flow", _C.REVENUE, _P, _L.QUARTERS, _HI,
        "Services revenue flows into operating cash flow."),
    # -- crude oil ---------------------------------------------------------------
    _oil("crude_oil_supply", "crude_oil_price", _C.SUPPLY_BALANCE, _N, _L.IMMEDIATE, _HI,
         "Less available crude against unchanged demand clears at a higher price."),
    _oil("crude_oil_price", "refined_product_prices", _C.COST_PASS_THROUGH, _P, _L.WEEKS, _HI,
         "Crude is the dominant input cost of refined products."),
    _oil("crude_oil_price", "oil_producer_revenue", _C.REVENUE, _P, _L.MONTHS, _MED,
         "Producers outside a cut realize the higher price on unchanged volumes; producers inside it lose volume."),
    _oil("refined_product_prices", "transport_fuel_costs", _C.COST_PASS_THROUGH, _P, _L.WEEKS, _HI,
         "Fuel prices are paid directly by transport and industrial users."),
    _oil("crude_oil_price", "headline_inflation", _C.COST_PASS_THROUGH, _P, _L.MONTHS, _MED,
         "Energy is a direct component of headline price indices; pass-through to core is smaller and slower."),
    _oil("headline_inflation", "policy_rate_path", _C.POLICY_REACTION, _P, _L.MONTHS, _LO,
         "A central bank may respond to inflation it judges persistent; supply-driven energy shocks are often "
         "looked through."),
    _oil("crude_oil_price", "shale_drilling_activity", _C.INVESTMENT_RESPONSE, _P, _L.QUARTERS, _MED,
         "Higher prices raise the return on new wells; operators respond with a lag."),
    _oil("shale_drilling_activity", "crude_oil_supply", _C.SUPPLY_RESPONSE, _P, _L.QUARTERS, _MED,
         "New wells add production -- the balancing response to a price rise."),
    # -- monetary policy -----------------------------------------------------------
    _mp("policy_rate_path", "treasury_yields", _C.POLICY_TRANSMISSION, _P, _L.IMMEDIATE, _HI,
        "Short and intermediate yields price the expected policy path."),
    _mp("treasury_yields", "equity_valuation_multiples", _C.DISCOUNT_RATE, _N, _L.IMMEDIATE, _MED,
        "Higher discount rates lower the present value of future earnings; growth expectations can offset."),
    _mp("policy_rate_path", "usd_exchange_value", _C.POLICY_TRANSMISSION, _P, _L.IMMEDIATE, _MED,
        "A higher relative rate path attracts capital into the currency."),
    _mp("treasury_yields", "gold_investment_demand", _C.OPPORTUNITY_COST, _N, _L.WEEKS, _LO,
        "Gold pays no yield, so higher yields raise its opportunity cost; the relationship has been unstable."),
    _mp("policy_rate_path", "borrowing_costs", _C.FINANCING_CONDITIONS, _P, _L.WEEKS, _HI,
        "Lending rates are priced off the policy path."),
    _mp("borrowing_costs", "housing_activity", _C.FINANCING_CONDITIONS, _N, _L.MONTHS, _HI,
        "Higher mortgage rates reduce affordability and transactions."),
    _mp("policy_rate_path", "bank_net_interest_margin", _C.REVENUE, _P, _L.QUARTERS, _MED,
        "Asset yields reprice faster than deposit costs, up to a point."),
    _mp("borrowing_costs", "credit_losses", _C.FINANCING_CONDITIONS, _P, _L.QUARTERS, _MED,
        "Heavier debt service raises delinquencies with a lag."),
    _mp("bank_net_interest_margin", "bank_earnings", _C.REVENUE, _P, _L.QUARTERS, _HI,
        "Net interest income is most banks' largest revenue line."),
    _mp("credit_losses", "bank_earnings", _C.COST_BURDEN, _N, _L.QUARTERS, _HI,
        "Loan-loss provisions are charged against earnings."),
)


# ---------------------------------------------------------------------------
# anchor rules: Phase A channel + direction shift -> root state + movement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorRule:
    """Attaches a Phase A `EconomicChannel` to the state it moves first, and
    maps each of that channel's direction shifts (`channels.DirectionCue
    .shift`) to the root state's movement. Every shift the channel declares
    must be mapped (checked at import)."""

    channel: EconomicChannel
    rule_id: str
    root_state: str
    shift_movement: tuple[tuple[str, Movement], ...]
    basis: str


ANCHOR_RULES: tuple[AnchorRule, ...] = (
    AnchorRule(
        EconomicChannel.TECHNOLOGY_INVESTMENT, AI_INFRASTRUCTURE_SEED, "ai_infrastructure_investment",
        (("SPENDING_INCREASE", Movement.UP), ("SPENDING_DECREASE", Movement.DOWN)),
        "The technology-investment channel is anchored at AI infrastructure investment, its seeded root.",
    ),
    AnchorRule(
        EconomicChannel.CRUDE_OIL_SUPPLY, CRUDE_OIL_SEED, "crude_oil_supply",
        (("TIGHTER_BALANCE", Movement.DOWN), ("LOOSER_BALANCE", Movement.UP)),
        "A tighter crude balance (cuts, disruptions, draws) is less available supply; a looser one is more.",
    ),
    AnchorRule(
        EconomicChannel.MONETARY_POLICY, MONETARY_POLICY_SEED, "policy_rate_path",
        (("TIGHTENING", Movement.UP), ("EASING", Movement.DOWN)),
        "Tightening raises the expected policy-rate path; easing lowers it.",
    ),
)


@dataclass(frozen=True)
class TransmissionLibrary:
    states: tuple[EconomicState, ...]
    claims: tuple[TransmissionClaim, ...]
    anchors: tuple[AnchorRule, ...]

    def anchor_rule(self, channel: EconomicChannel) -> AnchorRule | None:
        return next((a for a in self.anchors if a.channel is channel), None)


def _validate(library: TransmissionLibrary) -> None:
    """Structural checks at import (mirrors `channels._validate_channels`)."""
    ids = [s.state_id for s in library.states]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate state_id in STATE_CATALOG")
    names: dict[str, str] = {}
    for s in library.states:
        for ref in (s.state_id, s.label, *s.aliases):
            owner = names.setdefault(canonical_state_id(ref), s.state_id)
            if owner != s.state_id:
                raise ValueError(f"state name {ref!r} is ambiguous between {owner} and {s.state_id}")
    declared = set(ids)
    for c in library.claims:
        if c.source not in declared or c.target not in declared:
            raise ValueError(f"seeded claim {c.source} -> {c.target} references an undeclared state")
        if not c.has_required_provenance:
            raise ValueError(f"seeded claim {c.source} -> {c.target} lacks its seed rule provenance")
    shifts = {c.channel: {d.shift for d in c.direction_cues} for c in CHANNELS}
    for a in library.anchors:
        if a.root_state not in declared:
            raise ValueError(f"{a.rule_id}: undeclared root state {a.root_state}")
        if {s for s, _m in a.shift_movement} != shifts[a.channel]:
            raise ValueError(f"{a.rule_id}: must map exactly the shifts {sorted(shifts[a.channel])}")
    if len({a.channel for a in library.anchors}) != len(library.anchors):
        raise ValueError("more than one anchor rule for a channel")


DEFAULT_LIBRARY = TransmissionLibrary(states=STATE_CATALOG, claims=SEEDED_CLAIMS, anchors=ANCHOR_RULES)
_validate(DEFAULT_LIBRARY)

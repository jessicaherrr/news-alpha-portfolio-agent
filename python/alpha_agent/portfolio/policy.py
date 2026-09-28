"""News Alpha Phase G -- the construction POLICY and the CONSTRAINTS one plan is
built under.

`PortfolioConstructionPolicy` (``portfolio-construction/1``) holds every
choice that is a convention rather than a mandate: eligibility, count limits,
the target volatility per risk style, default caps, liquidity floors and the
risk-estimation window. It is fixed before a plan is built and never fitted
to the signals it allocates.

`resolve_constraints` turns a `ResearchMandate` + a policy into the concrete
limits the C++ allocator enforces, recording where EACH came from
(mandate, profile, or a labelled policy default). Two resolutions matter for
honesty:

* no leverage cap stated -> the book is UNLEVERED (1.0x), never "unlimited";
* no capital stated -> a REFERENCE capital, labelled as not the user's.

Nothing here reads a signal, a screen or a price.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from alpha_agent.news_alpha.mandate import LiquidityRequirement, MandateDomain, ResearchMandate
from alpha_agent.recommendation.profile import MaxDrawdown, RiskStyle

__all__ = [
    "EXECUTABLE_DOMAINS",
    "PORTFOLIO_CONSTRUCTION_METHOD",
    "ConstraintSource",
    "EligibilityMode",
    "PortfolioConstraints",
    "PortfolioConstructionPolicy",
    "ResolvedLimit",
    "resolve_constraints",
]

PORTFOLIO_CONSTRUCTION_METHOD = "portfolio-construction/1"

#: Domains the platform can SIZE and EXECUTE: real point-in-time bars, real
#: instrument economics and a C++ contract adapter. Equities have the adapter
#: but no acquired bars; options and crypto have neither.
EXECUTABLE_DOMAINS: tuple[MandateDomain, ...] = (MandateDomain.FUTURES, MandateDomain.ETF)


class EligibilityMode(str, Enum):
    #: Screening support only: screen outcome SCREEN_CONTINUE.
    QUALIFIED = "QUALIFIED"
    #: Explicit opt-in preview: not contradicted, evidence leaning the
    #: declared way (aligned t > 0). Labelled on every surface.
    EXPLORATORY = "EXPLORATORY"


ELIGIBILITY_TEXT: dict[EligibilityMode, str] = {
    EligibilityMode.QUALIFIED: "Qualified -- only signals whose screen supports continuing to validation",
    EligibilityMode.EXPLORATORY: (
        "Exploratory preview -- signals WITHOUT screening support whose evidence at least leans the declared way; "
        "shows how the allocator would treat them, not a qualified portfolio"
    ),
}


class PortfolioConstructionPolicy(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    method: Literal["portfolio-construction/1"] = PORTFOLIO_CONSTRUCTION_METHOD
    eligibility: EligibilityMode = EligibilityMode.QUALIFIED

    #: Alternates share their exposure's risk budget (never one of their own).
    include_alternates: bool = True
    #: Selection priority = rank: the best-ranked exposures and members enter first.
    max_exposures: int = Field(default=10, ge=1)
    max_signals_per_exposure: int = Field(default=4, ge=1)

    #: Conventional risk levels per stated risk style (annualized volatility).
    target_vol_by_risk_style: dict[RiskStyle, float] = Field(default_factory=lambda: {
        RiskStyle.CONSERVATIVE: 0.06, RiskStyle.BALANCED: 0.10, RiskStyle.AGGRESSIVE: 0.15,
    })
    #: Overrides the risk-style table when set.
    target_annual_volatility: float | None = Field(default=None, gt=0, le=1)

    #: Used only when the mandate states no leverage cap: unlevered.
    default_max_gross_leverage: float = Field(default=1.0, gt=0)
    #: |net| / capital; ``None`` = no separate net cap (net <= gross anyway).
    max_net_exposure: float | None = Field(default=None, gt=0)
    #: Shares of the gross exposure the mandate allows.
    max_instrument_gross_share: float = Field(default=0.4, gt=0, le=1)
    max_sector_gross_share: float = Field(default=0.6, gt=0, le=1)
    max_asset_class_gross_share: float = Field(default=0.8, gt=0, le=1)
    #: No single independent exposure may use more than this share of the
    #: risk budget (target volatility).
    max_exposure_risk_share: float = Field(default=0.5, gt=0, le=1)
    #: A position may be at most this fraction of the instrument's median
    #: daily traded notional.
    max_adv_participation: float = Field(default=0.01, gt=0, le=1)
    min_adv_usd_by_liquidity: dict[LiquidityRequirement, float] = Field(default_factory=lambda: {
        LiquidityRequirement.ANY: 0.0, LiquidityRequirement.STANDARD: 25e6, LiquidityRequirement.HIGH: 250e6,
    })
    #: Sum |trade| / capital vs current holdings; ``None`` = not limited.
    max_turnover: float | None = Field(default=None, gt=0)

    reference_capital_usd: float = Field(default=1_000_000.0, gt=0)
    risk_lookback_days: int = Field(default=252, ge=20)
    min_risk_observations: int = Field(default=200, ge=20)
    annualization_days: int = Field(default=252, ge=1)

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return "portpolicy1:" + hashlib.sha256(payload.encode()).hexdigest()


class ConstraintSource(str, Enum):
    MANDATE = "MANDATE"
    PROFILE = "PROFILE"
    POLICY = "POLICY"
    #: A policy default standing in for something the user has not stated.
    DEFAULT_FOR_UNSTATED = "DEFAULT_FOR_UNSTATED"


class ResolvedLimit(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    name: str
    value: str
    source: ConstraintSource
    note: str


#: The mandate's drawdown tolerance as the execution gate's drawdown kill
#: switch (a 20%+ tolerance is enforced at 20%).
MAX_DRAWDOWN_PCT: dict[MaxDrawdown, float] = {
    MaxDrawdown.PCT_5: 0.05, MaxDrawdown.PCT_10: 0.10, MaxDrawdown.PCT_15: 0.15, MaxDrawdown.PCT_20_PLUS: 0.20,
}


class PortfolioConstraints(BaseModel):
    """The limits one plan is built under -- exactly what the C++ allocator
    receives, plus where each came from."""

    model_config = {"frozen": True, "extra": "forbid"}

    capital_usd: float
    capital_source: ConstraintSource
    target_annual_vol: float
    allowed_domains: tuple[MandateDomain, ...]
    shorting_allowed: bool
    max_gross_leverage: float
    max_net_exposure: float | None
    max_instrument_gross_share: float
    max_sector_gross_share: float
    max_asset_class_gross_share: float
    max_exposure_risk_share: float
    max_adv_participation: float
    min_adv_usd: float
    max_turnover: float | None
    #: Futures only (a contract count); shares are never capped by a contract count.
    max_futures_contracts: int | None
    max_drawdown_pct: float
    limits: tuple[ResolvedLimit, ...]

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return "portconstraints1:" + hashlib.sha256(payload.encode()).hexdigest()


def resolve_constraints(mandate: ResearchMandate, policy: PortfolioConstructionPolicy) -> PortfolioConstraints:
    profile = mandate.risk_profile
    limits: list[ResolvedLimit] = []

    def add(name: str, value: str, source: ConstraintSource, note: str) -> None:
        limits.append(ResolvedLimit(name=name, value=value, source=source, note=note))

    if profile.approximate_capital_usd:
        capital, capital_source = float(profile.approximate_capital_usd), ConstraintSource.PROFILE
        add("Capital", f"${capital:,.0f}", capital_source, "your stated approximate capital")
    else:
        capital, capital_source = policy.reference_capital_usd, ConstraintSource.DEFAULT_FOR_UNSTATED
        add("Capital", f"${capital:,.0f}", capital_source,
            "reference capital -- you have not stated yours; sizes scale with it")

    if policy.target_annual_volatility is not None:
        target = policy.target_annual_volatility
        add("Target volatility", f"{target:.1%}", ConstraintSource.POLICY, "set explicitly for this plan")
    else:
        target = policy.target_vol_by_risk_style[profile.risk_style]
        add("Target volatility", f"{target:.1%}", ConstraintSource.PROFILE,
            f"the conventional level for the {profile.risk_style.value} risk style (a convention, not evidence)")

    allowed = tuple(d for d in mandate.allowed_domains if d in EXECUTABLE_DOMAINS)
    add("Asset domains", " + ".join(d.value for d in allowed) or "none executable", ConstraintSource.MANDATE,
        "your allowed domains that the platform can size and execute")
    add("Shorting", "allowed" if mandate.shorting_allowed else "not allowed", ConstraintSource.MANDATE,
        "a signal pointing short is flat under a no-shorting mandate")

    if mandate.max_gross_leverage is not None:
        gross, gross_source = float(mandate.max_gross_leverage), ConstraintSource.MANDATE
        add("Gross leverage", f"{gross:g}x", gross_source, "your stated cap")
    else:
        gross, gross_source = policy.default_max_gross_leverage, ConstraintSource.DEFAULT_FOR_UNSTATED
        add("Gross leverage", f"{gross:g}x", gross_source,
            "no cap stated -- the book is kept unlevered; state a cap in your mandate to allow more")

    if policy.max_net_exposure is not None:
        add("Net exposure", f"{policy.max_net_exposure:g}x", ConstraintSource.POLICY, "|long - short| / capital")
    add("Instrument concentration", f"{policy.max_instrument_gross_share:.0%} of allowed gross",
        ConstraintSource.POLICY, "no single instrument above this share of the gross exposure allowed")
    add("Sector concentration", f"{policy.max_sector_gross_share:.0%} of allowed gross", ConstraintSource.POLICY,
        "per risk sector (risk-classification/1)")
    add("Asset-class concentration", f"{policy.max_asset_class_gross_share:.0%} of allowed gross",
        ConstraintSource.POLICY, "per risk asset class")
    add("Risk per exposure", f"{policy.max_exposure_risk_share:.0%} of the risk budget", ConstraintSource.POLICY,
        "one independent exposure may use at most this share of the target volatility")
    min_adv = policy.min_adv_usd_by_liquidity[mandate.liquidity_requirement]
    add("Liquidity floor", f"${min_adv:,.0f}/day median traded notional" if min_adv else "none",
        ConstraintSource.MANDATE, f"your {mandate.liquidity_requirement.value.title()} liquidity requirement")
    add("Liquidity participation", f"{policy.max_adv_participation:.1%} of median daily traded notional",
        ConstraintSource.POLICY, "a position larger than this is cut")
    if policy.max_turnover is not None:
        add("Turnover", f"{policy.max_turnover:g}x capital", ConstraintSource.POLICY, "per rebalance")
    if profile.max_contracts is not None:
        add("Futures contracts", f"{profile.max_contracts} per contract", ConstraintSource.PROFILE,
            "your stated contract cap (futures only)")
    drawdown = MAX_DRAWDOWN_PCT[profile.max_drawdown]
    add("Drawdown kill switch", f"{drawdown:.0%}", ConstraintSource.PROFILE,
        "enforced by the execution risk gate, not by sizing")

    return PortfolioConstraints(
        capital_usd=capital, capital_source=capital_source, target_annual_vol=target, allowed_domains=allowed,
        shorting_allowed=mandate.shorting_allowed, max_gross_leverage=gross,
        max_net_exposure=policy.max_net_exposure, max_instrument_gross_share=policy.max_instrument_gross_share,
        max_sector_gross_share=policy.max_sector_gross_share,
        max_asset_class_gross_share=policy.max_asset_class_gross_share,
        max_exposure_risk_share=policy.max_exposure_risk_share, max_adv_participation=policy.max_adv_participation,
        min_adv_usd=min_adv, max_turnover=policy.max_turnover, max_futures_contracts=profile.max_contracts,
        max_drawdown_pct=drawdown, limits=tuple(limits),
    )

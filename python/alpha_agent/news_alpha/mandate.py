"""News Alpha Phase A -- the RESEARCH MANDATE: what the user permits the
research system to investigate.

A mandate is a set of CONSTRAINTS, never a scientific input. It scopes which
asset domains / instruments research may look at and carries the user's
horizon / risk / position constraints forward to later stages (Asset
Expression, Portfolio Construction). It never enters `ExperimentRegistry`,
`experiment_identity`, `ReliabilityPolicy`, BH-FDR/DSR, or any verdict --
exactly the boundary `alpha_agent.recommendation.profile.InvestorProfile`
already holds for presentation preferences.

REUSE, NOT DUPLICATION: horizon, risk style, drawdown tolerance, turnover
sensitivity and approximate capital already live on `InvestorProfile` (with a
UI editor and a local JSON store). `ResearchMandate` COMPOSES that object
verbatim as `risk_profile` rather than re-declaring any of those fields. The
genuinely new dimensions -- domain access, instrument constraints, shorting,
leverage, liquidity -- sit beside it.

ASSET ACCESS AND RISK TOLERANCE ARE INDEPENDENT AXES. Nothing here (or
downstream) derives risk from a domain or a domain from risk: a hedged
Futures position can be low-risk and a levered Equity book high-risk.
`allowed_domains` is pure access; `risk_profile` / `shorting_allowed` /
`max_gross_leverage` are pure constraints. `tests/python/test_news_alpha_phase_a.py`
holds this apart mechanically (varying only risk fields never changes which
domains are in scope or any impact level).

WHY A MANDATE-LEVEL DOMAIN VOCABULARY: `alpha_agent.registry.enums.AssetDomain`
is the registry's EVIDENCE namespace (FUTURES/ETF/EQUITY) and widening it is a
FROZEN RESEARCH SEMANTICS change (CLAUDE.md). A user must still be able to say
"Options and Crypto are off-limits" (or "allowed"), so `MandateDomain` is the
ACCESS vocabulary: every `AssetDomain` member appears here under the identical
value string (test-enforced, so the two can never drift), plus OPTIONS and
CRYPTO, which have no registry namespace and resolve to `None`.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from alpha_agent.core.instrument import InstrumentIdentity
from alpha_agent.recommendation.profile import (
    DEFAULT_PROFILE,
    REPO_ROOT,
    HoldingPeriod,
    InvestorProfile,
    RiskStyle,
    TurnoverSensitivity,
)
from alpha_agent.registry.enums import AssetDomain

__all__ = [
    "DEFAULT_MANDATE",
    "DEFAULT_MANDATE_PATH",
    "MANDATE_SCHEMA_VERSION",
    "LiquidityRequirement",
    "MandateDomain",
    "MandateStore",
    "ResearchMandate",
    "mandate_domain_for",
    "registry_domain_for",
]

MANDATE_SCHEMA_VERSION = "research-mandate/1"
DEFAULT_MANDATE_PATH = REPO_ROOT / "data" / "user_prefs" / "research_mandate.json"


class MandateDomain(str, Enum):
    """The asset-ACCESS vocabulary a mandate is written in. Declaration order
    is the canonical display/iteration order everywhere downstream."""

    FUTURES = "FUTURES"
    ETF = "ETF"
    EQUITY = "EQUITY"
    OPTIONS = "OPTIONS"
    CRYPTO = "CRYPTO"


#: Human labels for the UI -- never used as identity.
DOMAIN_LABELS: dict[MandateDomain, str] = {
    MandateDomain.FUTURES: "Futures",
    MandateDomain.ETF: "ETF",
    MandateDomain.EQUITY: "Equity",
    MandateDomain.OPTIONS: "Options",
    MandateDomain.CRYPTO: "Crypto",
}


def registry_domain_for(domain: MandateDomain) -> AssetDomain | None:
    """The registry evidence namespace for ``domain``, or ``None`` when the
    registry has none (OPTIONS, CRYPTO). Resolved by identical value string,
    never by a hand-maintained second table."""
    try:
        return AssetDomain(domain.value)
    except ValueError:
        return None


def mandate_domain_for(domain: AssetDomain) -> MandateDomain:
    return MandateDomain(domain.value)


class LiquidityRequirement(str, Enum):
    """How liquid a candidate instrument must be. Carried as a typed
    constraint; the Phase A universe resolver records honestly that no domain
    universe carries typed ADV/liquidity metadata yet (every declared universe
    was selected as liquid by construction), so it is not APPLIED as a filter
    until a later stage has real liquidity data to apply it against."""

    ANY = "ANY"
    STANDARD = "STANDARD"
    HIGH = "HIGH"


def _canonical_domains(domains: tuple[MandateDomain, ...]) -> tuple[MandateDomain, ...]:
    present = set(domains)
    return tuple(d for d in MandateDomain if d in present)


def _canonical_instruments(items: tuple[InstrumentIdentity, ...]) -> tuple[InstrumentIdentity, ...]:
    order = {d: i for i, d in enumerate(AssetDomain)}
    unique = {(i.asset_domain, i.symbol): i for i in items}
    return tuple(unique[k] for k in sorted(unique, key=lambda k: (order[k[0]], k[1])))


class ResearchMandate(BaseModel):
    """What the research system may investigate for this user. Frozen: a
    changed mandate is always a NEW instance. Canonicalized on construction
    (domains in `MandateDomain` order, instruments de-duplicated and sorted),
    so two mandates meaning the same thing serialize -- and fingerprint --
    identically."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = MANDATE_SCHEMA_VERSION

    #: ACCESS -- which asset domains research may look at. Never empty.
    allowed_domains: tuple[MandateDomain, ...] = Field(min_length=1)
    #: Optional instrument constraints. When the allowlist names any
    #: instrument of a domain, that domain is restricted to exactly those
    #: instruments; a domain with no allowlist entry is unrestricted. The
    #: denylist always removes.
    instrument_allowlist: tuple[InstrumentIdentity, ...] = ()
    instrument_denylist: tuple[InstrumentIdentity, ...] = ()

    #: POSITION CONSTRAINTS -- carried to Asset Expression / Portfolio
    #: Construction. Shorting defaults to NOT allowed: the system never
    #: assumes a user can hold net short exposure unless they say so.
    shorting_allowed: bool = False
    #: Maximum gross notional exposure as a multiple of capital. ``None`` =
    #: no cap stated (never interpreted as "unlimited leverage is fine").
    max_gross_leverage: float | None = Field(default=None, gt=0)
    liquidity_requirement: LiquidityRequirement = LiquidityRequirement.STANDARD

    #: RISK / HORIZON PREFERENCES -- the existing `InvestorProfile`, reused
    #: verbatim (holding period, risk style, drawdown, turnover, capital).
    risk_profile: InvestorProfile = DEFAULT_PROFILE

    @field_validator("allowed_domains")
    @classmethod
    def _canon_domains(cls, v: tuple[MandateDomain, ...]) -> tuple[MandateDomain, ...]:
        return _canonical_domains(v)

    @field_validator("instrument_allowlist", "instrument_denylist")
    @classmethod
    def _canon_instruments(cls, v: tuple[InstrumentIdentity, ...]) -> tuple[InstrumentIdentity, ...]:
        return _canonical_instruments(v)

    @model_validator(mode="after")
    def _consistent(self) -> ResearchMandate:
        allowed = set(self.allowed_domains)
        for inst in self.instrument_allowlist:
            if mandate_domain_for(inst.asset_domain) not in allowed:
                raise ValueError(
                    f"instrument_allowlist names {inst.asset_domain.value}:{inst.symbol}, but "
                    f"{inst.asset_domain.value} is not an allowed domain -- a mandate cannot both exclude a "
                    "domain and allow one of its instruments"
                )
        overlap = {(i.asset_domain, i.symbol) for i in self.instrument_allowlist} & {
            (i.asset_domain, i.symbol) for i in self.instrument_denylist
        }
        if overlap:
            names = ", ".join(f"{d.value}:{s}" for d, s in sorted(overlap))
            raise ValueError(f"instrument(s) both allowed and denied: {names}")
        return self

    # -- the prompt's named dimensions, read from the reused profile --------

    @property
    def investment_horizon(self) -> HoldingPeriod:
        return self.risk_profile.holding_period

    @property
    def risk_style(self) -> RiskStyle:
        return self.risk_profile.risk_style

    @property
    def turnover_sensitivity(self) -> TurnoverSensitivity | None:
        return self.risk_profile.turnover_sensitivity

    @property
    def capital_usd(self) -> float | None:
        return self.risk_profile.approximate_capital_usd

    def allows(self, domain: MandateDomain) -> bool:
        return domain in self.allowed_domains

    def fingerprint(self) -> str:
        """SHA-256 over the canonical JSON form -- provenance for "which
        mandate scoped this scan", never an identity input anywhere in the
        scientific plane."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "mandate1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> str:
        domains = " + ".join(DOMAIN_LABELS[d] for d in self.allowed_domains)
        shorting = "shorting allowed" if self.shorting_allowed else "no shorting"
        leverage = f"max {self.max_gross_leverage:g}x gross" if self.max_gross_leverage else "no leverage cap stated"
        return (
            f"{domains} · {self.investment_horizon.value} horizon · {self.risk_style.value} · "
            f"{shorting} · {leverage}"
        )


#: The not-yet-personalized default: every domain the registry has a real
#: evidence namespace for (derived from `AssetDomain`, never hand-listed), so
#: a new user sees triage across the platform's real universes. Options and
#: Crypto stay out until the user opts in.
DEFAULT_MANDATE = ResearchMandate(allowed_domains=tuple(mandate_domain_for(d) for d in AssetDomain))


class MandateStore:
    """Local JSON persistence for the ACCESS/CONSTRAINT half of a mandate,
    mirroring `ProfileStore` (gitignored `data/user_prefs/`, never the
    registry). The risk half is deliberately NOT stored here -- it is the
    user's saved `InvestorProfile`, composed in at load time, so the two can
    never drift into two disagreeing copies. Corrupt or missing state falls
    back to `DEFAULT_MANDATE`'s access fields."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_MANDATE_PATH

    def exists(self) -> bool:
        return self.path.exists()

    def load(self, *, risk_profile: InvestorProfile = DEFAULT_PROFILE) -> ResearchMandate:
        default = DEFAULT_MANDATE.model_copy(update={"risk_profile": risk_profile})
        if not self.path.exists():
            return default
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            data.pop("risk_profile", None)
            return ResearchMandate.model_validate({**data, "risk_profile": risk_profile.model_dump(mode="json")})
        except Exception:  # noqa: BLE001 -- a corrupt local preference file is not fatal
            return default

    def save(self, mandate: ResearchMandate) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = mandate.model_dump(mode="json", exclude={"risk_profile"})
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

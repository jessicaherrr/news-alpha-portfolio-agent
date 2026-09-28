"""Curated Practitioner Provider Registry (Release UX Part K, task spec
section 42).

`practitioner_connector.py`'s `INSTITUTION_ALLOWLIST` was a flat tuple of
name fragments; this module gives each entry an explicit category,
organization, allowed-domain hint, and default quality context, and
`INSTITUTION_ALLOWLIST` is now DERIVED from it (backward compatible -- the
connector's own matching logic is unchanged) so there is one source of truth
instead of two lists that could silently drift.

Brand reputation is never scientific truth (task spec section 42's own
warning): `default_quality_context` is research-PRIORITY metadata only,
identical in kind to `alpha_agent.knowledge.models.SourceQualityTier` --
never a validation threshold, never part of `ReliabilityPolicy`.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ProviderCategory(str, Enum):
    CENTRAL_BANK = "CENTRAL_BANK"
    EXCHANGE = "EXCHANGE"
    ASSET_MANAGER = "ASSET_MANAGER"
    INTERNATIONAL_ORG = "INTERNATIONAL_ORG"
    ACADEMIC_INSTITUTE = "ACADEMIC_INSTITUTE"


class PractitionerProvider(BaseModel):
    """One curated institutional source (task spec section 42's field list).
    `name_fragments` are the case-insensitive substrings matched against a
    paper's publisher/venue/host-organization string -- exactly what
    `practitioner_connector._matches_institution` already checks, just now
    traceable back to a named, categorized organization instead of a bare
    string in a flat tuple."""

    model_config = {"frozen": True, "extra": "forbid"}

    organization: str
    category: ProviderCategory
    name_fragments: tuple[str, ...]
    allowed_domains: tuple[str, ...] = ()
    default_quality_context: str = "Institutional research series; not independently audited here."


#: The curated registry. Broad and reviewed (task spec: "Do not hard-code
#: one institution as authoritative") -- no single entry is ever treated as
#: uniquely authoritative by the connector, which matches ANY of these.
PRACTITIONER_PROVIDERS: tuple[PractitionerProvider, ...] = (
    PractitionerProvider(
        organization="Federal Reserve System", category=ProviderCategory.CENTRAL_BANK,
        name_fragments=("federal reserve",), allowed_domains=("federalreserve.gov",),
    ),
    PractitionerProvider(
        organization="National Bureau of Economic Research", category=ProviderCategory.ACADEMIC_INSTITUTE,
        name_fragments=("national bureau of economic research", "nber"), allowed_domains=("nber.org",),
    ),
    PractitionerProvider(
        organization="Bank for International Settlements", category=ProviderCategory.INTERNATIONAL_ORG,
        name_fragments=("bank for international settlements", " bis "), allowed_domains=("bis.org",),
    ),
    PractitionerProvider(
        organization="International Monetary Fund", category=ProviderCategory.INTERNATIONAL_ORG,
        name_fragments=("international monetary fund", "imf working paper"), allowed_domains=("imf.org",),
    ),
    PractitionerProvider(
        organization="European Central Bank", category=ProviderCategory.CENTRAL_BANK,
        name_fragments=("european central bank",), allowed_domains=("ecb.europa.eu",),
    ),
    PractitionerProvider(
        organization="Bank of England", category=ProviderCategory.CENTRAL_BANK,
        name_fragments=("bank of england",), allowed_domains=("bankofengland.co.uk",),
    ),
    PractitionerProvider(
        organization="Bank of Canada", category=ProviderCategory.CENTRAL_BANK,
        name_fragments=("bank of canada",), allowed_domains=("bankofcanada.ca",),
    ),
    PractitionerProvider(
        organization="Bank of Japan", category=ProviderCategory.CENTRAL_BANK,
        name_fragments=("bank of japan",), allowed_domains=("boj.or.jp",),
    ),
    PractitionerProvider(
        organization="CME Group", category=ProviderCategory.EXCHANGE,
        name_fragments=("cme group", "chicago mercantile exchange"), allowed_domains=("cmegroup.com",),
    ),
    PractitionerProvider(
        organization="Intercontinental Exchange", category=ProviderCategory.EXCHANGE,
        name_fragments=("intercontinental exchange", "ice futures"), allowed_domains=("theice.com",),
    ),
    PractitionerProvider(
        organization="AQR Capital Management", category=ProviderCategory.ASSET_MANAGER,
        name_fragments=("aqr capital",), allowed_domains=("aqr.com",),
    ),
    PractitionerProvider(
        organization="Man Group / Man AHL", category=ProviderCategory.ASSET_MANAGER,
        name_fragments=("man group", "man ahl"), allowed_domains=("man.com",),
    ),
    PractitionerProvider(
        organization="Winton", category=ProviderCategory.ASSET_MANAGER,
        name_fragments=("winton",), allowed_domains=("winton.com",),
    ),
    PractitionerProvider(
        organization="Campbell & Company", category=ProviderCategory.ASSET_MANAGER,
        name_fragments=("campbell & company",), allowed_domains=("campbell.com",),
    ),
    PractitionerProvider(
        organization="Systematica Investments", category=ProviderCategory.ASSET_MANAGER,
        name_fragments=("systematica",), allowed_domains=("systematica.com",),
    ),
    PractitionerProvider(
        organization="World Bank", category=ProviderCategory.INTERNATIONAL_ORG,
        name_fragments=("world bank",), allowed_domains=("worldbank.org",),
    ),
    PractitionerProvider(
        organization="U.S. Office of Financial Research", category=ProviderCategory.CENTRAL_BANK,
        name_fragments=("office of financial research",), allowed_domains=("financialresearch.gov",),
    ),
)


def name_fragments_allowlist() -> tuple[str, ...]:
    """The flat allowlist `practitioner_connector.py` matches against --
    derived here so the connector and this registry can never silently
    drift apart."""
    return tuple(frag for provider in PRACTITIONER_PROVIDERS for frag in provider.name_fragments)


def providers_by_category(category: ProviderCategory) -> tuple[PractitionerProvider, ...]:
    return tuple(p for p in PRACTITIONER_PROVIDERS if p.category == category)


def provider_for_fragment(fragment: str) -> PractitionerProvider | None:
    lowered = fragment.lower()
    for provider in PRACTITIONER_PROVIDERS:
        if lowered in provider.name_fragments:
            return provider
    return None


__all__ = [
    "PRACTITIONER_PROVIDERS",
    "PractitionerProvider",
    "ProviderCategory",
    "name_fragments_allowlist",
    "provider_for_fragment",
    "providers_by_category",
]

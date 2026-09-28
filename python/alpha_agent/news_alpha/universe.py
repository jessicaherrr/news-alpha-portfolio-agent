"""News Alpha Phase A -- the ALLOWED ASSET UNIVERSE: a session-level SCOPE
over the existing per-domain universes, never a replacement for them.

Domain truth stays exactly where it already lives:

* Futures -- `alpha_agent.marketdata.product_catalog.PRODUCT_CATALOG` (the
  catalogued MARKET universe) with `RESEARCH_UNIVERSE` (the certified research
  roots, the same constant `translation.schemas.CERTIFIED_ROOTS` equals).
* ETF -- `alpha_agent.etf.universe.PILOT_UNIVERSE` (+ its `EXPOSURE` labels).
* Equity -- `alpha_agent.equities.universe.EQUITY_UNIVERSE` (+ its `SECTOR`s).
* Crypto -- `alpha_agent.crypto` is a synthetic-only scaffold (zero real
  vendor data); CME crypto futures are catalogued under FUTURES.
* Options -- no universe exists on this platform (Phase 9 proposal §C defers it).

`resolve_allowed_universe` is a PURE function of (mandate, capabilities):
the only filesystem fact it depends on (whether ETF/Equity raw bars were
acquired) arrives as an explicit `DomainCapabilities` snapshot, so the same
inputs always resolve to the same universe. No registry read, no network.
The same snapshot also carries each Futures root's acquired bar coverage
(read from the committed data catalog) for Phase D's measurement
resolution -- one capability snapshot for the whole pipeline, never a second.
"""
from __future__ import annotations

import json
from datetime import date
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from alpha_agent.equities.universe import EQUITY_UNIVERSE, SECTOR
from alpha_agent.etf.universe import EXPOSURE as ETF_EXPOSURE
from alpha_agent.etf.universe import PILOT_UNIVERSE
from alpha_agent.marketdata.product_catalog import (
    ASSET_CLASS_LABELS,
    PRODUCT_CATALOG,
    RESEARCH_UNIVERSE,
)
from alpha_agent.news_alpha.mandate import (
    LiquidityRequirement,
    MandateDomain,
    ResearchMandate,
)
from alpha_agent.recommendation.profile import REPO_ROOT

__all__ = [
    "FUTURES_CATALOG_PATH",
    "AllowedAssetUniverse",
    "DomainCapabilities",
    "DomainScope",
    "DomainSupport",
    "FuturesBarCoverage",
    "UniverseInstrument",
    "probe_domain_capabilities",
    "read_futures_bar_coverage",
    "resolve_allowed_universe",
]

UNIVERSE_SCHEMA_VERSION = "allowed-asset-universe/1"
FUTURES_CATALOG_SOURCE = "data/catalog/real_cme_dataset.json"
FUTURES_CATALOG_PATH = REPO_ROOT / FUTURES_CATALOG_SOURCE


class DomainSupport(str, Enum):
    """How far this PLATFORM can research a domain today -- platform truth,
    independent of any user's mandate."""

    #: Real acquired data and an end-to-end research path exist.
    RESEARCH_READY = "RESEARCH_READY"
    #: A declared universe exists, but its market data has not been acquired.
    DATA_NOT_ACQUIRED = "DATA_NOT_ACQUIRED"
    #: Data acquired, but the domain's research loop is not wired yet.
    FOUNDATION_ONLY = "FOUNDATION_ONLY"
    #: Synthetic scaffold only -- no real vendor data at all.
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    #: No instrument universe exists on this platform.
    NOT_SUPPORTED = "NOT_SUPPORTED"


class FuturesBarCoverage(BaseModel):
    """One Futures root's ACQUIRED research bars, read from the committed CME
    data catalog (`data/catalog/real_cme_dataset.json` -- the derived,
    human-readable index over the immutable raw store). Certification alone
    never implies coverage: a catalogued-but-unacquired root has no entry."""

    model_config = {"frozen": True, "extra": "forbid"}

    root: str
    dataset: str
    data_schema: str
    start: date
    #: Exclusive -- ``2025-01-01`` means data ending 2024-12-31 (CLAUDE.md).
    end_exclusive: date
    continuous_segments: int
    #: Raw-contract files around observed rolls: the only places two
    #: contract months exist side by side, i.e. the only curve observations.
    roll_overlap_artifacts: int
    source: str = FUTURES_CATALOG_SOURCE


class DomainCapabilities(BaseModel):
    """The only filesystem-dependent inputs to the News Alpha pipeline
    (universe resolution, and since Phase D measurement resolution),
    captured as an explicit snapshot (see `probe_domain_capabilities`)."""

    model_config = {"frozen": True, "extra": "forbid"}

    etf_market_data_acquired: bool
    equity_market_data_acquired: bool
    #: Futures roots with acquired research bars (Phase D, additive). Empty
    #: when the catalog is absent -- never filled in from certification.
    futures_bars: tuple[FuturesBarCoverage, ...] = ()

    def futures_coverage(self, root: str) -> FuturesBarCoverage | None:
        return next((c for c in self.futures_bars if c.root == root), None)


def read_futures_bar_coverage(path: Path = FUTURES_CATALOG_PATH) -> tuple[FuturesBarCoverage, ...]:
    """Per-root research-bar coverage from the committed catalog: the span of
    each root's CONTINUOUS_FRONT bars plus its roll-overlap raw-contract count.
    A catalog row reaching past the locked holdout boundary fails loudly."""
    from alpha_agent.registry.holdout_guard import HOLDOUT_START, HoldoutAccessError

    if not path.exists():
        return ()
    rows = json.loads(path.read_text(encoding="utf-8"))
    holdout = date.fromisoformat(HOLDOUT_START)
    out = []
    for root in sorted({r["root"] for r in rows}):
        mine = [r for r in rows if r["root"] == root]
        if any(date.fromisoformat(r["end"]) > holdout for r in mine):
            raise HoldoutAccessError(f"catalog row for {root} ends after the locked holdout start {HOLDOUT_START}")
        front = [r for r in mine if r["component"] == "CONTINUOUS_FRONT"]
        if not front:
            continue
        out.append(FuturesBarCoverage(
            root=root, dataset=front[0]["dataset"], data_schema=front[0]["schema"],
            start=min(date.fromisoformat(r["start"]) for r in front),
            end_exclusive=max(date.fromisoformat(r["end"]) for r in front),
            continuous_segments=len(front),
            roll_overlap_artifacts=sum(r["component"] == "ROLL_OVERLAP_RAW" for r in mine),
        ))
    return tuple(out)


def probe_domain_capabilities() -> DomainCapabilities:
    """Cheap path-existence probes against the immutable raw store (no
    decode, no network) plus one read of the committed futures catalog.
    Imported lazily so resolving a universe from an explicit snapshot never
    pulls in the data-loading stack."""
    from alpha_agent.equities.data_source import primary_listing_data_acquired as equity_acquired
    from alpha_agent.etf.data_source import primary_listing_data_acquired as etf_acquired

    return DomainCapabilities(
        etf_market_data_acquired=etf_acquired(PILOT_UNIVERSE[0]),
        equity_market_data_acquired=equity_acquired(EQUITY_UNIVERSE[0]),
        futures_bars=read_futures_bar_coverage(),
    )


class UniverseInstrument(BaseModel):
    """One instrument inside the allowed universe. ``group`` is the domain's
    own existing classification (Futures asset class, Equity sector, ETF
    exposure) -- the handle the impact scan's channel table selects by."""

    model_config = {"frozen": True, "extra": "forbid"}

    domain: MandateDomain
    symbol: str
    display_name: str
    group: str
    group_label: str
    #: True only when this instrument can enter today's research stack (a
    #: certified Futures root; an ETF with acquired pilot data).
    research_ready: bool
    readiness_note: str
    source: str


class DomainScope(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    domain: MandateDomain
    in_mandate: bool
    support: DomainSupport
    support_note: str
    #: The domain's universe after the mandate's instrument constraints.
    #: Always ``()`` for a domain outside the mandate -- the system never
    #: enumerates what the user did not permit it to investigate.
    instruments: tuple[UniverseInstrument, ...] = ()
    #: Instruments the mandate's allowlist/denylist removed from this domain.
    removed_by_constraints: tuple[UniverseInstrument, ...] = ()
    #: Allowlisted symbols that are not in this domain's universe at all --
    #: reported, never silently added.
    unresolved_allowlist: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


class AllowedAssetUniverse(BaseModel):
    """Every `MandateDomain`, in canonical order, each scoped by the
    mandate. Excluded domains stay visible (``in_mandate=False``) so a UI
    can say "excluded by your mandate" rather than silently omitting them."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = UNIVERSE_SCHEMA_VERSION
    mandate_fingerprint: str
    capabilities: DomainCapabilities
    scopes: tuple[DomainScope, ...]
    constraint_notes: tuple[str, ...] = ()

    def scope(self, domain: MandateDomain) -> DomainScope:
        return next(s for s in self.scopes if s.domain is domain)

    @property
    def allowed_domains(self) -> tuple[MandateDomain, ...]:
        return tuple(s.domain for s in self.scopes if s.in_mandate)

    def find(self, domain: MandateDomain, symbol: str) -> UniverseInstrument | None:
        return next((i for i in self.scope(domain).instruments if i.symbol == symbol), None)


# ---------------------------------------------------------------------------
# per-domain base universes -- read verbatim from each domain's own module
# ---------------------------------------------------------------------------

_FUTURES_SOURCE = "alpha_agent.marketdata.product_catalog.PRODUCT_CATALOG"
_ETF_SOURCE = "alpha_agent.etf.universe.PILOT_UNIVERSE"
_EQUITY_SOURCE = "alpha_agent.equities.universe.EQUITY_UNIVERSE"
_CRYPTO_SOURCE = "alpha_agent.crypto (synthetic scaffold)"


def _futures_base() -> tuple[tuple[UniverseInstrument, ...], DomainSupport, str]:
    certified = set(RESEARCH_UNIVERSE)
    instruments = tuple(
        UniverseInstrument(
            domain=MandateDomain.FUTURES, symbol=e.root_symbol, display_name=e.display_name,
            group=e.asset_class.value, group_label=ASSET_CLASS_LABELS[e.asset_class],
            research_ready=e.root_symbol in certified,
            readiness_note=(
                "Certified research root (validated data + research stack)."
                if e.root_symbol in certified
                else "Catalogued for observation only -- outside the certified research universe."
            ),
            source=_FUTURES_SOURCE,
        )
        for e in PRODUCT_CATALOG
    )
    note = (
        f"{len(certified)} certified research roots ({', '.join(RESEARCH_UNIVERSE)}); the other "
        f"{len(instruments) - len(certified)} catalogued roots are observation-only."
    )
    return instruments, DomainSupport.RESEARCH_READY, note


def _etf_base(caps: DomainCapabilities) -> tuple[tuple[UniverseInstrument, ...], DomainSupport, str]:
    ready = caps.etf_market_data_acquired
    instruments = tuple(
        UniverseInstrument(
            domain=MandateDomain.ETF, symbol=t, display_name=t, group=ETF_EXPOSURE[t], group_label=ETF_EXPOSURE[t],
            research_ready=ready,
            readiness_note="Phase 6 pilot ticker with acquired data." if ready else "Pilot data not present in this checkout.",
            source=_ETF_SOURCE,
        )
        for t in PILOT_UNIVERSE
    )
    if ready:
        return instruments, DomainSupport.RESEARCH_READY, f"Phase 6 ETF pilot: {len(instruments)} tickers, data acquired."
    return instruments, DomainSupport.DATA_NOT_ACQUIRED, "Phase 6 ETF pilot universe declared; raw data not present here."


def _equity_base(caps: DomainCapabilities) -> tuple[tuple[UniverseInstrument, ...], DomainSupport, str]:
    acquired = caps.equity_market_data_acquired
    instruments = tuple(
        UniverseInstrument(
            domain=MandateDomain.EQUITY, symbol=t, display_name=t, group=SECTOR[t], group_label=SECTOR[t],
            research_ready=False,
            readiness_note=(
                "Data acquired; the Equity research loop is Phase 9.2 scope."
                if acquired
                else "Declared Phase 9.1 universe -- market data not acquired yet (separate cost-approved step)."
            ),
            source=_EQUITY_SOURCE,
        )
        for t in EQUITY_UNIVERSE
    )
    if acquired:
        return instruments, DomainSupport.FOUNDATION_ONLY, "Phase 9.1 Equities foundation: data acquired, research loop not wired."
    return (
        instruments, DomainSupport.DATA_NOT_ACQUIRED,
        f"Phase 9.1 Equities foundation: {len(instruments)} declared names, market data not acquired yet.",
    )


def _base_for(domain: MandateDomain, caps: DomainCapabilities) -> tuple[tuple[UniverseInstrument, ...], DomainSupport, str, tuple[str, ...]]:
    if domain is MandateDomain.FUTURES:
        return (*_futures_base(), (_FUTURES_SOURCE,))
    if domain is MandateDomain.ETF:
        return (*_etf_base(caps), (_ETF_SOURCE,))
    if domain is MandateDomain.EQUITY:
        return (*_equity_base(caps), (_EQUITY_SOURCE,))
    if domain is MandateDomain.CRYPTO:
        return (
            (), DomainSupport.SYNTHETIC_ONLY,
            (
                "Synthetic scaffold only (Phase 22) -- no real crypto vendor data. CME crypto futures "
                "(BTC, MBT, ETH) are catalogued under Futures."
            ),
            (_CRYPTO_SOURCE,),
        )
    return (
        (), DomainSupport.NOT_SUPPORTED,
        "No options universe, chain, or pricing data exists on this platform (Options Lab deferred).", (),
    )


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def _scope_for(domain: MandateDomain, mandate: ResearchMandate, caps: DomainCapabilities) -> DomainScope:
    base, support, note, sources = _base_for(domain, caps)
    if not mandate.allows(domain):
        return DomainScope(domain=domain, in_mandate=False, support=support, support_note=note, sources=sources)

    allow = {i.symbol for i in mandate.instrument_allowlist if i.asset_domain.value == domain.value}
    deny = {i.symbol for i in mandate.instrument_denylist if i.asset_domain.value == domain.value}
    universe_symbols = {i.symbol for i in base}

    kept = tuple(i for i in base if (not allow or i.symbol in allow) and i.symbol not in deny)
    removed = tuple(i for i in base if i not in kept)
    unresolved = tuple(sorted(allow - universe_symbols))
    return DomainScope(
        domain=domain, in_mandate=True, support=support, support_note=note, instruments=kept,
        removed_by_constraints=removed, unresolved_allowlist=unresolved, sources=sources,
    )


def _constraint_notes(mandate: ResearchMandate) -> tuple[str, ...]:
    notes: list[str] = []
    if mandate.liquidity_requirement is not LiquidityRequirement.ANY:
        notes.append(
            f"Liquidity requirement {mandate.liquidity_requirement.value}: no domain universe carries typed "
            "ADV/liquidity metadata yet (each was selected as liquid by construction), so it is carried to "
            "Asset Expression / Portfolio Construction rather than applied as a filter here."
        )
    if mandate.max_gross_leverage is not None:
        notes.append(
            f"Max gross leverage {mandate.max_gross_leverage:g}x applies to total notional exposure across every "
            "domain alike at Portfolio Construction -- it does not exclude any domain."
        )
    if not mandate.shorting_allowed:
        notes.append(
            "Shorting not allowed: research may still study downward-pressure hypotheses, but they cannot be "
            "expressed as net short positions under this mandate."
        )
    return tuple(notes)


def resolve_allowed_universe(
    mandate: ResearchMandate, *, capabilities: DomainCapabilities | None = None,
) -> AllowedAssetUniverse:
    """Compose ``mandate`` with every existing domain universe. Deterministic
    for fixed ``capabilities``; ``None`` probes the local raw store once."""
    caps = capabilities or probe_domain_capabilities()
    return AllowedAssetUniverse(
        mandate_fingerprint=mandate.fingerprint(),
        capabilities=caps,
        scopes=tuple(_scope_for(d, mandate, caps) for d in MandateDomain),
        constraint_notes=_constraint_notes(mandate),
    )


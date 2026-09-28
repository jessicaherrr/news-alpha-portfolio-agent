"""News Alpha Phase D -- MEASUREMENT SPECS and point-in-time DATA FIELD
RESOLUTION.

    ... -> SignalPath -> Economic Consequence -> Asset Expression
        -> MEASUREMENT SPEC (this module) -> PIT DATA FIELD RESOLUTION (this module)

A `MeasurementSpec` is one measurement of one instrument (NQ price momentum,
NVDA revenue growth, HG open interest). Its IDEAL economic measurement is the
Phase 1 `translation.MeasurableVariable`, reused verbatim -- never a second
"measurable variable" type -- and its `DataFieldResolution` answers, from
repository state only: which dataset and field actually carry it, at what
frequency, over which window, whether each value was KNOWABLE at its
timestamp, with what publication lag, through which transform, as a proxy
or not, and -- when it cannot be used -- exactly why.

RESOLUTION (``field-resolution/1``). Each `DataRequirement` has one resolver
that reads existing authorities, never a parallel availability table:

* market bars -- the committed CME data catalog (Futures, via the Phase A
  `DomainCapabilities` snapshot), `etf.universe.DATASET_WINDOWS` and
  `equities.universe.DATASET_WINDOWS` with the snapshot's acquired flags;
* fundamentals / analyst estimates -- the Phase 9.1
  `equities.data_availability` findings (the status follows the recorded
  finding, never a re-assertion);
* executability -- `translation.researchability.classify_factor` over the
  live `FeatureRegistry`, and each registered kind's own point-in-time flag;
* Options / Crypto -- the universe's `DomainSupport` (no options universe;
  crypto is a synthetic scaffold), with `crypto.provenance.DataProvenanceRole`
  marking synthetic data so it can never be promoted to real.

Typed `ResolutionIssue`s decide the status by fixed precedence:
DOMAIN_UNAVAILABLE > NOT_PIT_SAFE > MISSING > NOT_EXECUTABLE > PARTIAL >
AVAILABLE_WITH_PROXY > AVAILABLE. A structural point-in-time violation
outranks missing data because acquiring the data would not fix it.

POINT IN TIME. A value that can be queried today is not automatically
usable historically. `pit_safe` is True only when a typed rule establishes
that each value was knowable at its timestamp, False for a known violation
(a hindsight constituent list, a retrospective feature), and None (UNKNOWN)
when there is no data to judge -- never guessed. Only a resolution that is
usable (`historically_usable`: a data-present status, PIT-safe, executable,
REAL data) may ever feed a historical candidate signal, and
`available_on` refuses any date inside the locked 2025 holdout.

HYPOTHESIS PLANE ONLY -- no registry, network or LLM access; nothing here
loads a bar.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from pydantic import BaseModel, model_validator

from alpha_agent.crypto.provenance import DataProvenanceRole
from alpha_agent.equities import data_availability as equity_findings
from alpha_agent.equities import universe as equity_universe
from alpha_agent.etf import universe as etf_universe
from alpha_agent.features.registry import REGISTRY, FeatureRegistry
from alpha_agent.news_alpha.expression_library import (
    DataRequirement,
    MeasurementRole,
    MeasurementTemplate,
)
from alpha_agent.news_alpha.mandate import DOMAIN_LABELS, MandateDomain
from alpha_agent.news_alpha.universe import (
    FUTURES_CATALOG_SOURCE,
    DomainCapabilities,
    DomainSupport,
)
from alpha_agent.registry.holdout_guard import HOLDOUT_START, HoldoutAccessError
from alpha_agent.translation.researchability import classify_factor
from alpha_agent.translation.schemas import MeasurableVariable, ResearchabilityStatus

__all__ = [
    "FIELD_RESOLUTION_RULE",
    "ISSUE_TEXT",
    "REQUIREMENT_LABELS",
    "DataFieldResolution",
    "DataFrequency",
    "MeasurementSpec",
    "PublicationLag",
    "ResolutionGap",
    "ResolutionIssue",
    "ResolutionStatus",
    "build_measurement_spec",
    "resolve_field",
    "status_for",
]

FIELD_RESOLUTION_RULE = "field-resolution/1"
_HOLDOUT = date.fromisoformat(HOLDOUT_START)


class ResolutionStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    AVAILABLE_WITH_PROXY = "AVAILABLE_WITH_PROXY"
    #: Real, PIT-safe and executable over a contiguous window shorter than
    #: the research window.
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    #: Not established as knowable at each timestamp (a known violation, or
    #: unverified) -- `pit_safe` keeps False apart from None.
    NOT_PIT_SAFE = "NOT_PIT_SAFE"
    DOMAIN_UNAVAILABLE = "DOMAIN_UNAVAILABLE"
    NOT_EXECUTABLE = "NOT_EXECUTABLE"


class ResolutionIssue(str, Enum):
    # -- DOMAIN_UNAVAILABLE ------------------------------------------------------
    DOMAIN_NOT_SUPPORTED = "DOMAIN_NOT_SUPPORTED"
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    # -- NOT_PIT_SAFE ------------------------------------------------------------
    HINDSIGHT_MEMBERSHIP = "HINDSIGHT_MEMBERSHIP"
    RETROSPECTIVE_FEATURE = "RETROSPECTIVE_FEATURE"
    PIT_UNVERIFIED = "PIT_UNVERIFIED"
    # -- MISSING -------------------------------------------------------------------
    DATA_NOT_ACQUIRED = "DATA_NOT_ACQUIRED"
    CONNECTOR_NOT_BUILT = "CONNECTOR_NOT_BUILT"
    PAID_SOURCE_ONLY = "PAID_SOURCE_ONLY"
    NO_SOURCE_FOUND = "NO_SOURCE_FOUND"
    NOT_INVESTIGATED = "NOT_INVESTIGATED"
    # -- NOT_EXECUTABLE ----------------------------------------------------------
    CROSS_INSTRUMENT = "CROSS_INSTRUMENT"
    FEATURE_NOT_REGISTERED = "FEATURE_NOT_REGISTERED"
    # -- PARTIAL / AVAILABLE_WITH_PROXY -------------------------------------------
    PARTIAL_HISTORY = "PARTIAL_HISTORY"
    PROXY = "PROXY"


_S, _I = ResolutionStatus, ResolutionIssue
_ISSUE_STATUS: dict[ResolutionIssue, ResolutionStatus] = {
    _I.DOMAIN_NOT_SUPPORTED: _S.DOMAIN_UNAVAILABLE, _I.SYNTHETIC_ONLY: _S.DOMAIN_UNAVAILABLE,
    _I.HINDSIGHT_MEMBERSHIP: _S.NOT_PIT_SAFE, _I.RETROSPECTIVE_FEATURE: _S.NOT_PIT_SAFE,
    _I.PIT_UNVERIFIED: _S.NOT_PIT_SAFE,
    _I.DATA_NOT_ACQUIRED: _S.MISSING, _I.CONNECTOR_NOT_BUILT: _S.MISSING, _I.PAID_SOURCE_ONLY: _S.MISSING,
    _I.NO_SOURCE_FOUND: _S.MISSING, _I.NOT_INVESTIGATED: _S.MISSING,
    _I.CROSS_INSTRUMENT: _S.NOT_EXECUTABLE, _I.FEATURE_NOT_REGISTERED: _S.NOT_EXECUTABLE,
    _I.PARTIAL_HISTORY: _S.PARTIAL,
    _I.PROXY: _S.AVAILABLE_WITH_PROXY,
}
_PRECEDENCE = (
    _S.DOMAIN_UNAVAILABLE, _S.NOT_PIT_SAFE, _S.MISSING, _S.NOT_EXECUTABLE, _S.PARTIAL, _S.AVAILABLE_WITH_PROXY,
)
_USABLE = frozenset({_S.AVAILABLE, _S.AVAILABLE_WITH_PROXY, _S.PARTIAL})
#: Issues that qualify a usable measurement rather than block it.
_NON_BLOCKING = frozenset({_I.PARTIAL_HISTORY, _I.PROXY})
_DATA_ISSUES = frozenset({
    _I.DOMAIN_NOT_SUPPORTED, _I.SYNTHETIC_ONLY, _I.DATA_NOT_ACQUIRED, _I.CONNECTOR_NOT_BUILT, _I.PAID_SOURCE_ONLY,
    _I.NO_SOURCE_FOUND, _I.NOT_INVESTIGATED,
})

#: Short human text per issue -- the UI and the blocker summary read it.
ISSUE_TEXT: dict[ResolutionIssue, str] = {
    _I.DOMAIN_NOT_SUPPORTED: "no universe or data exists for this domain on the platform",
    _I.SYNTHETIC_ONLY: "only a synthetic scaffold exists -- never real evidence",
    _I.HINDSIGHT_MEMBERSHIP: "the only constituent list was chosen with hindsight (not point-in-time)",
    _I.RETROSPECTIVE_FEATURE: "a required feature looks forward in time",
    _I.PIT_UNVERIFIED: "knowability at each timestamp is not established",
    _I.DATA_NOT_ACQUIRED: "not acquired",
    _I.CONNECTOR_NOT_BUILT: "a source exists, but no connector ingests it",
    _I.PAID_SOURCE_ONLY: "point-in-time history is available only from paid vendors",
    _I.NO_SOURCE_FOUND: "no usable source was found",
    _I.NOT_INVESTIGATED: "no source has been investigated",
    _I.CROSS_INSTRUMENT: "needs more than one series at once -- no registered feature does that",
    _I.FEATURE_NOT_REGISTERED: "no registered feature computes it",
    _I.PARTIAL_HISTORY: "covers only part of the research window",
    _I.PROXY: "a proxy stands in for the ideal quantity",
}

REQUIREMENT_LABELS: dict[DataRequirement, str] = {
    DataRequirement.MARKET_BARS: "Market bars",
    DataRequirement.BENCHMARK_BARS: "Instrument and SPY benchmark bars",
    DataRequirement.CURVE_CONTRACTS: "Simultaneous contract months",
    DataRequirement.SPOT_REFERENCE: "Spot reference prices",
    DataRequirement.OPEN_INTEREST: "Exchange open interest",
    DataRequirement.COT_REPORTS: "CFTC Commitments of Traders reports",
    DataRequirement.PHYSICAL_INVENTORY: "Official inventory data",
    DataRequirement.PIT_FUNDAMENTALS: "Point-in-time fundamentals",
    DataRequirement.CONSENSUS_HISTORY: "Point-in-time analyst estimates",
    DataRequirement.FUND_FLOWS: "Fund flow history",
    DataRequirement.CONSTITUENT_MEMBERSHIP: "Point-in-time constituents",
    DataRequirement.OPTION_CHAIN: "Option chains",
    DataRequirement.CRYPTO_DERIVATIVES: "Crypto derivatives data",
    DataRequirement.ON_CHAIN_METRICS: "On-chain data",
}


class DataFrequency(str, Enum):
    MINUTE = "MINUTE"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    QUARTERLY = "QUARTERLY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PublicationLag(str, Enum):
    """When a value becomes knowable relative to the period it describes."""

    #: A bar is stamped at its interval START; its values are knowable at
    #: the interval END.
    AT_BAR_CLOSE = "AT_BAR_CLOSE"
    #: Exchange statistics for a session, published the next session.
    NEXT_SESSION = "NEXT_SESSION"
    #: Released days after its reference date (CFTC COT: Tuesday positions,
    #: released Friday).
    WEEKLY_RELEASE = "WEEKLY_RELEASE"
    #: Knowable from the filing's SEC acceptance timestamp, weeks after the
    #: fiscal period ends.
    FILING_ACCEPTANCE = "FILING_ACCEPTANCE"
    #: Knowable from each release's own publication timestamp.
    AT_PUBLICATION = "AT_PUBLICATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


def status_for(issues: tuple[ResolutionIssue, ...] | set[ResolutionIssue]) -> ResolutionStatus:
    """``field-resolution/1``: the status the issues imply, by precedence."""
    implied = {_ISSUE_STATUS[i] for i in issues}
    return next((s for s in _PRECEDENCE if s in implied), _S.AVAILABLE)


class ResolutionGap(BaseModel):
    """One named blocker: the typed issue and the gap that closing would
    remove ("Equity daily bars not acquired")."""

    model_config = {"frozen": True, "extra": "forbid"}

    issue: ResolutionIssue
    label: str


def _rank(issue: ResolutionIssue) -> int:
    return _PRECEDENCE.index(_ISSUE_STATUS[issue])


class DataFieldResolution(BaseModel):
    """Where one measurement's data actually comes from -- or exactly why it
    cannot be used. Every field is read from repository state."""

    model_config = {"frozen": True, "extra": "forbid"}

    status: ResolutionStatus
    issues: tuple[ResolutionIssue, ...] = ()
    dataset: str | None = None
    data_schema: str | None = None
    field: str | None = None
    frequency: DataFrequency = DataFrequency.NOT_APPLICABLE
    coverage_start: date | None = None
    #: Exclusive; never later than the locked holdout start.
    coverage_end_exclusive: date | None = None
    coverage_note: str = ""
    #: True: each value was knowable at its timestamp under a typed rule.
    #: False: a known violation. None: UNKNOWN (no data to judge).
    pit_safe: bool | None = None
    pit_rule: str
    publication_lag: PublicationLag = PublicationLag.NOT_APPLICABLE
    transform: str
    feature_kinds: tuple[str, ...] = ()
    #: Today's research stack (FeatureRegistry -> StrategySpec) can compute it.
    executable: bool = False
    #: What the available quantity stands in for, when it is not the ideal.
    proxy_note: str | None = None
    missing_reason: str | None = None
    #: EVERY named blocker, primary first (the one whose issue decides the
    #: status), then any others by status precedence. A measurement can wait
    #: on several gaps at once (equity relative strength needs equity bars
    #: AND a cross-instrument feature), so closing one gap removes one
    #: blocker -- it only makes the measurement usable when it was the last.
    #: Empty exactly when usable.
    gaps: tuple[ResolutionGap, ...] = ()
    #: REAL or SYNTHETIC data behind it; None when there is none.
    data_role: DataProvenanceRole | None = None
    provenance: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> DataFieldResolution:
        issues = set(self.issues)
        if self.status is not status_for(issues):
            raise ValueError(f"status {self.status.value} does not follow from its issues")
        if self.data_role is DataProvenanceRole.SYNTHETIC and _I.SYNTHETIC_ONLY not in issues:
            raise ValueError("synthetic data can only ever resolve as DOMAIN_UNAVAILABLE -- never promoted to real")
        if issues & {_I.HINDSIGHT_MEMBERSHIP, _I.RETROSPECTIVE_FEATURE} and self.pit_safe is not False:
            raise ValueError("a known point-in-time violation must set pit_safe=False")
        if _I.PIT_UNVERIFIED in issues and self.pit_safe is not None:
            raise ValueError("unverified knowability is pit_safe=None")
        if _I.PROXY in issues and not self.proxy_note:
            raise ValueError("a proxy resolution must say what the proxy stands in for")
        if self.executable == bool(issues & {_I.CROSS_INSTRUMENT, _I.FEATURE_NOT_REGISTERED}):
            raise ValueError("executable exactly when no executability issue is present")
        if self.coverage_end_exclusive is not None and self.coverage_end_exclusive > _HOLDOUT:
            raise ValueError("coverage may never reach into the locked holdout")
        if self.status in _USABLE:
            if self.pit_safe is not True or self.data_role is not DataProvenanceRole.REAL:
                raise ValueError("a usable resolution must be point-in-time safe and backed by REAL data")
            if not (self.dataset and self.field and self.coverage_start and self.coverage_end_exclusive):
                raise ValueError("a usable resolution names its dataset, field and coverage window")
            if self.gaps:
                raise ValueError("a usable resolution has no gap")
        else:
            if not (self.missing_reason and self.gaps):
                raise ValueError("an unusable resolution must say why, and name its gaps")
            if status_for((self.gaps[0].issue,)) is not self.status:
                raise ValueError("the primary gap must be the one that decides the status")
            if not {g.issue for g in self.gaps} <= issues:
                raise ValueError("every gap names one of the resolution's own issues")
            if not (issues - _NON_BLOCKING) <= {g.issue for g in self.gaps}:
                raise ValueError("every blocking issue is named by a gap -- none is dropped")
            ranks = [_rank(g.issue) for g in self.gaps]
            if ranks != sorted(ranks):
                raise ValueError("gaps are ordered primary first, then by status precedence")
        return self

    @property
    def gap(self) -> str | None:
        """The primary blocker's label (``None`` when usable)."""
        return self.gaps[0].label if self.gaps else None

    @property
    def additional_gaps(self) -> tuple[str, ...]:
        """Blockers that remain even if the primary gap is closed."""
        return tuple(g.label for g in self.gaps[1:])

    @property
    def historically_usable(self) -> bool:
        """May feed a HISTORICAL candidate signal: real data, knowable at each
        timestamp, computable by today's research stack."""
        return self.status in _USABLE

    def available_on(self, day: date) -> bool:
        """Whether a value for ``day`` exists and was knowable -- the
        "which measurements did we actually have then?" question. A date
        inside the locked holdout is refused, never answered."""
        if day >= _HOLDOUT:
            raise HoldoutAccessError(f"{day} is inside the locked holdout (>= {HOLDOUT_START})")
        return (
            self.historically_usable and self.coverage_start is not None
            and self.coverage_end_exclusive is not None and self.coverage_start <= day < self.coverage_end_exclusive
        )


class MeasurementSpec(BaseModel):
    """One measurement of one instrument. `variable` is the IDEAL economic
    measurement (`translation.MeasurableVariable`, reused); its point-in-time
    fields always equal the resolution's (enforced)."""

    model_config = {"frozen": True, "extra": "forbid"}

    spec_id: str
    measurement_id: str
    label: str
    role: MeasurementRole
    domain: MandateDomain
    #: For a domain with no instrument universe (Options, Crypto) the
    #: underlying the expression names, or ``None`` when it names none.
    symbol: str | None
    requirement: DataRequirement
    variable: MeasurableVariable
    resolution: DataFieldResolution
    #: The asset expressions this measurement serves.
    expression_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _variable_matches_resolution(self) -> MeasurementSpec:
        if self.variable.point_in_time_available != self.resolution.pit_safe:
            raise ValueError("the ideal variable's point-in-time claim must equal the resolution's")
        return self

    @property
    def target(self) -> str:
        return self.symbol or DOMAIN_LABELS[self.domain]


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

_BAR_RULE = (
    "A bar is stamped at its interval START (ts_event, schemas.market_data); its close and volume are knowable only "
    "at the interval end, so bar t may drive a decision no earlier than the next permitted execution point."
)
_PIT_RULES: dict[DataRequirement, str] = {
    DataRequirement.MARKET_BARS: _BAR_RULE,
    DataRequirement.BENCHMARK_BARS: _BAR_RULE + " The benchmark joins backward in time only (features.crossmarket).",
    DataRequirement.CURVE_CONTRACTS: _BAR_RULE + " Each contract's own raw bars -- never a back-adjusted series.",
    DataRequirement.SPOT_REFERENCE: "A spot assessment is knowable from its publication time.",
    DataRequirement.OPEN_INTEREST: "Reported for a session and published the next: key each value to its "
                                   "publication time.",
    DataRequirement.COT_REPORTS: "Positions as of Tuesday are released on Friday: a value is knowable from its "
                                 "release timestamp only (features.macro.PointInTimeDatum.published_ts_ns).",
    DataRequirement.PHYSICAL_INVENTORY: "An official release describes an earlier period: knowable only from its "
                                        "publication timestamp.",
    DataRequirement.PIT_FUNDAMENTALS: "Knowable from each filing's SEC acceptance timestamp, weeks after the fiscal "
                                      "period ends; restated as-of-today snapshots are never point-in-time.",
    DataRequirement.CONSENSUS_HISTORY: "Needs each estimate as it stood on the date; a source exposing only today's "
                                       "consensus would leak later revisions.",
    DataRequirement.FUND_FLOWS: "Shares outstanding are published after the session: knowable from publication.",
    DataRequirement.CONSTITUENT_MEMBERSHIP: "Needs the constituents as they were on each date.",
    DataRequirement.OPTION_CHAIN: "Quotes are knowable at their own timestamps.",
    DataRequirement.CRYPTO_DERIVATIVES: "Knowable from each record's available_ts_ns (crypto.schemas).",
    DataRequirement.ON_CHAIN_METRICS: "Knowable from each record's available_ts_ns (crypto.schemas).",
}
_LAGS: dict[DataRequirement, PublicationLag] = {
    DataRequirement.MARKET_BARS: PublicationLag.AT_BAR_CLOSE,
    DataRequirement.BENCHMARK_BARS: PublicationLag.AT_BAR_CLOSE,
    DataRequirement.CURVE_CONTRACTS: PublicationLag.AT_BAR_CLOSE,
    DataRequirement.SPOT_REFERENCE: PublicationLag.AT_PUBLICATION,
    DataRequirement.OPEN_INTEREST: PublicationLag.NEXT_SESSION,
    DataRequirement.COT_REPORTS: PublicationLag.WEEKLY_RELEASE,
    DataRequirement.PHYSICAL_INVENTORY: PublicationLag.AT_PUBLICATION,
    DataRequirement.PIT_FUNDAMENTALS: PublicationLag.FILING_ACCEPTANCE,
    DataRequirement.CONSENSUS_HISTORY: PublicationLag.AT_PUBLICATION,
    DataRequirement.FUND_FLOWS: PublicationLag.NEXT_SESSION,
    DataRequirement.CONSTITUENT_MEMBERSHIP: PublicationLag.AT_PUBLICATION,
}
_FREQUENCIES: dict[DataRequirement, DataFrequency] = {
    DataRequirement.SPOT_REFERENCE: DataFrequency.DAILY,
    DataRequirement.OPEN_INTEREST: DataFrequency.DAILY,
    DataRequirement.COT_REPORTS: DataFrequency.WEEKLY,
    DataRequirement.PHYSICAL_INVENTORY: DataFrequency.WEEKLY,
    DataRequirement.PIT_FUNDAMENTALS: DataFrequency.QUARTERLY,
    DataRequirement.CONSENSUS_HISTORY: DataFrequency.DAILY,
    DataRequirement.FUND_FLOWS: DataFrequency.DAILY,
}
_FINDING_ISSUE = {
    equity_findings.DataSourceAvailability.AVAILABLE_FREE_SELF_SERVE: _I.CONNECTOR_NOT_BUILT,
    equity_findings.DataSourceAvailability.AVAILABLE_PAID_ONLY: _I.PAID_SOURCE_ONLY,
    equity_findings.DataSourceAvailability.NOT_AVAILABLE: _I.NO_SOURCE_FOUND,
}
_FINDING_GAP = {
    equity_findings.DataSourceAvailability.AVAILABLE_FREE_SELF_SERVE: "no connector built",
    equity_findings.DataSourceAvailability.AVAILABLE_PAID_ONLY: "paid vendors only",
    equity_findings.DataSourceAvailability.NOT_AVAILABLE: "no source found",
}
_CONNECTOR_GAP = {
    DataRequirement.COT_REPORTS: "No CFTC COT connector",
    DataRequirement.PHYSICAL_INVENTORY: "No official inventory-data connector",
}
_UNINVESTIGATED_GAP = {
    DataRequirement.SPOT_REFERENCE: "No spot reference source investigated",
    DataRequirement.FUND_FLOWS: "No fund-flow source investigated",
}
_EXECUTABILITY_GAP = {
    _I.CROSS_INSTRUMENT: "No cross-instrument features in the feature registry",
    _I.FEATURE_NOT_REGISTERED: "No registered feature computes it",
}
_FINDING_CONCEPT = {
    DataRequirement.PIT_FUNDAMENTALS: "point_in_time_fundamentals",
    DataRequirement.CONSENSUS_HISTORY: "analyst_consensus_expectations",
}
#: What would close each gap -- guidance shown beside the reason, never an
#: action this module takes.
_REMEDY: dict[DataRequirement, str] = {
    DataRequirement.OPEN_INTEREST: "Databento's GLBX.MDP3 `statistics` schema carries it; acquiring it needs a cost "
                                   "estimate and your approval.",
    DataRequirement.COT_REPORTS: "CFTC publishes the reports free; a connector would align them on their release "
                                 "timestamps (features.macro).",
    DataRequirement.PHYSICAL_INVENTORY: "Official releases reach the platform only as market_intel headlines; no "
                                        "point-in-time numeric history is stored.",
    DataRequirement.PIT_FUNDAMENTALS: "SEC EDGAR XBRL filings are a free candidate whose acceptance timestamps are "
                                      "point-in-time; no connector is built.",
    DataRequirement.CONSENSUS_HISTORY: "Free sources expose only today's consensus, which is not point-in-time.",
}


@dataclass
class _Route:
    issues: list[ResolutionIssue] = field(default_factory=list)
    dataset: str | None = None
    data_schema: str | None = None
    frequency: DataFrequency = DataFrequency.NOT_APPLICABLE
    start: date | None = None
    end: date | None = None
    coverage_note: str = ""
    pit_safe: bool | None = None
    data_role: DataProvenanceRole | None = None
    reasons: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    gaps: list[tuple[ResolutionIssue, str]] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return self.data_role is DataProvenanceRole.REAL


def _span(start: date, end: date) -> str:
    return f"{start.isoformat()} → {(end - timedelta(days=1)).isoformat()}"


def _declared_window(windows: tuple[tuple[str, str, str], ...], dataset: str) -> tuple[date, date]:
    """A declared (dataset, start, end) window; the end is the vendor
    request's exclusive bound (the ETF pilot's last acquired bar is
    2024-12-30 for an end of 2024-12-31)."""
    start, end = next((s, e) for d, s, e in windows if d == dataset)
    return date.fromisoformat(start), date.fromisoformat(end)


def _bars(domain: MandateDomain, symbol: str, caps: DomainCapabilities) -> _Route:
    if domain is MandateDomain.FUTURES:
        cov = caps.futures_coverage(symbol)
        if cov is None:
            return _Route(
                issues=[_I.DATA_NOT_ACQUIRED], frequency=DataFrequency.MINUTE,
                gaps=[(_I.DATA_NOT_ACQUIRED, "Futures history not acquired (GLBX.MDP3)")],
                reasons=[(f"no {symbol} history is acquired -- catalogued for observation only; acquiring GLBX.MDP3 "
                          "history needs a Databento cost estimate and your approval")],
                provenance=[f"{FUTURES_CATALOG_SOURCE} (no {symbol} entry)",
                            "alpha_agent.marketdata.product_catalog.PRODUCT_CATALOG"],
            )
        # Shorter than the span the platform holds for its other roots:
        # usable, but only over its own window.
        span_start = min(c.start for c in caps.futures_bars)
        span_end = max(c.end_exclusive for c in caps.futures_bars)
        short = cov.start > span_start or cov.end_exclusive < span_end
        return _Route(
            issues=[_I.PARTIAL_HISTORY] if short else [],
            dataset=cov.dataset, data_schema=cov.data_schema, frequency=DataFrequency.MINUTE, start=cov.start,
            end=cov.end_exclusive, pit_safe=True, data_role=DataProvenanceRole.REAL,
            coverage_note=f"{_span(cov.start, cov.end_exclusive)} · {cov.continuous_segments} continuous-front "
                          "segments with raw contracts at each roll"
                          + (f" (other roots: {_span(span_start, span_end)})" if short else ""),
            provenance=[f"{cov.source} ({symbol} CONTINUOUS_FRONT)"],
        )
    universe, acquired, source = (
        (etf_universe, caps.etf_market_data_acquired, "alpha_agent.etf")
        if domain is MandateDomain.ETF else (equity_universe, caps.equity_market_data_acquired, "alpha_agent.equities")
    )
    dataset = universe.primary_listing_dataset(symbol)
    start, end = _declared_window(universe.DATASET_WINDOWS, dataset)
    provenance = [f"{source}.universe.DATASET_WINDOWS ({dataset})", f"{source}.data_source.load_primary_listing_bars"]
    if not acquired:
        return _Route(
            issues=[_I.DATA_NOT_ACQUIRED], frequency=DataFrequency.DAILY, provenance=provenance,
            gaps=[(_I.DATA_NOT_ACQUIRED, f"{DOMAIN_LABELS[domain]} daily bars not acquired")],
            reasons=[(f"{DOMAIN_LABELS[domain]} daily bars ({dataset}, declared {_span(start, end)}) are not "
                      "acquired -- a separate, cost-approved acquisition")],
        )
    return _Route(
        dataset=dataset, data_schema="ohlcv-1d", frequency=DataFrequency.DAILY, start=start, end=end, pit_safe=True,
        data_role=DataProvenanceRole.REAL, coverage_note=f"{_span(start, end)} · primary-listing venue",
        provenance=provenance,
    )


def _benchmark(domain: MandateDomain, symbol: str, caps: DomainCapabilities) -> _Route:
    route = _bars(domain, symbol, caps)
    bench = _bars(MandateDomain.ETF, "SPY", caps)
    route.provenance += bench.provenance
    if not bench.has_data:
        route.issues += [i for i in bench.issues if i not in route.issues]
        route.reasons += ["the SPY benchmark bars are not acquired"]
        route.gaps += [g for g in bench.gaps if g not in route.gaps]
        route.pit_safe = route.data_role = None
    elif route.has_data:
        route.start, route.end = max(route.start, bench.start), min(route.end, bench.end)
        route.coverage_note = f"{_span(route.start, route.end)} · joined backward against SPY ({bench.dataset})"
    return route


def _curve(symbol: str, caps: DomainCapabilities) -> _Route:
    route = _bars(MandateDomain.FUTURES, symbol, caps)
    cov = caps.futures_coverage(symbol)
    if cov is None:
        return route
    if not cov.roll_overlap_artifacts:
        return _Route(issues=[_I.DATA_NOT_ACQUIRED], frequency=DataFrequency.MINUTE, provenance=route.provenance,
                      gaps=[(_I.DATA_NOT_ACQUIRED, "Simultaneous contract months not acquired")],
                      reasons=[f"no simultaneous {symbol} contract months are acquired"])
    if _I.PARTIAL_HISTORY not in route.issues:
        route.issues.append(_I.PARTIAL_HISTORY)
    route.coverage_note = (
        f"two contract months side by side only around {cov.roll_overlap_artifacts} observed rolls "
        "(roll-overlap raw contracts) -- no continuous curve history"
    )
    route.provenance.append(f"{cov.source} ({symbol} ROLL_OVERLAP_RAW)")
    return route


def _finding(requirement: DataRequirement) -> _Route:
    f = equity_findings.finding(_FINDING_CONCEPT[requirement])
    return _Route(
        issues=[_FINDING_ISSUE[f.availability]],
        gaps=[(_FINDING_ISSUE[f.availability], f"{REQUIREMENT_LABELS[requirement]}: {_FINDING_GAP[f.availability]}")],
        reasons=[(f"{ISSUE_TEXT[_FINDING_ISSUE[f.availability]]} (data-availability finding "
                  f"'{f.concept}', {f.investigated_at})")],
        provenance=[f"alpha_agent.equities.data_availability:{f.concept}"],
    )


def _membership(caps: DomainCapabilities) -> _Route:
    route = _Route(
        issues=[_I.HINDSIGHT_MEMBERSHIP], pit_safe=False,
        gaps=[(_I.HINDSIGHT_MEMBERSHIP, "No point-in-time constituent history")],
        reasons=[(f"the only constituent list on the platform is the equity universe declared "
                  f"{equity_universe.UNIVERSE_DECLARED_AT.isoformat()} from hindsight (names that survived to "
                  "today); point-in-time fund holdings are not ingested")],
        provenance=["alpha_agent.equities.universe.UNIVERSE_DECLARED_AT"],
    )
    if not caps.equity_market_data_acquired:
        route.issues.append(_I.DATA_NOT_ACQUIRED)
        route.gaps.append((_I.DATA_NOT_ACQUIRED, "Equity daily bars not acquired"))
        route.reasons.append("constituent bars are not acquired either")
    return route


def _route(template: MeasurementTemplate, symbol: str | None, caps: DomainCapabilities) -> _Route:
    req, domain = template.requirement, template.domain
    if req in (DataRequirement.MARKET_BARS, DataRequirement.BENCHMARK_BARS, DataRequirement.CURVE_CONTRACTS) and (
        symbol is None
    ):
        return _Route(issues=[_I.DOMAIN_NOT_SUPPORTED], reasons=["no instrument to measure"],
                      gaps=[(_I.DOMAIN_NOT_SUPPORTED, "No instrument to measure")])
    if req is DataRequirement.MARKET_BARS:
        return _bars(domain, symbol, caps)
    if req is DataRequirement.BENCHMARK_BARS:
        return _benchmark(domain, symbol, caps)
    if req is DataRequirement.CURVE_CONTRACTS:
        return _curve(symbol, caps)
    if req in _FINDING_CONCEPT:
        return _finding(req)
    if req is DataRequirement.CONSTITUENT_MEMBERSHIP:
        return _membership(caps)
    if req is DataRequirement.OPEN_INTEREST:
        return _Route(issues=[_I.DATA_NOT_ACQUIRED], reasons=["exchange open interest is not acquired for any root"],
                      gaps=[(_I.DATA_NOT_ACQUIRED, "Exchange open interest not acquired")])
    if req in (DataRequirement.COT_REPORTS, DataRequirement.PHYSICAL_INVENTORY):
        return _Route(issues=[_I.CONNECTOR_NOT_BUILT], reasons=[ISSUE_TEXT[_I.CONNECTOR_NOT_BUILT]],
                      gaps=[(_I.CONNECTOR_NOT_BUILT, _CONNECTOR_GAP[req])])
    if req in (DataRequirement.SPOT_REFERENCE, DataRequirement.FUND_FLOWS):
        return _Route(issues=[_I.NOT_INVESTIGATED], reasons=["nothing is ingested and no source has been investigated"],
                      gaps=[(_I.NOT_INVESTIGATED, _UNINVESTIGATED_GAP[req])])
    return _Route(issues=[_I.DOMAIN_NOT_SUPPORTED], reasons=[ISSUE_TEXT[_I.DOMAIN_NOT_SUPPORTED]],
                  gaps=[(_I.DOMAIN_NOT_SUPPORTED, f"{DOMAIN_LABELS[domain]}: {ISSUE_TEXT[_I.DOMAIN_NOT_SUPPORTED]}")])


def resolve_field(
    template: MeasurementTemplate,
    symbol: str | None,
    *,
    support: DomainSupport,
    capabilities: DomainCapabilities,
    registry: FeatureRegistry | None = None,
) -> DataFieldResolution:
    """``field-resolution/1`` for one measurement of one instrument. Pure
    for a fixed capability snapshot and registry."""
    reg = registry or REGISTRY
    label = REQUIREMENT_LABELS[template.requirement]
    if support is DomainSupport.NOT_SUPPORTED:
        route = _Route(issues=[_I.DOMAIN_NOT_SUPPORTED],
                       gaps=[(_I.DOMAIN_NOT_SUPPORTED, f"No {DOMAIN_LABELS[template.domain].lower()} universe")],
                       reasons=[(f"no {DOMAIN_LABELS[template.domain].lower()} universe, chain or pricing data "
                                 "exists on this platform")])
    elif support is DomainSupport.SYNTHETIC_ONLY:
        route = _Route(issues=[_I.SYNTHETIC_ONLY], data_role=DataProvenanceRole.SYNTHETIC,
                       gaps=[(_I.SYNTHETIC_ONLY, f"{DOMAIN_LABELS[template.domain]}: synthetic scaffold only")],
                       reasons=[("only the synthetic Phase 22 scaffold exists -- no real vendor data; synthetic "
                                 "rows are never evidence")],
                       provenance=["alpha_agent.crypto.provenance.DataProvenanceRole.SYNTHETIC"])
    else:
        route = _route(template, symbol, capabilities)
    if route.end is not None and route.end > _HOLDOUT:
        raise HoldoutAccessError(f"{template.measurement_id} coverage reaches {route.end}, inside the locked holdout")

    issues = list(route.issues)
    kinds = template.feature_kinds
    if template.cross_instrument:
        issues.append(_I.CROSS_INSTRUMENT)
    else:
        status, _, available, _ = classify_factor(
            proposed_feature_kinds=kinds, required_external_data=(), registry=reg,
        )
        if status is not ResearchabilityStatus.AVAILABLE:
            issues.append(_I.FEATURE_NOT_REGISTERED)
        if any(not reg.get(k).point_in_time_safe for k in available):
            issues.append(_I.RETROSPECTIVE_FEATURE)
            route.pit_safe = False
            route.reasons.append("a registered feature it needs is retrospective")
            route.gaps.append((_I.RETROSPECTIVE_FEATURE, "Retrospective (look-ahead) feature"))
    executable = not ({_I.CROSS_INSTRUMENT, _I.FEATURE_NOT_REGISTERED} & set(issues))
    if template.proxy and route.has_data:
        issues.append(_I.PROXY)
    status = status_for(issues)
    usable = status in _USABLE

    missing = None
    gaps: list[tuple[ResolutionIssue, str]] = []
    if not usable:
        reasons = list(route.reasons)
        if not set(issues) & _DATA_ISSUES:
            # The data is there (or not the blocker): say why it still cannot be used.
            reasons += [ISSUE_TEXT[i] for i in issues if i in (_I.CROSS_INSTRUMENT, _I.FEATURE_NOT_REGISTERED)]
        missing = f"{label}: " + "; ".join(dict.fromkeys(reasons))
        if template.requirement in _REMEDY and set(issues) & _DATA_ISSUES:
            missing += f". {_REMEDY[template.requirement]}"
        gaps = list(route.gaps)
        gaps += [(i, _EXECUTABILITY_GAP[i]) for i in dict.fromkeys(issues) if i in _EXECUTABILITY_GAP]
        gaps = sorted(dict.fromkeys(gaps), key=lambda g: _rank(g[0]))  # stable: primary first
        others = [label for _, label in gaps[1:]]
        if others:
            missing += f". Also blocked by: {'; '.join(others)}"
    provenance = list(dict.fromkeys(route.provenance))
    if kinds:
        provenance.append(f"alpha_agent.features.registry: {', '.join(kinds)}")
    return DataFieldResolution(
        status=status, issues=tuple(dict.fromkeys(issues)),
        dataset=route.dataset if route.has_data else None,
        data_schema=route.data_schema if route.has_data else None,
        field=template.field if route.has_data else None,
        frequency=route.frequency if route.frequency is not DataFrequency.NOT_APPLICABLE
        else _FREQUENCIES.get(template.requirement, DataFrequency.NOT_APPLICABLE),
        coverage_start=route.start if route.has_data else None,
        coverage_end_exclusive=route.end if route.has_data else None,
        coverage_note=route.coverage_note, pit_safe=route.pit_safe,
        pit_rule=_PIT_RULES[template.requirement],
        publication_lag=_LAGS.get(template.requirement, PublicationLag.NOT_APPLICABLE),
        transform=template.transform, feature_kinds=kinds, executable=executable,
        proxy_note=template.proxy, missing_reason=missing,
        gaps=tuple(ResolutionGap(issue=i, label=label) for i, label in gaps),
        data_role=route.data_role,
        provenance=tuple(provenance),
    )


def build_measurement_spec(
    template: MeasurementTemplate,
    symbol: str | None,
    *,
    support: DomainSupport,
    capabilities: DomainCapabilities,
    expression_ids: tuple[str, ...] = (),
    registry: FeatureRegistry | None = None,
) -> MeasurementSpec:
    resolution = resolve_field(template, symbol, support=support, capabilities=capabilities, registry=registry)
    target = symbol or DOMAIN_LABELS[template.domain]
    key = f"{template.measurement_id}|{template.domain.value}|{target}"
    spec_id = "ms-" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return MeasurementSpec(
        spec_id=spec_id, measurement_id=template.measurement_id, label=template.label, role=template.role,
        domain=template.domain, symbol=symbol, requirement=template.requirement,
        variable=MeasurableVariable(
            name=f"{template.measurement_id}:{target}",
            economic_meaning=template.meaning.format(instrument=target),
            required_source=REQUIREMENT_LABELS[template.requirement],
            point_in_time_available=resolution.pit_safe,
            point_in_time_note=resolution.pit_rule,
        ),
        resolution=resolution, expression_ids=expression_ids,
    )

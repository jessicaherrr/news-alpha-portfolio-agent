"""Alpha Discovery campaign, Part C -- pluggable EXTERNAL knowledge-source
adapters (task spec sections 18-25).

SAFE EXTERNAL-SOURCE BOUNDARY (task spec sections 18/19/25) -- read before
touching this file:

    GitHub / academic / community / practitioner source
        -> inspect README / documentation / abstract (human or agent, offline)
        -> identify strategy logic, economic mechanism, parameters, data assumptions
        -> normalize into StrategyKnowledgeItem (THIS module's job)
        -> Claude / an internal translator proposes a closed DSL hypothesis
        -> the internal StrategyCompiler validates it
        -> the internal C++ engine backtests it

This module and everything downstream of it NEVER:
    * calls `eval` / `exec` on anything sourced externally,
    * imports a package fetched from a GitHub repository,
    * runs a repository's `setup.py` / install scripts / shell scripts,
    * executes a notebook cell,
    * runs `pip install` against an arbitrary repository.

An external source is TEXT / METADATA describing an idea. The only thing
that ever reaches execution is an internally-authored, independently
compiled `StrategySpec` -- see `alpha_agent.strategy.compiler`.

`default_adapters()` remains every source `NotConnectedAdapter`-shaped, on
purpose -- every existing caller and test that does not explicitly opt into
live research keeps its exact prior behaviour. `live_adapters()` (Alpha
Discovery live-research campaign, Checkpoints 5-7) is the real, opt-in
factory: it binds `GitHubStrategyConnector` for GITHUB (a real HTTP client
behind the exact same `ExternalSourceAdapter` protocol this module always
declared) and falls back to `NotConnectedAdapter` for any source category
that does not yet have a real live implementation, or whose real
implementation reports it cannot connect (missing credential, network
failure, rate limit) -- never a fabricated result either way.
"""
from __future__ import annotations

from typing import Protocol

from alpha_agent.knowledge.models import IngestionResult, IngestionStatus, SourceType

#: Forbidden operations this module (and every adapter built against its
#: protocol) must never perform. Documentation constant, asserted by
#: `tests/python/test_phase_alpha_discovery_knowledge_base.py` via static
#: source-grep -- the enforcement is "this code literally does not contain
#: these tokens", not merely a comment.
FORBIDDEN_OPERATIONS: frozenset[str] = frozenset({
    "eval(", "exec(", "os.system(", "subprocess.Popen(", "subprocess.call(",
    "subprocess.run(", "pip install", "importlib.import_module(",
})


class ExternalSourceAdapter(Protocol):
    """One pluggable external knowledge source. `ingest` never raises for a
    routine "not connected" outcome -- it returns a typed `IngestionResult`
    instead, so a caller can always render an honest status."""

    @property
    def source_type(self) -> SourceType: ...

    def ingest(self, *, query: str, markets: tuple[str, ...] = ()) -> IngestionResult: ...


class DisabledAdapter:
    """A source category that was never even ATTEMPTED because the caller's
    own research-sources selection excluded it for this campaign (Release UX
    bugfix pass, issue 1) -- e.g. the Discover page's "Internal + Classic
    only" mode. Distinct from `NotConnectedAdapter`: this adapter's `ingest`
    makes no decision at all about connectivity, it simply reports that
    nothing was asked of it. Never used for a source category that live
    research WAS requested for -- see `disabled_adapters` vs. `live_adapters`."""

    def __init__(self, source_type: SourceType):
        self._source_type = source_type

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def ingest(self, *, query: str, markets: tuple[str, ...] = ()) -> IngestionResult:
        return IngestionResult(
            source_type=self._source_type,
            status=IngestionStatus.DISABLED,
            items=(),
            detail=(
                f"{self._source_type.value.title()} was not requested for this campaign -- "
                "no live connection was attempted."
            ),
        )


class NotConnectedAdapter:
    """The default adapter for every external source type in this
    environment: architecturally real (implements the protocol, is queried
    the same way a live adapter would be), but always reports
    `NOT_CONNECTED` rather than a fabricated result (task spec section 23:
    "do NOT fake external search... report external live ingestion as
    NOT_CONNECTED")."""

    def __init__(self, source_type: SourceType):
        self._source_type = source_type

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def ingest(self, *, query: str, markets: tuple[str, ...] = ()) -> IngestionResult:
        return IngestionResult(
            source_type=self._source_type,
            status=IngestionStatus.NOT_CONNECTED,
            items=(),
            detail=(
                f"No live {self._source_type.value.title()} ingestion adapter is connected "
                "in this environment. The typed architecture (StrategyKnowledgeItem / "
                "ExternalSourceAdapter) is real and ready for a future live adapter to be "
                "bound behind the same interface -- this default never fabricates a result."
            ),
        )


def default_adapters() -> dict[SourceType, ExternalSourceAdapter]:
    """One `NotConnectedAdapter` per external source category (task spec
    section 14): GITHUB, ACADEMIC, COMMUNITY, PRACTITIONER. INTERNAL and
    CLASSIC are not "external" and are built by
    `alpha_agent.knowledge.registry_source` / `.classic_library` instead.

    UNCHANGED since Part C -- every existing caller/test that builds a
    `StrategyKnowledgeBase` without an explicit `adapters=` keeps its exact
    prior behaviour (every external source NOT_CONNECTED)."""
    return {
        st: NotConnectedAdapter(st)
        for st in (SourceType.GITHUB, SourceType.ACADEMIC, SourceType.COMMUNITY, SourceType.PRACTITIONER)
    }


def disabled_adapters() -> dict[SourceType, ExternalSourceAdapter]:
    """One `DisabledAdapter` per external source category -- the caller's
    research-sources selection did not request live research at all for this
    campaign (Release UX bugfix pass, issue 1). Use this, never
    `default_adapters()`, whenever the reason a source shows no items is
    "not requested" rather than "attempted and unreachable"."""
    return {
        st: DisabledAdapter(st)
        for st in (SourceType.GITHUB, SourceType.ACADEMIC, SourceType.COMMUNITY, SourceType.PRACTITIONER)
    }


def live_adapters() -> dict[SourceType, ExternalSourceAdapter]:
    """Alpha Discovery live-research campaign -- the REAL adapter set,
    opt-in only. Each source category is bound to its real connector when
    one exists; a source category without a real connector yet keeps the
    honest `NotConnectedAdapter` default. A connector's OWN health is
    resolved lazily, inside `ingest()` -- this function never makes a network
    call itself, so building the dict is always cheap and side-effect-free.
    """
    adapters = default_adapters()
    from alpha_agent.knowledge.academic_connector import AcademicResearchConnector
    from alpha_agent.knowledge.community_connector import CommunityResearchConnector
    from alpha_agent.knowledge.github_connector import GitHubStrategyConnector
    from alpha_agent.knowledge.practitioner_connector import PractitionerResearchConnector

    adapters[SourceType.GITHUB] = GitHubStrategyConnector()
    adapters[SourceType.ACADEMIC] = AcademicResearchConnector()
    adapters[SourceType.COMMUNITY] = CommunityResearchConnector()
    adapters[SourceType.PRACTITIONER] = PractitionerResearchConnector()
    return adapters


def ingest_all(
    adapters: dict[SourceType, ExternalSourceAdapter], *, query: str, markets: tuple[str, ...] = ()
) -> tuple[IngestionResult, ...]:
    """Query every configured adapter. Never raises: an adapter that fails is
    caught and turned into its own typed NOT_CONNECTED-shaped result rather
    than aborting the whole knowledge-base build."""
    results: list[IngestionResult] = []
    for source_type, adapter in adapters.items():
        try:
            results.append(adapter.ingest(query=query, markets=markets))
        except Exception as exc:  # noqa: BLE001 -- an external adapter failure is data, not a crash
            results.append(
                IngestionResult(
                    source_type=source_type, status=IngestionStatus.NOT_CONNECTED, items=(),
                    detail=f"adapter raised {type(exc).__name__}: {exc}",
                )
            )
    return tuple(results)

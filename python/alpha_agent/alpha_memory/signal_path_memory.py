"""News Alpha Phase H -- SIGNAL-PATH MEMORY: the Alpha Memory read model over the
registry's signal-path evidence (schema v7).

Alpha Memory stays a read model -- this module writes nothing. For one
transmission ROUTE (a path signature: the same ordered links, from any event)
it answers:

* how often the route has come up, from which events and research runs;
* how far its hypotheses got (the furthest stage) and what stopped them
  (typed reasons), split by expression domain -- an equity expression can be
  blocked by missing data where a futures expression of the same route was
  screened;
* which registry experiments tested a portfolio containing one of its
  signals (Evidence Plane ids -- linked, never merged);
* what would change the picture (`MemoryGuidance`).

It never blacklists. A failure belongs to the (path, expression, measurement,
candidate, portfolio) it happened at: another route of the same event, the
same transmission family, the same event type or the same mechanism is not
touched by it, and `blocked` is always False. A PORTFOLIO-scope success is
shown as such -- "a portfolio containing this signal passed validation" --
never as a proven route or an established mechanism.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from alpha_agent.registry.enums import HypothesisStage, PathEvidenceReason
from alpha_agent.registry.models import SignalPathEvidenceRecord

__all__ = [
    "SIGNAL_PATH_MEMORY_SCHEMA",
    "DomainOutcome",
    "LineageLink",
    "MemoryGuidance",
    "SignalPathMemory",
    "lineage",
    "signal_path_memory",
    "summarize_path_memory",
]

SIGNAL_PATH_MEMORY_SCHEMA = "signal-path-memory/1"
_STAGE_ORDER = {s: i for i, s in enumerate(HypothesisStage)}


class MemoryGuidance(str, Enum):
    """What would change the picture -- never "avoid this route"."""

    RETRY_WHEN_DATA_EXISTS = "RETRY_WHEN_DATA_EXISTS"
    NEEDS_SIGNAL_RULE = "NEEDS_SIGNAL_RULE"
    NEEDS_EVENT_CONDITIONING = "NEEDS_EVENT_CONDITIONING"
    NEEDS_DIAGNOSTIC_METHOD = "NEEDS_DIAGNOSTIC_METHOD"
    NEEDS_A_SCREEN = "NEEDS_A_SCREEN"
    OUTSIDE_CURRENT_MANDATE = "OUTSIDE_CURRENT_MANDATE"
    NOT_EXPRESSIBLE_YET = "NOT_EXPRESSIBLE_YET"
    SCREEN_EVIDENCE_AGAINST = "SCREEN_EVIDENCE_AGAINST"
    AWAITING_VALIDATION = "AWAITING_VALIDATION"
    PORTFOLIO_EVIDENCE_AGAINST = "PORTFOLIO_EVIDENCE_AGAINST"
    PORTFOLIO_EVIDENCE_FOR = "PORTFOLIO_EVIDENCE_FOR"


GUIDANCE_TEXT: dict[MemoryGuidance, str] = {
    MemoryGuidance.RETRY_WHEN_DATA_EXISTS: "Blocked by missing data, not by evidence -- retry once the data exists.",
    MemoryGuidance.NEEDS_SIGNAL_RULE: "No reviewed rule turns its measurement into a factor yet.",
    MemoryGuidance.NEEDS_EVENT_CONDITIONING: (
        "Its measurement is confirmation-only -- it needs an event-conditioned study, not an unconditional factor."),
    MemoryGuidance.NEEDS_DIAGNOSTIC_METHOD: "No diagnostic method exists for its domain yet.",
    MemoryGuidance.NEEDS_A_SCREEN: "A candidate exists but has not been screened.",
    MemoryGuidance.OUTSIDE_CURRENT_MANDATE: "Excluded by a mandate, not by evidence -- another mandate may admit it.",
    MemoryGuidance.NOT_EXPRESSIBLE_YET: "No reviewed asset expression takes the route up (or the route is unresolved).",
    MemoryGuidance.SCREEN_EVIDENCE_AGAINST: (
        "In-sample unconditional screening did not support (or contradicted) a factor on it -- screening evidence "
        "about that factor, not a test of the route."),
    MemoryGuidance.AWAITING_VALIDATION: "Screen-supported; not yet validated inside an eligible portfolio.",
    MemoryGuidance.PORTFOLIO_EVIDENCE_AGAINST: (
        "A portfolio containing one of its signals failed validation -- evidence about that portfolio only."),
    MemoryGuidance.PORTFOLIO_EVIDENCE_FOR: (
        "A portfolio containing one of its signals passed validation -- evidence about that portfolio only, never a "
        "proven route."),
}

_REASON_GUIDANCE: dict[PathEvidenceReason, MemoryGuidance] = {
    PathEvidenceReason.PATH_UNRESOLVED: MemoryGuidance.NOT_EXPRESSIBLE_YET,
    PathEvidenceReason.PATH_NOT_EXPRESSED: MemoryGuidance.NOT_EXPRESSIBLE_YET,
    PathEvidenceReason.EXPRESSION_EXCLUDED_BY_MANDATE: MemoryGuidance.OUTSIDE_CURRENT_MANDATE,
    PathEvidenceReason.SIGNAL_EXCLUDED_BY_MANDATE: MemoryGuidance.OUTSIDE_CURRENT_MANDATE,
    PathEvidenceReason.REMOVED_BY_CONSTRAINTS: MemoryGuidance.OUTSIDE_CURRENT_MANDATE,
    PathEvidenceReason.DOMAIN_UNAVAILABLE: MemoryGuidance.RETRY_WHEN_DATA_EXISTS,
    PathEvidenceReason.NO_INSTRUMENT: MemoryGuidance.RETRY_WHEN_DATA_EXISTS,
    PathEvidenceReason.MEASUREMENT_UNAVAILABLE: MemoryGuidance.RETRY_WHEN_DATA_EXISTS,
    PathEvidenceReason.NO_SIGNAL_RULE: MemoryGuidance.NEEDS_SIGNAL_RULE,
    PathEvidenceReason.CONFIRMATION_ONLY_MEASUREMENT: MemoryGuidance.NEEDS_EVENT_CONDITIONING,
    PathEvidenceReason.NOT_SCREENED: MemoryGuidance.NEEDS_A_SCREEN,
    PathEvidenceReason.NO_DIAGNOSTIC_METHOD: MemoryGuidance.NEEDS_DIAGNOSTIC_METHOD,
    PathEvidenceReason.SCREEN_NO_SUPPORT: MemoryGuidance.SCREEN_EVIDENCE_AGAINST,
    PathEvidenceReason.SCREEN_CONTRADICTS_SIGN: MemoryGuidance.SCREEN_EVIDENCE_AGAINST,
    PathEvidenceReason.SCREEN_INSUFFICIENT_DATA: MemoryGuidance.RETRY_WHEN_DATA_EXISTS,
    PathEvidenceReason.NOT_RANKABLE: MemoryGuidance.NEEDS_A_SCREEN,
    PathEvidenceReason.NOT_ELIGIBLE_FOR_VALIDATION: MemoryGuidance.AWAITING_VALIDATION,
    PathEvidenceReason.EXECUTION_UNSUPPORTED: MemoryGuidance.AWAITING_VALIDATION,
    PathEvidenceReason.COST_MODEL_INCOMPLETE: MemoryGuidance.AWAITING_VALIDATION,
    PathEvidenceReason.PORTFOLIO_REJECTED: MemoryGuidance.PORTFOLIO_EVIDENCE_AGAINST,
    PathEvidenceReason.PORTFOLIO_COST_FRAGILE: MemoryGuidance.PORTFOLIO_EVIDENCE_AGAINST,
    PathEvidenceReason.PORTFOLIO_INCONCLUSIVE: MemoryGuidance.AWAITING_VALIDATION,
    PathEvidenceReason.PORTFOLIO_VALIDATED: MemoryGuidance.PORTFOLIO_EVIDENCE_FOR,
    PathEvidenceReason.PORTFOLIO_PRIOR_RESULT_CITED: MemoryGuidance.AWAITING_VALIDATION,
}


class DomainOutcome(BaseModel):
    """One expression domain's share of a route's evidence."""

    model_config = {"frozen": True, "extra": "forbid"}

    domain: str
    records: int
    furthest_stage: HypothesisStage
    outcomes: dict[str, int]
    reasons: dict[str, int]


class SignalPathMemory(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = SIGNAL_PATH_MEMORY_SCHEMA
    plane: Literal["HYPOTHESIS"] = "HYPOTHESIS"
    path_signature: str
    path_type: str | None
    transmission_depth: int | None
    consequence_states: tuple[str, ...]
    event_ids: tuple[str, ...]
    event_headlines: tuple[str, ...]
    research_runs: int
    records: int
    furthest_stage: HypothesisStage
    outcomes: dict[str, int]
    scopes: dict[str, int]
    reasons: dict[str, int]
    #: The most frequent reason (ties: alphabetical) -- a display summary, not a verdict.
    primary_reason: PathEvidenceReason
    by_expression_domain: tuple[DomainOutcome, ...]
    candidate_signal_ids: tuple[str, ...]
    #: Evidence Plane links -- registry experiments of portfolios containing its signals.
    experiment_identities: tuple[str, ...]
    guidance: tuple[MemoryGuidance, ...]
    #: Always False: memory never blacklists a route.
    blocked: Literal[False] = False
    scope_note: str = (
        "Evidence about hypotheses on this route at the stage each reached. A portfolio result is about that "
        "portfolio; a screen is in-sample and unconditional. Nothing here proves or rules out the route or its "
        "mechanism."
    )

    def headline(self) -> str:
        seen = f"seen in {len(self.event_ids)} event(s), {self.research_runs} run(s)"
        to = " / ".join(c.replace("_", " ").lower() for c in self.consequence_states) or "?"
        return (f"{self.path_type or 'UNCLASSIFIED'} route to {to} (depth {self.transmission_depth}) -- {seen}; "
                f"furthest stage {self.furthest_stage.value.replace('_', ' ').lower()}.")


def _furthest(records: Iterable[SignalPathEvidenceRecord]) -> HypothesisStage:
    return max((r.stage_reached for r in records), key=lambda s: _STAGE_ORDER[s])


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize_path_memory(records: Sequence[SignalPathEvidenceRecord]) -> tuple[SignalPathMemory, ...]:
    """Pure: group records by route signature (records without a route are
    not route memory) and summarize each group."""
    groups: dict[str, list[SignalPathEvidenceRecord]] = {}
    for r in records:
        if r.path_signature:
            groups.setdefault(r.path_signature, []).append(r)
    out = []
    for signature, rs in groups.items():
        domains: dict[str, list[SignalPathEvidenceRecord]] = {}
        for r in rs:
            domains.setdefault(r.expression_domain or "NONE", []).append(r)
        reasons = _counts(r.reason_code.value for r in rs)
        primary = PathEvidenceReason(min(reasons, key=lambda k: (-reasons[k], k)))
        # the primary reason's guidance first, the rest in vocabulary order
        guidance = sorted({_REASON_GUIDANCE[r.reason_code] for r in rs},
                          key=lambda g: (g is not _REASON_GUIDANCE[primary], list(MemoryGuidance).index(g)))
        out.append(SignalPathMemory(
            path_signature=signature, path_type=rs[0].path_type, transmission_depth=rs[0].transmission_depth,
            consequence_states=tuple(sorted({r.consequence_state for r in rs if r.consequence_state})),
            event_ids=tuple(sorted({r.event_id for r in rs})),
            event_headlines=tuple(sorted({r.event_headline for r in rs})),
            research_runs=len({r.research_run_id for r in rs}), records=len(rs), furthest_stage=_furthest(rs),
            outcomes=_counts(r.outcome.value for r in rs), scopes=_counts(r.evidence_scope.value for r in rs),
            reasons=reasons, primary_reason=primary,
            by_expression_domain=tuple(DomainOutcome(
                domain=d, records=len(ds), furthest_stage=_furthest(ds), outcomes=_counts(x.outcome.value for x in ds),
                reasons=_counts(x.reason_code.value for x in ds)) for d, ds in sorted(domains.items())),
            candidate_signal_ids=tuple(sorted({r.candidate_signal_id for r in rs if r.candidate_signal_id})),
            experiment_identities=tuple(sorted({r.experiment_identity for r in rs if r.experiment_identity})),
            guidance=tuple(guidance),
        ))
    return tuple(sorted(out, key=lambda m: (-_STAGE_ORDER[m.furthest_stage], m.path_type or "", m.path_signature)))


def signal_path_memory(registry, *, path_signature: str | None = None,
                       event_id: str | None = None) -> tuple[SignalPathMemory, ...]:
    """Route memory from the registry. With ``event_id``, the routes that
    event produced -- each summarized over EVERY recorded run of that route,
    from any event (that is what makes it memory)."""
    if event_id is None:
        return summarize_path_memory(registry.signal_path_evidence(path_signature=path_signature))
    signatures = {r.path_signature for r in registry.signal_path_evidence(event_id=event_id) if r.path_signature}
    if path_signature is not None:
        signatures &= {path_signature}
    records = [r for s in sorted(signatures) for r in registry.signal_path_evidence(path_signature=s)]
    return summarize_path_memory(records)


class LineageLink(BaseModel):
    """One hop of the Event -> ... -> Evidence chain; ``ref`` is the id the
    hop is recorded under (``None`` when the chain stopped before it)."""

    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["EVENT", "TRANSMISSION", "SIGNAL_PATH", "PATH_TYPE", "ASSET_EXPRESSION", "MEASUREMENT",
                  "CANDIDATE_SIGNAL", "PORTFOLIO", "EXPERIMENT", "EVIDENCE"]
    ref: str | None
    label: str


def lineage(r: SignalPathEvidenceRecord) -> tuple[LineageLink, ...]:
    """The record's chain, hop by hop -- every hop it did not reach says so."""
    outcome = (f"{r.outcome.value} at {r.stage_reached.value.replace('_', ' ').lower()} "
               f"({r.evidence_scope.value.lower()} evidence): {r.reason_code.value}")
    return (
        LineageLink(kind="EVENT", ref=r.event_id, label=r.event_headline),
        LineageLink(kind="TRANSMISSION", ref=r.mechanism_graph_id, label="economic mechanism graph"),
        LineageLink(kind="SIGNAL_PATH", ref=r.path_id, label=r.consequence_state or "--"),
        LineageLink(kind="PATH_TYPE", ref=r.path_type,
                    label=f"{r.path_type or 'unclassified'}, depth {r.transmission_depth}"),
        LineageLink(kind="ASSET_EXPRESSION", ref=r.expression_id,
                    label=f"{r.expression_concept} ({r.expression_domain}, {r.expression_fidelity})"
                    if r.expression_id else "not expressed"),
        LineageLink(kind="MEASUREMENT", ref=r.measurement_id,
                    label=r.measurement_status or ("resolved" if r.candidate_signal_id else "not reached")),
        LineageLink(kind="CANDIDATE_SIGNAL", ref=r.candidate_signal_id,
                    label=(r.instrument or "--") if r.candidate_signal_id else "no candidate"),
        LineageLink(kind="PORTFOLIO", ref=r.portfolio_strategy_fingerprint,
                    label="tested portfolio" if r.portfolio_strategy_fingerprint else "not in a tested portfolio"),
        LineageLink(kind="EXPERIMENT", ref=r.experiment_identity,
                    label="registry experiment" if r.experiment_identity else "no experiment"),
        LineageLink(kind="EVIDENCE", ref=r.evidence_id, label=outcome),
    )

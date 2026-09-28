"""News Alpha Phase H -- LINKS between the Hypothesis Plane and this graph's
Evidence Plane.

The two planes stay separate:

* HYPOTHESIS plane -- news-alpha transmission paths and candidate signals
  (`alpha_agent.news_alpha`), recorded as typed evidence in the registry's
  ``signal_path_evidence`` (schema v7);
* EVIDENCE plane -- this graph's Instrument nodes and registry experiments.

A `HypothesisEvidenceLink` carries ids only: which candidate signal, from
which event and routes, points at which Instrument node, and which registry
experiments (portfolio validations) it took part in. No verdict, coverage or
maturity crosses the link -- the Instrument node keeps its own registry
evidence (`builder.build_mechanism_graph`), and a hypothesis keeps its own
typed outcome. Nothing here writes, fetches or scores.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from alpha_agent.alpha_graph.builder import list_considered_instruments
from alpha_agent.alpha_graph.schemas import InstrumentRef
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.models import SignalPathEvidenceRecord

__all__ = [
    "HypothesisEvidenceLink",
    "hypotheses_for_instrument",
    "hypothesis_links",
    "link_records",
]


class HypothesisEvidenceLink(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    plane_from: Literal["HYPOTHESIS"] = "HYPOTHESIS"
    plane_to: Literal["EVIDENCE"] = "EVIDENCE"
    candidate_signal_id: str
    event_ids: tuple[str, ...]
    path_signatures: tuple[str, ...]
    path_types: tuple[str, ...]
    #: The Evidence-Plane Instrument node, when the graph considers the instrument.
    instrument: InstrumentRef | None
    instrument_symbol: str | None
    expression_domain: str | None
    #: Registry experiments this hypothesis took part in (portfolio validations).
    experiment_identities: tuple[str, ...]
    research_run_ids: tuple[str, ...]
    note: str = "A link between planes by id -- no evidence is transferred across it in either direction."


def _instrument_ref(domain: str | None, symbol: str | None) -> InstrumentRef | None:
    if domain is None or symbol is None or domain not in AssetDomain.__members__:
        return None
    ref = InstrumentRef(root_symbol=symbol, asset_domain=AssetDomain(domain))
    return ref if ref in set(list_considered_instruments()) else None


def link_records(records: Sequence[SignalPathEvidenceRecord]) -> tuple[HypothesisEvidenceLink, ...]:
    """Pure: one link per candidate signal across the given records."""
    by_candidate: dict[str, list[SignalPathEvidenceRecord]] = {}
    for r in records:
        if r.candidate_signal_id:
            by_candidate.setdefault(r.candidate_signal_id, []).append(r)
    out = []
    for cid, rs in sorted(by_candidate.items()):
        first = rs[0]
        out.append(HypothesisEvidenceLink(
            candidate_signal_id=cid, event_ids=tuple(sorted({r.event_id for r in rs})),
            path_signatures=tuple(sorted({r.path_signature for r in rs if r.path_signature})),
            path_types=tuple(sorted({r.path_type for r in rs if r.path_type})),
            instrument=_instrument_ref(first.expression_domain, first.instrument), instrument_symbol=first.instrument,
            expression_domain=first.expression_domain,
            experiment_identities=tuple(sorted({r.experiment_identity for r in rs if r.experiment_identity})),
            research_run_ids=tuple(sorted({r.research_run_id for r in rs})),
        ))
    return tuple(out)


def hypothesis_links(registry, *, event_id: str | None = None) -> tuple[HypothesisEvidenceLink, ...]:
    return link_records(registry.signal_path_evidence(event_id=event_id))


def hypotheses_for_instrument(registry, instrument: InstrumentRef) -> tuple[HypothesisEvidenceLink, ...]:
    """Reverse navigation: from an Evidence-Plane Instrument node to the
    news-alpha hypotheses that point at it."""
    return tuple(link for link in link_records(registry.signal_path_evidence()) if link.instrument == instrument)

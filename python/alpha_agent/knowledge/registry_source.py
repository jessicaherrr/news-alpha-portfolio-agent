"""Alpha Discovery campaign, Part C -- the INTERNAL knowledge source.

Turns already-committed registry evidence (canonical trials, real
scientific/engineering history) into `StrategyKnowledgeItem`s so it can be
displayed and prioritized alongside CLASSIC/ACADEMIC/GITHUB/COMMUNITY
references (task spec section 13's "1. Internal ExperimentRegistry").

This module reads REGISTRY-ROW-SHAPED plain dicts -- the exact shape
`alpha_agent.ui.services.list_experiments()` already returns and the Phase B1
recommendation package already tests against -- rather than importing the
sqlite registry itself, so it stays dependency-light and trivially testable
against synthetic fixtures (no database needed). A caller that has a live
registry (e.g. `alpha_agent.discovery`) passes `services.list_experiments()`
rows straight through.

Never re-derives a verdict, a Sharpe, or a p-value here: every numeric field
copied into `reported_results` is copied VERBATIM from the committed
`ResultRecord`, and it is still, structurally, `reported_results` -- a
description of a PRIOR internal experiment for research-inspiration purposes,
never itself re-entered as new evidence for a new hypothesis.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from alpha_agent.knowledge.models import (
    EconomicMechanism,
    IngestionStatus,
    SourceQualityTier,
    SourceType,
    StrategyKnowledgeItem,
)

#: The same family -> mechanism-category mapping
#: `alpha_agent.recommendation.fit` uses for User Fit, reused here so the
#: knowledge base's mechanism labels line up with the rest of the platform's
#: vocabulary. A family absent from this table is skipped (task spec
#: section 14: never guess a mechanism for a family with no documented
#: mapping) rather than mis-labelled.
_FAMILY_MECHANISM: dict[str, EconomicMechanism] = {
    "tsmom": EconomicMechanism.MOMENTUM,
    "ma_trend": EconomicMechanism.TREND,
    "breakout": EconomicMechanism.BREAKOUT,
    "mean_reversion": EconomicMechanism.MEAN_REVERSION,
    "silver_bullet": EconomicMechanism.OPENING_RANGE,
}


def _hash(row_key: str, payload: dict) -> str:
    canon = json.dumps({"id": row_key, **payload}, sort_keys=True, separators=(",", ":"), default=str)
    return "internal1:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:32]


def internal_items_from_registry_rows(
    rows: list[dict[str, Any]],
) -> tuple[StrategyKnowledgeItem, ...]:
    """One `StrategyKnowledgeItem` per (strategy_family, root_symbol)
    combination present in `rows` that maps to a known mechanism -- a real
    row missing a mapped family is simply not represented (never mis-tagged).
    Deterministic ordering: grouped then sorted by (family, root)."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        family = r.get("strategy_family")
        root = r.get("root_symbol")
        if not family or not root or family not in _FAMILY_MECHANISM:
            continue
        grouped.setdefault((family, root), []).append(r)

    items: list[StrategyKnowledgeItem] = []
    for (family, root), group in sorted(grouped.items()):
        verdicts = sorted({g.get("verdict") or "NOT_ADJUDICATED" for g in group})
        n_trials = len(group)
        knowledge_id = f"internal-{family}-{root}"
        payload = {"family": family, "root": root, "verdicts": verdicts, "n": n_trials}
        items.append(
            StrategyKnowledgeItem(
                knowledge_id=knowledge_id,
                source_type=SourceType.INTERNAL,
                source_quality=SourceQualityTier.TIER_A,  # our own directly-measured evidence
                ingestion_status=IngestionStatus.CONNECTED,
                title=f"Internal history: {family} on {root}",
                markets=(root,),
                asset_classes=("futures",),
                economic_mechanism=_FAMILY_MECHANISM[family],
                entry_logic_summary=f"Already-compiled '{family}' strategy family.",
                reported_results=(
                    f"{n_trials} prior committed trial(s); verdict(s) observed: {', '.join(verdicts)}. "
                    "This describes PAST internal experiments only -- it is not new evidence for any "
                    "future hypothesis and never re-enters a BH/FDR family on its own."
                ),
                implementation_notes="Sourced from the live ExperimentRegistry, not re-derived.",
                candidate_dsl_template=family,
                provenance_hash=_hash(knowledge_id, payload),
            )
        )
    return tuple(items)

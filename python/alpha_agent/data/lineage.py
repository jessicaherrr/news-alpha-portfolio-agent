"""Lineage sidecars for processed artifacts (layers 2 & 5).

Every processed file gets a ``<file>.lineage.json`` recording where it came from
and how it was produced, so a backtest can record dataset provenance
(CLAUDE backtest rule 5).
"""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from alpha_agent.schemas.market_data import CANONICAL_SCHEMA_VERSION

LINEAGE_SUFFIX = ".lineage.json"
SCHEMA_VERSION_BY_KIND = {
    "canonical_bars": CANONICAL_SCHEMA_VERSION,
    "contracts": CANONICAL_SCHEMA_VERSION,
}


def code_commit() -> str | None:
    """Best-effort short git commit; ``None`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- provenance is best-effort; never fail the pipeline over git
        return None


class Lineage(BaseModel):
    """Provenance sidecar for any processed artifact (layers 2-6).

    Layer-2/5 fields (schema/stype/normalization) and layer-3/4/6 fields
    (source_canonical_paths / roll_rule / adjustment_*) are both optional; each
    producer fills the ones that apply.
    """

    artifact_kind: str  # canonical_bars | contracts | continuous | backadjusted | rolls
    code_commit: str | None = None
    timezone: str = "UTC"
    price_domain: str | None = None
    schema_version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    diagnostics_summary: dict = Field(default_factory=dict)
    extra: dict = Field(default_factory=dict)

    # --- layer 2 / 5 (canonical pipeline) ---
    source_raw_sha256: str | None = None
    source_manifest_path: str | None = None
    dataset: str | None = None
    requested_symbols: list[str] = Field(default_factory=list)
    schema_: str | None = Field(default=None, alias="schema")
    stype_in: str | None = None
    stype_out: str | None = None
    price_scale_policy: str | None = None
    normalization_policy: dict = Field(default_factory=dict)
    canonical_schema_version: str = CANONICAL_SCHEMA_VERSION

    # --- layer 3 / 4 / 6 (futures history) ---
    source_canonical_paths: list[str] = Field(default_factory=list)
    source_contract_paths: list[str] = Field(default_factory=list)
    continuous_symbol: str | None = None
    roll_rule: str | None = None
    roll_price_policy: str | None = None
    adjustment_method: str | None = None
    adjustment_mode: str | None = None          # point_in_time | retrospective_research
    as_of_ts_ns: int | None = None
    calendar_version: str | None = None
    first_notice_policy: dict = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


def write_lineage(artifact_path: str | Path, lineage: Lineage) -> Path:
    p = Path(artifact_path)
    out = p.with_name(p.name + LINEAGE_SUFFIX)
    out.write_text(
        json.dumps(lineage.model_dump(by_alias=True), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


def load_lineage(artifact_path: str | Path) -> Lineage:
    p = Path(artifact_path)
    out = p if p.name.endswith(LINEAGE_SUFFIX) else p.with_name(p.name + LINEAGE_SUFFIX)
    return Lineage.model_validate_json(out.read_text(encoding="utf-8"))

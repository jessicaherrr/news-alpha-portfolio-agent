"""Phase 13.5A -- real CME futures dataset design + acquisition manifest.

This module holds the *design* only. It never downloads. It carries:

* :class:`RealDatasetPlan`     -- the frozen, predeclared acquisition plan
  (roots, dataset, schema, split date windows). Loaded from
  ``configs/real_dataset.yaml``.
* :class:`RealDatasetComponent` -- one Databento request in the plan
  (continuous front bars / roll-overlap raw bars / instrument definitions),
  with its exact symbology and role.
* :class:`RealDatasetManifest`  -- the frozen typed record written **after** a
  component is acquired, binding raw bytes + lineage + query identity. A cosmetic
  path relocation must not change ``semantic_identity()``.

None of this is the Phase 14 experiment registry.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

REAL_DATASET_SCHEMA_VERSION = "real-dataset/1"
DATASET = "GLBX.MDP3"                 # CLAUDE market-data rule 1
RESEARCH_SCHEMA = "ohlcv-1m"          # primary research resolution (13.5A)


class SplitRole(str, Enum):
    """The role a date window plays in Phase 13 validation. The 2025 window is
    LOCKED_HOLDOUT and must not be acquired until the strategy / spec / policy /
    commit are frozen (Phase 13 sections 5, 20; 13.5A section 14)."""

    RESEARCH = "research"
    VALIDATION = "validation"
    LOCKED_HOLDOUT = "locked_holdout"


class ComponentKind(str, Enum):
    """A component is one kind of Databento request in the acquisition plan."""

    # <ROOT>.v.0 ohlcv-1m, stype_in=continuous, stype_out=instrument_id.
    # The volume-ranked FRONT contract's real raw-contract bars for the whole
    # window (instrument_id rolls at each transition). Canonicalised per raw
    # contract -> PriceDomain.RAW_CONTRACT execution bars for the front contract;
    # the unadjusted-continuous (RAW_CONTINUOUS) and point-in-time back-adjusted
    # (BACK_ADJUSTED) SIGNAL series are derived locally, never purchased.
    CONTINUOUS_FRONT = "continuous_front_ohlcv1m"
    # <OUTGOING_CONTRACT> ohlcv-1m, stype_in=raw_symbol, a tight window around a
    # detected roll transition. Supplies the contemporaneous old-contract bar the
    # engine needs to price a roll close-leg via the "roll" path rather than
    # RejectDefer (docs/EXECUTION_MODEL.md). The exact symbols + windows are
    # determined FROM the observed CONTINUOUS_FRONT instrument_id transitions --
    # a dependency, not a guess (13.5A section 7).
    ROLL_OVERLAP_RAW = "roll_overlap_raw_ohlcv1m"
    # <ROOT>.FUT schema=definition, stype_in=parent, requested as short periodic
    # snapshots (definition cost scales with the requested span). instrument_id
    # -> raw_symbol / tick / expiration / economics via the Phase 03.5 / 04.5
    # robust derivation (NEVER the vendor sentinel multiplier).
    DEFINITIONS = "instrument_definitions"


class DateWindow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    role: SplitRole
    start: str                       # inclusive, "YYYY-MM-DD"
    end: str                         # EXCLUSIVE, "YYYY-MM-DD" (Databento convention)

    @model_validator(mode="after")
    def _order(self) -> DateWindow:
        if self.end <= self.start:
            raise ValueError(f"window end {self.end} must be after start {self.start}")
        return self


class RealDatasetComponent(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "populate_by_name": True}

    kind: ComponentKind
    root_symbol: str = Field(pattern=r"^[A-Z0-9]{1,12}$")
    # the exact request symbology
    symbols: tuple[str, ...]
    schema_: str = Field(alias="schema")
    stype_in: str
    stype_out: str = "instrument_id"
    window: DateWindow
    # ROLL_OVERLAP_RAW only: filled after the CONTINUOUS_FRONT transitions are
    # observed. Empty here means "to be resolved from the continuous download".
    resolved_from_transitions: bool = False

    def query_identity(self) -> str:
        if self.resolved_from_transitions and not self.symbols:
            # placeholder: the real per-contract windows are resolved from the
            # observed CONTINUOUS_FRONT transitions (13.5A section 7).
            payload: dict = {
                "unresolved": True, "kind": self.kind.value, "root": self.root_symbol,
                "schema": self.schema_, "stype_in": self.stype_in,
                "window": [self.window.role.value, self.window.start, self.window.end],
            }
        else:
            payload = {
                "dataset": DATASET,
                "schema": self.schema_,
                "stype_in": self.stype_in,
                "stype_out": self.stype_out,
                "symbols": sorted(self.symbols),
                "start": self.window.start,
                "end": self.window.end,
            }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"dbnq1:{digest[:24]}"


class RealDatasetPlan(BaseModel):
    """The frozen, predeclared acquisition plan. Dates must not change after any
    performance is seen (13.5A section 4)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = REAL_DATASET_SCHEMA_VERSION
    plan_id: str                     # "A" | "B"
    dataset: str = DATASET
    research_schema: str = RESEARCH_SCHEMA
    roots: tuple[str, ...]
    windows: tuple[DateWindow, ...]
    calendar_version: str            # SessionCalendar version bound to this plan
    # exchange-standard roll cadence per root (documented, for the overlap estimate)
    rolls_per_year: dict[str, int]
    roll_overlap_trading_days: int = 5

    @model_validator(mode="after")
    def _check(self) -> RealDatasetPlan:
        roles = [w.role for w in self.windows]
        if roles.count(SplitRole.LOCKED_HOLDOUT) != 1:
            raise ValueError("exactly one LOCKED_HOLDOUT window is required")
        if self.windows[-1].role != SplitRole.LOCKED_HOLDOUT:
            raise ValueError("the LOCKED_HOLDOUT window must be chronologically last")
        for a, b in zip(self.windows, self.windows[1:]):
            if b.start < a.end:
                raise ValueError("plan windows must be chronological and non-overlapping")
        if set(self.rolls_per_year) != set(self.roots):
            raise ValueError("rolls_per_year must cover exactly the plan roots")
        return self

    def window(self, role: SplitRole) -> DateWindow:
        for w in self.windows:
            if w.role == role:
                return w
        raise KeyError(role)

    def research_validation_span(self) -> tuple[str, str]:
        rs = [w for w in self.windows if w.role != SplitRole.LOCKED_HOLDOUT]
        return (min(w.start for w in rs), max(w.end for w in rs))

    def holdout_span(self) -> tuple[str, str]:
        w = self.window(SplitRole.LOCKED_HOLDOUT)
        return (w.start, w.end)

    def plan_fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "dataset": self.dataset,
            "research_schema": self.research_schema,
            "roots": sorted(self.roots),
            "windows": sorted([w.role.value, w.start, w.end] for w in self.windows),
            "calendar_version": self.calendar_version,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"realdsplan1:{digest[:32]}"

    def components(self, *, include_holdout: bool = False) -> tuple[RealDatasetComponent, ...]:
        """Every Databento request the plan implies. ROLL_OVERLAP_RAW components
        are placeholders (one per root per non-holdout window) -- their exact
        symbols/windows are resolved from the observed continuous transitions."""
        out: list[RealDatasetComponent] = []
        for w in self.windows:
            if w.role == SplitRole.LOCKED_HOLDOUT and not include_holdout:
                continue
            for root in self.roots:
                out.append(RealDatasetComponent(
                    kind=ComponentKind.CONTINUOUS_FRONT, root_symbol=root,
                    symbols=(f"{root}.v.0",), schema=self.research_schema,
                    stype_in="continuous", window=w,
                ))
                out.append(RealDatasetComponent(
                    kind=ComponentKind.DEFINITIONS, root_symbol=root,
                    symbols=(f"{root}.FUT",), schema="definition",
                    stype_in="parent", window=w,
                ))
                out.append(RealDatasetComponent(
                    kind=ComponentKind.ROLL_OVERLAP_RAW, root_symbol=root,
                    symbols=(), schema=self.research_schema,
                    stype_in="raw_symbol", window=w, resolved_from_transitions=True,
                ))
        return tuple(out)


class RealDatasetManifest(BaseModel):
    """Frozen typed record of ONE acquired component (13.5A section 13). Written
    after download; binds raw bytes + lineage + query + calendar + versions. A
    cosmetic path move must not change ``semantic_identity()``."""

    model_config = {"frozen": True, "extra": "forbid", "populate_by_name": True}

    schema_version: str = REAL_DATASET_SCHEMA_VERSION
    vendor: str = "databento"
    dataset: str = DATASET
    databento_schema: str            # ohlcv-1m | definition
    component_kind: ComponentKind
    root_symbol: str
    split_role: SplitRole

    # exact request
    symbol_request: tuple[str, ...]
    stype_in: str
    stype_out: str
    start: str
    end: str
    query_identity: str              # deterministic hash of the request

    price_domain: str                # raw_contract | (n/a for definitions)
    raw_or_continuous_role: str      # "continuous_front" | "roll_overlap_raw" | "definitions"

    # file identity (semantic identity does NOT include the path)
    raw_artifact_relpath: str
    raw_sha256: str
    row_count: int | None = None

    # versions / lineage
    calendar_identity: str
    ingestion_version: str
    canonicalization_version: str
    contract_definition_identity: str
    code_commit: str | None = None
    download_cost_usd: float | None = None
    downloaded_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def semantic_identity(self) -> str:
        """Identity that survives a cosmetic path relocation. Path + timestamp +
        cost are excluded; content hash + query + versions are in."""
        payload = {
            "schema_version": self.schema_version,
            "vendor": self.vendor,
            "dataset": self.dataset,
            "databento_schema": self.databento_schema,
            "component_kind": self.component_kind.value,
            "root_symbol": self.root_symbol,
            "split_role": self.split_role.value,
            "symbol_request": sorted(self.symbol_request),
            "stype_in": self.stype_in,
            "stype_out": self.stype_out,
            "start": self.start,
            "end": self.end,
            "query_identity": self.query_identity,
            "price_domain": self.price_domain,
            "raw_sha256": self.raw_sha256,
            "calendar_identity": self.calendar_identity,
            "ingestion_version": self.ingestion_version,
            "canonicalization_version": self.canonicalization_version,
            "contract_definition_identity": self.contract_definition_identity,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"realdsmani1:{digest}"

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=indent)


def load_plan(path: str, plan_id: str) -> RealDatasetPlan:
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["plans"][plan_id]
    windows = tuple(
        DateWindow(role=SplitRole(w["role"]), start=w["start"], end=w["end"])
        for w in p["windows"]
    )
    return RealDatasetPlan(
        plan_id=plan_id,
        roots=tuple(cfg["roots"]),
        windows=windows,
        calendar_version=cfg["calendar_version"],
        rolls_per_year=dict(cfg["rolls_per_year"]),
        roll_overlap_trading_days=int(cfg.get("roll_overlap_trading_days", 5)),
    )

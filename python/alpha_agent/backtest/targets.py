"""The deterministic Python -> C++ target schedule (Phase 11 sections 7-14).

``targets.csv`` is a **separate** research boundary next to the frozen Phase 02.5
``bars.csv`` / ``contracts.csv`` -- it is never overloaded onto them. It carries
**target-position intent only**:

    ts_event_ns, root_symbol, target_units, strategy_fingerprint, matched_rule_id

and explicitly **never** a fill price, execution price, raw_symbol, instrument_id,
slippage, spread, commission or a risk decision. ``matched_rule_id`` is a
non-executable audit field (not part of the schedule hash).

Semantics (section 11):
* **No row at T** => no new strategy decision. The C++ ScheduledTargetStrategy
  re-emits its previous target; it does **not** treat absence as flat.
* **A row with ``target_units = 0``** => an explicit FLAT decision.

``KEEP_PREVIOUS_TARGET`` is fully resolved here by the Python reference evaluator
before the schedule is written (section 12); the C++ side only replays validated
integer targets.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from alpha_agent.features.frame import FeatureFrame
from alpha_agent.features.registry import FEATURE_ENGINE_VERSION
from alpha_agent.strategy.evaluator import ReferenceEvaluator
from alpha_agent.strategy.plan import CompiledStrategyPlan

TARGET_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "ts_event_ns",
    "root_symbol",
    "target_units",
    "strategy_fingerprint",
    "matched_rule_id",
)

# Execution-plane names that must never appear as a target-schedule column. The
# bridge carries target intent only; execution belongs to the C++ Quant Core.
FORBIDDEN_TARGET_COLUMNS = frozenset(
    {
        "fill_price",
        "execution_price",
        "reference_price",
        "price",
        "raw_symbol",
        "instrument_id",
        "contract_month",
        "contract",
        "slippage",
        "slippage_ticks",
        "spread",
        "spread_ticks",
        "commission",
        "commission_usd",
        "latency",
        "latency_bars",
        "margin",
        "risk_decision",
        "risk_verdict",
        "order_quantity",
        "quantity",
        "pnl",
    }
)

_SCHEDULE_HASH_PREFIX = "targsched1"
_RAW_CONTRACT_RE = re.compile(r"^[A-Z]{1,3}[FGHJKMNQUVXZ]\d{1,2}$")
_ROOT_RE = re.compile(r"^[A-Z0-9]{1,12}$")


def _looks_like_raw_contract(symbol: str) -> bool:
    """A bare root is ``NQ`` / ``CL`` / ``ES``; a raw contract is ``NQU6`` /
    ``CLZ26``. The Python bridge must only ever name a root (section 14)."""
    return bool(_RAW_CONTRACT_RE.match(symbol)) or "." in symbol


class TargetScheduleRow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    ts_event_ns: int = Field(gt=0)
    root_symbol: str
    target_units: int
    strategy_fingerprint: str
    # Non-executable audit only: which DSL rule produced the target (or None for
    # the default action). Never part of the schedule hash / executable intent.
    matched_rule_id: str | None = None

    @field_validator("target_units", mode="before")
    @classmethod
    def _strict_int(cls, v: object) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(  # noqa: TRY004 -- ValueError for parity with the DSL's strict scalar typing
                f"target_units must be a plain int, got {type(v).__name__} {v!r}"
            )
        return v

    @field_validator("root_symbol")
    @classmethod
    def _root_only(cls, v: str) -> str:
        if not _ROOT_RE.match(v):
            raise ValueError(f"root_symbol {v!r} is not a bare root ([A-Z0-9]{{1,12}})")
        if _looks_like_raw_contract(v):
            raise ValueError(
                f"root_symbol {v!r} looks like a raw contract; the target schedule names a "
                "ROOT only -- the C++ engine resolves the real execution contract via "
                "MarketState -> ActiveContractResolver (section 14)"
            )
        return v

    @field_validator("strategy_fingerprint")
    @classmethod
    def _fp(cls, v: str) -> str:
        if not v.startswith("stratdsl1:"):
            raise ValueError(f"strategy_fingerprint {v!r} is not a stratdsl1 fingerprint")
        return v


class TargetSchedule(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    root_symbol: str
    strategy_fingerprint: str
    strategy_id: str
    strategy_dsl_version: str
    feature_engine_version: str
    warmup_bars: int = Field(ge=0)
    rows: tuple[TargetScheduleRow, ...]

    @model_validator(mode="after")
    def _consistency(self) -> TargetSchedule:
        if _looks_like_raw_contract(self.root_symbol) or not _ROOT_RE.match(self.root_symbol):
            raise ValueError(f"schedule root_symbol {self.root_symbol!r} must be a bare root")
        prev = 0
        for r in self.rows:
            if r.strategy_fingerprint != self.strategy_fingerprint:
                raise ValueError(
                    "target schedule contains mixed strategy fingerprints "
                    f"({r.strategy_fingerprint!r} != {self.strategy_fingerprint!r}); a schedule "
                    "carries exactly one StrategySpec fingerprint (section 13)"
                )
            if r.root_symbol != self.root_symbol:
                raise ValueError(
                    f"target row root {r.root_symbol!r} != schedule root {self.root_symbol!r}; "
                    "a one-root strategy cannot emit a target for another root (section 14)"
                )
            if r.ts_event_ns <= prev:
                raise ValueError(
                    f"target rows must be strictly time-ordered; {r.ts_event_ns} <= {prev}"
                )
            prev = r.ts_event_ns
        return self

    # -- derived ---------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def date_range_ns(self) -> tuple[int, int] | None:
        if not self.rows:
            return None
        return (self.rows[0].ts_event_ns, self.rows[-1].ts_event_ns)

    def schedule_hash(self) -> str:
        """Deterministic hash of the **executable** intent only (ts, root,
        target_units) plus the fingerprint and DSL/engine versions. Stable across
        processes and independent of ``matched_rule_id`` / row object identity
        (section 6/F)."""
        payload = {
            "prefix": _SCHEDULE_HASH_PREFIX,
            "root_symbol": self.root_symbol,
            "strategy_fingerprint": self.strategy_fingerprint,
            "strategy_dsl_version": self.strategy_dsl_version,
            "feature_engine_version": self.feature_engine_version,
            "rows": [[r.ts_event_ns, r.root_symbol, r.target_units] for r in self.rows],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"{_SCHEDULE_HASH_PREFIX}:{digest}"

    # -- serialization -------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_event_ns": [r.ts_event_ns for r in self.rows],
                "root_symbol": [r.root_symbol for r in self.rows],
                "target_units": [r.target_units for r in self.rows],
                "strategy_fingerprint": [r.strategy_fingerprint for r in self.rows],
                "matched_rule_id": [
                    "" if r.matched_rule_id is None else r.matched_rule_id for r in self.rows
                ],
            },
            columns=list(TARGET_SCHEDULE_COLUMNS),
        )

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        root_symbol: str,
        strategy_fingerprint: str,
        strategy_id: str = "",
        strategy_dsl_version: str = "",
        feature_engine_version: str = FEATURE_ENGINE_VERSION,
        warmup_bars: int = 0,
    ) -> TargetSchedule:
        df = pd.read_csv(path)
        cols = tuple(df.columns)
        forbidden = FORBIDDEN_TARGET_COLUMNS & {c.lower() for c in cols}
        if forbidden:
            raise ValueError(
                f"targets.csv contains execution-plane column(s) {sorted(forbidden)}; the "
                "target schedule carries target intent only"
            )
        if cols != TARGET_SCHEDULE_COLUMNS:
            raise ValueError(
                f"targets.csv header {cols} != required {TARGET_SCHEDULE_COLUMNS}"
            )
        rows = tuple(
            TargetScheduleRow(
                ts_event_ns=int(rec.ts_event_ns),
                root_symbol=str(rec.root_symbol),
                target_units=int(rec.target_units),
                strategy_fingerprint=str(rec.strategy_fingerprint),
                matched_rule_id=(
                    None
                    if pd.isna(rec.matched_rule_id) or str(rec.matched_rule_id) == ""
                    else str(rec.matched_rule_id)
                ),
            )
            for rec in df.itertuples(index=False)
        )
        return cls(
            root_symbol=root_symbol,
            strategy_fingerprint=strategy_fingerprint,
            strategy_id=strategy_id,
            strategy_dsl_version=strategy_dsl_version,
            feature_engine_version=feature_engine_version,
            warmup_bars=warmup_bars,
            rows=rows,
        )


def build_target_schedule(
    plan: CompiledStrategyPlan,
    frame: FeatureFrame,
    *,
    evaluator: ReferenceEvaluator | None = None,
) -> TargetSchedule:
    """Run the deterministic Phase 10 reference evaluator and turn its
    target-position decisions into a target schedule.

    Schedule policy (section 11): a row is emitted for bar ``T`` **only** when
    every feature value referenced that bar was present and every rule was
    evaluable -- i.e. the strategy genuinely had enough information to decide.
    Warm-up bars and bars inside a data gap produce **no row** (NO DECISION, not
    a flat decision). ``KEEP_PREVIOUS_TARGET`` is already resolved into a
    concrete integer by the evaluator.
    """
    ev = evaluator or ReferenceEvaluator(plan)
    decisions = ev.evaluate_frame(frame)
    rows: list[TargetScheduleRow] = []
    for d in decisions:
        if d.missing_features or d.not_evaluable_rule_ids:
            continue
        rows.append(
            TargetScheduleRow(
                ts_event_ns=d.ts_event_ns,
                root_symbol=plan.root_symbol,
                target_units=d.target_units,
                strategy_fingerprint=plan.fingerprint,
                matched_rule_id=d.matched_rule_id,
            )
        )
    return TargetSchedule(
        root_symbol=plan.root_symbol,
        strategy_fingerprint=plan.fingerprint,
        strategy_id=plan.strategy_id,
        strategy_dsl_version=plan.dsl_version,
        feature_engine_version=frame.lineage.engine_version,
        warmup_bars=plan.warmup_bars,
        rows=tuple(rows),
    )

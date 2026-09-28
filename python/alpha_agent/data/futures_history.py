"""Futures research history orchestrator (Phase 04).

    canonical raw-contract bars (one continuous symbol's history)
        -> layer 6 roll map        data/processed/rolls/<sym>.parquet
        -> layer 3 unadjusted       data/processed/continuous/<sym>.parquet
        -> layer 4 back-adjusted    data/processed/backadjusted/<sym>.parquet   (research only)

Every artifact gets a lineage sidecar. No network. Execution still resolves only
to real raw contracts -- layers 3/4 are research views.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha_agent.data.backadjust import AdjustmentMode, build_back_adjusted_series
from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.data.continuous import build_continuous_series
from alpha_agent.data.contract_lifecycle import FirstNoticePolicy
from alpha_agent.data.definitions import DefinitionRegistry
from alpha_agent.data.diagnostics import DiagnosticsReport
from alpha_agent.data.lineage import Lineage, code_commit, write_lineage
from alpha_agent.data.rolls import RollPricePolicy, build_roll_events, rolls_frame
from alpha_agent.schemas.market_data import (
    BACKADJUSTED_SCHEMA_VERSION,
    CONTINUOUS_SCHEMA_VERSION,
    ROLL_SCHEMA_VERSION,
    RollEvent,
)

PROCESSED_ROOT = Path("data/processed")


@dataclass
class FuturesHistoryResult:
    continuous_symbol: str
    rolls: list[RollEvent]
    continuous: pd.DataFrame
    back_adjusted: pd.DataFrame
    paths: dict[str, Path] = field(default_factory=dict)
    diagnostics: dict[str, DiagnosticsReport] = field(default_factory=dict)


def build_futures_history(
    canonical_bars: pd.DataFrame,
    *,
    continuous_symbol: str,
    registry: DefinitionRegistry,
    calendar: SessionCalendar | None = None,
    roll_rule: str = "instrument_id_transition",
    price_policy: RollPricePolicy = RollPricePolicy.SAME_TIMESTAMP_CLOSE_CLOSE,
    overlap_bars: pd.DataFrame | None = None,
    adjustment_mode: AdjustmentMode = AdjustmentMode.RETROSPECTIVE_RESEARCH,
    as_of_ts_ns: int | None = None,
    first_notice_policy: FirstNoticePolicy | None = None,
    processed_root: str | Path = PROCESSED_ROOT,
    write: bool = True,
) -> FuturesHistoryResult:
    calendar = calendar or default_calendar()
    first_notice_policy = first_notice_policy or FirstNoticePolicy()
    processed_root = Path(processed_root)
    commit = code_commit()

    rolls, roll_report = build_roll_events(
        canonical_bars, continuous_symbol=continuous_symbol, registry=registry,
        rule=roll_rule, price_policy=price_policy, overlap_bars=overlap_bars,
    )
    continuous, cont_report = build_continuous_series(
        canonical_bars, continuous_symbol=continuous_symbol, registry=registry, rolls=rolls,
    )
    back_adjusted, adj_report = build_back_adjusted_series(
        continuous, rolls, continuous_symbol=continuous_symbol,
        mode=adjustment_mode, as_of_ts_ns=as_of_ts_ns,
    )

    result = FuturesHistoryResult(
        continuous_symbol=continuous_symbol, rolls=rolls,
        continuous=continuous, back_adjusted=back_adjusted,
        diagnostics={"rolls": roll_report, "continuous": cont_report, "back_adjusted": adj_report},
    )
    if not write:
        return result

    n_fallback = sum(1 for r in rolls if r.used_fallback)

    def _lin(kind: str, price_domain: str, schema_version: str, diag: DiagnosticsReport) -> Lineage:
        return Lineage(
            artifact_kind=kind, code_commit=commit, price_domain=price_domain,
            schema_version=schema_version, continuous_symbol=continuous_symbol,
            roll_rule=roll_rule, roll_price_policy=price_policy.value,
            adjustment_method="additive" if kind == "backadjusted" else None,
            adjustment_mode=adjustment_mode.value if kind == "backadjusted" else None,
            as_of_ts_ns=as_of_ts_ns if kind == "backadjusted" else None,
            calendar_version=calendar.version,
            first_notice_policy={
                "apply_first_notice": first_notice_policy.apply_first_notice,
                "safety_buffer_days": first_notice_policy.safety_buffer_days,
                "safety_buffer_business_days": first_notice_policy.safety_buffer_business_days,
            },
            extra={"n_rolls": len(rolls), "n_rolls_used_fallback": n_fallback},
            diagnostics_summary=diag.summary(),
        )

    safe = continuous_symbol.replace("/", "_")
    targets = {
        "rolls": (processed_root / "rolls" / f"{safe}.parquet", rolls_frame(rolls),
                  "n/a", ROLL_SCHEMA_VERSION, roll_report),
        "continuous": (processed_root / "continuous" / f"{safe}.parquet", continuous,
                       "raw_continuous", CONTINUOUS_SCHEMA_VERSION, cont_report),
        "backadjusted": (processed_root / "backadjusted" / f"{safe}.parquet", back_adjusted,
                         "back_adjusted", BACKADJUSTED_SCHEMA_VERSION, adj_report),
    }
    for kind, (path, frame, domain, ver, diag) in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        write_lineage(path, _lin(kind, domain, ver, diag))
        result.paths[kind] = path
    return result

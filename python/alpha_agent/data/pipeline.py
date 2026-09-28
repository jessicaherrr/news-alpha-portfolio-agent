"""Canonical futures data pipeline (Phase 03).

    raw OHLCV artifact  +  raw definition artifact
            -> verify manifests
            -> parse definitions -> DefinitionRegistry -> contracts parquet
            -> canonicalize OHLCV -> canonical bars parquet (per contract)
            -> lineage sidecars

No network, no back-adjustment, no roll math -- those are later phases.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_agent.data.calendars import SessionCalendar, default_calendar
from alpha_agent.data.canonicalize import NormalizationPolicy, canonicalize
from alpha_agent.data.definitions import (
    DefinitionRegistry,
    parse_definition_frame,
    write_contracts_parquet,
)
from alpha_agent.data.diagnostics import DiagnosticsReport, PipelineError
from alpha_agent.data.lineage import Lineage, code_commit, write_lineage
from alpha_agent.data.raw_store import RawArtifact, verify_manifest
from alpha_agent.schemas.market_data import CANONICAL_BAR_COLUMNS, PRICE_SCALE, UNDEF_PRICE

PROCESSED_ROOT = Path("data/processed")


@dataclass
class PipelineResult:
    contracts_paths: list[Path]
    bars_paths: list[Path]
    diagnostics: DiagnosticsReport
    registry: DefinitionRegistry
    canonical: pd.DataFrame


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported raw artifact format: {path.suffix}")


def run_canonical_pipeline(
    ohlcv_artifact: str | Path,
    definition_artifact: str | Path,
    *,
    processed_root: str | Path = PROCESSED_ROOT,
    calendar: SessionCalendar | None = None,
    root_map: dict[str, str] | None = None,
    policy: NormalizationPolicy | None = None,
    require_clean: bool = False,
) -> PipelineResult:
    """Run the full pipeline. ``require_clean=True`` fails on *any* diagnostic
    error (not just fatal ones)."""
    policy = policy or NormalizationPolicy()
    calendar = calendar or default_calendar()
    processed_root = Path(processed_root)

    ohlcv_raw = verify_manifest(ohlcv_artifact)
    def_raw = verify_manifest(definition_artifact)

    # --- contracts (layer 5)
    specs = parse_definition_frame(_read_frame(def_raw.path), root_map=root_map)
    registry = DefinitionRegistry(specs)
    contracts_dir = processed_root / "contracts"
    contracts_paths = write_contracts_parquet(specs, contracts_dir)

    commit = code_commit()
    policy_dict = dataclasses.asdict(policy)

    def _lineage(kind: str, source: RawArtifact, diag: dict) -> Lineage:
        return Lineage(
            artifact_kind=kind,
            source_raw_sha256=source.manifest.sha256,
            source_manifest_path=str(source.manifest_path),
            dataset=source.manifest.dataset,
            requested_symbols=source.manifest.symbols,
            schema=source.manifest.schema_,
            stype_in=source.manifest.stype_in,
            stype_out=source.manifest.stype_out,
            code_commit=commit,
            price_scale_policy=f"fixed_point/{PRICE_SCALE} -> float64; UNDEF_PRICE={UNDEF_PRICE}",
            normalization_policy=policy_dict,
            diagnostics_summary=diag,
        )

    for cp in contracts_paths:
        write_lineage(cp, _lineage("contracts", def_raw, {}))

    # --- canonical bars (layer 2)
    canonical, report = canonicalize(_read_frame(ohlcv_raw.path), registry, calendar, policy=policy)

    if require_clean and report.errors:
        raise PipelineError(
            f"{len(report.errors)} diagnostic error(s) with require_clean=True", report
        )

    bars_dir = processed_root / "bars"
    bars_paths: list[Path] = []
    for (root, raw_symbol), grp in canonical.groupby(["root_symbol", "raw_symbol"], sort=True):
        out = bars_dir / str(root) / f"{raw_symbol}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = grp.loc[:, list(CANONICAL_BAR_COLUMNS)].reset_index(drop=True)
        frame.to_parquet(out, index=False)
        write_lineage(out, _lineage("canonical_bars", ohlcv_raw, report.summary()))
        bars_paths.append(out)

    return PipelineResult(
        contracts_paths=contracts_paths,
        bars_paths=bars_paths,
        diagnostics=report,
        registry=registry,
        canonical=canonical,
    )

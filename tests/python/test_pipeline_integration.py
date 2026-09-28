"""Phase 03 -- full raw -> canonical -> C++ CLI integration."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from alpha_agent.adapters.cpp_cli import canonical_bars_to_boundary, write_boundary_bundle
from alpha_agent.data.definitions import contracts_frame
from alpha_agent.data.lineage import load_lineage
from alpha_agent.data.pipeline import run_canonical_pipeline
from alpha_agent.schemas.market_data import CANONICAL_BAR_COLUMNS, CANONICAL_SCHEMA_VERSION
from vendor_fixtures import (
    store_raw_definitions,
    store_raw_ohlcv,
    vendor_definition_frame,
    vendor_ohlcv_frame,
)


def _run_pipeline(tmp_path, *, n_bars=60):
    raw_dir = tmp_path / "raw"
    ohlcv_art = store_raw_ohlcv(vendor_ohlcv_frame(n_bars=n_bars), raw_dir)
    def_art = store_raw_definitions(vendor_definition_frame(), raw_dir)
    result = run_canonical_pipeline(
        ohlcv_art.path,
        def_art.path,
        processed_root=tmp_path / "processed",
        require_clean=True,
    )
    return ohlcv_art, def_art, result


def test_pipeline_writes_bars_contracts_and_lineage(tmp_path):
    ohlcv_art, _, result = _run_pipeline(tmp_path)

    assert len(result.bars_paths) == 1
    assert len(result.contracts_paths) == 1
    bars_path = result.bars_paths[0]
    assert bars_path == tmp_path / "processed" / "bars" / "NQ" / "NQM5.parquet"

    bars = pd.read_parquet(bars_path)
    assert list(bars.columns) == list(CANONICAL_BAR_COLUMNS)
    assert len(bars) == 60

    # lineage points back at the exact raw artifact
    lin = load_lineage(bars_path)
    assert lin.source_raw_sha256 == ohlcv_art.manifest.sha256
    assert lin.source_manifest_path == str(ohlcv_art.manifest_path)
    assert lin.timezone == "UTC"
    assert lin.canonical_schema_version == CANONICAL_SCHEMA_VERSION
    assert lin.stype_out == "instrument_id"
    assert "by_kind" in lin.diagnostics_summary

    con_lin = load_lineage(result.contracts_paths[0])
    assert con_lin.artifact_kind == "contracts"


def test_processed_bar_resolves_through_registry(tmp_path):
    _, _, result = _run_pipeline(tmp_path)
    bars = pd.read_parquet(result.bars_paths[0])
    for iid in bars["instrument_id"].unique():
        assert result.registry.is_tradable(int(iid))


def test_full_python_to_cpp_cli_integration(tmp_path):
    exe = Path("build/cpp/cpp/quant_backtest_csv")
    if not exe.exists():
        pytest.skip("C++ core not built")

    _, _, result = _run_pipeline(tmp_path, n_bars=60)
    bars_boundary = canonical_bars_to_boundary(result.canonical)
    contracts_boundary = contracts_frame(result.registry.specs())

    bundle_dir = tmp_path / "bundle"
    bars_csv, contracts_csv = write_boundary_bundle(bars_boundary, contracts_boundary, bundle_dir)

    proc = subprocess.run(
        [str(exe), str(bars_csv), str(contracts_csv), "5", "0.0"],
        check=True, capture_output=True, text=True,
    )
    out = json.loads(proc.stdout)
    assert out["bars"] == 60
    assert out["contracts_resolved"] == 1

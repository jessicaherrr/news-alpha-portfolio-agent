"""Phase 03 -- write-once raw store + manifest verification."""
from __future__ import annotations

import pytest
from alpha_agent.data.manifest import ManifestMismatchError
from alpha_agent.data.raw_store import load_raw_manifest, store_raw, verify_manifest
from vendor_fixtures import store_raw_ohlcv, vendor_ohlcv_frame


def test_store_raw_writes_artifact_and_manifest(tmp_path):
    art = store_raw_ohlcv(vendor_ohlcv_frame(n_bars=10), tmp_path)
    assert art.path.exists()
    assert art.manifest_path.exists()
    assert art.manifest.sha256
    assert art.manifest.row_count == 10
    assert art.manifest.stype_out == "instrument_id"
    assert art.manifest.dataset == "GLBX.MDP3"
    verify_manifest(art.path)  # no raise


def test_raw_artifact_cannot_be_overwritten(tmp_path):
    store_raw_ohlcv(vendor_ohlcv_frame(n_bars=10), tmp_path)
    with pytest.raises(FileExistsError):
        store_raw_ohlcv(vendor_ohlcv_frame(n_bars=10), tmp_path)


def test_modified_raw_file_fails_verification(tmp_path):
    art = store_raw_ohlcv(vendor_ohlcv_frame(n_bars=10), tmp_path)
    art.path.write_bytes(art.path.read_bytes() + b"tampered")
    with pytest.raises(ManifestMismatchError):
        verify_manifest(art.path)
    with pytest.raises(ManifestMismatchError):
        art.verify()


def test_load_raw_manifest_roundtrip(tmp_path):
    art = store_raw(
        b"hello",
        vendor="databento", dataset="GLBX.MDP3", schema="ohlcv-1m",
        stype_in="continuous", stype_out="instrument_id", symbols=["NQ.v.0"],
        start="2025-03-17", end="2025-03-18", artifact_format="csv", row_count=None,
        root_dir=tmp_path,
    )
    m = load_raw_manifest(art.path)
    assert m.schema_ == "ohlcv-1m"
    assert m.symbols == ["NQ.v.0"]
    assert m.artifact_format == "csv"
    assert "created_at" in m.model_dump()

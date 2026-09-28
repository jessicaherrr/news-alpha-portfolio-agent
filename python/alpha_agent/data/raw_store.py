"""Write-once raw vendor storage (layer 1 of the data model).

A raw artifact is the vendor's bytes, stored verbatim under ``data/raw/<vendor>/``
and never modified. Every artifact gets a sidecar SHA-256 manifest. Processed
writers must never touch these files; they only read + ``verify_manifest``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from alpha_agent.data.manifest import ManifestMismatchError, sha256_file

RAW_ROOT = Path("data/raw")
MANIFEST_SUFFIX = ".manifest.json"


class RawManifest(BaseModel):
    """Sidecar describing one immutable raw artifact."""

    vendor: str
    dataset: str
    schema_: str = Field(alias="schema")
    stype_in: str
    stype_out: str
    symbols: list[str]
    start: str
    end: str
    artifact_format: str            # "parquet" | "dbn.zst" | "csv" | ...
    artifact_filename: str
    sha256: str
    row_count: int | None = None
    created_at: str
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class RawArtifact(BaseModel):
    path: Path
    manifest_path: Path
    manifest: RawManifest

    model_config = {"arbitrary_types_allowed": True}

    def verify(self) -> None:
        verify_manifest(self.path)


def _request_hash(dataset: str, schema: str, stype_in: str, stype_out: str,
                  symbols: list[str], start: str, end: str) -> str:
    canonical = json.dumps(
        {
            "dataset": dataset, "schema": schema, "stype_in": stype_in,
            "stype_out": stype_out, "symbols": sorted(symbols),
            "start": start, "end": end,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def raw_artifact_dir(
    *,
    vendor: str,
    dataset: str,
    schema: str,
    stype_in: str,
    stype_out: str,
    symbols: list[str],
    start: str,
    end: str,
    root_dir: str | Path = RAW_ROOT,
) -> Path:
    """Deterministic directory for a request: same request -> same directory,
    so a re-request trips the write-once guard instead of silently diverging."""
    h = _request_hash(dataset, schema, stype_in, stype_out, symbols, start, end)
    return Path(root_dir) / vendor / dataset / schema / h


def store_raw(
    data: bytes,
    *,
    vendor: str,
    dataset: str,
    schema: str,
    stype_in: str,
    stype_out: str,
    symbols: list[str],
    start: str,
    end: str,
    artifact_format: str,
    filename: str | None = None,
    row_count: int | None = None,
    extra: dict[str, Any] | None = None,
    root_dir: str | Path = RAW_ROOT,
) -> RawArtifact:
    """Write ``data`` once and produce its manifest.

    Raises ``FileExistsError`` if the target artifact already exists -- raw data
    is never overwritten.
    """
    out_dir = raw_artifact_dir(
        vendor=vendor, dataset=dataset, schema=schema, stype_in=stype_in,
        stype_out=stype_out, symbols=symbols, start=start, end=end, root_dir=root_dir,
    )
    name = filename or f"{schema}.{artifact_format}"
    artifact_path = out_dir / name
    manifest_path = artifact_path.with_name(artifact_path.name + MANIFEST_SUFFIX)

    if artifact_path.exists():
        raise FileExistsError(f"raw artifact already exists, refusing to overwrite: {artifact_path}")
    if manifest_path.exists():
        raise FileExistsError(f"raw manifest already exists, refusing to overwrite: {manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Write atomically so a crash can't leave a half-written "immutable" file.
    tmp = artifact_path.with_name(artifact_path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(artifact_path)

    manifest = RawManifest(
        vendor=vendor,
        dataset=dataset,
        schema=schema,
        stype_in=stype_in,
        stype_out=stype_out,
        symbols=list(symbols),
        start=start,
        end=end,
        artifact_format=artifact_format,
        artifact_filename=name,
        sha256=sha256_file(artifact_path),
        row_count=row_count,
        created_at=datetime.now(UTC).isoformat(),
        extra=extra or {},
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(by_alias=True), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return RawArtifact(path=artifact_path, manifest_path=manifest_path, manifest=manifest)


def manifest_path_for(artifact_path: str | Path) -> Path:
    p = Path(artifact_path)
    return p.with_name(p.name + MANIFEST_SUFFIX)


def load_raw_manifest(artifact_or_manifest_path: str | Path) -> RawManifest:
    p = Path(artifact_or_manifest_path)
    if not p.name.endswith(MANIFEST_SUFFIX):
        p = manifest_path_for(p)
    if not p.exists():
        raise FileNotFoundError(f"no raw manifest at {p}")
    return RawManifest.model_validate_json(p.read_text(encoding="utf-8"))


def verify_manifest(artifact_or_manifest_path: str | Path) -> RawArtifact:
    """Recompute the artifact SHA-256 and compare it to the manifest.

    Raises ``ManifestMismatchError`` on any mismatch, ``FileNotFoundError`` if
    either file is missing.
    """
    p = Path(artifact_or_manifest_path)
    manifest_path = p if p.name.endswith(MANIFEST_SUFFIX) else manifest_path_for(p)
    manifest = load_raw_manifest(manifest_path)
    artifact_path = manifest_path.with_name(manifest.artifact_filename)
    if not artifact_path.exists():
        raise FileNotFoundError(f"raw artifact missing: {artifact_path}")

    actual = sha256_file(artifact_path)
    if actual != manifest.sha256:
        raise ManifestMismatchError(
            f"raw artifact {artifact_path} does not match its manifest:\n"
            f"  manifest sha256 {manifest.sha256}\n"
            f"  actual   sha256 {actual}"
        )
    return RawArtifact(path=artifact_path, manifest_path=manifest_path, manifest=manifest)

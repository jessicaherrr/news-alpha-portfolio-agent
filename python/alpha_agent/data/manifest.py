from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(data_path: str | Path, metadata: dict[str, Any]) -> Path:
    data_path = Path(data_path)
    payload = dict(metadata)
    payload["sha256"] = sha256_file(data_path)
    manifest = data_path.with_suffix(data_path.suffix + ".manifest.json")
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


class ManifestMismatchError(RuntimeError):
    """Raised when a file's SHA-256 does not match its recorded manifest."""


def verify_file_sha256(path: str | Path, expected_sha256: str) -> None:
    """Recompute the SHA-256 of ``path`` and fail loudly on any mismatch."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ManifestMismatchError(
            f"SHA-256 mismatch for {Path(path)}:\n  expected {expected_sha256}\n  actual   {actual}"
        )

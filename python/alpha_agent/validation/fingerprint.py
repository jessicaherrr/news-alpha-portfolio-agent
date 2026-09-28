"""Deterministic fingerprints for every Phase 13 semantic object (section 21).

A fingerprint is ``<prefix>:`` + SHA-256 over a canonical JSON payload. Canonical
JSON == ``json.dumps(payload, sort_keys=True, separators=(",", ":"))`` with
NaN/inf rejected. Cosmetic fields (display names, timestamps, free-text
rationale, RNG-independent labels) are never included by the callers here.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

VALIDATION_FRAMEWORK_VERSION = "reliability-validation/1"


def _walk(obj: Any, *, on_non_finite: str) -> Any:
    """``on_non_finite``: 'reject' (semantic-choice fingerprints -- a non-finite
    value is a bug) or 'sentinel' (report fingerprints -- an undefined statistic
    like a Sharpe with zero variance is legitimately NaN and must hash stably)."""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if on_non_finite == "reject":
            raise ValueError(f"non-finite float {obj!r} cannot enter a validation fingerprint")
        return f"__nonfinite__:{obj!r}"
    if isinstance(obj, dict):
        return {k: _walk(v, on_non_finite=on_non_finite) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk(v, on_non_finite=on_non_finite) for v in obj]
    return obj


def canonical_json(payload: dict, *, allow_non_finite: bool = False) -> str:
    """Deterministic, cross-process-stable JSON. Rejects non-finite floats unless
    ``allow_non_finite`` (then they hash as a stable sentinel string)."""
    clean = _walk(payload, on_non_finite="sentinel" if allow_non_finite else "reject")
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(prefix: str, payload: dict, *, allow_non_finite: bool = False) -> str:
    digest = hashlib.sha256(
        canonical_json(payload, allow_non_finite=allow_non_finite).encode("utf-8")
    ).hexdigest()
    return f"{prefix}:{digest}"

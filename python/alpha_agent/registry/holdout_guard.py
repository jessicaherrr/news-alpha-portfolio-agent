"""Fail-loud locked-holdout guard for everything written into the registry.

Phase 14 is metadata / memory infrastructure: there is zero reason for a 2025
market-data or performance timestamp to appear anywhere in it (prompt 14
section 25). Any such value is a hard error, not a warning.

The guard is deliberately narrow about *what counts*: a **market-data or
performance** timestamp. A registry row also carries system bookkeeping times
(when an artifact was generated, when a record was created); those are ordinary
wall-clock values and are stored in their own columns, never inside a guarded
payload. The one 2025 string the registry is allowed to hold is the declarative
``locked_holdout_2025`` *label* asserting the window was never touched -- it
carries no timestamp value and is checked for exactly that.

Runtime-integration release note: two real defects were found in this module
(never in ``ReliabilityPolicy`` / BH / DSR / null methodology -- this file is
pure text/int scanning, not a scientific semantic):

1. **Under-detection.** The old date regex used ``\\b...\\b`` word-boundary
   anchors. ``\\b`` requires a transition between a word char and a non-word
   char; the character right after an ISO date's day digits in a full
   timestamp (``2025-06-15T10:30:00Z``) is ``T`` -- also a word character -- so
   there was NO boundary there and the trailing anchor silently failed to
   match. A full ISO-8601 *datetime* (as opposed to a bare date) sailed
   straight through the old guard. Fixed with digit-only lookaround
   (``(?<!\\d)...(?!\\d)``), which anchors on "not another digit" instead of
   "any non-word char" and so is insensitive to what follows the date.
2. **Over-rejection, case A (observational context).** Once (1) is fixed, a
   live, non-scientific, tagged ``OBSERVATIONAL_CONTEXT_ONLY`` note (see
   ``alpha_agent.ui.market_context.observational_context_note``) -- a delayed
   IBKR quote carrying TODAY's wall-clock timestamp -- would trip the SAME
   guard the moment today is on/after 2025-01-01, even though it is
   conversational color, never scientific evidence. Disabling or loosening the
   guard is not the fix (that would reopen the real hole). Instead
   :func:`assert_no_holdout_market_data` accepts an explicit,
   caller-declared allowlist of PATHS (``observational_context_paths``) where a
   strictly-typed, whole-string-matched observational-context note is exempt.
   Both conditions are required: an untagged path never gets the exemption
   (a scientific payload accidentally containing the tag text is still
   scanned), and a tagged path only exempts a string that matches the exact,
   closed grammar below -- not "contains the tag somewhere".
3. **Over-rejection, case B (bookkeeping timestamps).** The same fix (1)
   surfaced a second, independent, pre-existing defect: several call sites
   pass a WHOLE model/artifact dump -- including its ordinary
   ``created_at`` / ``generated_at``-style wall-clock field -- straight into
   this guard, instead of the model's own scientific-content-only payload
   (``ExperimentRecord.scientific_payload()`` and friends already do this
   correctly; a few callers that dump the full model or an ad-hoc JSON
   artifact did not). This was always wrong -- the module docstring above
   already says bookkeeping times are "stored in their own columns, never
   inside a guarded payload" -- but the old regex's failure to match a
   trailing ``T`` happened to mask it for every ``isoformat()`` timestamp.
   Rather than hunt down and fix every such call site (an artifact-wide sweep
   in particular has no fixed shape to special-case by path), a narrow, named
   set of KNOWN, code-reviewed bookkeeping-timestamp key names
   (``BOOKKEEPING_TIMESTAMP_KEYS``) is skipped by default. This is a key-NAME
   exemption, not a content exemption: unlike the observational-context case,
   nothing under one of these keys is ever inspected, because in every model
   in this codebase these keys hold only a ``datetime.now(UTC).isoformat()``
   capture, never market data.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

#: 2025-01-01T00:00:00Z in nanoseconds. Mirrors
#: ``alpha_agent.data.real_market_dataset.HOLDOUT_START_NS`` without importing
#: the (pandas/databento-touching) data layer into the registry.
HOLDOUT_START = "2025-01-01"
HOLDOUT_START_NS = 1_735_689_600_000_000_000

#: Anything at or above ~2001-09 in epoch-ns. Below this an integer is a count
#: (bars, trades, fills), not a timestamp.
_NS_TIMESTAMP_FLOOR = 1_000_000_000_000_000_000

#: A bare ISO date (``YYYY-MM-DD``) OR the date component of a full ISO-8601
#: datetime (``YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+HH:MM]``). Digit-only lookaround
#: (not ``\b``) so a following ``T`` never hides the match -- see module
#: docstring, defect 1.
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")


class HoldoutAccessError(AssertionError):
    """A locked-holdout (>= 2025-01-01) market-data / performance value reached
    the registry. Never downgrade this to a warning."""


def _fail(path: str, value: Any) -> None:
    raise HoldoutAccessError(
        f"locked-holdout value at {path!r}: {value!r} is >= {HOLDOUT_START}. "
        "2025 was never downloaded, queried, cost-fetched, loaded, featured or "
        "evaluated and must not enter the experiment registry."
    )


# ---------------------------------------------------------------------------
# provenance-aware observational-context exemption (see module docstring,
# defect 2). This recognises exactly ONE closed string shape -- the note
# `alpha_agent.ui.market_context.observational_context_note` builds -- and
# nothing else. It is not a general "trust this key" mechanism.
# ---------------------------------------------------------------------------
OBSERVATIONAL_CONTEXT_TAG = "OBSERVATIONAL_CONTEXT_ONLY"

#: Providers / feed states that are non-scientific by construction. Keep this
#: list in lockstep with `alpha_agent.marketdata` -- a new provider or a feed
#: state that could ever mean "this is real/live tradeable data" must NOT be
#: added here without re-reviewing this boundary.
#:
#: "DATABENTO" (Release UX Part D) added alongside "IBKR": both are read-only
#: OBSERVATION-plane providers whose notes are built by a single closed
#: template function (`alpha_agent.ui.market_context.observational_context_note`
#: / `alpha_agent.ui.databento_context.observational_context_note`) -- never
#: free text. "LIVE" is deliberately never added to the feeds set below: no
#: provider in this codebase has ever proven a live/streaming entitlement (see
#: `alpha_agent.marketdata.databento_schemas.DatabentoCapability`'s docstring),
#: so a note claiming LIVE would have no code path that could honestly produce
#: it -- adding the label here first would be a standing invitation to.
_ALLOWED_OBSERVATIONAL_PROVIDERS = frozenset({"IBKR", "DATABENTO"})
_ALLOWED_OBSERVATIONAL_FEEDS = frozenset({"DELAYED", "LATEST_AVAILABLE", "HISTORICAL_ONLY"})

#: Must match `alpha_agent.ui.market_context.observational_context_note`'s
#: trailing sentence EXACTLY. Anchoring on the fixed suffix (not `.+$`) closes
#: an adversarial hole: a string cannot smuggle a second, real date past this
#: exemption by appending it after the tag -- the whole string must equal this
#: one closed template, not merely start with the tag.
_OBSERVATIONAL_CONTEXT_SUFFIX = (
    "-- delayed market color for conversational context only; not scientific "
    "evidence, not a feature, not used in any validation or paper-eligibility "
    "decision."
)

_OBSERVATIONAL_CONTEXT_RE = re.compile(
    r"^OBSERVATIONAL_CONTEXT_ONLY "
    r"provider=(?P<provider>[A-Za-z0-9_]+) "
    r"feed=(?P<feed>[A-Za-z0-9_]+) "
    r"root=(?P<root>[A-Za-z0-9_]+) "
    r"last=(?P<last>\S+) "
    r"as_of=(?P<as_of>\S+) "
    + re.escape(_OBSERVATIONAL_CONTEXT_SUFFIX)
    + r"$"
)


# ---------------------------------------------------------------------------
# bookkeeping-timestamp key exemption (see module docstring, defect 3)
# ---------------------------------------------------------------------------
#: Every field name in this codebase that is EXCLUSIVELY a
#: ``datetime.now(UTC).isoformat()``-style "when was this record/artifact
#: produced" capture -- audited by hand (grep for the field name + its
#: assignment) before being added here, never guessed. Adding a name to this
#: set means its value is never inspected by this guard, so add one only after
#: confirming every use of that key is bookkeeping, never market/performance
#: data.
BOOKKEEPING_TIMESTAMP_KEYS: frozenset[str] = frozenset(
    {
        "created_at",
        "generated_at",
        "started_at",
        "finished_at",
        "stopped_at",
        "downloaded_at",
        "frozen_at",
    }
)


def _is_valid_observational_context_string(value: str) -> bool:
    """True iff `value` is, in its ENTIRETY, a well-formed observational-context
    note: a known provider, a known non-scientific feed state, and a
    syntactically real timestamp. A string that merely contains the tag as a
    substring of something else does not match -- the regex is anchored to the
    whole string (`^...$`), so this can never be satisfied by embedding the tag
    inside a longer, otherwise-scientific payload."""
    m = _OBSERVATIONAL_CONTEXT_RE.match(value)
    if not m:
        return False
    if m.group("provider") not in _ALLOWED_OBSERVATIONAL_PROVIDERS:
        return False
    if m.group("feed") not in _ALLOWED_OBSERVATIONAL_FEEDS:
        return False
    try:
        datetime.fromisoformat(m.group("as_of"))
    except ValueError:
        return False
    return True


def _path_is_declared_observational(path: str, declared_paths: frozenset[str]) -> bool:
    for prefix in declared_paths:
        if path == prefix or path.startswith((prefix + "[", prefix + ".")):
            return True
    return False


def assert_no_holdout_market_data(
    payload: Any,
    *,
    path: str = "$",
    observational_context_paths: frozenset[str] = frozenset(),
    bookkeeping_timestamp_keys: frozenset[str] = BOOKKEEPING_TIMESTAMP_KEYS,
) -> None:
    """Recursively refuse any locked-holdout market-data / performance value.

    Rules, applied to every leaf:

    * an ``int`` at or above the epoch-ns floor is treated as a market-data
      timestamp and must be ``< HOLDOUT_START_NS``;
    * a ``str`` containing an ISO ``YYYY-MM-DD`` date (bare, or as the date
      component of a full ISO-8601 datetime) must not contain a date
      ``>= 2025-01-01`` -- UNLESS it sits at one of ``observational_context_paths``
      (exact match, or a child of that path) AND it is, in its entirety, a
      well-formed ``OBSERVATIONAL_CONTEXT_ONLY`` note (see above). Both
      conditions are required; neither alone grants the exemption;
    * a dict value under a key named in ``bookkeeping_timestamp_keys`` is never
      inspected at all (see :data:`BOOKKEEPING_TIMESTAMP_KEYS`).

    The second rule is why the exclusive boundary ``2025-01-01`` is never stored
    as a bare date string: registry market windows are stored inclusively
    (``2023-01-01..2024-12-31``), matching the human-readable convention in
    CLAUDE.md.

    ``observational_context_paths`` defaults to empty: every existing call site
    that does not explicitly declare a knowledge-base-shaped path keeps that
    part of the guard fully closed, exactly as before.
    ``bookkeeping_timestamp_keys`` defaults to the module's audited set (opt
    OUT with ``frozenset()`` for a caller that wants the stricter old
    behaviour, never opt further IN without auditing a new name first).
    """
    if isinstance(payload, bool):
        return
    if isinstance(payload, int):
        if payload >= _NS_TIMESTAMP_FLOOR and payload >= HOLDOUT_START_NS:
            _fail(path, payload)
        return
    if isinstance(payload, str):
        if _path_is_declared_observational(
            path, observational_context_paths
        ) and _is_valid_observational_context_string(payload):
            return
        for m in _ISO_DATE.finditer(payload):
            if m.group(0) >= HOLDOUT_START:
                _fail(path, payload)
        return
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k in bookkeeping_timestamp_keys:
                continue
            assert_no_holdout_market_data(
                v,
                path=f"{path}.{k}",
                observational_context_paths=observational_context_paths,
                bookkeeping_timestamp_keys=bookkeeping_timestamp_keys,
            )
        return
    if isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            assert_no_holdout_market_data(
                v,
                path=f"{path}[{i}]",
                observational_context_paths=observational_context_paths,
                bookkeeping_timestamp_keys=bookkeeping_timestamp_keys,
            )
        return


def assert_holdout_declaration(label: str) -> None:
    """The one allowed mention of the holdout: a declarative never-accessed
    label with no timestamp value in it."""
    if _ISO_DATE.search(label):
        raise HoldoutAccessError(
            f"holdout declaration {label!r} carries a concrete date; it must be a "
            "statement that 2025 was never accessed, not a window value"
        )

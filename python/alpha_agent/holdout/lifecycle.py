"""Alpha Discovery campaign, Part L -- the 2025 holdout lifecycle (task spec
sections 61-65).

2025 is not locked because it is in the future (it is not -- see CLAUDE.md /
this module's own docstring: today is 2026). It is locked because it is a
PRISTINE, UNSEEN FINAL HOLDOUT: research/fast-screen sees 2018-2022, strict
validation sees 2023-2024, and 2025 is reserved for exactly one final,
untouched evaluation.

This module is PURE BOOKKEEPING -- it has no market-data-loading code
anywhere in it (no import of `alpha_agent.data.real_market_dataset`, no
Databento client, nothing reads a bar or a price). It cannot become a vector
for holdout access even in principle; the actual data-layer boundary remains
`alpha_agent.registry.holdout_guard.assert_no_holdout_market_data` /
`HOLDOUT_START_NS`, completely unchanged by this module.

States (task spec section 62):

    SEALED            -- untouched, no final manifest frozen yet (the default,
                          and this campaign's real 2025 state, forever, in
                          this session)
    FINAL_EVAL_READY  -- an immutable `FinalHoldoutManifest` has been frozen
                          (code/registry/candidate/policy state pinned) and
                          an explicit authorization mechanism is engaged, but
                          no evaluation has actually run
    CONSUMED          -- the one, single, final evaluation has run; 2025 is
                          now historical evidence, never "unseen" again
    ROLLED_FORWARD    -- this epoch's consumption is finished AND a NEW
                          prospective holdout epoch has been established as
                          its successor

Transitions are one-directional and irreversible in the direction that
matters: CONSUMED can never return to SEALED (task spec section 62/77), and
neither can FINAL_EVAL_READY (freezing a manifest is not itself
undo-able -- a fresh SEALED epoch is a NEW epoch, never the same one
un-frozen). `HoldoutLifecycle` persists one JSON record per epoch under
`data/holdout_lifecycle/` (gitignored, like the other Alpha Discovery
operational directories) so the state survives process restarts and is
independently auditable.

REAL 2025 CONSUMPTION IS NOT PERFORMED BY THIS MODULE, EVER, IN THIS
CODEBASE STATE. `consume()` requires an explicit authorization token that
must equal the `ALLOW_FINAL_2025_HOLDOUT_EVALUATION` environment variable's
own current value (never a hardcoded default, never inferred) -- and no
caller anywhere in this repository ever sets that variable or calls
`consume()` for the real "2025" epoch. Tests exercise this machinery only
against synthetic epoch labels.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIFECYCLE_DIR = REPO_ROOT / "data" / "holdout_lifecycle"

#: The one environment variable that may ever authorize a real consumption
#: (task spec section 64's own example name, used verbatim). Read at call
#: time only, inside `consume()` -- never cached, never defaulted to a truthy
#: value.
AUTHORIZATION_ENV_VAR = "ALLOW_FINAL_2025_HOLDOUT_EVALUATION"


class HoldoutState(str, Enum):
    SEALED = "SEALED"
    FINAL_EVAL_READY = "FINAL_EVAL_READY"
    CONSUMED = "CONSUMED"
    ROLLED_FORWARD = "ROLLED_FORWARD"


class HoldoutLifecycleError(RuntimeError):
    """Base class for this module's own errors -- an illegal transition, a
    missing/invalid authorization, or a manifest built against the wrong
    state."""


class HoldoutAuthorizationError(HoldoutLifecycleError):
    """`consume()` was called without a matching `ALLOW_FINAL_2025_HOLDOUT_
    EVALUATION` environment value. This is the ONE gate a real consumption
    must pass; it is never bypassable from inside this module."""


class FinalHoldoutManifest(BaseModel):
    """The immutable, frozen-before-evaluation snapshot (task spec section
    63). Every field is pinned BEFORE `FINAL_EVAL_READY` is reached -- no
    field here may change after freezing without becoming a different
    manifest (a new `manifest_hash`)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "final-holdout-manifest/1"
    epoch_label: str
    git_commit: str = ""
    code_version: str = ""
    registry_digest: str = ""
    candidate_experiment_identities: tuple[str, ...] = ()
    strategy_spec_fingerprints: tuple[str, ...] = ()
    validation_policy_fingerprint: str = ""
    execution_config_identity: str = ""
    cost_config_identity: str = ""
    data_identity: str = ""
    allowed_strategy_families: tuple[str, ...] = ()
    frozen_at: str = ""

    @model_validator(mode="after")
    def _no_dates_in_epoch_label(self) -> FinalHoldoutManifest:
        # the epoch label itself (e.g. "2025") is a category name, never a
        # holdout-guard-shaped concrete window value -- this module never
        # imports holdout_guard (no market-data concept exists here at all),
        # so this is a light, local, structural sanity check only.
        if "T" in self.epoch_label and ":" in self.epoch_label:
            raise ValueError("epoch_label must be a short category label, not a timestamp")
        return self

    def manifest_hash(self) -> str:
        import hashlib

        payload = self.model_dump(mode="json")
        payload.pop("frozen_at", None)
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class HoldoutEpochRecord(BaseModel):
    """One epoch's current lifecycle state + history. `state` is the only
    field that changes across the record's life; every manifest/consumption
    event that happened is preserved, never overwritten (task spec section
    64/77: failed/completed history is first-class evidence, same principle
    as the experiment registry)."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "holdout-epoch-record/1"
    epoch_label: str
    state: HoldoutState = HoldoutState.SEALED
    manifest: FinalHoldoutManifest | None = None
    consumed_at: str = ""
    successor_epoch_label: str | None = None
    accessed: bool = Field(
        default=False,
        description="True IFF a real evaluation actually ran (state == CONSUMED or ROLLED_FORWARD).",
    )

    def status(self) -> dict:
        """The exact honest pair the task spec's closing report line wants:
        '2025 STATUS: SEALED, 2025 ACCESSED: NO'."""
        return {"epoch": self.epoch_label, "status": self.state.value, "accessed": self.accessed}


class HoldoutLifecycle:
    """File-persisted state machine, one JSON record per epoch label."""

    def __init__(self, *, base_dir: Path | None = None):
        self._dir = base_dir or DEFAULT_LIFECYCLE_DIR

    def _path(self, epoch_label: str) -> Path:
        safe = epoch_label.replace("/", "_")
        return self._dir / f"{safe}.json"

    def get(self, epoch_label: str) -> HoldoutEpochRecord:
        """The current record, or a fresh SEALED default if none was ever
        written -- "no record on disk" and "SEALED" are the same honest
        state: nothing has ever happened to this epoch."""
        path = self._path(epoch_label)
        if not path.exists():
            return HoldoutEpochRecord(epoch_label=epoch_label)
        return HoldoutEpochRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _write(self, record: HoldoutEpochRecord) -> HoldoutEpochRecord:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(record.epoch_label).write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
        )
        return record

    def freeze_final_manifest(self, epoch_label: str, *, manifest: FinalHoldoutManifest) -> HoldoutEpochRecord:
        """SEALED -> FINAL_EVAL_READY. Requires the epoch currently SEALED
        (task spec section 63: freezing happens once, before any access)."""
        if manifest.epoch_label != epoch_label:
            raise HoldoutLifecycleError("manifest.epoch_label does not match the epoch being frozen")
        current = self.get(epoch_label)
        if current.state is not HoldoutState.SEALED:
            raise HoldoutLifecycleError(
                f"epoch {epoch_label!r} is {current.state.value}, not SEALED -- a manifest may "
                "only be frozen once, from SEALED"
            )
        return self._write(
            current.model_copy(update={"state": HoldoutState.FINAL_EVAL_READY, "manifest": manifest})
        )

    def consume(self, epoch_label: str, *, authorization_token: str | None) -> HoldoutEpochRecord:
        """FINAL_EVAL_READY -> CONSUMED. The ONE real-access gate: the
        supplied token must exactly equal the CURRENT value of the
        `ALLOW_FINAL_2025_HOLDOUT_EVALUATION` environment variable, read
        fresh at call time -- never a default, never cached. This function
        itself still touches no market data; it only flips a bookkeeping
        state and records that a (real, elsewhere-executed) evaluation was
        authorized to happen."""
        current = self.get(epoch_label)
        if current.state is not HoldoutState.FINAL_EVAL_READY:
            raise HoldoutLifecycleError(
                f"epoch {epoch_label!r} is {current.state.value}, not FINAL_EVAL_READY -- freeze a "
                "manifest first"
            )
        env_value = os.environ.get(AUTHORIZATION_ENV_VAR)
        if not env_value or authorization_token != env_value:
            raise HoldoutAuthorizationError(
                f"consuming epoch {epoch_label!r} requires authorization_token to equal the current "
                f"{AUTHORIZATION_ENV_VAR} environment value; refusing"
            )
        return self._write(
            current.model_copy(update={
                "state": HoldoutState.CONSUMED, "accessed": True,
                "consumed_at": datetime.now(UTC).isoformat(),
            })
        )

    def roll_forward(self, epoch_label: str, *, new_epoch_label: str) -> tuple[HoldoutEpochRecord, HoldoutEpochRecord]:
        """CONSUMED -> ROLLED_FORWARD, and creates `new_epoch_label` fresh at
        SEALED (task spec section 65: a new immutable prospective cutoff,
        never retroactively applied to already-observed data). Returns
        (old_record, new_record)."""
        current = self.get(epoch_label)
        if current.state is not HoldoutState.CONSUMED:
            raise HoldoutLifecycleError(
                f"epoch {epoch_label!r} is {current.state.value}, not CONSUMED -- an epoch may only "
                "roll forward after it has actually been consumed"
            )
        new_current = self.get(new_epoch_label)
        if new_current.state is not HoldoutState.SEALED or new_current.manifest is not None:
            raise HoldoutLifecycleError(
                f"successor epoch {new_epoch_label!r} already has lifecycle history -- refusing to "
                "reuse it as a fresh cutoff"
            )
        old = self._write(
            current.model_copy(update={
                "state": HoldoutState.ROLLED_FORWARD, "successor_epoch_label": new_epoch_label,
            })
        )
        new = self._write(HoldoutEpochRecord(epoch_label=new_epoch_label))
        return old, new

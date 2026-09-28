"""Structured QA diagnostics for the canonical pipeline."""
from __future__ import annotations

from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field


class DiagnosticKind(str, Enum):
    DUPLICATE_BAR = "duplicate_bar"
    NON_MONOTONIC_TIMESTAMP = "non_monotonic_timestamp"
    INVALID_OHLC = "invalid_ohlc"
    # Retained for compatibility. A normalized futures price may be zero or
    # NEGATIVE (historical CL). Canonicalize now emits IMPLAUSIBLE_NORMALIZED_PRICE
    # for the real failure modes (non-finite / un-normalized fixed-point scale).
    NON_POSITIVE_PRICE = "non_positive_price"
    IMPLAUSIBLE_NORMALIZED_PRICE = "implausible_normalized_price"
    NEGATIVE_VOLUME = "negative_volume"
    UNKNOWN_INSTRUMENT_ID = "unknown_instrument_id"
    RAW_SYMBOL_MISMATCH = "raw_symbol_mismatch"
    # Retained for compatibility. No longer emitted by canonicalize: a continuous
    # vendor "symbol" on a continuous request is expected (Databento does not
    # support continuous -> raw_symbol; raw_symbol comes from the registry).
    CONTINUOUS_SYMBOL_AS_RAW = "continuous_symbol_as_raw"
    UNDEFINED_PRICE_SENTINEL = "undefined_price_sentinel"
    MISSING_MINUTE_GAP = "missing_minute_gap"
    ROOT_UNDETERMINED = "root_undetermined"
    SESSION_CALENDAR_MISSING = "session_calendar_missing"
    TIMESTAMPS_REORDERED = "timestamps_reordered"

    # --- Phase 04: rolls / continuous / back-adjustment / calendar / lifecycle
    UNKNOWN_INSTRUMENT_DURING_ROLL = "unknown_instrument_during_roll"
    ROLL_ROOT_MISMATCH = "roll_root_mismatch"
    DUPLICATE_ROLL = "duplicate_roll"
    BACKWARD_ROLL_TIMESTAMP = "backward_roll_timestamp"
    ROLL_SAME_INSTRUMENT = "roll_same_instrument"
    MISSING_ALIGNED_ROLL_REFERENCE_PRICE = "missing_aligned_roll_reference_price"
    ROLL_BASIS_FALLBACK_USED = "roll_basis_fallback_used"
    OVERLAPPING_ACTIVE_CONTRACTS = "overlapping_active_contracts"
    CONTINUOUS_MISSING_ACTIVE_CONTRACT = "continuous_missing_active_contract"
    ADJUSTED_SERIES_HAS_EXECUTABLE_IDENTITY = "adjusted_series_has_executable_identity"
    ADJUSTED_EXECUTION_ATTEMPT = "adjusted_execution_attempt"
    SESSION_CALENDAR_INCONSISTENCY = "session_calendar_inconsistency"
    UNEXPECTED_BARS_DURING_MAINTENANCE = "unexpected_bars_during_maintenance"
    INVALID_CONTRACT_LIFECYCLE_ORDERING = "invalid_contract_lifecycle_ordering"
    CONTRACT_ROLLED_PAST_TRADABLE_WINDOW = "contract_rolled_past_tradable_window"
    MISSING_FIRST_NOTICE_FOR_DELIVERABLE = "missing_first_notice_for_deliverable"


# Kinds that always mean the dataset cannot be trusted -> the producer raises.
FATAL_KINDS: frozenset[DiagnosticKind] = frozenset(
    {
        DiagnosticKind.DUPLICATE_BAR,
        DiagnosticKind.DUPLICATE_ROLL,
        DiagnosticKind.BACKWARD_ROLL_TIMESTAMP,
        DiagnosticKind.ROLL_SAME_INSTRUMENT,
        DiagnosticKind.ROLL_ROOT_MISMATCH,
        DiagnosticKind.UNKNOWN_INSTRUMENT_DURING_ROLL,
        DiagnosticKind.OVERLAPPING_ACTIVE_CONTRACTS,
        DiagnosticKind.ADJUSTED_SERIES_HAS_EXECUTABLE_IDENTITY,
        DiagnosticKind.ADJUSTED_EXECUTION_ATTEMPT,
        DiagnosticKind.INVALID_CONTRACT_LIFECYCLE_ORDERING,
    }
)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class Diagnostic(BaseModel):
    kind: DiagnosticKind
    severity: Severity
    message: str
    count: int = 1
    instrument_id: int | None = None
    ts_event_ns: int | None = None
    context: dict = Field(default_factory=dict)


class DiagnosticsReport(BaseModel):
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    n_input_rows: int = 0
    n_output_rows: int = 0
    n_rejected_rows: int = 0

    def add(
        self,
        kind: DiagnosticKind,
        message: str,
        *,
        severity: Severity = Severity.ERROR,
        count: int = 1,
        instrument_id: int | None = None,
        ts_event_ns: int | None = None,
        **context,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                kind=kind, severity=severity, message=message, count=count,
                instrument_id=instrument_id, ts_event_ns=ts_event_ns, context=context,
            )
        )

    @property
    def by_kind(self) -> dict[str, int]:
        c: Counter[str] = Counter()
        for d in self.diagnostics:
            c[d.kind.value] += d.count
        return dict(c)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True when nothing fatal was found (row-level errors may still have
        been dropped -- see ``n_rejected_rows``)."""
        return not any(d.kind in FATAL_KINDS for d in self.diagnostics)

    def has_kind(self, kind: DiagnosticKind) -> bool:
        return any(d.kind is kind for d in self.diagnostics)

    def summary(self) -> dict:
        return {
            "n_input_rows": self.n_input_rows,
            "n_output_rows": self.n_output_rows,
            "n_rejected_rows": self.n_rejected_rows,
            "by_kind": self.by_kind,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
        }


class PipelineError(RuntimeError):
    """Raised when the pipeline is configured to fail on the diagnostics it saw."""

    def __init__(self, message: str, report: DiagnosticsReport):
        super().__init__(message)
        self.report = report

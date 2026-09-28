"""Typed compiler / evaluator diagnostics (sections 14, 17)."""
from __future__ import annotations

from pydantic import BaseModel


class StrategyDiagnostic(BaseModel):
    """One typed compiler finding. ``code`` is a stable machine slug."""

    model_config = {"frozen": True}

    code: str
    message: str
    location: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        where = f" [{self.location}]" if self.location else ""
        return f"{self.code}{where}: {self.message}"


# Stable diagnostic codes (section 14).
UNKNOWN_FEATURE = "unknown_feature"
INVALID_FEATURE_PARAMS = "invalid_feature_params"
UNSAFE_FEATURE = "unsafe_feature"
DUPLICATE_FEATURE_ALIAS = "duplicate_feature_alias"
CANONICAL_NAME_COLLISION = "canonical_name_collision"
UNKNOWN_FEATURE_REF = "unknown_feature_ref"
UNKNOWN_OPERATOR = "unknown_operator"
UNKNOWN_ACTION = "unknown_action"
TOO_MANY_RULES = "too_many_rules"
TOO_MANY_FEATURES = "too_many_features"
MALFORMED_CONDITION = "malformed_condition"
INVALID_CONSTANT = "invalid_constant"
INVALID_TARGET_UNITS = "invalid_target_units"
CONDITION_TOO_DEEP = "condition_too_deep"
CONDITION_TOO_MANY_NODES = "condition_too_many_nodes"
FUTURE_TEMPORAL_REFERENCE = "future_temporal_reference"
DUPLICATE_RULE_ID = "duplicate_rule_id"
ILLEGAL_DSL_FIELD = "illegal_dsl_field"
NO_RULES = "no_rules"


class StrategyCompileError(ValueError):
    """Raised by :class:`StrategyCompiler` when a spec is rejected.

    Carries every :class:`StrategyDiagnostic` found in one pass so an Agent gets
    the full list, not just the first failure.
    """

    def __init__(self, diagnostics: list[StrategyDiagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        joined = "; ".join(str(d) for d in self.diagnostics)
        super().__init__(f"strategy spec rejected ({len(self.diagnostics)} issue(s)): {joined}")

    def codes(self) -> list[str]:
        return [d.code for d in self.diagnostics]

    def has(self, code: str) -> bool:
        return code in self.codes()


class StrategyEvaluationError(ValueError):
    """Raised by the reference evaluator for a structural failure (a required
    feature column absent from the frame, a not-signal-safe frame, a root
    mismatch)."""

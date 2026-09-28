"""The closed allow-lists of the Strategy DSL.

Every operator, node kind, action and policy the DSL understands is enumerated
here in source. A :class:`StrategySpec` that names anything outside these enums
fails typed validation before it ever reaches the compiler -- there is no
dynamic operator table, no plugin hook, no string dispatched to code.
"""
from __future__ import annotations

from enum import Enum


class Comparator(str, Enum):
    """Closed predicate vocabulary (section 7).

    Only ordered comparisons. Float equality (``eq`` / ``ne``) is deliberately
    **not** implemented: exact equality of two floating-point feature values is
    almost never a meaningful trading condition and invites silent
    non-determinism across platforms/dtypes. Use a banded ``gt`` / ``lt`` pair
    instead.
    """

    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"

    def apply(self, left: float, right: float) -> bool:
        if self is Comparator.GT:
            return left > right
        if self is Comparator.GTE:
            return left >= right
        if self is Comparator.LT:
            return left < right
        return left <= right


class BooleanOp(str, Enum):
    """Typed boolean composition (section 8)."""

    ALL = "all"   # AND across children (>= 1 child)
    ANY = "any"   # OR across children (>= 1 child)


class NodeType(str, Enum):
    """Discriminator tags for the condition tree."""

    COMPARISON = "comparison"
    BOOLEAN = "boolean"
    NOT = "not"


class OperandType(str, Enum):
    """Discriminator tags for a comparison operand."""

    FEATURE = "feature"       # current point-in-time value of a declared feature
    LAG = "lag"               # value of a declared feature N>=1 bars ago (causal)
    CONST = "const"           # a finite numeric literal


class DefaultAction(str, Enum):
    """Behaviour when no rule matches (section 12). Never implicit."""

    FLAT = "flat"                             # target_units = 0
    KEEP_PREVIOUS_TARGET = "keep_previous_target"  # re-emit the last emitted target


class MissingRulePolicy(str, Enum):
    """What happens when a rule's condition is NOT EVALUABLE (section 10).

    A condition is NOT EVALUABLE when it references a feature value that is
    missing / NaN for the current bar (insufficient history, data gap, ...).
    NaN never silently becomes ``True`` or ``False``.

    ``SKIP_RULE``  -- the non-evaluable rule is skipped; evaluation continues to
                      the next rule ("first matching *evaluable* rule wins").
    ``HOLD``       -- the first non-evaluable rule encountered stops rule
                      evaluation for this bar; the default action is applied and
                      the bar is flagged not-evaluable.
    """

    SKIP_RULE = "skip_rule"
    HOLD = "hold"

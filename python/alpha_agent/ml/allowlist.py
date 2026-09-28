"""The explicit MODEL INPUT ALLOWLIST (prompt 15A.1 s.8).

The label may look forward. The features may not.

A :class:`~alpha_agent.ml.labels.MetaLabelEvent` deliberately carries both kinds
of field side by side: the causal decision-time inputs, and the realised episode
outcome the label is derived from. That is right for an auditable event record
and wrong for a model matrix, so the boundary between the two is made explicit
here rather than left to whoever assembles ``X``.

Allowlist, not blacklist
------------------------
A blacklist protects only against the leaks somebody already thought of. This
module inverts that: :func:`assert_model_inputs_allowed` accepts a column ONLY
if it is one of the frozen feature set's own ordered aliases or a declared
categorical identity column. Anything else -- a new audit field, a debugging
column, a join artifact, a field a future phase adds to the event schema -- fails
closed with :class:`~alpha_agent.ml.errors.ForbiddenModelInput`.

:data:`KNOWN_AUDIT_ONLY_FIELDS` exists on top of that, purely so the error
message can say *why* a recognised offender is forbidden. It is diagnostic
sugar; the allowlist is the actual protection.
"""
from __future__ import annotations

from collections.abc import Iterable

from alpha_agent.ml.errors import ForbiddenModelInput
from alpha_agent.ml.features import MLFeatureSetSpec

#: Audit / label-derived fields, with the reason each is forbidden. Every one of
#: them is knowable only AFTER the decision the model is making.
KNOWN_AUDIT_ONLY_FIELDS: dict[str, str] = {
    "label": "the supervised target itself",
    "episode_net_pnl_usd": "realised episode PnL -- the label's own source",
    "episode_gross_pnl_usd": "realised episode gross PnL",
    "episode_costs_usd": "realised episode costs",
    "net_pnl_usd": "realised PnL",
    "gross_pnl_usd": "realised gross PnL",
    "costs_usd": "realised costs",
    "episode_duration_ns": "episode duration -- known only once the episode ends",
    "episode_duration_days": "episode duration -- known only once the episode ends",
    "exit_timestamp": "the exit instant -- in the future at decision time",
    "exit_decision_ts_ns": "the exit decision -- in the future at decision time",
    "exit_row_index": "the exit row -- in the future at decision time",
    "exit_price": "an execution price the decision never saw",
    "label_end_timestamp": "the label horizon -- a future instant",
    "information_horizon_ts_ns": "the label's information horizon -- a future instant",
    "n_attributed_trades": "how many fills the episode eventually produced",
    "n_attributed_fills": "future fills",
    "n_roll_closes": "future roll count",
    "n_rolls": "future roll count",
    "mae_usd": "maximum adverse excursion -- realised over the episode's future",
    "mfe_usd": "maximum favourable excursion -- realised over the episode's future",
    "max_adverse_excursion": "realised over the episode's future",
    "max_favourable_excursion": "realised over the episode's future",
    "realised_regime": "a regime label resolved with future information",
    "realized_regime": "a regime label resolved with future information",
    "future_regime": "a regime label resolved with future information",
    "contracts_traded": "which contracts the episode eventually traded",
    "close_reason": "how the episode eventually closed",
}


def model_input_allowlist(feature_set: MLFeatureSetSpec) -> tuple[str, ...]:
    """The ONLY columns permitted in the model matrix, in matrix column order.

    Ordered numeric features first -- their order is semantic -- then the
    declared categorical identity columns.
    """
    return (*feature_set.ordered_aliases, *feature_set.categoricals)


def assert_model_inputs_allowed(
    columns: Iterable[str], feature_set: MLFeatureSetSpec, *, what: str = "model matrix"
) -> None:
    """Refuse any model-matrix column outside the frozen allowlist.

    Order is not checked here (that is the feature set's own invariant); presence
    is. A duplicate column is refused too: a repeated column silently reweights a
    feature.
    """
    allowed = set(model_input_allowlist(feature_set))
    seen: set[str] = set()
    for col in columns:
        if col in seen:
            raise ForbiddenModelInput(
                f"{what}: column {col!r} appears more than once; a duplicated column silently "
                "reweights a feature"
            )
        seen.add(col)
        if col in allowed:
            continue
        reason = KNOWN_AUDIT_ONLY_FIELDS.get(col)
        detail = (
            f" -- {reason}: it is AUDIT/LABEL-only and is knowable only after the decision"
            if reason
            else " -- it is not one of the frozen feature set's declared inputs"
        )
        raise ForbiddenModelInput(
            f"{what}: column {col!r} is not on the model input ALLOWLIST{detail}. "
            f"Permitted columns are {sorted(allowed)}. The label may look forward; the "
            "features may not."
        )


def assert_no_audit_field_is_allowlisted(feature_set: MLFeatureSetSpec) -> None:
    """Defence in depth: the allowlist itself must contain no audit-only field.

    Catches the one way an allowlist can fail -- somebody adding a forward-looking
    column to the feature set and thereby legitimising it.
    """
    offenders = sorted(set(model_input_allowlist(feature_set)) & set(KNOWN_AUDIT_ONLY_FIELDS))
    if offenders:
        raise ForbiddenModelInput(
            f"the frozen feature set declares AUDIT/LABEL-only field(s) as model inputs: "
            f"{offenders}. Adding a forward-looking column to the feature set does not make it "
            "causal; it makes the allowlist wrong."
        )

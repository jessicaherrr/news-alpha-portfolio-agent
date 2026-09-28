"""Alpha Discovery campaign, Part H -- multi-period / multi-market
robustness (task spec sections 45-47).

Multi-PERIOD robustness (section 45 -- walk-forward folds, parameter
neighbourhoods, regime evidence) needs NO new code here: it is already the
frozen `alpha_agent.validation` pipeline's job, unchanged by this campaign --
`ValidationEngine.run_research()` already builds walk-forward folds
(`WalkForwardResult`/`fold_consistency`), `ParameterStabilityResult` already
evaluates a predeclared parameter neighbourhood, and `RegimeStabilityResult`
already buckets performance by regime. Parameter DISCIPLINE (section 47 --
"prefer broad plateaus, penalize isolated optima") is likewise already
enforced by the frozen `ReliabilityPolicy.param_stability_reject_isolated_spike`
gate (default `True`) inside `evaluate_policy` -- adding a second, competing
notion of "broad enough" here would be exactly the kind of silent
`ReliabilityPolicy` duplication CLAUDE.md forbids.

What genuinely does not exist yet is multi-MARKET replication (section 46):
a deterministic, mechanical suggestion of which OTHER markets an already-
frozen mechanism could reasonably be re-tested on, scoped to economically
related instruments only ("do not force a crude-specific mechanism onto NQ
just to increase sample count"). This module is that suggestion layer ONLY
-- it never re-compiles, re-executes, or auto-freezes a replicated
hypothesis; running the suggested markets through discovery again (Parts
D-G) is a deliberate, separate research decision.
"""
from __future__ import annotations

from pydantic import BaseModel

from alpha_agent.agents.orchestrator import FamilyManifest

#: Economically related root-symbol groups, fixed and documented -- NOT
#: derived from any measured correlation (that would make this a scientific
#: claim; it is deliberately just an economic-category grouping). Today's
#: 5-root universe gives one genuine multi-member group (index futures);
#: the others are singletons until more roots are onboarded, and a
#: singleton's own group correctly proposes no replication target at all.
RELATED_MARKET_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ES", "NQ"),   # equity index futures
    ("CL",),        # energy
    ("GC",),        # metals
    ("ZN",),        # rates
)


def _group_for(root: str) -> tuple[str, ...]:
    for group in RELATED_MARKET_GROUPS:
        if root in group:
            return group
    return (root,)


def propose_replication_targets(root: str, *, universe: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Other roots in `root`'s economic group, optionally filtered to an
    approved `universe`. Deterministic, order-preserving (group order, then
    universe order), never includes `root` itself."""
    group = tuple(m for m in _group_for(root) if m != root)
    if not universe:
        return group
    universe_set = set(universe)
    return tuple(m for m in group if m in universe_set)


class ReplicationCandidate(BaseModel):
    """One deterministic replication suggestion for an already-frozen family
    member. This is NOT a new hypothesis, NOT a new experiment_identity, and
    NOT itself executed -- it is a pointer for a future discovery pass."""

    model_config = {"frozen": True, "extra": "forbid"}

    source_experiment_identity: str
    strategy_family: str
    source_root: str
    suggested_roots: tuple[str, ...]
    rationale: str


def build_replication_suggestions(
    manifest: FamilyManifest, *, universe: tuple[str, ...] = ()
) -> tuple[ReplicationCandidate, ...]:
    """One suggestion per frozen member that has at least one economically
    related, approved-universe target it does not already cover."""
    covered_roots = {m.root_symbol for m in manifest.members}
    out: list[ReplicationCandidate] = []
    for member in manifest.members:
        targets = tuple(
            r for r in propose_replication_targets(member.root_symbol, universe=universe)
            if r not in covered_roots
        )
        if not targets:
            continue
        out.append(
            ReplicationCandidate(
                source_experiment_identity=member.experiment_identity,
                strategy_family=member.strategy_family,
                source_root=member.root_symbol,
                suggested_roots=targets,
                rationale=(
                    f"'{member.strategy_family}' was frozen on {member.root_symbol}; "
                    f"{', '.join(targets)} share the same economic category "
                    f"(RELATED_MARKET_GROUPS) and are not already covered by this family -- "
                    "cross-market replication is mechanism evidence, not brute-force mining, "
                    "and requires its own discovery/fast-screen/freeze/validation pass, never "
                    "an automatic re-execution."
                ),
            )
        )
    return tuple(out)


__all__ = [
    "RELATED_MARKET_GROUPS",
    "ReplicationCandidate",
    "build_replication_suggestions",
    "propose_replication_targets",
]

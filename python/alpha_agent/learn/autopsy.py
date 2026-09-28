"""Phase 4 -- Failure Autopsy: what looked promising, what failed, why the
gate matters, other experiments sharing the same committed reason code
(across the WHOLE registry, never presented as proven Factor/Strategy/
Mechanism similarity), engineering lessons (`FailureMemory`, with their own
real `scope`/root/family metadata always exposed rather than implied),
and what a next experiment would need to change. Built ONLY from
already-committed registry evidence (`ResultRecord`, `FailureRecord`) --
this module assigns no verdict of its own and applies no threshold; the
verdict and every gate state it narrates are read verbatim from evidence
the caller already computed with `alpha_agent.ui.services.gate_table` /
`gate_evidence_text` (the same, already-tested `gate_state` resolution the
Validation page renders).

Semantic hardening patch (Phase 4 review): explicit `PASS`/`REJECT`/
`INCONCLUSIVE`/`NOT_ADJUDICATED` handling throughout -- `NOT_ADJUDICATED`
(the registry's own value for a predeclared parameter neighbour, or a real
typed refusal that the frozen policy never headline-adjudicates, e.g. a
Phase 15B ML trial refused for insufficient training events) is NEVER
described as a failure; a `verdict is None` trial (genuinely never
executed) is described differently again. Descriptive backtest facts
(`descriptive_evidence`) are now surfaced separately from validated gate
passes (`what_looked_promising`) and are always labeled as such by the
caller -- a positive raw number here is explicitly NOT validated alpha.

Any free-text `narrative` field is filled ONLY by
`alpha_agent.learn.narration_agent` and is always additive commentary next to
these typed facts -- never a replacement for them (CLAUDE.md: "Claude
narrates but never assigns verdict"). `narration_agent` receives only
`root_symbol`/`strategy_family`/`verdict`/`reason_codes`/`first_failed_gate`/
`what_looked_promising`/`what_failed` -- deliberately never `engineering_notes`,
so Claude can never present an unrelated (often SYSTEM-scope, not
experiment-specific) engineering lesson as a causal explanation for this
one experiment's result.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from alpha_agent.learn.concepts import GATE_FAIL_CODE_TO_CONCEPT
from alpha_agent.registry.enums import FailureScope, TrialRole
from alpha_agent.registry.failure_memory import FailureMemory
from alpha_agent.registry.sqlite_registry import ExperimentRegistry

__all__ = [
    "EngineeringNote",
    "FailureAutopsy",
    "GateOutcome",
    "SimilarFailure",
    "build_failure_autopsy",
]

#: Fixed, deterministic caveat (not derived from any one experiment's data --
#: the same text every time) satisfying "Factor != Strategy != Experiment":
#: one rejected `StrategySpec` implementation must never be read as evidence
#: against the broader economic idea. See CLAUDE.md's own `alpha_memory`
#: Factor-identity discipline -- a Factor is mechanism-scoped and can span
#: several Strategy variants; this trial speaks to exactly one of them.
SCOPE_CAVEAT = (
    "This verdict applies to this experiment / StrategySpec implementation only. It does not, by "
    "itself, invalidate the broader Factor or Economic Mechanism -- a different implementation of the "
    "same idea, or the same idea on a different market, may behave differently."
)

#: fail_code (GATE_DEFINITIONS order, as `gate_table` rows carry it via their
#: label) -> a deterministic statement of what a next experiment would need
#: to demonstrate. Keyed by the exact gate LABEL (`services.GATE_DEFINITIONS`
#: strings) since that is what a `gate_table` row exposes, not the raw code.
_WHAT_WOULD_NEED_TO_CHANGE: dict[str, str] = {
    "Null Hypothesis (Bootstrap)": (
        "A materially larger or more consistent edge across the sample -- the observed return pattern "
        "was not statistically distinguishable from reshuffled noise of the same block structure."
    ),
    "Multiple Testing (BH-FDR)": (
        "A q-value low enough to survive correction for the WHOLE predeclared family -- either stronger "
        "evidence for this specific hypothesis, or a smaller, better-justified family of hypotheses "
        "tested alongside it (never a post-hoc narrower family chosen after seeing results)."
    ),
    "Deflated Sharpe Ratio": (
        "A higher risk-adjusted edge relative to how many parameter variants were effectively searched -- "
        "trying fewer variants (reducing the effective trial count) or finding a genuinely larger effect, "
        "not a friendlier deflation assumption."
    ),
    "Positive OOS Net PnL": (
        "A larger gross edge than round-trip trading costs at the SAME cost assumptions -- lower "
        "turnover or a bigger average price move captured per trade."
    ),
    "Walk-Forward Consistency": (
        "The edge would need to show up consistently across MOST walk-forward folds, not a subset -- a "
        "mechanism that only works in specific historical periods is not yet a robust, repeatable edge."
    ),
    "Cost Stress": (
        "An edge that survives higher commission/slippage/spread assumptions -- usually means lower "
        "turnover or a materially larger average edge per trade, not a rosier cost model."
    ),
    "Parameter Stability": (
        "Nearby parameter settings would need to perform similarly well -- an isolated spike is more "
        "consistent with fitting to noise than with a smooth, genuine underlying effect."
    ),
    "Regime Robustness": (
        "PnL would need to be spread across volatility regimes, not concentrated in one -- evidence the "
        "mechanism is not just one unusual historical period."
    ),
    "Cross-Market Evidence": (
        "The same mechanism would need to show up on other related roots tested in the same family, not "
        "just this one."
    ),
}

_DEFAULT_WHAT_WOULD_NEED_TO_CHANGE = (
    "No single gate's explicit FAIL code is committed for this trial -- see the reason codes below for "
    "the exact committed evidence; a next attempt would need to resolve whatever specifically blocked "
    "adjudication (e.g. more trades/days/folds) before any gate-level change can even be evaluated."
)


class GateOutcome(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    state: str
    concept_id: str | None = None
    evidence: str = ""


class SimilarFailure(BaseModel):
    """One OTHER canonical experiment sharing >=1 committed reason code with
    this one. This is proof of shared VALIDATION REASON CODES ONLY -- it is
    NOT Factor, Strategy, or Mechanism similarity (no structured similarity
    evidence, e.g. `alpha_agent.registry.similarity`, backs any such claim
    here); the UI must render this under a label that says exactly that
    (`services.py`'s "Other Experiments with Shared Failure Gates", never
    "Similar Prior Failures" -- that name overclaimed hypothesis similarity
    from what is really a shared statistical outcome)."""

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_id: str
    root_symbol: str
    strategy_family: str
    verdict: str | None = None
    shared_reason_codes: tuple[str, ...] = Field(default_factory=tuple)


class EngineeringNote(BaseModel):
    """One `FailureRecord` from `FailureMemory.related_engineering_failures`,
    with its OWN real scope metadata always exposed rather than implied by
    the section it happens to render in. `scope` is the record's real
    `FailureScope` (`EXPERIMENT` or `SYSTEM`, per `alpha_agent.registry.enums`);
    `applies_to_this_experiment` is `True` only when `scope == EXPERIMENT`
    AND the record's own `experiment_identity` matches the trial this
    autopsy is FOR -- never inferred from free text, and never assumed just
    because the note was returned by a market/family-scoped query (a
    SYSTEM-scope record, e.g. a general contract-economics methodology fix
    whose summary happens to describe a different, concrete root, is
    returned for every market it is not root/family-tagged against)."""

    model_config = {"frozen": True, "extra": "forbid"}

    failure_code: str
    summary: str
    scope: str
    applies_to_this_experiment: bool = False
    root_symbol: str | None = None
    strategy_family: str | None = None


class FailureAutopsy(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "failure-autopsy/2"
    experiment_id: str
    experiment_identity: str
    root_symbol: str
    strategy_family: str
    trial_role: str
    verdict: str | None = None
    reason_codes: tuple[str, ...] = Field(default_factory=tuple)

    gates: tuple[GateOutcome, ...] = Field(default_factory=tuple)
    first_failed_gate: str | None = None
    first_failed_gate_concept_id: str | None = None

    #: raw, UNGATED facts read directly off the committed `ResultRecord`
    #: (e.g. a positive gross/net PnL, a positive raw Sharpe, trade count) --
    #: shown regardless of which gates passed or failed. These are
    #: DESCRIPTIVE EVIDENCE ONLY, never validated alpha; the caller must
    #: always render them under that explicit label, never merged silently
    #: into `what_looked_promising` (a later validation failure does not
    #: erase the fact that the raw backtest may have looked interesting).
    descriptive_evidence: tuple[str, ...] = Field(default_factory=tuple)

    #: gates EXPLICITLY satisfied (state == "PASS") before the first FAIL, in
    #: `GATE_DEFINITIONS` order -- real, validated, statistically-adjudicated
    #: evidence, distinct in kind from `descriptive_evidence` above.
    what_looked_promising: tuple[str, ...] = Field(default_factory=tuple)
    what_failed: str = ""
    what_would_need_to_change: str = ""

    similar_failures: tuple[SimilarFailure, ...] = Field(default_factory=tuple)

    #: `FailureMemory.related_engineering_failures` for this market/family,
    #: each with its own real scope metadata (see `EngineeringNote`) -- never
    #: presented as proven causal to this specific experiment unless that
    #: scope metadata itself says so.
    engineering_notes: tuple[EngineeringNote, ...] = Field(default_factory=tuple)

    #: fixed, deterministic Factor/Strategy/Experiment caveat -- see
    #: `SCOPE_CAVEAT`'s own docstring. Always the same text; not derived from
    #: this experiment's own data, so it can never be wrong about "this"
    #: trial, only ever a reminder of what the verdict does and doesn't cover.
    scope_note: str = SCOPE_CAVEAT

    #: additive, optional Claude commentary -- see `narration_agent`. Never
    #: consulted by this module and never influences any field above it.
    narrative: str | None = None


def _promising_statements(gates: Sequence[GateOutcome]) -> tuple[str, ...]:
    """Every gate explicitly PASSed before the first FAIL -- in `gate_table`
    order (`services.GATE_DEFINITIONS`'s own declared order), so this reads
    as "here is what held up before the thing that didn't"."""
    out: list[str] = []
    for g in gates:
        if g.state == "PASS":
            out.append(f"{g.label} was explicitly satisfied" + (f" ({g.evidence})" if g.evidence else "") + ".")
        elif g.state == "FAIL":
            break
    return tuple(out)


def _descriptive_evidence(result: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Raw, ungated backtest facts read directly off the committed
    `ResultRecord` -- shown regardless of gate state. A number here is a
    plain fact, never a validated claim: the caller must render these under
    an explicit "descriptive evidence, not validated alpha" label. Only
    surfaces a PnL/Sharpe fact when it is actually positive (that is what
    makes it "interesting" to contrast against a failed gate); trade count
    and fold-positivity are neutral descriptive context, shown whenever
    committed, positive or not."""
    if not result:
        return ()
    out: list[str] = []
    gross = result.get("gross_pnl_usd")
    if gross is not None and gross > 0:
        out.append(f"Gross PnL before costs was positive: ${gross:,.0f}.")
    net = result.get("net_pnl_usd")
    if net is not None and net > 0:
        out.append(f"Net PnL after costs was positive: ${net:,.0f}.")
    sharpe = result.get("daily_sharpe")
    if sharpe is not None and sharpe > 0:
        out.append(f"Raw daily Sharpe ratio was positive: {sharpe:.4f}.")
    n_trades = result.get("n_trades")
    if n_trades is not None:
        out.append(f"{n_trades} trades were executed in the out-of-sample window.")
    fold_consistency = result.get("fold_consistency")
    if fold_consistency is not None and fold_consistency > 0:
        out.append(f"{fold_consistency:.0%} of walk-forward folds showed a positive result.")
    return tuple(out)


def build_failure_autopsy(
    reg: ExperimentRegistry,
    experiment_id: str,
    *,
    gate_table: Sequence[Mapping[str, str]],
    gate_evidence: Mapping[str, str],
) -> FailureAutopsy:
    """``gate_table``/``gate_evidence`` are exactly what
    ``alpha_agent.ui.services.gate_table`` / ``gate_evidence_text`` already
    compute for this same experiment -- this function reinterprets that
    already-adjudicated evidence into a narrative, it never recomputes a gate
    state itself."""
    view = reg.get(experiment_id)
    result = view.result
    verdict = view.verdict.value if view.verdict else None
    reason_codes = tuple(result.reason_codes) if result else ()

    gates = tuple(
        GateOutcome(
            label=row["label"],
            state=row["state"],
            concept_id=GATE_FAIL_CODE_TO_CONCEPT.get(_label_to_fail_code(row["label"])),
            evidence=gate_evidence.get(row["label"], ""),
        )
        for row in gate_table
    )

    first_failed = next((g for g in gates if g.state == "FAIL"), None)
    refused_before_gate = any(g.state == "REFUSED_BEFORE_GATE" for g in gates)

    # `NOT_ADJUDICATED` is checked FIRST and takes absolute priority over any
    # individual gate's FAIL state: it is the registry's own value for a
    # predeclared parameter neighbour (never given a headline verdict by
    # frozen-policy design) or a real typed refusal the frozen policy
    # deliberately never adjudicates (e.g. a Phase 15B ML trial refused for
    # insufficient training events, still entered into BH at the
    # conservative p=1 per CLAUDE.md) -- either way, a real, non-executed,
    # non-failed state that must never read as "failed" (semantic hardening
    # patch; see `test_not_adjudicated_is_never_described_as_a_failure`).
    if verdict == "NOT_ADJUDICATED":
        what_failed = (
            "This trial has not been headline-adjudicated. NOT_ADJUDICATED is the frozen policy's own "
            "value for a predeclared parameter neighbour, or a real typed refusal the policy deliberately "
            "never gives a headline verdict to -- a real, deliberately-not-adjudicated outcome, never a "
            "REJECT or INCONCLUSIVE scientific result."
        )
        change = (
            "Not applicable in the way it is for a REJECT/INCONCLUSIVE trial: a predeclared neighbour "
            "never becomes headline-adjudicated by design; a typed refusal (see the committed reason "
            "codes below) would need the specific blocking condition resolved before any gate could run."
        )
    elif first_failed is not None:
        what_failed = f"{first_failed.label} failed" + (f" ({first_failed.evidence})" if first_failed.evidence else "") + "."
        change = _WHAT_WOULD_NEED_TO_CHANGE.get(first_failed.label, _DEFAULT_WHAT_WOULD_NEED_TO_CHANGE)
    elif refused_before_gate:
        what_failed = (
            "Evaluation was refused before any statistical gate ran -- an insufficient sample "
            "(trades, out-of-sample days, or walk-forward folds)."
        )
        change = "A materially larger out-of-sample sample (more trades/days/folds) before any gate can even be evaluated."
    elif verdict == "INCONCLUSIVE":
        what_failed = "Committed evidence was insufficient to reach a PASS or REJECT verdict."
        change = _DEFAULT_WHAT_WOULD_NEED_TO_CHANGE
    elif verdict == "PASS":
        what_failed = "No gate failed -- this trial's headline verdict is PASS."
        change = "Nothing further is required for this trial; see Validation for the full gate detail."
    elif verdict is None:
        what_failed = "This experiment has no committed result yet -- it has not been executed."
        change = "This experiment needs to be run to a valid, adjudicated result before any gate-level evidence exists."
    else:  # pragma: no cover -- RegistryVerdict is a closed enum; defensive only
        what_failed = f"Unrecognized committed verdict {verdict!r}."
        change = _DEFAULT_WHAT_WOULD_NEED_TO_CHANGE

    promising = _promising_statements(gates)
    descriptive = _descriptive_evidence(result.model_dump(mode="json") if result else None)

    # "Similar failures": OTHER canonical, authoritative experiments ANYWHERE
    # in the registry (not just this family/root) that share at least one
    # reason code -- deliberately broader than `FailureMemory.lookup`'s own
    # family+root-scoped `prior_experiments` (which answers "has THIS EXACT
    # proposal been tried before", not "where else does this failure mode
    # show up"), ranked by the most shared codes first.
    similar: tuple[SimilarFailure, ...] = ()
    if reason_codes:
        target_codes = set(reason_codes)
        candidates = reg.experiments(trial_role=TrialRole.CANONICAL, authoritative_only=True)
        matches: list[SimilarFailure] = []
        for cand in candidates:
            if cand.experiment_id == experiment_id or cand.result is None:
                continue
            shared = tuple(sorted(target_codes.intersection(cand.result.reason_codes)))
            if not shared:
                continue
            matches.append(
                SimilarFailure(
                    experiment_id=cand.experiment_id,
                    root_symbol=cand.experiment.root_symbol,
                    strategy_family=cand.experiment.strategy_family,
                    verdict=cand.verdict.value if cand.verdict else None,
                    shared_reason_codes=shared,
                )
            )
        matches.sort(key=lambda m: len(m.shared_reason_codes), reverse=True)
        similar = tuple(matches[:5])

    # Each note keeps its OWN real scope metadata (`FailureRecord.scope` /
    # `.experiment_identity` / `.root_symbol` / `.strategy_family`) rather
    # than being flattened into a bare string that the UI would otherwise
    # have to render under one blanket "for this market and family" claim --
    # a SYSTEM-scope record (the common case: a general methodology lesson)
    # is real, relevant historical context, but it is not proof the note
    # caused or affected THIS experiment specifically; only an EXPERIMENT-
    # scope record whose own `experiment_identity` matches this trial's is.
    engineering_notes = tuple(
        EngineeringNote(
            failure_code=f.failure_code,
            summary=f.summary,
            scope=f.scope.value,
            applies_to_this_experiment=(
                f.scope is FailureScope.EXPERIMENT and f.experiment_identity == view.experiment_identity
            ),
            root_symbol=f.root_symbol,
            strategy_family=f.strategy_family,
        )
        for f in FailureMemory(reg).related_engineering_failures(
            root_symbol=view.experiment.root_symbol, strategy_family=view.experiment.strategy_family,
        )
    )

    return FailureAutopsy(
        experiment_id=view.experiment_id,
        experiment_identity=view.experiment_identity,
        root_symbol=view.experiment.root_symbol,
        strategy_family=view.experiment.strategy_family,
        trial_role=view.experiment.trial_role.value,
        verdict=verdict,
        reason_codes=reason_codes,
        gates=gates,
        first_failed_gate=first_failed.label if first_failed else None,
        first_failed_gate_concept_id=first_failed.concept_id if first_failed else None,
        descriptive_evidence=descriptive,
        what_looked_promising=promising,
        what_failed=what_failed,
        what_would_need_to_change=change,
        similar_failures=similar,
        engineering_notes=engineering_notes,
    )


#: `gate_table` rows carry only the display label (`services.GATE_DEFINITIONS`
#: strings); this reverse-maps a label back to its fail_code so
#: `GATE_FAIL_CODE_TO_CONCEPT` (keyed by fail_code) can resolve a Concept.
#: Duplicated, not imported, from `services.GATE_DEFINITIONS` to keep this
#: backend module free of a `ui` import -- proven byte-identical in
#: `test_phase4_failure_autopsy.py`.
_LABEL_TO_FAIL_CODE: dict[str, str] = {
    "Null Hypothesis (Bootstrap)": "null_hypothesis_not_rejected",
    "Multiple Testing (BH-FDR)": "fdr_qvalue_above_threshold",
    "Deflated Sharpe Ratio": "deflated_sharpe_below_threshold",
    "Positive OOS Net PnL": "negative_oos_net_pnl",
    "Walk-Forward Consistency": "fold_consistency_below_threshold",
    "Cost Stress": "cost_stress_degradation_exceeds_limit",
    "Parameter Stability": "parameter_neighbourhood_unstable",
    "Regime Robustness": "performance_concentrated_in_one_regime",
    "Cross-Market Evidence": "performance_concentrated_in_one_root",
}


def _label_to_fail_code(label: str) -> str | None:
    return _LABEL_TO_FAIL_CODE.get(label)

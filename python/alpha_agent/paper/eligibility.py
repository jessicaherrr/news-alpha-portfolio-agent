"""Phase 21 -- the gate between the experiment registry and paper trading.

CLAUDE.md: "after a strategy passes frozen validation, route it only to a
simulated paper portfolio." This module is the ONLY place that decision is
made, and it makes it from committed registry evidence alone:

1. the experiment must exist, be the AUTHORITATIVE (non-superseded) record for
   its identity, and carry a VALID authoritative attempt;
2. its committed headline verdict must be ``PASS`` with the frozen policy's own
   ``all_required_gates_satisfied`` reason code present -- the exact positive
   proof :func:`alpha_agent.ui.services.gate_state` requires, never inferred
   from the absence of a failure code;
3. its ``StrategySpec`` must be deterministically reconstructable from the
   committed ``strategy_family`` + parameter dict (Phase 13.5C's own
   ``candidates_phase_13_5c.spec_for_params`` rebuild rule, CLAUDE.md's
   "IDENTITY USES THE ACTUAL PROPOSED StrategySpec's FEATURE SEMANTICS"
   section), and the rebuilt strategy's fingerprint must equal the one
   committed on the experiment record. A strategy this cannot be done for
   (e.g. an arbitrary Phase 17-compiled blueprint with no committed full
   ``StrategySpec`` artifact) is out of this MVP's scope and is refused, never
   silently approximated.

Nothing here computes PnL, a fill, or a risk decision -- this module only
decides whether a strategy is ALLOWED to be paper-traded, using the same kind
of read-only registry evidence CLAUDE.md's "Rebuild + inspect" workflow
already sanctions.
"""
from __future__ import annotations

from pydantic import BaseModel

from alpha_agent.paper.errors import PaperTradingEligibilityError
from alpha_agent.registry.enums import Authority, RegistryVerdict
from alpha_agent.registry.sqlite_registry import (
    ExperimentRegistry,
    ExperimentView,
    UnknownExperiment,
)
from alpha_agent.strategy import StrategySpec, strategy_fingerprint
from alpha_agent.strategy.candidates_phase_13_5c import BASELINE_FAMILIES, spec_for_params

#: The frozen policy's own positive-proof reason code for a headline PASS
#: (alpha_agent.ui.services._ALL_GATES_SATISFIED -- duplicated, not imported,
#: to keep this module import-light and because the UI module is presentation,
#: not a registry-semantics authority).
_ALL_GATES_SATISFIED = "all_required_gates_satisfied"

#: The frozen policy's own reason codes for "a required evidence category was
#: never evaluated" (alpha_agent.validation.enums.ReasonCode -- duplicated as
#: plain strings for the same reason `_ALL_GATES_SATISFIED` is: this module
#: stays import-light and is not a registry-semantics authority). Presence of
#: ANY of these alongside a committed PASS means the PASS was not actually
#: built on complete required evidence -- defense in depth, never relied on
#: as the only guard (see `evaluate_policy`, which the validation-safety fix
#: makes structurally incapable of emitting PASS alongside these codes).
_INCOMPLETE_EVIDENCE_CODES = frozenset(
    {"parameter_stability_not_evaluated", "ablation_not_evaluated"}
)

#: Strategy families whose StrategySpec this MVP can deterministically rebuild
#: from a committed parameter dict + a fingerprint check (Phase 13.5C's daily
#: baselines only). ``silver_bullet`` (native 1m, session-labelled) needs its
#: own schedule-window builder and is out of scope until a later phase.
REBUILDABLE_FAMILIES: frozenset[str] = frozenset(BASELINE_FAMILIES)


class PaperEligibility(BaseModel):
    """Everything :mod:`alpha_agent.paper.engine` needs to start a paper run,
    derived entirely from committed registry evidence."""

    model_config = {"frozen": True, "extra": "forbid", "arbitrary_types_allowed": True}

    experiment_id: str
    experiment_identity: str
    strategy_family: str
    strategy_id: str
    root_symbol: str
    params: dict
    strategy_fingerprint: str
    strategy_spec: StrategySpec
    backtest_daily_sharpe: float | None
    backtest_annualized_sharpe: float | None
    backtest_net_pnl_usd: float | None
    backtest_n_trades: int | None
    validation_report_reason_codes: tuple[str, ...]


def assert_paper_trading_eligible(
    registry: ExperimentRegistry, experiment_key: str
) -> PaperEligibility:
    """Return the reconstructed, fingerprint-verified eligibility bundle for
    ``experiment_key``, or raise :class:`PaperTradingEligibilityError`.

    ``experiment_key`` is anything :meth:`ExperimentRegistry.get` accepts (an
    ``experiment_id`` friendly slug or a raw ``experiment_identity``).
    """
    try:
        view: ExperimentView = registry.get(experiment_key)
    except UnknownExperiment as exc:
        raise PaperTradingEligibilityError(
            f"no experiment {experiment_key!r} in the registry"
        ) from exc

    if view.authority is not Authority.AUTHORITATIVE:
        raise PaperTradingEligibilityError(
            f"experiment {view.experiment_id} has been superseded "
            f"(by {list(view.superseded_by)}); paper trading always uses the "
            "current authoritative record for an identity, never a superseded one"
        )

    if view.result is None or not view.has_valid_authoritative_result:
        raise PaperTradingEligibilityError(
            f"experiment {view.experiment_id} has no VALID authoritative execution "
            "attempt yet; paper trading needs a completed, adjudicated result"
        )

    reason_codes = tuple(view.result.reason_codes)
    if view.verdict is not RegistryVerdict.PASS or _ALL_GATES_SATISFIED not in reason_codes:
        raise PaperTradingEligibilityError(
            f"experiment {view.experiment_id} has not passed frozen validation "
            f"(headline_verdict={view.verdict.value if view.verdict else None!r}, "
            f"reason_codes={reason_codes!r}); CLAUDE.md: route to paper trading "
            "only after a strategy passes frozen validation"
        )
    # Defense in depth (validation-safety fix): the upstream `evaluate_policy`
    # fix makes PASS + ALL_GATES_SATISFIED structurally impossible while a
    # required evidence category was never evaluated, but this module never
    # relies on that alone -- it independently re-checks the same committed
    # `reason_codes` for any "required evidence not evaluated" code. No new
    # registry metadata is invented here; this reads only what is already
    # committed. Historical PASS rows are never rewritten -- there are none.
    incomplete_evidence = _INCOMPLETE_EVIDENCE_CODES.intersection(reason_codes)
    if incomplete_evidence:
        raise PaperTradingEligibilityError(
            f"experiment {view.experiment_id} is committed as headline PASS but its "
            f"own reason_codes also carry {sorted(incomplete_evidence)!r}, meaning "
            "required scientific evidence was not evaluated; refusing paper "
            "eligibility rather than trusting a PASS built on incomplete evidence"
        )

    exp = view.experiment
    family = exp.strategy_family
    if family not in REBUILDABLE_FAMILIES:
        raise PaperTradingEligibilityError(
            f"experiment {view.experiment_id}'s strategy_family {family!r} is not "
            f"one of the Phase 21 MVP's deterministically-reconstructable families "
            f"{sorted(REBUILDABLE_FAMILIES)}; its full StrategySpec is not committed "
            "as a standalone artifact this phase can safely rebuild and "
            "fingerprint-verify, so it is refused rather than approximated"
        )

    params = dict(exp.strategy_spec_json.get("params", {}))
    if "root_symbol" not in params:
        params = {**params, "root_symbol": exp.root_symbol}
    try:
        spec = spec_for_params(family, params)
    except Exception as exc:
        raise PaperTradingEligibilityError(
            f"experiment {view.experiment_id}'s committed params {params!r} could not "
            f"be rebuilt into a StrategySpec for family {family!r}: {exc}"
        ) from exc

    rebuilt_fp = strategy_fingerprint(spec)
    if rebuilt_fp != exp.strategy_fingerprint:
        raise PaperTradingEligibilityError(
            f"experiment {view.experiment_id}: the StrategySpec rebuilt from its "
            f"committed params fingerprints to {rebuilt_fp!r}, which does not match "
            f"the committed strategy_fingerprint {exp.strategy_fingerprint!r}; refusing "
            "rather than paper-trading an unverified strategy"
        )

    result = view.result
    return PaperEligibility(
        experiment_id=view.experiment_id,
        experiment_identity=view.experiment_identity,
        strategy_family=family,
        strategy_id=exp.strategy_id,
        root_symbol=exp.root_symbol,
        params=params,
        strategy_fingerprint=rebuilt_fp,
        strategy_spec=spec,
        backtest_daily_sharpe=result.daily_sharpe,
        backtest_annualized_sharpe=result.annualized_sharpe,
        backtest_net_pnl_usd=result.net_pnl_usd,
        backtest_n_trades=result.n_trades,
        validation_report_reason_codes=reason_codes,
    )

"""Failure memory: the structured answer a future Research Agent gets *before*
it proposes or runs a hypothesis (section 11).

Everything here is typed. A future agent consumes
:class:`FailureMemoryResponse` directly -- it never parses Markdown, never
re-reads a Phase 13 report, and never asks an LLM what a prior result meant.

This layer answers; it does not decide. It returns prior evidence, related
work and lessons. Whether a proposal should proceed is future agent *policy*,
and the deterministic ``ReliabilityPolicy`` remains the only thing that can
gate a verdict.
"""
from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from alpha_agent.registry.enums import (
    AssetDomain,
    Authority,
    FailureClass,
    RegistryVerdict,
    TrialRole,
)
from alpha_agent.registry.models import FailureRecord
from alpha_agent.registry.sqlite_registry import (
    ExactDuplicate,
    ExperimentRegistry,
    RelatedExperiment,
)


class PriorExperimentEvidence(BaseModel):
    """One prior experiment, flattened for agent consumption."""

    model_config = {"frozen": True, "extra": "forbid"}

    experiment_id: str
    experiment_identity: str
    phase: str
    root_symbol: str
    strategy_family: str
    trial_role: TrialRole
    params: dict = Field(default_factory=dict)
    headline_verdict: RegistryVerdict | None = None
    reason_codes: tuple[str, ...] = ()
    net_pnl_usd: float | None = None
    daily_sharpe: float | None = None
    bh_q: float | None = None
    dsr_probability: float | None = None
    authority: Authority = Authority.AUTHORITATIVE
    superseded_by: tuple[str, ...] = ()


class FailureMemoryResponse(BaseModel):
    """The complete structured memory for one (family, root, params) query.

    Three distinct kinds of count live here and must never be conflated:

    * **execution-attempt counts** (``*_execution_attempts``) count unique runs
      of a hypothesis. An ``INVALID_EXECUTION`` attempt is an engineering /
      data-pipeline defect and says nothing about the science.
    * **valid scientific-outcome counts** (``valid_scientific_*``) count unique
      results produced by a ``VALID`` attempt -- at most one per experiment.
    * **failure/reason record histograms** (``failure_class_counts``,
      ``reason_code_counts``, ``invalidation_class_counts``) count typed
      *records*. There can be several records per attempt or per outcome, so
      these numbers are evidence detail, never a count of attempts or
      hypotheses.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "failure-memory-response/2"
    query: dict = Field(default_factory=dict)

    exact_duplicate: ExactDuplicate | None = None
    prior_experiments: tuple[PriorExperimentEvidence, ...] = ()
    related_experiments: tuple[RelatedExperiment, ...] = ()

    verdict_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    failure_class_counts: dict[str, int] = Field(default_factory=dict)
    parameter_variants_tested: tuple[dict, ...] = ()
    markets_tested: tuple[str, ...] = ()

    # -- execution-attempt accounting (unique attempts, NOT failure records) --
    execution_attempts_total: int = 0
    valid_execution_attempts: int = 0
    invalid_execution_attempts: int = 0
    invalidation_class_counts: dict[str, int] = Field(default_factory=dict)

    # -- valid scientific outcomes (unique results from a VALID attempt) --
    valid_scientific_outcomes: int = 0
    valid_scientific_refusals: int = 0
    scientific_verdict_counts: dict[str, int] = Field(default_factory=dict)

    experiment_failures: tuple[FailureRecord, ...] = ()
    engineering_failures: tuple[FailureRecord, ...] = ()
    lessons: tuple[str, ...] = ()

    n_superseded_related: int = 0
    holdout_eligible_prior: int = 0


def relevant_failure_memory(
    registry: ExperimentRegistry, *, market_universe: Sequence[str],
    asset_domain: AssetDomain,
) -> tuple[FailureMemoryResponse, ...]:
    """PRE-PROPOSAL failure memory (Agent runtime-integration release, section
    3): every ``(strategy_family, root_symbol)`` combination already tested
    for a root in ``market_universe``, looked up and returned as one
    aggregated :class:`FailureMemoryResponse` digest per combination -- BEFORE
    any hypothesis exists, and bounded by what has actually been tried (never
    a raw dump of every registry row).

    Shared by :meth:`ResearchOrchestrator._planning_context` and the
    lightweight propose/inspect UI path (``alpha_agent.ui.llm_demo``) so both
    surfaces build identical pre-proposal memory from the same registry
    evidence -- there is exactly one sweep implementation, not two that could
    drift apart.

    ``asset_domain`` is REQUIRED (Phase 6 ETF Research Pilot): this sweep
    groups experiments by ``(strategy_family, root_symbol)``, so it must be
    scoped to one domain -- a BH/FDR-family-adjacent aggregation must never
    pool ETF and Futures evidence merely because a family label matches.
    """
    combos = sorted(
        {
            (v.experiment.strategy_family, v.experiment.root_symbol)
            for v in registry.experiments(authoritative_only=False, asset_domain=asset_domain)
            if v.experiment.root_symbol in set(market_universe)
        }
    )
    fm = FailureMemory(registry)
    return tuple(
        fm.lookup(strategy_family=f, root_symbol=r, asset_domain=asset_domain)
        for f, r in combos
    )


class FailureMemory:
    """Read-only query facade over an :class:`ExperimentRegistry`."""

    def __init__(self, registry: ExperimentRegistry):
        self.registry = registry

    def lookup(
        self,
        *,
        strategy_family: str,
        asset_domain: AssetDomain,
        root_symbol: str | None = None,
        strategy_spec: dict | None = None,
        experiment_identity: str | None = None,
        signal_cadence: str = "",
        execution_cadence: str = "",
        feature_fingerprints: Sequence[str] = (),
        top_k_related: int = 10,
        include_superseded: bool = False,
    ) -> FailureMemoryResponse:
        """Structured prior evidence for a proposed hypothesis.

        ``strategy_spec`` is a plain params-carrying dict (``{"params": {...}}``
        or the params themselves) so an agent can ask before it has compiled a
        full ``StrategySpec``.

        ``asset_domain`` is REQUIRED (Phase 6 ETF Research Pilot), not
        inferred: every prior-evidence / near-duplicate / exact-duplicate query
        this method issues is scoped to it, so a same-named ``strategy_family``
        under a different domain never contributes to this response.
        """
        params = _params_of(strategy_spec)

        exact = (
            self.registry.find_exact_duplicate(experiment_identity, asset_domain=asset_domain)
            if experiment_identity
            else None
        )

        priors = self.registry.experiments(
            strategy_family=strategy_family,
            root_symbol=root_symbol,
            asset_domain=asset_domain,
            authoritative_only=not include_superseded,
            include_superseded=include_superseded,
        )

        related: tuple[RelatedExperiment, ...] = ()
        if root_symbol is not None:
            related = self.registry.find_related(
                strategy_family=strategy_family,
                root_symbol=root_symbol,
                asset_domain=asset_domain,
                params=params,
                signal_cadence=signal_cadence,
                execution_cadence=execution_cadence,
                feature_fingerprints=feature_fingerprints,
                top_k=top_k_related,
                authoritative_only=not include_superseded,
            )

        verdicts: dict[str, int] = {}
        reasons: dict[str, int] = {}
        variants: list[dict] = []
        markets: set[str] = set()
        evidence: list[PriorExperimentEvidence] = []
        experiment_failures: list[FailureRecord] = []
        holdout_eligible = 0

        # execution-attempt / valid-outcome accounting -- distinct from the
        # failure-record histograms below.
        n_attempts_total = 0
        n_valid_attempts = 0
        n_invalid_attempts = 0
        invalidation_classes: dict[str, int] = {}
        n_valid_outcomes = 0
        n_valid_refusals = 0
        sci_verdicts: dict[str, int] = {}

        for v in priors:
            markets.add(v.experiment.root_symbol)
            p = dict(v.experiment.strategy_spec_json.get("params", {}))
            if p not in variants:
                variants.append(p)
            if v.verdict is not None:
                verdicts[v.verdict.value] = verdicts.get(v.verdict.value, 0) + 1
            if v.result is not None:
                for code in v.result.reason_codes:
                    reasons[code] = reasons.get(code, 0) + 1
                holdout_eligible += int(v.result.holdout_eligible)

            for att in v.attempts:
                n_attempts_total += 1
                if att.is_valid:
                    n_valid_attempts += 1
                    # a valid scientific outcome is the result of a VALID attempt
                    # -- at most one per experiment. An INVALID attempt's result
                    # row is engineering-run bookkeeping and is never counted here.
                    if att.result is not None:
                        n_valid_outcomes += 1
                        sci_verdicts[att.result.headline_verdict.value] = (
                            sci_verdicts.get(att.result.headline_verdict.value, 0) + 1
                        )
                        if _is_scientific_refusal(att.result):
                            n_valid_refusals += 1
                else:
                    n_invalid_attempts += 1
                    cls = att.invalidation_class or "UNSPECIFIED"
                    invalidation_classes[cls] = invalidation_classes.get(cls, 0) + 1

            evidence.append(
                PriorExperimentEvidence(
                    experiment_id=v.experiment_id,
                    experiment_identity=v.experiment_identity,
                    phase=v.experiment.phase,
                    root_symbol=v.experiment.root_symbol,
                    strategy_family=v.experiment.strategy_family,
                    trial_role=v.experiment.trial_role,
                    params=p,
                    headline_verdict=v.verdict,
                    reason_codes=v.result.reason_codes if v.result else (),
                    net_pnl_usd=v.result.net_pnl_usd if v.result else None,
                    daily_sharpe=v.result.daily_sharpe if v.result else None,
                    bh_q=v.result.bh_q if v.result else None,
                    dsr_probability=v.result.dsr_probability if v.result else None,
                    authority=v.authority,
                    superseded_by=v.superseded_by,
                )
            )
            experiment_failures.extend(
                self.registry.failures(experiment_identity=v.experiment_identity)
            )

        classes: dict[str, int] = {}
        for f in experiment_failures:
            classes[f.failure_class.value] = classes.get(f.failure_class.value, 0) + 1

        engineering = self.related_engineering_failures(
            root_symbol=root_symbol, strategy_family=strategy_family
        )
        for f in engineering:
            classes[f.failure_class.value] = classes.get(f.failure_class.value, 0) + 1

        return FailureMemoryResponse(
            query={
                "strategy_family": strategy_family,
                "root_symbol": root_symbol,
                "asset_domain": asset_domain.value,
                "params": params,
                "include_superseded": include_superseded,
            },
            exact_duplicate=exact,
            prior_experiments=tuple(evidence),
            related_experiments=related,
            verdict_counts=verdicts,
            reason_code_counts=reasons,
            failure_class_counts=classes,
            parameter_variants_tested=tuple(variants),
            markets_tested=tuple(sorted(markets)),
            execution_attempts_total=n_attempts_total,
            valid_execution_attempts=n_valid_attempts,
            invalid_execution_attempts=n_invalid_attempts,
            invalidation_class_counts=invalidation_classes,
            valid_scientific_outcomes=n_valid_outcomes,
            valid_scientific_refusals=n_valid_refusals,
            scientific_verdict_counts=sci_verdicts,
            experiment_failures=tuple(
                sorted(experiment_failures, key=lambda f: f.failure_id)
            ),
            engineering_failures=engineering,
            lessons=tuple(f"{f.failure_code}: {f.summary}" for f in engineering),
            n_superseded_related=sum(
                1 for e in evidence if e.authority is Authority.SUPERSEDED
            ),
            holdout_eligible_prior=holdout_eligible,
        )

    def related_engineering_failures(
        self, *, root_symbol: str | None = None, strategy_family: str | None = None
    ) -> tuple[FailureRecord, ...]:
        """System-scope data / execution / procedural lessons that constrain any
        new work on this market, regardless of strategy family.

        NOTE (Phase 6 ETF Research Pilot): the ``failures`` table has no
        ``asset_domain`` column -- schema v6 adds that column to ``experiments``
        only (see ``alpha_agent.registry.schema``). This method therefore still
        matches on ``root_symbol`` / ``strategy_family`` text alone, exactly as
        before; it is not a regression (every real failure record today is
        Futures), but it is not yet structural domain isolation either. A
        future phase that wants full defense-in-depth here would need to add
        ``asset_domain`` to ``failures`` too.
        """
        engineering_classes = (
            FailureClass.DATA_QUALITY_FAILURE,
            FailureClass.CONTRACT_ECONOMICS_FAILURE,
            FailureClass.ROLL_DATA_FAILURE,
            FailureClass.EXECUTION_INTEGRITY_FAILURE,
            FailureClass.SOFTWARE_FAILURE,
            FailureClass.SUPERSEDED_RESULT,
        )
        out: list[FailureRecord] = []
        for cls in engineering_classes:
            for f in self.registry.failures(failure_class=cls):
                if f.root_symbol and root_symbol and f.root_symbol != root_symbol:
                    continue
                if (
                    f.strategy_family
                    and strategy_family
                    and f.strategy_family != strategy_family
                ):
                    continue
                out.append(f)
        return tuple(sorted(out, key=lambda f: (f.failure_class.value, f.failure_id)))


#: reason-code substrings that mark a VALID attempt's result as a scientific
#: *refusal* -- a real outcome, but "not enough evidence to adjudicate" rather
#: than a pass or a clean reject. Matched case-insensitively.
_REFUSAL_REASON_MARKERS: tuple[str, ...] = ("REFUSED", "INSUFFICIENT")


def _is_scientific_refusal(result) -> bool:
    """A refusal is a genuine scientific outcome of a VALID attempt where the
    protocol declined to adjudicate for lack of evidence (e.g. insufficient
    event density). It is still evidence: a repeat proposal must say what is
    materially different."""
    if result.headline_verdict is RegistryVerdict.INCONCLUSIVE:
        return True
    codes = " ".join(result.reason_codes).upper()
    return any(marker in codes for marker in _REFUSAL_REASON_MARKERS)


def _params_of(strategy_spec: dict | None) -> dict:
    if not strategy_spec:
        return {}
    if "params" in strategy_spec and isinstance(strategy_spec["params"], dict):
        return dict(strategy_spec["params"])
    return {k: v for k, v in strategy_spec.items() if not isinstance(v, (dict, list))}

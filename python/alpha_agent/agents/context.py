"""The typed, closed input bundle a runtime :class:`ResearchAgent` is allowed to
see (Phase 16).

Prompt 16 is explicit about the agent's inputs -- feature catalog, approved
market universe, research knowledge base, holdout-excluded registry summaries,
and validation feedback from prior experiments -- and nothing else. This module
builds that bundle from the deterministic layers and freezes it.

Two invariants are enforced here, not left to prompt wording:

1. **No locked holdout.** Every string / int in the serialised context passes
   :func:`assert_no_holdout_market_data`. A 2025 market-data or performance
   value anywhere in the bundle is a hard error.
2. **Failure memory keeps the engineering / science distinction.** Phase 15B is
   the canonical lesson: 60 ``INVALID_EXECUTION`` attempts (a feature-pipeline
   defect) are *not* evidence against the ML meta-labeling hypothesis, whereas
   the corrected valid attempt's 60 typed refusals (insufficient event density)
   *are* a real scientific outcome. :class:`FailureMemoryDigest` carries both
   counts separately so the agent cannot conflate them.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field, model_validator

from alpha_agent.registry.enums import FailureClass
from alpha_agent.registry.failure_memory import FailureMemoryResponse
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data
from alpha_agent.registry.sqlite_registry import RegistrySummary

#: Failure classes that describe an *engineering / execution* defect. An attempt
#: that failed for one of these reasons says nothing about the scientific merit
#: of the hypothesis -- it must be re-executed, not abandoned.
ENGINEERING_FAILURE_CLASSES: frozenset[FailureClass] = frozenset(
    {
        FailureClass.INVALID_EXECUTION_ATTEMPT,
        FailureClass.FEATURE_PIPELINE_FAILURE,
        FailureClass.SOFTWARE_FAILURE,
        FailureClass.EXECUTION_INTEGRITY_FAILURE,
        FailureClass.DATA_QUALITY_FAILURE,
        FailureClass.CONTRACT_ECONOMICS_FAILURE,
        FailureClass.ROLL_DATA_FAILURE,
    }
)

#: Failure classes that are a genuine scientific / statistical outcome. These
#: *are* evidence: a future proposal that repeats the mechanism must say what is
#: materially different.
SCIENTIFIC_FAILURE_CLASSES: frozenset[FailureClass] = frozenset(
    {
        FailureClass.SCIENTIFIC_REJECTION,
        FailureClass.STATISTICAL_INCONCLUSIVE,
        FailureClass.INSUFFICIENT_TRADES,
        FailureClass.NULL_NOT_REJECTED,
        FailureClass.FDR_NOT_PASSED,
        FailureClass.DSR_NOT_PASSED,
        FailureClass.PARAMETER_INSTABILITY,
        FailureClass.COST_FRAGILITY,
        FailureClass.CROSS_MARKET_WEAKNESS,
    }
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FeatureCatalogEntry(BaseModel):
    """One registered feature the agent may reference in ``required_features``.

    Built from ``alpha_agent.features.REGISTRY`` -- the agent never sees a
    callable, an expression, or an import name, only the closed ``kind`` label
    and its declared parameters.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    kind: str
    family: str
    description: str = ""
    params: tuple[str, ...] = ()


class FailureMemoryDigest(BaseModel):
    """Prior evidence for one (family, root) neighbourhood.

    Three kinds of count, never conflated (Phase 16.1):

    * ``*_execution_attempts`` -- unique runs of a hypothesis. An
      ``INVALID_EXECUTION`` attempt is an engineering / data-pipeline defect and
      is **not** evidence against the science; re-execution under the same
      identity is permitted.
    * ``valid_scientific_*`` -- unique results from a ``VALID`` attempt, at most
      one per experiment. A refusal (insufficient evidence to adjudicate) is
      still a real outcome: a repeat proposal must say what is materially
      different.
    * ``*_failure_records`` -- typed record histograms. Several records can
      attach to one attempt or one outcome, so these are evidence detail and
      never a count of attempts or hypotheses.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    strategy_family: str
    root_symbol: str | None = None
    markets_tested: tuple[str, ...] = ()
    parameter_variants_tested: int = 0
    verdict_counts: dict[str, int] = Field(default_factory=dict)

    # -- unique execution attempts (engineering, not science) --
    invalid_execution_attempts: int = 0
    valid_execution_attempts: int = 0
    invalidation_class_counts: dict[str, int] = Field(default_factory=dict)

    # -- unique valid scientific outcomes --
    valid_scientific_outcomes: int = 0
    valid_scientific_refusals: int = 0
    scientific_verdict_counts: dict[str, int] = Field(default_factory=dict)

    # -- typed record histograms (evidence detail; >=1 per attempt/outcome) --
    engineering_failure_records: dict[str, int] = Field(default_factory=dict)
    scientific_failure_records: dict[str, int] = Field(default_factory=dict)

    exact_duplicate_blocks_reexecution: bool = False
    lessons: tuple[str, ...] = ()
    guidance: str = ""


def build_feature_catalog(registry: object | None = None) -> tuple[FeatureCatalogEntry, ...]:
    """Project the feature registry into the closed catalog the agent sees."""
    if registry is None:
        import alpha_agent.features.compute  # noqa: F401 - register defs
        from alpha_agent.features import REGISTRY as registry

    out: list[FeatureCatalogEntry] = []
    for kind in registry.kinds():  # type: ignore[union-attr]
        fdef = registry.get(kind)  # type: ignore[union-attr]
        out.append(
            FeatureCatalogEntry(
                kind=kind,
                family=getattr(fdef.family, "value", str(fdef.family)),
                description=getattr(fdef, "description", "") or "",
                params=tuple(getattr(fdef, "param_order", ()) or ()),
            )
        )
    return tuple(out)


def _split_failure_records(failure_class_counts: dict[str, int]) -> tuple[dict, dict]:
    """Partition the typed failure-*record* histogram into engineering and
    scientific buckets. Record counts stay record counts -- they are never
    summed into an attempt or outcome count."""
    engineering: dict[str, int] = {}
    scientific: dict[str, int] = {}
    for cls_name, count in failure_class_counts.items():
        try:
            cls = FailureClass(cls_name)
        except ValueError:
            continue
        if cls in ENGINEERING_FAILURE_CLASSES:
            engineering[cls_name] = count
        elif cls in SCIENTIFIC_FAILURE_CLASSES:
            scientific[cls_name] = count
    return engineering, scientific


def build_failure_memory_digest(response: FailureMemoryResponse) -> FailureMemoryDigest:
    """Collapse a :class:`FailureMemoryResponse` into the agent-facing digest.

    Attempt counts come from the response's execution-attempt accounting (unique
    runs); refusal counts come from its valid-scientific-outcome accounting
    (unique results). The typed failure-record histograms are carried through
    unchanged as evidence detail.

    Phase 15B flows through exactly: 60 ``invalid_execution_attempts`` (a
    feature-pipeline defect, not a refutation) and 60 ``valid_scientific_refusals``
    (insufficient event density), while ``scientific_failure_records`` still
    shows the 120 underlying ``STATISTICAL_INCONCLUSIVE`` records.
    """
    query = response.query or {}
    engineering_records, scientific_records = _split_failure_records(
        response.failure_class_counts
    )
    blocks = response.exact_duplicate is not None and getattr(
        response.exact_duplicate, "blocks_reexecution", False
    )

    parts: list[str] = []
    if response.invalid_execution_attempts:
        parts.append(
            f"{response.invalid_execution_attempts} prior execution attempt(s) were "
            "INVALID_EXECUTION (engineering / data-pipeline defect). This is not a "
            "scientific verdict -- the hypothesis may be re-executed under the same "
            "identity and must not be treated as disproven."
        )
    if response.valid_scientific_refusals:
        parts.append(
            f"{response.valid_scientific_refusals} valid scientific refusal(s) "
            "(protocol declined to adjudicate for lack of evidence, e.g. "
            "insufficient event density). This is a real outcome: a repeat "
            "proposal must state what is materially different."
        )
    adjudicated = {
        k: v
        for k, v in response.scientific_verdict_counts.items()
        if k in {"PASS", "REJECT", "INCONCLUSIVE"}
    }
    if adjudicated:
        parts.append(f"prior adjudicated verdicts: {adjudicated}.")
    if blocks:
        parts.append(
            "An existing experiment_identity already has a VALID authoritative "
            "result: cite it, do not re-run it."
        )
    if not parts:
        parts.append("No prior experiments in this neighbourhood.")

    return FailureMemoryDigest(
        strategy_family=str(query.get("strategy_family", "")),
        root_symbol=query.get("root_symbol"),
        markets_tested=tuple(response.markets_tested),
        parameter_variants_tested=len(response.parameter_variants_tested),
        verdict_counts=dict(response.verdict_counts),
        invalid_execution_attempts=response.invalid_execution_attempts,
        valid_execution_attempts=response.valid_execution_attempts,
        invalidation_class_counts=dict(response.invalidation_class_counts),
        valid_scientific_outcomes=response.valid_scientific_outcomes,
        valid_scientific_refusals=response.valid_scientific_refusals,
        scientific_verdict_counts=dict(response.scientific_verdict_counts),
        engineering_failure_records=engineering_records,
        scientific_failure_records=scientific_records,
        exact_duplicate_blocks_reexecution=blocks,
        lessons=tuple(response.lessons),
        guidance=" ".join(parts),
    )


class RegistryDigest(BaseModel):
    """Holdout-excluded registry summary. Counts only -- no timestamps, no PnL
    windows, nothing that could carry a 2025 value."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: int
    identity_schema: str
    authoritative_hypotheses: int
    canonical: int
    neighbour: int
    superseded_experiments: int
    canonical_verdict_counts: dict[str, int] = Field(default_factory=dict)
    valid_execution_attempts: int = 0
    invalid_execution_attempts: int = 0
    failure_class_counts: dict[str, int] = Field(default_factory=dict)
    holdout_eligible: int = 0

    @classmethod
    def from_summary(cls, summary: RegistrySummary) -> RegistryDigest:
        return cls(
            schema_version=summary.schema_version,
            identity_schema=summary.identity_schema,
            authoritative_hypotheses=summary.authoritative_statistical_hypotheses,
            canonical=summary.canonical,
            neighbour=summary.neighbour,
            superseded_experiments=summary.superseded_experiments,
            canonical_verdict_counts=dict(summary.canonical_verdict_counts),
            valid_execution_attempts=summary.valid_execution_attempts,
            invalid_execution_attempts=summary.invalid_execution_attempts,
            failure_class_counts=dict(summary.failure_class_counts),
            holdout_eligible=summary.holdout_eligible,
        )


class ResearchContext(BaseModel):
    """The complete, frozen input the research agent is given for one proposal.

    Anything not in this model, the agent does not see: no filesystem, no
    registry handle, no market data, no holdout, no ability to run a backtest.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "research-context/1"
    objective: str
    market_universe: tuple[str, ...]
    feature_catalog: tuple[FeatureCatalogEntry, ...]
    knowledge_base: tuple[str, ...] = ()
    registry_digest: RegistryDigest | None = None
    failure_memory: tuple[FailureMemoryDigest, ...] = ()
    validation_feedback: tuple[str, ...] = ()
    avoid_repeating: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _no_holdout(self) -> ResearchContext:
        if not self.market_universe:
            raise ValueError("market_universe must be non-empty")
        if not self.feature_catalog:
            raise ValueError("feature_catalog must be non-empty")
        assert_no_holdout_market_data(
            self.model_dump(mode="json"),
            path="$.research_context",
            observational_context_paths=frozenset({"$.research_context.knowledge_base"}),
        )
        return self

    # -- derived views -------------------------------------------------------

    @property
    def feature_kinds(self) -> frozenset[str]:
        return frozenset(e.kind for e in self.feature_catalog)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return _sha256(self.canonical_json())

    def render_user_message(self) -> str:
        """Deterministic prompt body. Stable ordering so the prompt hash is
        reproducible for a given context."""
        payload = {
            "objective": self.objective,
            "approved_market_universe": list(self.market_universe),
            "feature_catalog": [e.model_dump(mode="json") for e in self.feature_catalog],
            "research_knowledge_base": list(self.knowledge_base),
            "experiment_registry_digest": (
                self.registry_digest.model_dump(mode="json") if self.registry_digest else None
            ),
            "failure_memory": [d.model_dump(mode="json") for d in self.failure_memory],
            "validation_feedback": list(self.validation_feedback),
            "avoid_repeating_without_material_change": list(self.avoid_repeating),
        }
        return (
            "Propose ONE falsifiable futures research hypothesis as JSON.\n\n"
            + json.dumps(payload, indent=2, sort_keys=True)
        )


def build_research_context(
    *,
    objective: str,
    market_universe: Iterable[str],
    knowledge_base: Sequence[str] = (),
    registry_summary: RegistrySummary | None = None,
    failure_memory: Sequence[FailureMemoryResponse | FailureMemoryDigest] = (),
    validation_feedback: Sequence[str] = (),
    avoid_repeating: Sequence[str] = (),
    feature_registry: object | None = None,
) -> ResearchContext:
    """Assemble a :class:`ResearchContext` from the deterministic layers."""
    digests = tuple(
        d if isinstance(d, FailureMemoryDigest) else build_failure_memory_digest(d)
        for d in failure_memory
    )
    # Fail loud on a locked-holdout value in any raw input, before pydantic can
    # wrap it in a ValidationError.
    assert_no_holdout_market_data(
        {
            "objective": objective,
            "market_universe": list(market_universe),
            "knowledge_base": list(knowledge_base),
            "failure_memory": [d.model_dump(mode="json") for d in digests],
            "validation_feedback": list(validation_feedback),
            "avoid_repeating": list(avoid_repeating),
        },
        path="$.research_context_inputs",
        observational_context_paths=frozenset({"$.research_context_inputs.knowledge_base"}),
    )
    return ResearchContext(
        objective=objective,
        market_universe=tuple(market_universe),
        feature_catalog=build_feature_catalog(feature_registry),
        knowledge_base=tuple(knowledge_base),
        registry_digest=(
            RegistryDigest.from_summary(registry_summary) if registry_summary is not None else None
        ),
        failure_memory=digests,
        validation_feedback=tuple(validation_feedback),
        avoid_repeating=tuple(avoid_repeating),
    )

"""Runtime Strategy Compiler Agent (Phase 17, architecture-corrected in 17.1).

Turns an **approved** :class:`~alpha_agent.schemas.hypothesis.HypothesisSpec`
into a safe, schema-valid :class:`~alpha_agent.strategy.spec.StrategySpec` -- or a
typed rejection.

The LLM never authors strategy code, a free-form expression, a callable, an
import, execution logic, or an arbitrary DSL extension. It fills one **closed
typed intermediate blueprint**, and deterministic code builds the
:class:`StrategySpec` and passes it through the **UNCHANGED** Phase 10
:class:`StrategyCompiler`.

Two ways to express a hypothesis (exactly one per plan):

* ``template`` -- a Phase 11 baseline family (``tsmom`` / ``ma_trend`` /
  ``breakout`` / ``mean_reversion``) with numeric parameters inside a **typed**
  bound table (`PHASE_11_PARAM_BOUNDS`, transcribed from the frozen Phase 11
  declared grid ranges -- no prose parsing on the policy path). A convenience
  shortcut, not the limit of what the compiler can express.
* ``blueprint`` -- a full composition of the frozen Phase 10 closed Strategy
  DSL: declared :class:`FeatureSpec` s (registered ``kind`` + scalar params
  only), a condition tree from the closed comparator vocabulary
  (``gt``/``gte``/``lt``/``lte``), boolean (``all``/``any``) and ``not`` nodes,
  ``feature`` / ``lag`` / ``const`` operands, and integer target-position rules.

Everything downstream is deterministic:

1. root ∈ approved universe ∩ hypothesis universe;
2. template params are bounds-checked against the typed table; blueprint feature
   ``kind`` s must be registered;
3. the :class:`StrategySpec` is built deterministically (frozen Phase 11 factory
   for a template, node-by-node translation for a blueprint) -- no market data,
   no backtest;
4. locked-2025-holdout guard on the input hypothesis and the produced spec;
5. compile through the **unchanged** Phase 10 :class:`StrategyCompiler` -- the
   real safety gate (unknown / unsafe features, execution-plane fields,
   non-causal references, unbounded trees are all rejected there);
6. **HypothesisSpec fidelity**: the set of feature KINDS in the compiled spec
   must equal ``hypothesis.required_features`` -- a required kind not referenced
   by a rule is ``REQUIRED_FEATURE_NOT_REPRESENTED``; a feature kind the
   hypothesis did not approve is ``UNAPPROVED_FEATURE_ADDED``. Several
   ``FeatureSpec`` s of the same approved kind with different parameters, plus
   lags / constants / boolean composition, are all fine. The ResearchAgent
   decides what mechanism is tested; the CompilerAgent only translates it;
7. de-duplication: a matching ``strategy_fingerprint`` is surfaced as pure prior
   :class:`DuplicateEvidence` and **NEVER** rejects compilation. Phase 17 does
   not construct the complete scientific ``experiment_identity``, so even a
   fingerprint with prior VALID authoritative results is only evidence. Phase 18
   builds the full identity and queries the schema-v5 registry: exact identity +
   VALID authoritative result => block; exact identity + INVALID_EXECUTION-only
   history => permit another attempt; same fingerprint, different scientific
   identity => permit as a distinct experiment.

The agent holds no filesystem, experiment-registry, process-spawning or
order-routing surface. The only file it reads is its own prompt template, at
construction.

Retry policy mirrors Phase 16: a retry happens **only** on a schema failure
(unparseable JSON, or a ``StrategyPlanRequest`` validation error). A
semantically valid plan that fails a deterministic guardrail is *rejected*, not
retried.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from alpha_agent.agents.context import FeatureCatalogEntry, build_feature_catalog
from alpha_agent.agents.llm import DEFAULT_MODEL, LLMClient, LLMResponse

# _extract_json / _sha256 are the shared runtime-agent helpers introduced in
# Phase 16; reusing them keeps JSON extraction and hashing identical across the
# two agents.
from alpha_agent.agents.research_agent import (
    AttemptRecord,
    PromptLog,
    _extract_json,
    _sha256,
)
from alpha_agent.execution.capability import SUPPORTED_CADENCES
from alpha_agent.features.spec import FeatureSpec
from alpha_agent.registry.holdout_guard import assert_no_holdout_market_data
from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.strategy import (
    DEFAULT_COMPILE_LIMITS,
    StrategyCompileError,
    StrategyCompiler,
    strategy_fingerprint,
)
from alpha_agent.strategy.baselines.families import baseline_family_docs
from alpha_agent.strategy.candidates_phase_13_5c import (
    EXECUTION_ASSUMPTIONS,
    EXECUTION_CADENCE,
    ROOTS,
    SIGNAL_CADENCE,
    baseline_neighbour_params,
    baseline_params,
    spec_for_params,
)
from alpha_agent.strategy.enums import (
    BooleanOp,
    Comparator,
    DefaultAction,
    MissingRulePolicy,
)
from alpha_agent.strategy.spec import (
    BooleanNode,
    ComparisonNode,
    ConstOperand,
    FeatureDeclaration,
    FeatureOperand,
    LagOperand,
    NotNode,
    Rule,
    StrategySpec,
    TargetAction,
)

DEFAULT_PROMPT_PATH = Path(__file__).parent / "runtime_prompts" / "strategy_compiler.md"

#: Phase 11 baseline families exposed as convenience templates. Silver Bullet is
#: deliberately excluded: it is a frozen single-root benchmark hypothesis whose
#: NQ session / window semantics are part of its identity, not a parametric
#: template.
COMPILER_FAMILY_KEYS: tuple[str, ...] = ("tsmom", "ma_trend", "breakout", "mean_reversion")

#: Schema hard cap on |target_units| (mirrors ``spec._MAX_ABS_TARGET_UNITS_SCHEMA``).
#: The Phase 10 compiler enforces a tighter, configurable limit on top of this.
_SCHEMA_MAX_TARGET_UNITS = 1000


# -- typed errors -------------------------------------------------------------


class StrategyCompilerAgentError(RuntimeError):
    """Base class for Phase 17 strategy-compiler-agent failures."""


class CompilerSchemaRetryExhausted(StrategyCompilerAgentError):
    """The model never returned a schema-valid ``StrategyPlanRequest`` within the
    retry budget. Carries every attempt for the orchestrator / registry."""

    def __init__(self, attempts: tuple[AttemptRecord, ...]):
        self.attempts = attempts
        last = attempts[-1].error if attempts else "no attempts"
        super().__init__(
            f"no schema-valid StrategyPlanRequest after {len(attempts)} attempt(s); "
            f"last error: {last}"
        )


class CompilerBudgetExceeded(StrategyCompilerAgentError):
    """A turn or token budget was hit before a valid plan was produced."""


class _BuildRejected(Exception):
    """Internal: a deterministic guardrail rejected the plan while building the
    :class:`StrategySpec`. Carried up to :meth:`_finalize` and turned into a
    typed :class:`CompiledStrategyProposal` rejection."""

    def __init__(self, code: str, detail: str, *, diagnostics: Sequence[str] = ()):
        self.code = code
        self.detail = detail
        self.diagnostics = tuple(diagnostics)
        super().__init__(f"{code}: {detail}")


# -- rejection codes (stable machine slugs) ---------------------------------

HYPOTHESIS_NOT_EXPRESSIBLE = "HYPOTHESIS_NOT_EXPRESSIBLE"
HYPOTHESIS_FEATURE_UNAVAILABLE = "HYPOTHESIS_FEATURE_UNAVAILABLE"
UNKNOWN_STRATEGY_FAMILY = "UNKNOWN_STRATEGY_FAMILY"
MARKET_OUTSIDE_UNIVERSE = "MARKET_OUTSIDE_UNIVERSE"
ROOT_OUTSIDE_HYPOTHESIS_UNIVERSE = "ROOT_OUTSIDE_HYPOTHESIS_UNIVERSE"
UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
PARAMETER_OUT_OF_BOUNDS = "PARAMETER_OUT_OF_BOUNDS"
NON_INTEGER_PARAMETER = "NON_INTEGER_PARAMETER"
INVALID_STRATEGY_PARAMETERS = "INVALID_STRATEGY_PARAMETERS"
UNKNOWN_FEATURE_KIND = "UNKNOWN_FEATURE_KIND"
INVALID_BLUEPRINT = "INVALID_BLUEPRINT"
STRATEGY_COMPILE_REJECTED = "STRATEGY_COMPILE_REJECTED"
EXECUTION_SEMANTICS_VIOLATION = "EXECUTION_SEMANTICS_VIOLATION"
REQUIRED_FEATURE_NOT_REPRESENTED = "REQUIRED_FEATURE_NOT_REPRESENTED"
UNAPPROVED_FEATURE_ADDED = "UNAPPROVED_FEATURE_ADDED"

REJECTION_CODES: tuple[str, ...] = (
    HYPOTHESIS_NOT_EXPRESSIBLE,
    HYPOTHESIS_FEATURE_UNAVAILABLE,
    UNKNOWN_STRATEGY_FAMILY,
    MARKET_OUTSIDE_UNIVERSE,
    ROOT_OUTSIDE_HYPOTHESIS_UNIVERSE,
    UNKNOWN_PARAMETER,
    PARAMETER_OUT_OF_BOUNDS,
    NON_INTEGER_PARAMETER,
    INVALID_STRATEGY_PARAMETERS,
    UNKNOWN_FEATURE_KIND,
    INVALID_BLUEPRINT,
    STRATEGY_COMPILE_REJECTED,
    EXECUTION_SEMANTICS_VIOLATION,
    REQUIRED_FEATURE_NOT_REPRESENTED,
    UNAPPROVED_FEATURE_ADDED,
)


# -- typed scalar bounds (replaces prose parsing on the policy path) --------


class ScalarBound(BaseModel):
    """A closed numeric interval for one strategy parameter."""

    model_config = {"frozen": True, "extra": "forbid"}

    lo: float
    hi: float
    integer: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> ScalarBound:
        if self.hi < self.lo:
            raise ValueError(f"bound hi {self.hi} < lo {self.lo}")
        return self

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


#: Typed transcription of the Phase 11 declared parameter grid ranges
#: (``alpha_agent.strategy.baselines.families``). The typed table is
#: authoritative for the compiler; :func:`assert_phase_11_bounds_match_declared_ranges`
#: proves it reproduces the numeric endpoints declared in prose.
PHASE_11_PARAM_BOUNDS: dict[str, dict[str, ScalarBound]] = {
    "tsmom": {
        "fast_horizon": ScalarBound(lo=5, hi=60, integer=True),
        "slow_horizon": ScalarBound(lo=20, hi=250, integer=True),
        "size": ScalarBound(lo=1, hi=3, integer=True),
    },
    "ma_trend": {
        "fast_window": ScalarBound(lo=5, hi=50, integer=True),
        "slow_window": ScalarBound(lo=20, hi=200, integer=True),
        "size": ScalarBound(lo=1, hi=3, integer=True),
    },
    "breakout": {
        "lookback": ScalarBound(lo=10, hi=100, integer=True),
        "size": ScalarBound(lo=1, hi=3, integer=True),
    },
    "mean_reversion": {
        "zscore_window": ScalarBound(lo=10, hi=120, integer=True),
        "entry_z": ScalarBound(lo=1.5, hi=3.0, integer=False),
        "exit_z": ScalarBound(lo=0.0, hi=1.0, integer=False),
        "size": ScalarBound(lo=1, hi=3, integer=True),
    },
}

_PROSE_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def assert_phase_11_bounds_match_declared_ranges() -> None:
    """Consistency check -- NOT on the policy path.

    The typed :data:`PHASE_11_PARAM_BOUNDS` table must reproduce the numeric
    endpoints declared in prose in ``alpha_agent.strategy.baselines.families``.
    The typed table is authoritative; this only proves faithful transcription so
    a later edit to either side is caught.
    """
    for doc in baseline_family_docs():
        if doc.key not in PHASE_11_PARAM_BOUNDS:
            continue
        typed = PHASE_11_PARAM_BOUNDS[doc.key]
        if set(typed) != set(doc.parameters):
            raise AssertionError(
                f"{doc.key}: typed bound keys {sorted(typed)} != declared parameters "
                f"{sorted(doc.parameters)}"
            )
        for pname, prose in doc.param_grid_ranges.items():
            nums = _PROSE_NUMBER_RE.findall(prose)
            if len(nums) < 2:
                raise AssertionError(
                    f"{doc.key}.{pname}: declared range {prose!r} has no numeric box to check"
                )
            lo, hi = float(nums[0]), float(nums[1])
            got = typed[pname]
            if (got.lo, got.hi) != (lo, hi):
                raise AssertionError(
                    f"{doc.key}.{pname}: typed [{got.lo}, {got.hi}] != declared [{lo}, {hi}]"
                )


# -- family (template) catalog -------------------------------------------


class StrategyFamilyCard(BaseModel):
    """One Phase 11 convenience template, projected for the LLM: the closed
    ``family_key``, its economic mechanism and exact timing, and the typed
    numeric bound for each parameter. No factory or callable is exposed."""

    model_config = {"frozen": True, "extra": "forbid"}

    family_key: str
    name: str
    economic_mechanism: str
    formula_and_timing: str
    parameters: tuple[str, ...]
    param_bounds: dict[str, ScalarBound]
    default_action: str
    failure_regimes: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def build_family_catalog() -> tuple[StrategyFamilyCard, ...]:
    """Deterministic projection of the frozen Phase 11 family docs + the typed
    bound table into the convenience-template catalog."""
    docs = {d.key: d for d in baseline_family_docs()}
    cards: list[StrategyFamilyCard] = []
    for key in sorted(COMPILER_FAMILY_KEYS):
        doc = docs[key]
        bounds = PHASE_11_PARAM_BOUNDS[key]
        if set(doc.parameters) != set(bounds):  # pragma: no cover - guards a frozen-data change
            raise ValueError(
                f"family {key!r} parameters {sorted(doc.parameters)} != typed bound keys "
                f"{sorted(bounds)}"
            )
        cards.append(
            StrategyFamilyCard(
                family_key=key,
                name=doc.name,
                economic_mechanism=doc.economic_mechanism,
                formula_and_timing=doc.formula_and_timing,
                parameters=tuple(doc.parameters),
                param_bounds={p: bounds[p] for p in doc.parameters},
                default_action=doc.default_action,
                failure_regimes=tuple(doc.failure_regimes),
            )
        )
    return tuple(cards)


def _family_fingerprints(family: str, root: str) -> Iterator[str]:
    yield strategy_fingerprint(spec_for_params(family, baseline_params(family, root)))
    for neigh in baseline_neighbour_params(family, root):
        yield strategy_fingerprint(spec_for_params(family, neigh))


# -- known-strategy evidence -------------------------------------------


class KnownStrategyRecord(BaseModel):
    """One prior strategy the compiler agent is told about, for de-duplication
    *evidence*. Authority is supplied by the caller (Phase 18 / the registry);
    the compiler never infers it."""

    model_config = {"frozen": True, "extra": "forbid"}

    strategy_fingerprint: str
    source: str  # "frozen_candidate_manifest" | "experiment_registry" | ...
    experiment_id: str | None = None
    experiment_identity: str | None = None
    has_valid_authoritative_result: bool = False
    valid_execution_attempts: int = 0
    invalid_execution_attempts: int = 0


@lru_cache(maxsize=1)
def frozen_template_strategy_records() -> tuple[KnownStrategyRecord, ...]:
    """Every frozen Phase 13.5C canonical + pre-declared-neighbour strategy
    fingerprint, rebuilt deterministically (no market data, no backtest). These
    carry **no authority claim** -- they are "seen" evidence only. Phase 18
    supplies real authority from the live registry."""
    out: list[KnownStrategyRecord] = []
    seen: set[str] = set()
    for family in COMPILER_FAMILY_KEYS:
        for root in ROOTS:
            for fp in _family_fingerprints(family, root):
                if fp in seen:
                    continue
                seen.add(fp)
                out.append(
                    KnownStrategyRecord(
                        strategy_fingerprint=fp, source="frozen_candidate_manifest"
                    )
                )
    return tuple(out)


def known_strategy_fingerprints() -> frozenset[str]:
    """Convenience: just the fingerprint strings of the frozen template set."""
    return frozenset(r.strategy_fingerprint for r in frozen_template_strategy_records())


#: Fixed advisory attached to every :class:`DuplicateEvidence`. Phase 17 does not
#: know the complete scientific experiment identity, so a strategy fingerprint --
#: even one with prior VALID authoritative results -- is only prior EVIDENCE. The
#: compiler never turns it into a re-execution decision or a rejection.
_DUPLICATE_ADVISORY = (
    "Strategy-fingerprint evidence only. Phase 17 does not construct the complete "
    "scientific experiment_identity (ValidationSpec / ReliabilityPolicy / split / "
    "cost + risk config / feature set), so this NEVER blocks compilation. Phase 18 "
    "must build the full identity and query the schema-v5 registry: exact identity "
    "+ VALID authoritative result => block; exact identity + INVALID_EXECUTION-only "
    "history => permit another attempt; same strategy fingerprint but a different "
    "scientific identity => permit as a distinct experiment."
)


class DuplicateEvidence(BaseModel):
    """What the compiler found about this strategy fingerprint in the supplied
    known-strategy set. Pure prior EVIDENCE -- it carries no decision. Phase 18 /
    the registry decides re-execution from the complete scientific experiment
    identity and execution-attempt authority."""

    model_config = {"frozen": True, "extra": "forbid"}

    strategy_fingerprint: str
    strategy_fingerprint_seen: bool
    matches: tuple[KnownStrategyRecord, ...] = ()
    prior_valid_authoritative_results: int = 0
    prior_valid_execution_attempts: int = 0
    prior_invalid_execution_attempts: int = 0
    prior_experiment_identities: tuple[str, ...] = ()
    prior_experiment_ids: tuple[str, ...] = ()
    advisory: str = _DUPLICATE_ADVISORY


def _duplicate_evidence(fingerprint: str, context: CompilerContext) -> DuplicateEvidence:
    matches = tuple(
        r for r in context.known_strategies if r.strategy_fingerprint == fingerprint
    )
    return DuplicateEvidence(
        strategy_fingerprint=fingerprint,
        strategy_fingerprint_seen=bool(matches),
        matches=matches,
        prior_valid_authoritative_results=sum(
            1 for r in matches if r.has_valid_authoritative_result
        ),
        prior_valid_execution_attempts=sum(r.valid_execution_attempts for r in matches),
        prior_invalid_execution_attempts=sum(
            r.invalid_execution_attempts for r in matches
        ),
        prior_experiment_identities=tuple(
            dict.fromkeys(r.experiment_identity for r in matches if r.experiment_identity)
        ),
        prior_experiment_ids=tuple(
            dict.fromkeys(r.experiment_id for r in matches if r.experiment_id)
        ),
    )


# -- execution semantics (deterministic, not the LLM's to choose) ----------


#: The closed set of `signal_cadence` values a BLUEPRINT may declare -- the
#: LLM DECLARES cadence from this typed vocabulary, it never invents a prose
#: string. This is deliberately the SAME two values `SIGNAL_CADENCE` already
#: uses for the five legacy families (`alpha_agent.execution.capability.
#: SUPPORTED_CADENCES`), not a re-derived or larger set: those are the only
#: two cadences this release's execution path (`_CompiledSpecAdapter` /
#: `daily_baseline_schedule` / `native_1m_schedule`) actually knows how to
#: schedule. Adding a THIRD real cadence (hourly / 15m / 5m / 1m-non-daily
#: session-event, etc.) needs a new schedule builder in
#: `alpha_agent.validation.phase_13_5c_matrix` first -- out of scope here, and
#: not silently faked by widening this Literal.
BlueprintSignalCadence = Literal["daily_trading_day", "native_1m"]


def assert_blueprint_cadences_match_supported() -> None:
    """Consistency check -- NOT on the policy path. `BlueprintSignalCadence`
    must name exactly the cadences this release's execution path can actually
    schedule, so a blueprint can never declare a cadence the compiler accepts
    but the capability gate / execution service would then reject."""
    from typing import get_args

    declared = frozenset(get_args(BlueprintSignalCadence))
    if declared != SUPPORTED_CADENCES:
        raise AssertionError(
            f"BlueprintSignalCadence {sorted(declared)} != SUPPORTED_CADENCES "
            f"{sorted(SUPPORTED_CADENCES)}"
        )


class ExecutionSemantics(BaseModel):
    """The execution contract that applies to every compiled strategy. Fixed by
    the architecture -- the LLM chooses none of it (it declares `signal_cadence`
    on a BLUEPRINT, from a closed vocabulary; a TEMPLATE's cadence is always the
    frozen per-family value)."""

    model_config = {"frozen": True, "extra": "forbid"}

    signal_cadence: str
    execution_cadence: str = EXECUTION_CADENCE
    entry_timing: str
    exit_timing: str
    same_bar_fills_prohibited: bool = True
    cost_assumptions: dict[str, float]
    note: str = (
        "The StrategySpec expresses target-position intent only. Execution "
        "prices, fills, slippage, commission, latency, margin and risk "
        "decisions belong to the deterministic C++ Quant Core, never to the "
        "strategy."
    )


def _execution_semantics(*, signal_cadence: str, default_action: str) -> ExecutionSemantics:
    return ExecutionSemantics(
        signal_cadence=signal_cadence,
        entry_timing=(
            "The decision uses information through bar T only. The C++ engine "
            "fills at the next eligible bar (T+1+latency_bars). Same-bar fills "
            "are prohibited: no rule references bar T's own execution price, and "
            "the DSL has no lead / future primitive (lag periods are >= 1)."
        ),
        exit_timing=(
            "There are no stop / target / bracket orders in the DSL. An exit is "
            f"a rule that targets 0, or the default_action ({default_action}); the "
            "C++ Quant Core computes order_delta = target - current and fills it "
            "on the next eligible bar."
        ),
        cost_assumptions={
            "commission_per_contract_usd": float(
                EXECUTION_ASSUMPTIONS["commission_per_contract_usd"]
            ),
            "slippage_ticks": float(EXECUTION_ASSUMPTIONS["slippage_ticks"]),
            "spread_ticks": float(EXECUTION_ASSUMPTIONS["spread_ticks"]),
            "latency_bars": float(EXECUTION_ASSUMPTIONS["latency_bars"]),
        },
    )


# -- LLM output schema: the closed typed blueprint ----------------------


class _Closed(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}


def _scalar_number_dict(v: object, *, what: str) -> dict[str, float]:
    if not isinstance(v, dict):
        raise ValueError(  # noqa: TRY004 - pydantic maps ValueError to a retryable schema failure
            f"{what} must be a JSON object of name -> number"
        )
    out: dict[str, float] = {}
    for key, val in v.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(  # noqa: TRY004 - retryable schema failure, not a hard error
                f"{what} value {key!r} must be a bare number (got {type(val).__name__} "
                f"{val!r}); the compiler agent chooses parameters, not expressions"
            )
        f = float(val)
        if not math.isfinite(f):
            raise ValueError(f"{what} value {key!r} must be finite")
        out[str(key)] = f
    return out


class TemplatePlan(_Closed):
    """A Phase 11 convenience family + numeric parameters."""

    family_key: str
    root_symbol: str
    params: dict[str, float] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    @classmethod
    def _numeric(cls, v: object) -> dict[str, float]:
        return _scalar_number_dict(v, what="template params")


class BlueprintFeature(_Closed):
    """One declared feature: a registered ``kind`` and scalar parameters. The
    alias namespace is closed -- a condition can only reference a declared
    alias, never an arbitrary column."""

    alias: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=64)
    kind: str = Field(min_length=1, max_length=64)
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)
    price_field: str | None = Field(default=None, max_length=32)

    @field_validator("params", mode="before")
    @classmethod
    def _scalar_params(cls, v: object) -> dict:
        if not isinstance(v, dict):
            raise ValueError(  # noqa: TRY004 - retryable schema failure
                "feature params must be a JSON object of name -> scalar"
            )
        out: dict = {}
        for key, val in v.items():
            if isinstance(val, bool):
                out[str(key)] = val
            elif isinstance(val, float) and val.is_integer():
                out[str(key)] = int(val)
            elif isinstance(val, (int, float, str)):
                out[str(key)] = val
            else:
                raise ValueError(
                    f"feature param {key!r} must be a scalar number / string / bool, "
                    f"got {type(val).__name__}"
                )
        return out


class BlueprintOperand(_Closed):
    """A comparison operand: the current value of a declared feature
    (``feature``), its value ``periods`` bars ago (``lag``, strictly causal
    ``periods >= 1``), or a finite numeric literal (``const``)."""

    type: Literal["feature", "lag", "const"]
    feature: str | None = Field(default=None, max_length=64)
    periods: int | None = Field(default=None, ge=1, le=100_000)
    value: float | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _finite_number(cls, v: object) -> object:
        if v is None:
            return v
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(  # noqa: TRY004 - retryable schema failure
                "const operand 'value' must be a bare finite number"
            )
        if not math.isfinite(float(v)):
            raise ValueError("const operand 'value' must be finite (no NaN / inf)")
        return v

    @model_validator(mode="after")
    def _shape(self) -> BlueprintOperand:
        if self.type in ("feature", "lag"):
            if not self.feature:
                raise ValueError(f"{self.type} operand requires 'feature' (a declared alias)")
            if self.value is not None:
                raise ValueError(f"{self.type} operand must not set 'value'")
        if self.type == "lag" and self.periods is None:
            raise ValueError("lag operand requires 'periods' (>= 1)")
        if self.type != "lag" and self.periods is not None:
            raise ValueError("only a lag operand may set 'periods'")
        if self.type == "const":
            if self.value is None:
                raise ValueError("const operand requires 'value'")
            if self.feature is not None:
                raise ValueError("const operand must not set 'feature'")
        return self


class BlueprintComparison(_Closed):
    type: Literal["comparison"] = "comparison"
    op: Literal["gt", "gte", "lt", "lte"]
    left: BlueprintOperand
    right: BlueprintOperand


class BlueprintNot(_Closed):
    type: Literal["not"] = "not"
    node: BlueprintCondition


class BlueprintBoolean(_Closed):
    type: Literal["boolean"] = "boolean"
    op: Literal["all", "any"]
    nodes: list[BlueprintCondition] = Field(min_length=1, max_length=32)


BlueprintCondition = Annotated[
    BlueprintComparison | BlueprintBoolean | BlueprintNot,
    Field(discriminator="type"),
]

BlueprintNot.model_rebuild()
BlueprintBoolean.model_rebuild()


class BlueprintRule(_Closed):
    rule_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    when: BlueprintCondition
    target_units: int = Field(ge=-_SCHEMA_MAX_TARGET_UNITS, le=_SCHEMA_MAX_TARGET_UNITS)

    @field_validator("target_units", mode="before")
    @classmethod
    def _strict_int(cls, v: object) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(  # noqa: TRY004 - retryable schema failure
                "target_units must be a plain integer (no 1.0 / '1' / true coercion)"
            )
        return v


class StrategyBlueprint(_Closed):
    """A full composition of the frozen Phase 10 closed Strategy DSL. Translated
    deterministically, node by node, into a :class:`StrategySpec`.

    ``signal_cadence`` is the blueprint's OWN declared execution cadence -- a
    real, typed, pre-run quantity (not the placeholder string this used to
    fall back to for every non-template build). ``daily_trading_day`` means
    the strategy decides once per trading day off the causal daily aggregation
    of the 1-minute forward-adjusted continuous (matching the four legacy
    baseline families); ``native_1m`` means it decides on the native 1-minute
    signal (matching Silver Bullet). Both cadences always EXECUTE fills on the
    native 1-minute raw contract (``execution_cadence``, unchanged, fixed by
    the architecture) -- cadence only changes how often/at what granularity
    the strategy is allowed to look at the world and decide, never how a fill
    is priced. Defaults to ``daily_trading_day`` (the common case, and every
    pre-existing blueprint fixture's intended cadence) -- a real agent SHOULD
    still declare it explicitly for a mechanism that genuinely needs
    intraday/session timing (opening range, overnight gap, etc.).
    """

    root_symbol: str = Field(pattern=r"^[A-Z0-9]{1,12}$")
    features: list[BlueprintFeature] = Field(min_length=1, max_length=32)
    rules: list[BlueprintRule] = Field(min_length=1, max_length=64)
    default_action: Literal["flat", "keep_previous_target"]
    on_missing: Literal["skip_rule", "hold"] = "skip_rule"
    signal_cadence: BlueprintSignalCadence = "daily_trading_day"

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )


class StrategyPlanRequest(_Closed):
    """The ONLY thing the LLM returns. A closed choice, not a program.

    ``expressible=True``  -> exactly one of ``template`` or ``blueprint``.
    ``expressible=False`` -> ``not_expressible_reason`` explaining why no
    registered feature + closed-DSL composition can represent the hypothesis.
    That is a legitimate typed outcome -- inventing a strategy is forbidden.
    """

    expressible: bool = True
    template: TemplatePlan | None = None
    blueprint: StrategyBlueprint | None = None
    not_expressible_reason: str = ""
    rationale: str = ""

    @model_validator(mode="after")
    def _shape(self) -> StrategyPlanRequest:
        if self.expressible:
            n = (self.template is not None) + (self.blueprint is not None)
            if n != 1:
                raise ValueError(
                    "an expressible plan requires exactly one of 'template' or 'blueprint'"
                )
        else:
            if not self.not_expressible_reason.strip():
                raise ValueError(
                    "not_expressible_reason is required when expressible is false"
                )
            if self.template is not None or self.blueprint is not None:
                raise ValueError(
                    "a non-expressible plan must not carry a template or blueprint"
                )
        return self


# -- proposal ----------------------------------------------------------------


class FeatureCoverage(BaseModel):
    """HypothesisSpec fidelity: the set of feature KINDS in the compiled spec
    must equal ``hypothesis.required_features``.

    * ``missing`` -- a required kind not referenced by any rule (a required kind
      is only "represented" when a rule actually reads it).
    * ``unapproved`` -- a feature kind present in the compiled spec that the
      hypothesis did not approve. Multiple :class:`FeatureSpec` s of the same
      approved kind with different parameters, lags, constants and boolean
      composition are all fine -- only a *new kind* is a fidelity violation.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    required: tuple[str, ...]
    referenced_kinds: tuple[str, ...]
    declared_kinds: tuple[str, ...]
    represented: tuple[str, ...]
    missing: tuple[str, ...]
    unapproved: tuple[str, ...]
    used_canonical_by_kind: dict[str, str]


class CompiledStrategyProposal(BaseModel):
    """The outcome of one compile round. Either an accepted, schema-valid
    :class:`StrategySpec` or a typed rejection -- never a raw model string."""

    model_config = {"frozen": True, "extra": "forbid"}

    accepted: bool
    build_mode: str | None = None  # "template" | "blueprint"
    strategy_spec: StrategySpec | None = None
    strategy_fingerprint: str | None = None
    family_key: str | None = None
    root_symbol: str | None = None
    resolved_params: dict[str, float] = Field(default_factory=dict)
    blueprint_feature_kinds: tuple[str, ...] = ()
    plan_summary: dict | None = None
    execution_semantics: ExecutionSemantics | None = None
    feature_coverage: FeatureCoverage | None = None
    duplicate_evidence: DuplicateEvidence | None = None

    rejection_code: str | None = None
    rejection_detail: str | None = None
    compile_diagnostics: tuple[str, ...] = ()

    prompt_log: PromptLog
    attempts: tuple[AttemptRecord, ...]

    @property
    def is_duplicate(self) -> bool:
        return bool(
            self.duplicate_evidence and self.duplicate_evidence.strategy_fingerprint_seen
        )


_OUTPUT_CONTRACT = """
## Output contract (Phase 17.1, enforced deterministically downstream)

Return ONLY a single JSON object -- no prose, no code, no markdown fence
required. It must validate against this JSON schema:

{schema}

Hard rules:
- `expressible: false` + `not_expressible_reason` when no registered feature
  plus a closed Strategy-DSL composition can faithfully represent the hypothesis
  mechanism. Never approximate it with an unrelated strategy.
- Otherwise return EXACTLY ONE of:
  - `template`: a Phase 11 convenience family (`family_key`) + `root_symbol` +
    numeric `params`, each inside the bound given in the family catalog;
  - `blueprint`: declared `features` (each a registered feature `kind` from the
    catalog + scalar params only), `rules` whose `when` is a condition tree
    of `comparison` (op in gt/gte/lt/lte), `boolean` (op in all/any) and `not`
    nodes over `feature` / `lag` (periods >= 1) / `const` operands, with an
    integer `target_units`, and a declared `signal_cadence`
    (`daily_trading_day` or `native_1m` -- pick `native_1m` only when the
    mechanism genuinely needs intraday/session timing; otherwise
    `daily_trading_day`).
- You never emit code, an expression, a callable, an import, a new operator, a
  new feature kind, or any execution-plane concept (fill / price / slippage /
  commission / latency / margin / risk / PnL). Entry timing, exit timing, costs
  and the same-bar-fill prohibition are fixed by the platform.
- `root_symbol` must be in the approved universe AND the hypothesis `universe`.
- Every feature `kind` named in `hypothesis.required_features` MUST be referenced
  by at least one rule in your strategy -- declaring it is not enough.
- Never reference any date on or after 2025-01-01.
"""


class StrategyCompilerAgent:
    """LLM-backed hypothesis -> StrategySpec compiler. The LLM fills a closed
    blueprint; deterministic code builds the spec and the unchanged Phase 10
    compiler is the safety gate."""

    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        model: str = DEFAULT_MODEL,
        max_schema_retries: int = 2,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        token_budget: int | None = 40_000,
        log_sink: Callable[[PromptLog], None] | None = None,
    ):
        if max_schema_retries < 0:
            raise ValueError("max_schema_retries must be >= 0")
        self._client = client
        self._prompt_path = Path(prompt_path)
        self._prompt_template = self._prompt_path.read_text(encoding="utf-8")
        self._model = model
        self._max_schema_retries = max_schema_retries
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._token_budget = token_budget
        self._log_sink = log_sink

        schema_json = json.dumps(
            StrategyPlanRequest.model_json_schema(), indent=2, sort_keys=True
        )
        self._system_prompt = (
            self._prompt_template.rstrip()
            + "\n"
            + _OUTPUT_CONTRACT.format(schema=schema_json)
        )
        self._prompt_version = _sha256(self._prompt_template)[:16]

    # -- public API --------------------------------------------------------

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def compile_hypothesis(
        self, hypothesis: HypothesisSpec, context: CompilerContext
    ) -> CompiledStrategyProposal:
        if not isinstance(hypothesis, HypothesisSpec):
            raise TypeError("hypothesis must be a HypothesisSpec")
        if not isinstance(context, CompilerContext):
            raise TypeError("context must be a CompilerContext")

        # The hypothesis is an INPUT here; re-assert the locked-holdout guard on
        # it (fail loud, never a soft rejection).
        assert_no_holdout_market_data(
            hypothesis.model_dump(mode="json"), path="$.hypothesis"
        )

        system = self._system_prompt
        messages: list[dict] = [
            {"role": "user", "content": self._user_message(hypothesis, context)}
        ]
        attempts: list[AttemptRecord] = []
        total_in = 0
        total_out = 0

        for i in range(self._max_schema_retries + 1):
            if self._token_budget is not None and (total_in + total_out) >= self._token_budget:
                raise CompilerBudgetExceeded(
                    f"token budget {self._token_budget} reached after {i} attempt(s)"
                )

            resp: LLMResponse = self._client.complete(
                system=system,
                messages=messages,
                model=self._model,
                max_tokens=self._max_output_tokens,
                temperature=self._temperature,
            )
            total_in += resp.input_tokens
            total_out += resp.output_tokens

            error: str | None = None
            plan: StrategyPlanRequest | None = None
            try:
                payload = _extract_json(resp.text)
                plan = StrategyPlanRequest.model_validate(payload)
            except (ValueError, ValidationError) as exc:  # JSON or schema failure only
                error = f"{type(exc).__name__}: {exc}"

            attempts.append(
                AttemptRecord(
                    index=i,
                    raw_text=resp.text,
                    parsed_ok=plan is not None,
                    error=error,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    stop_reason=resp.stop_reason,
                )
            )

            if plan is None:
                if i >= self._max_schema_retries:
                    self._emit_log(context, hypothesis, attempts, total_in, total_out)
                    raise CompilerSchemaRetryExhausted(tuple(attempts))
                messages.append({"role": "assistant", "content": resp.text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That did not validate against the StrategyPlanRequest schema:\n"
                            f"{error}\n"
                            "Return only a single corrected JSON object."
                        ),
                    }
                )
                continue

            prompt_log = self._emit_log(context, hypothesis, attempts, total_in, total_out)
            return self._finalize(plan, hypothesis, context, prompt_log, tuple(attempts))

        raise StrategyCompilerAgentError(  # pragma: no cover - loop returns or raises
            "compile loop exited without a result"
        )

    # -- internals --------------------------------------------------------

    def _user_message(self, hypothesis: HypothesisSpec, context: CompilerContext) -> str:
        payload = {
            "approved_hypothesis": hypothesis.model_dump(mode="json"),
            "approved_market_universe": list(context.approved_universe),
            "feature_catalog": [e.model_dump(mode="json") for e in context.feature_catalog],
            "strategy_family_catalog": [
                c.model_dump(mode="json") for c in context.family_catalog
            ],
            "compiler_limits": dict(context.compiler_limits),
        }
        return (
            "Compile this approved hypothesis into ONE StrategyPlanRequest JSON "
            "object (or a typed non-expressible result).\n\n"
            + json.dumps(payload, indent=2, sort_keys=True)
        )

    def _finalize(
        self,
        plan: StrategyPlanRequest,
        hypothesis: HypothesisSpec,
        context: CompilerContext,
        prompt_log: PromptLog,
        attempts: tuple[AttemptRecord, ...],
    ) -> CompiledStrategyProposal:
        def reject(
            code: str, detail: str, *, diagnostics: Sequence[str] = ()
        ) -> CompiledStrategyProposal:
            return CompiledStrategyProposal(
                accepted=False,
                rejection_code=code,
                rejection_detail=detail,
                compile_diagnostics=tuple(diagnostics),
                prompt_log=prompt_log,
                attempts=attempts,
            )

        if not plan.expressible:
            return reject(HYPOTHESIS_NOT_EXPRESSIBLE, plan.not_expressible_reason.strip())

        # feature availability: every required feature must be a registered kind.
        known = context.feature_kinds
        if not hypothesis.required_features:
            return reject(HYPOTHESIS_FEATURE_UNAVAILABLE, "hypothesis lists no required features")
        missing = sorted(f for f in hypothesis.required_features if f not in known)
        if missing:
            return reject(
                HYPOTHESIS_FEATURE_UNAVAILABLE,
                f"hypothesis requires features not in the registry: {missing}",
            )

        try:
            if plan.template is not None:
                spec, build_mode, family_key, resolved = self._build_from_template(
                    plan.template, hypothesis, context
                )
                blueprint_kinds: tuple[str, ...] = ()
            else:
                assert plan.blueprint is not None
                spec, build_mode, family_key, resolved = self._build_from_blueprint(
                    plan.blueprint, hypothesis, context
                )
                blueprint_kinds = tuple(sorted({f.kind for f in plan.blueprint.features}))
        except _BuildRejected as exc:
            return reject(exc.code, exc.detail, diagnostics=exc.diagnostics)

        # locked-holdout guard on the produced spec.
        assert_no_holdout_market_data(spec.model_dump(mode="json"), path="$.strategy_spec")

        # compile with the UNCHANGED Phase 10 compiler -- the real safety gate.
        try:
            compiled = StrategyCompiler(limits=DEFAULT_COMPILE_LIMITS).compile(spec)
        except StrategyCompileError as exc:
            return reject(
                STRATEGY_COMPILE_REJECTED,
                "the Phase 10 StrategyCompiler rejected the spec",
                diagnostics=[str(d) for d in exc.diagnostics],
            )

        # execution-semantics gate.
        safety = compiled.safety
        if not (
            safety.require_signal_safe
            and safety.require_point_in_time_safe
            and safety.forbid_execution_price_use
        ):
            return reject(
                EXECUTION_SEMANTICS_VIOLATION,
                f"compiled plan safety requirements are not all set: {safety.model_dump()}",
            )

        # HypothesisSpec fidelity: the compiled spec's feature KINDS must equal
        # hypothesis.required_features -- no missing required kind, no new kind.
        coverage = _required_feature_coverage(hypothesis, compiled)
        if coverage.missing:
            return reject(
                REQUIRED_FEATURE_NOT_REPRESENTED,
                "hypothesis required feature(s) not referenced by any rule in the "
                f"compiled spec: {list(coverage.missing)}",
            )
        if coverage.unapproved:
            return reject(
                UNAPPROVED_FEATURE_ADDED,
                "the compiled spec introduces feature kind(s) the hypothesis did not "
                f"approve in required_features: {list(coverage.unapproved)}. The "
                "ResearchAgent decides what mechanism is tested; the CompilerAgent "
                "only translates the approved hypothesis.",
            )

        fingerprint = strategy_fingerprint(spec)
        # determinism re-check: a second build must fingerprint identically.
        try:
            if plan.template is not None:
                spec2, *_ = self._build_from_template(plan.template, hypothesis, context)
            else:
                spec2, *_ = self._build_from_blueprint(plan.blueprint, hypothesis, context)
        except _BuildRejected:  # pragma: no cover - first build already succeeded
            spec2 = spec
        if strategy_fingerprint(spec2) != fingerprint:  # pragma: no cover - non-determinism
            return reject(
                EXECUTION_SEMANTICS_VIOLATION,
                "the spec build is not deterministic for this plan",
            )

        evidence = _duplicate_evidence(fingerprint, context)
        plan_summary = {
            "dsl_version": compiled.dsl_version,
            "schema_version": compiled.schema_version,
            "strategy_id": compiled.strategy_id,
            "strategy_name": compiled.strategy_name,
            "root_symbol": compiled.root_symbol,
            "fingerprint": compiled.fingerprint,
            "warmup_bars": compiled.warmup_bars,
            "declared_features": list(compiled.declared_features),
            "required_features": list(compiled.required_features),
            "n_rules": len(compiled.rules),
            "default_action": compiled.default_action.value,
            "on_missing": compiled.on_missing.value,
            "safety": safety.model_dump(),
        }
        cadence = (
            SIGNAL_CADENCE[family_key]
            if build_mode == "template" and family_key in SIGNAL_CADENCE
            else plan.blueprint.signal_cadence  # type: ignore[union-attr]
        )
        execution_semantics = _execution_semantics(
            signal_cadence=cadence, default_action=compiled.default_action.value
        )

        # Strategy-fingerprint evidence NEVER rejects compilation: Phase 17 does
        # not know the complete scientific experiment identity. The successfully
        # compiled spec is returned as accepted with DuplicateEvidence attached;
        # Phase 18 builds the full identity and queries the schema-v5 registry to
        # decide re-execution.
        return CompiledStrategyProposal(
            accepted=True,
            build_mode=build_mode,
            strategy_spec=spec,
            strategy_fingerprint=fingerprint,
            family_key=family_key,
            root_symbol=spec.root_symbol,
            resolved_params=resolved,
            blueprint_feature_kinds=blueprint_kinds,
            plan_summary=plan_summary,
            execution_semantics=execution_semantics,
            feature_coverage=coverage,
            duplicate_evidence=evidence,
            prompt_log=prompt_log,
            attempts=attempts,
        )

    # -- spec builders ----------------------------------------------------

    @staticmethod
    def _check_root(root: str, hypothesis: HypothesisSpec, context: CompilerContext) -> None:
        if root not in context.approved_universe:
            raise _BuildRejected(
                MARKET_OUTSIDE_UNIVERSE,
                f"root_symbol {root!r} is not in the approved universe "
                f"{sorted(context.approved_universe)}",
            )
        if root not in set(hypothesis.universe):
            raise _BuildRejected(
                ROOT_OUTSIDE_HYPOTHESIS_UNIVERSE,
                f"root_symbol {root!r} is not in the approved hypothesis universe "
                f"{sorted(hypothesis.universe)}",
            )

    def _build_from_template(
        self, tpl: TemplatePlan, hypothesis: HypothesisSpec, context: CompilerContext
    ) -> tuple[StrategySpec, str, str, dict[str, float]]:
        cards = {c.family_key: c for c in context.family_catalog}
        card = cards.get(tpl.family_key)
        if card is None:
            raise _BuildRejected(
                UNKNOWN_STRATEGY_FAMILY,
                f"family_key {tpl.family_key!r} is not a registered template family; "
                f"allowed: {sorted(cards)}",
            )
        self._check_root(tpl.root_symbol, hypothesis, context)

        unknown = sorted(set(tpl.params) - set(card.parameters))
        if unknown:
            raise _BuildRejected(
                UNKNOWN_PARAMETER,
                f"params {unknown} are not parameters of family {card.family_key!r} "
                f"(allowed: {list(card.parameters)})",
            )
        oob: list[str] = []
        for pname, value in tpl.params.items():
            bound = card.param_bounds[pname]
            if not bound.contains(value):
                oob.append(f"{pname}={value:g} outside [{bound.lo:g}, {bound.hi:g}]")
        if oob:
            raise _BuildRejected(
                PARAMETER_OUT_OF_BOUNDS,
                "parameter(s) outside the declared bound: " + "; ".join(oob),
            )
        coerced: dict[str, object] = {}
        non_int: list[str] = []
        for pname, value in tpl.params.items():
            bound = card.param_bounds[pname]
            if bound.integer:
                if not float(value).is_integer():
                    non_int.append(f"{pname}={value:g}")
                else:
                    coerced[pname] = int(value)
            else:
                coerced[pname] = float(value)
        if non_int:
            raise _BuildRejected(
                NON_INTEGER_PARAMETER,
                f"integer parameter(s) given a fractional value: {non_int}",
            )

        try:
            spec = spec_for_params(
                card.family_key, {"root_symbol": tpl.root_symbol, **coerced}
            )
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            raise _BuildRejected(
                INVALID_STRATEGY_PARAMETERS,
                f"the frozen {card.family_key!r} factory rejected the parameters: {exc}",
            ) from exc
        return spec, "template", card.family_key, dict(tpl.params)

    def _build_from_blueprint(
        self,
        blueprint: StrategyBlueprint,
        hypothesis: HypothesisSpec,
        context: CompilerContext,
    ) -> tuple[StrategySpec, str, None, dict[str, float]]:
        self._check_root(blueprint.root_symbol, hypothesis, context)

        known = context.feature_kinds
        bad_kinds = sorted({f.kind for f in blueprint.features} - known)
        if bad_kinds:
            raise _BuildRejected(
                UNKNOWN_FEATURE_KIND,
                f"blueprint features name kinds not in the registry catalog: {bad_kinds}",
            )
        aliases = [f.alias for f in blueprint.features]
        if len(set(aliases)) != len(aliases):
            raise _BuildRejected(INVALID_BLUEPRINT, "duplicate feature alias in the blueprint")
        declared = set(aliases)
        referenced: set[str] = set()
        for rule in blueprint.rules:
            referenced |= set(_blueprint_condition_aliases(rule.when))
        undeclared = sorted(referenced - declared)
        if undeclared:
            raise _BuildRejected(
                INVALID_BLUEPRINT,
                f"blueprint rules reference undeclared feature aliases: {undeclared}",
            )

        canonical = blueprint.canonical_json()
        strategy_id = f"CMP-{blueprint.root_symbol}-{_sha256(canonical)[:16]}"
        try:
            feature_decls = [
                FeatureDeclaration(
                    alias=f.alias,
                    spec=FeatureSpec(
                        kind=f.kind,
                        params=dict(f.params),
                        **({"price_field": f.price_field} if f.price_field else {}),
                    ),
                )
                for f in blueprint.features
            ]
            rules = [
                Rule(
                    rule_id=rule.rule_id,
                    when=_translate_condition(rule.when),
                    action=TargetAction(target_units=rule.target_units),
                )
                for rule in blueprint.rules
            ]
            spec = StrategySpec(
                strategy_name=f"compiled DSL strategy ({blueprint.root_symbol})",
                strategy_id=strategy_id,
                root_symbol=blueprint.root_symbol,
                features=feature_decls,
                rules=rules,
                default_action=DefaultAction(blueprint.default_action),
                on_missing=MissingRulePolicy(blueprint.on_missing),
                rationale=(
                    "Phase 17.1 -- deterministically compiled from a closed "
                    "StrategyBlueprint. Not claimed alpha."
                ),
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise _BuildRejected(
                INVALID_BLUEPRINT,
                f"the blueprint could not be assembled into a StrategySpec: {exc}",
            ) from exc
        return spec, "blueprint", None, {}

    def _emit_log(
        self,
        context: CompilerContext,
        hypothesis: HypothesisSpec,
        attempts: list[AttemptRecord],
        total_in: int,
        total_out: int,
    ) -> PromptLog:
        log = PromptLog(
            prompt_version=self._prompt_version,
            prompt_path=str(self._prompt_path),
            system_prompt_sha256=_sha256(self._system_prompt),
            context_sha256=context.sha256_with_hypothesis(hypothesis),
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            n_attempts=len(attempts),
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )
        if self._log_sink is not None:
            self._log_sink(log)
        return log


# -- blueprint -> DSL translation --------------------------------------


def _blueprint_condition_aliases(node: object) -> Iterator[str]:
    if isinstance(node, BlueprintComparison):
        for operand in (node.left, node.right):
            if operand.type in ("feature", "lag") and operand.feature:
                yield operand.feature
    elif isinstance(node, BlueprintNot):
        yield from _blueprint_condition_aliases(node.node)
    elif isinstance(node, BlueprintBoolean):
        for child in node.nodes:
            yield from _blueprint_condition_aliases(child)


def _translate_operand(operand: BlueprintOperand):
    if operand.type == "feature":
        return FeatureOperand(feature=operand.feature)
    if operand.type == "lag":
        return LagOperand(feature=operand.feature, periods=operand.periods)
    return ConstOperand(value=operand.value)


def _translate_condition(node: object):
    if isinstance(node, BlueprintComparison):
        return ComparisonNode(
            op=Comparator(node.op),
            left=_translate_operand(node.left),
            right=_translate_operand(node.right),
        )
    if isinstance(node, BlueprintNot):
        return NotNode(node=_translate_condition(node.node))
    if isinstance(node, BlueprintBoolean):
        return BooleanNode(
            op=BooleanOp(node.op),
            nodes=[_translate_condition(child) for child in node.nodes],
        )
    raise TypeError(type(node))  # pragma: no cover


def _required_feature_coverage(hypothesis: HypothesisSpec, compiled) -> FeatureCoverage:
    """HypothesisSpec fidelity check.

    A required kind is *referenced* only when a rule actually reads a
    :class:`FeatureSpec` of that kind. ``declared_kinds`` covers every feature in
    the compiled spec (declared features enter the strategy fingerprint even when
    unused), so a declared-but-unused new kind is still a fidelity violation.
    Only the KIND matters -- several ``FeatureSpec`` s of the same approved kind
    with different parameters are fine.
    """
    required = list(hypothesis.required_features)
    required_set = set(required)
    used_canonical = set(compiled.required_features)

    used_by_kind: dict[str, str] = {}
    referenced_kinds: set[str] = set()
    declared_kinds: set[str] = set()
    for binding in compiled.feature_bindings:
        declared_kinds.add(binding.spec.kind)
        if binding.canonical_name in used_canonical:
            referenced_kinds.add(binding.spec.kind)
            used_by_kind.setdefault(binding.spec.kind, binding.canonical_name)

    represented = tuple(k for k in required if k in referenced_kinds)
    missing = tuple(k for k in required if k not in referenced_kinds)
    unapproved = tuple(sorted(declared_kinds - required_set))
    return FeatureCoverage(
        required=tuple(required),
        referenced_kinds=tuple(sorted(referenced_kinds)),
        declared_kinds=tuple(sorted(declared_kinds)),
        represented=represented,
        missing=missing,
        unapproved=unapproved,
        used_canonical_by_kind=dict(used_by_kind),
    )


# -- compiler context -----------------------------------------------------


class CompilerContext(BaseModel):
    """The typed, closed input bundle a :class:`StrategyCompilerAgent` is allowed
    to see: the approved universe, the registered feature catalog, the
    convenience-template catalog with typed parameter bounds, the compiler
    limits, and the known strategies (with any authority the caller supplies)
    for de-duplication evidence.

    Anything not in this model, the agent does not see: no filesystem, no
    registry handle, no market data, no holdout, no ability to run a backtest.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "compiler-context/2"
    approved_universe: tuple[str, ...]
    feature_catalog: tuple[FeatureCatalogEntry, ...]
    family_catalog: tuple[StrategyFamilyCard, ...]
    compiler_limits: dict[str, int] = Field(default_factory=dict)
    known_strategies: tuple[KnownStrategyRecord, ...] = ()

    @model_validator(mode="after")
    def _guard(self) -> CompilerContext:
        if not self.approved_universe:
            raise ValueError("approved_universe must be non-empty")
        if not self.family_catalog:
            raise ValueError("family_catalog must be non-empty")
        if not self.feature_catalog:
            raise ValueError("feature_catalog must be non-empty")
        assert_no_holdout_market_data(
            self.model_dump(mode="json"), path="$.compiler_context"
        )
        return self

    @property
    def feature_kinds(self) -> frozenset[str]:
        return frozenset(e.kind for e in self.feature_catalog)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    def sha256(self) -> str:
        return _sha256(self.canonical_json())

    def sha256_with_hypothesis(self, hypothesis: HypothesisSpec) -> str:
        blob = self.canonical_json() + "\n" + json.dumps(
            hypothesis.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return _sha256(blob)


def build_compiler_context(
    *,
    approved_universe: Iterable[str],
    feature_registry: object | None = None,
    known_strategies: Sequence[KnownStrategyRecord] = (),
    include_frozen_templates: bool = True,
) -> CompilerContext:
    """Assemble a :class:`CompilerContext` from the deterministic layers.

    ``known_strategies`` lets the orchestrator (Phase 18) fold in strategy
    fingerprints from the live experiment registry **with their authority**
    (valid authoritative result vs INVALID_EXECUTION-only history), without the
    agent ever holding a registry handle.
    """
    universe = tuple(approved_universe)
    supplied = list(known_strategies)
    # Fail loud on a locked-holdout value in any raw input before pydantic can
    # wrap it in a ValidationError.
    assert_no_holdout_market_data(
        {
            "approved_universe": list(universe),
            "known_strategies": [r.model_dump(mode="json") for r in supplied],
        },
        path="$.compiler_context_inputs",
    )
    records: list[KnownStrategyRecord] = []
    if include_frozen_templates:
        records.extend(frozen_template_strategy_records())
    records.extend(supplied)
    return CompilerContext(
        approved_universe=universe,
        feature_catalog=build_feature_catalog(feature_registry),
        family_catalog=build_family_catalog(),
        compiler_limits=DEFAULT_COMPILE_LIMITS.model_dump(),
        known_strategies=tuple(records),
    )

"""``StrategySpec`` -> :class:`CompiledStrategyPlan`.

The compiler is the single gate between an Agent-authored spec and anything that
can act on it. It:

* parses the spec with strict typed schemas (unknown / illegal fields, unknown
  operators/actions, NaN/inf constants, non-integral target units, future
  temporal references are all rejected here);
* runs ``REGISTRY.validate_spec`` on every referenced :class:`FeatureSpec`
  (unknown kind / unknown or missing or mistyped parameter / out-of-range /
  cross-parameter constraint -- Phase 09 section 6);
* enforces the **feature signal-safety gate** (section 5): a feature may drive a
  Signal only if it is point-in-time & signal safe (a retrospective future-roll
  feature is rejected; the retrospective back-adjusted *source* case is caught at
  evaluation time by ``FeatureFrame.assert_signal_safe``);
* resolves feature aliases deterministically and rejects unknown references,
  duplicate aliases, canonical-name collisions and duplicate rule ids;
* bounds condition-tree depth / node count to stop pathological specs;
* emits a closed typed IR with a deterministic fingerprint. No source code is
  generated.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from alpha_agent.features.registry import REGISTRY, FeatureRegistry
from alpha_agent.strategy import errors as E
from alpha_agent.strategy.enums import NodeType
from alpha_agent.strategy.errors import (
    StrategyCompileError,
    StrategyDiagnostic,
)
from alpha_agent.strategy.fingerprint import (
    STRATEGY_DSL_VERSION,
    canonical_strategy_payload,
    strategy_fingerprint,
)
from alpha_agent.strategy.plan import (
    CompiledCondition,
    CompiledOperand,
    CompiledRule,
    CompiledStrategyPlan,
    FeatureBinding,
    SafetyRequirements,
)
from alpha_agent.strategy.spec import (
    BooleanNode,
    ComparisonNode,
    ConstOperand,
    FeatureOperand,
    LagOperand,
    NotNode,
    StrategySpec,
)


class CompileLimits(BaseModel):
    """Configurable ceilings that keep Agent-generated specs bounded (section 8)."""

    model_config = {"frozen": True}

    max_condition_depth: int = Field(default=8, ge=1, le=64)
    max_condition_nodes: int = Field(default=64, ge=1, le=4096)
    max_rules: int = Field(default=64, ge=1, le=4096)
    max_features: int = Field(default=32, ge=1, le=1024)
    max_abs_target_units: int = Field(default=10, ge=1, le=1000)


DEFAULT_COMPILE_LIMITS = CompileLimits()


# --------------------------------------------------------------------------
# tree helpers
# --------------------------------------------------------------------------
def _iter_operands(node):
    if isinstance(node, ComparisonNode):
        yield node.left
        yield node.right
    elif isinstance(node, NotNode):
        yield from _iter_operands(node.node)
    elif isinstance(node, BooleanNode):
        for c in node.nodes:
            yield from _iter_operands(c)


def _depth(node) -> int:
    if isinstance(node, ComparisonNode):
        return 1
    if isinstance(node, NotNode):
        return 1 + _depth(node.node)
    if isinstance(node, BooleanNode):
        return 1 + max(_depth(c) for c in node.nodes)
    raise TypeError  # pragma: no cover


def _count(node) -> int:
    if isinstance(node, ComparisonNode):
        return 1
    if isinstance(node, NotNode):
        return 1 + _count(node.node)
    if isinstance(node, BooleanNode):
        return 1 + sum(_count(c) for c in node.nodes)
    raise TypeError  # pragma: no cover


def _diag_from_validation_error(exc: ValidationError) -> list[StrategyDiagnostic]:
    out: list[StrategyDiagnostic] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        etype = str(err["type"])
        msg = str(err["msg"])
        if etype == "extra_forbidden":
            code = E.ILLEGAL_DSL_FIELD
        elif etype in ("literal_error", "enum"):
            code = E.UNKNOWN_ACTION if "action" in loc or "default" in loc else E.UNKNOWN_OPERATOR
        elif etype.startswith("union_tag"):
            code = E.UNKNOWN_OPERATOR
        elif "value" in err["loc"] and (
            "const" in err["loc"] or "finite" in msg or "NaN" in msg or "inf" in msg
        ):
            code = E.INVALID_CONSTANT
        elif "target_units" in loc:
            code = E.INVALID_TARGET_UNITS
        elif "periods" in loc:
            code = E.FUTURE_TEMPORAL_REFERENCE
        elif "execution-plane field" in msg:
            code = E.ILLEGAL_DSL_FIELD
        else:
            code = E.MALFORMED_CONDITION
        out.append(StrategyDiagnostic(code=code, message=f"{etype}: {msg}", location=loc))
    return out


# --------------------------------------------------------------------------
# compiler
# --------------------------------------------------------------------------
class StrategyCompiler:
    def __init__(
        self,
        registry: FeatureRegistry = REGISTRY,
        limits: CompileLimits = DEFAULT_COMPILE_LIMITS,
    ) -> None:
        self.registry = registry
        self.limits = limits

    # -- public ----------------------------------------------------------
    def compile(self, spec: StrategySpec | dict | str | bytes) -> CompiledStrategyPlan:
        parsed = self._parse(spec)
        diags: list[StrategyDiagnostic] = []

        if len(parsed.rules) > self.limits.max_rules:
            diags.append(
                StrategyDiagnostic(
                    code=E.TOO_MANY_RULES,
                    message=f"{len(parsed.rules)} rules > limit {self.limits.max_rules}",
                    location="rules",
                )
            )
        if len(parsed.features) > self.limits.max_features:
            diags.append(
                StrategyDiagnostic(
                    code=E.TOO_MANY_FEATURES,
                    message=f"{len(parsed.features)} features > limit {self.limits.max_features}",
                    location="features",
                )
            )

        alias_to_canonical, alias_to_meta = self._validate_features(parsed, diags)
        self._check_references(parsed, set(alias_to_canonical), diags)
        self._check_condition_shape(parsed, diags)
        self._check_rules(parsed, diags)

        if diags:
            raise StrategyCompileError(diags)

        return self._build_plan(parsed, alias_to_canonical, alias_to_meta)

    __call__ = compile

    # -- steps ---------------------------------------------------------------
    def _parse(self, spec: StrategySpec | dict | str | bytes) -> StrategySpec:
        if isinstance(spec, StrategySpec):
            return spec
        try:
            if isinstance(spec, (str, bytes)):
                import json

                return StrategySpec.model_validate(json.loads(spec))
            return StrategySpec.model_validate(spec)
        except ValidationError as exc:
            raise StrategyCompileError(_diag_from_validation_error(exc)) from exc

    def _validate_features(
        self, spec: StrategySpec, diags: list[StrategyDiagnostic]
    ) -> tuple[dict[str, str], dict[str, object]]:
        alias_to_canonical: dict[str, str] = {}
        alias_to_meta: dict[str, object] = {}
        canonical_owner: dict[str, tuple[str, str]] = {}  # name -> (alias, spec_json)

        for decl in spec.features:
            loc = f"features.{decl.alias}"
            if decl.alias in alias_to_canonical:
                diags.append(
                    StrategyDiagnostic(
                        code=E.DUPLICATE_FEATURE_ALIAS,
                        message=f"feature alias {decl.alias!r} declared more than once",
                        location=loc,
                    )
                )
                continue
            try:
                meta = self.registry.validate_spec(decl.spec)
            except KeyError as exc:
                diags.append(
                    StrategyDiagnostic(
                        code=E.UNKNOWN_FEATURE, message=str(exc).strip('"'), location=loc
                    )
                )
                continue
            except (ValueError, TypeError) as exc:
                diags.append(
                    StrategyDiagnostic(
                        code=E.INVALID_FEATURE_PARAMS, message=str(exc), location=loc
                    )
                )
                continue

            if not (meta.signal_safe and meta.point_in_time_safe):
                diags.append(
                    StrategyDiagnostic(
                        code=E.UNSAFE_FEATURE,
                        message=(
                            f"feature {meta.feature_name!r} is not signal-safe "
                            f"(point_in_time_safe={meta.point_in_time_safe}, "
                            f"signal_safe={meta.signal_safe}); a strategy may reference "
                            "only causal / point-in-time features"
                        ),
                        location=loc,
                    )
                )
                continue

            name = meta.feature_name
            if name in canonical_owner and canonical_owner[name][1] != decl.spec.canonical_json():
                diags.append(
                    StrategyDiagnostic(
                        code=E.CANONICAL_NAME_COLLISION,
                        message=(
                            f"aliases {canonical_owner[name][0]!r} and {decl.alias!r} both "
                            f"resolve to canonical feature name {name!r} from different specs"
                        ),
                        location=loc,
                    )
                )
                continue
            canonical_owner.setdefault(name, (decl.alias, decl.spec.canonical_json()))
            alias_to_canonical[decl.alias] = name
            alias_to_meta[decl.alias] = meta

        return alias_to_canonical, alias_to_meta

    def _check_references(
        self, spec: StrategySpec, known_aliases: set[str], diags: list[StrategyDiagnostic]
    ) -> None:
        for rule in spec.rules:
            for operand in _iter_operands(rule.when):
                is_ref = isinstance(operand, (FeatureOperand, LagOperand))
                if is_ref and operand.feature not in known_aliases:
                    diags.append(
                        StrategyDiagnostic(
                            code=E.UNKNOWN_FEATURE_REF,
                            message=(
                                f"rule {rule.rule_id!r} references undeclared feature "
                                f"alias {operand.feature!r}"
                            ),
                            location=f"rules.{rule.rule_id}",
                        )
                    )

    def _check_condition_shape(
        self, spec: StrategySpec, diags: list[StrategyDiagnostic]
    ) -> None:
        for rule in spec.rules:
            d, n = _depth(rule.when), _count(rule.when)
            if d > self.limits.max_condition_depth:
                diags.append(
                    StrategyDiagnostic(
                        code=E.CONDITION_TOO_DEEP,
                        message=f"rule {rule.rule_id!r} condition depth {d} > "
                        f"{self.limits.max_condition_depth}",
                        location=f"rules.{rule.rule_id}",
                    )
                )
            if n > self.limits.max_condition_nodes:
                diags.append(
                    StrategyDiagnostic(
                        code=E.CONDITION_TOO_MANY_NODES,
                        message=f"rule {rule.rule_id!r} condition has {n} nodes > "
                        f"{self.limits.max_condition_nodes}",
                        location=f"rules.{rule.rule_id}",
                    )
                )

    def _check_rules(self, spec: StrategySpec, diags: list[StrategyDiagnostic]) -> None:
        seen: set[str] = set()
        for rule in spec.rules:
            if rule.rule_id in seen:
                diags.append(
                    StrategyDiagnostic(
                        code=E.DUPLICATE_RULE_ID,
                        message=f"rule id {rule.rule_id!r} used more than once",
                        location="rules",
                    )
                )
            seen.add(rule.rule_id)
            if abs(rule.action.target_units) > self.limits.max_abs_target_units:
                diags.append(
                    StrategyDiagnostic(
                        code=E.INVALID_TARGET_UNITS,
                        message=(
                            f"rule {rule.rule_id!r} target_units {rule.action.target_units} "
                            f"exceeds |{self.limits.max_abs_target_units}|"
                        ),
                        location=f"rules.{rule.rule_id}",
                    )
                )

    # -- plan build --------------------------------------------------------
    def _compile_condition(self, node, alias_to_canonical: dict[str, str]) -> CompiledCondition:
        if isinstance(node, ComparisonNode):
            return CompiledCondition(
                node=NodeType.COMPARISON,
                op=node.op,
                left=self._compile_operand(node.left, alias_to_canonical),
                right=self._compile_operand(node.right, alias_to_canonical),
            )
        if isinstance(node, NotNode):
            return CompiledCondition(
                node=NodeType.NOT,
                children=(self._compile_condition(node.node, alias_to_canonical),),
            )
        if isinstance(node, BooleanNode):
            return CompiledCondition(
                node=NodeType.BOOLEAN,
                bool_op=node.op,
                children=tuple(
                    self._compile_condition(c, alias_to_canonical) for c in node.nodes
                ),
            )
        raise TypeError(type(node))  # pragma: no cover

    @staticmethod
    def _compile_operand(operand, alias_to_canonical: dict[str, str]) -> CompiledOperand:
        if isinstance(operand, FeatureOperand):
            return CompiledOperand(
                kind=operand.type, feature=alias_to_canonical[operand.feature]
            )
        if isinstance(operand, LagOperand):
            return CompiledOperand(
                kind=operand.type,
                feature=alias_to_canonical[operand.feature],
                periods=operand.periods,
            )
        if isinstance(operand, ConstOperand):
            return CompiledOperand(kind=operand.type, value=operand.value)
        raise TypeError(type(operand))  # pragma: no cover

    def _build_plan(
        self,
        spec: StrategySpec,
        alias_to_canonical: dict[str, str],
        alias_to_meta: dict[str, object],
    ) -> CompiledStrategyPlan:
        bindings = tuple(
            FeatureBinding(
                alias=decl.alias,
                canonical_name=alias_to_canonical[decl.alias],
                spec=decl.spec,
                metadata=alias_to_meta[decl.alias],
            )
            for decl in sorted(spec.features, key=lambda d: d.alias)
        )

        # referenced features + warmup (feature warmup + any lag offset)
        referenced: set[str] = set()
        warmup = 0
        for rule in spec.rules:
            for operand in _iter_operands(rule.when):
                if isinstance(operand, (FeatureOperand, LagOperand)):
                    canonical = alias_to_canonical[operand.feature]
                    referenced.add(canonical)
                    meta = alias_to_meta[operand.feature]
                    lag = operand.periods if isinstance(operand, LagOperand) else 0
                    warmup = max(warmup, int(meta.minimum_observations) + lag)

        compiled_rules = tuple(
            CompiledRule(
                rule_id=rule.rule_id,
                condition=self._compile_condition(rule.when, alias_to_canonical),
                target_units=rule.action.target_units,
                rationale_label=rule.rationale,
            )
            for rule in spec.rules
        )

        return CompiledStrategyPlan(
            dsl_version=STRATEGY_DSL_VERSION,
            schema_version=spec.schema_version,
            strategy_name=spec.strategy_name,
            strategy_id=spec.strategy_id,
            root_symbol=spec.root_symbol,
            fingerprint=strategy_fingerprint(spec),
            feature_bindings=bindings,
            declared_features=tuple(sorted(alias_to_canonical.values())),
            required_features=tuple(sorted(referenced)),
            rules=compiled_rules,
            default_action=spec.default_action,
            on_missing=spec.on_missing,
            warmup_bars=warmup,
            safety=SafetyRequirements(),
            canonical_payload=canonical_strategy_payload(spec),
        )


def compile_strategy(
    spec: StrategySpec | dict | str | bytes,
    *,
    registry: FeatureRegistry = REGISTRY,
    limits: CompileLimits = DEFAULT_COMPILE_LIMITS,
) -> CompiledStrategyPlan:
    return StrategyCompiler(registry, limits).compile(spec)

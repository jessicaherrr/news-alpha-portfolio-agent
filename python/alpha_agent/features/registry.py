"""Feature registry: typed specs in, computed columns out. No code execution.

    FeatureSpec  ->  FeatureRegistry  ->  FeatureDef.compute  ->  pd.Series

A ``FeatureDef`` is registered *in Python source* via the :func:`feature`
decorator. A :class:`FeatureSpec` can only reference a ``kind`` that is already
registered and pass scalar parameters that pass the def's declared schema.
There is deliberately no way to inject a callable, an expression, or an import
name through the registry.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from alpha_agent.features.enums import FeatureFamily, RollFeatureMode, SessionPolicy
from alpha_agent.features.qa import FeatureQAReport
from alpha_agent.features.safety import SourceSafety, combine
from alpha_agent.features.source import SourceSeries
from alpha_agent.features.spec import FeatureMetadata, FeatureSpec, _fmt_param
from alpha_agent.schemas.market_data import PriceDomain

FEATURE_ENGINE_VERSION = "0.9.2"


@dataclass
class FeatureComputeContext:
    """Everything a compute function is allowed to see. Strictly point-in-time:
    ``frame`` is already truncated to the as-of timestamp by the computer."""

    source: SourceSeries
    frame: pd.DataFrame
    segment_id: pd.Series
    params: dict
    price_field: str
    spec: FeatureSpec
    name: str
    qa: FeatureQAReport
    interval_ns: int
    min_obs: int = 1

    def price(self, field_name: str | None = None) -> pd.Series:
        return self.source.price(field_name or self.price_field)

    def column(self, name: str) -> pd.Series:
        return pd.to_numeric(self.frame[name], errors="coerce").astype("float64")


@dataclass(frozen=True)
class ParamRule:
    kind: type
    min: float | None = None
    max: float | None = None
    required: bool = True
    default: object = None

    def check(self, name: str, value):
        if value is None:
            if self.required:
                raise ValueError(f"missing required param {name!r}")
            return self.default
        if self.kind is int and isinstance(value, bool):
            raise ValueError(f"param {name!r} must be a positive integer, not bool")
        int_for_float = (
            self.kind is float and isinstance(value, int) and not isinstance(value, bool)
        )
        if not isinstance(value, self.kind) and not int_for_float:
            raise ValueError(
                f"param {name!r} must be {self.kind.__name__}, got {type(value).__name__} {value!r}"
            )
        if self.kind in (int, float):
            num = float(value)
            if self.min is not None and num < self.min:
                raise ValueError(f"param {name!r}={value} below minimum {self.min}")
            if self.max is not None and num > self.max:
                raise ValueError(f"param {name!r}={value} above maximum {self.max}")
            return self.kind(value)
        return value


@dataclass(frozen=True)
class FeatureDef:
    kind: str
    family: FeatureFamily
    param_rules: dict[str, ParamRule]
    param_order: tuple[str, ...]
    default_price_domain: tuple[PriceDomain, ...]
    default_session_policy: SessionPolicy
    lookback_fn: Callable[[dict], int]
    compute: Callable[[FeatureComputeContext], pd.Series]
    name_stem: str = ""
    point_in_time_safe: bool = True
    description: str = ""
    # roll features that behave differently in PIT vs retrospective mode
    retrospective_kinds: bool = False
    # cross-parameter contracts: each returns an error message or None
    param_constraints: tuple[Callable[[dict], str | None], ...] = ()

    def stem(self) -> str:
        return self.name_stem or self.kind

    def resolve_params(self, raw: dict) -> dict:
        if not isinstance(raw, dict):
            raise ValueError(  # noqa: TRY004 -- ValueError for parity with pydantic param validation
                f"params for {self.kind!r} must be a dict, got {type(raw).__name__}"
            )
        unknown = set(raw) - set(self.param_rules)
        if unknown:
            raise ValueError(
                f"unknown param(s) for {self.kind!r}: {sorted(unknown)} "
                f"(allowed: {sorted(self.param_rules)})"
            )
        out: dict = {}
        for pname, rule in self.param_rules.items():
            out[pname] = rule.check(pname, raw.get(pname))
        for constraint in self.param_constraints:
            msg = constraint(out)
            if msg:
                raise ValueError(f"invalid params for {self.kind!r}: {msg}")
        return out

    def canonical_name(self, spec: FeatureSpec) -> str:
        params = self.resolve_params(spec.params)
        parts = [self.stem()]
        parts += [_fmt_param(params[p]) for p in self.param_order]
        name = "_".join(parts)
        if spec.price_field != "close":
            name = f"{name}_{spec.price_field}"
        if self.retrospective_kinds and spec.roll_mode is RollFeatureMode.RETROSPECTIVE:
            name = f"{name}_retro"
        return name

    def effective_session_policy(self, spec: FeatureSpec) -> SessionPolicy:
        return spec.session_policy or self.default_session_policy

    def effective_domains(self, spec: FeatureSpec) -> tuple[PriceDomain, ...]:
        return spec.required_price_domain or self.default_price_domain

    def effective_min_obs(self, spec: FeatureSpec) -> int:
        params = self.resolve_params(spec.params)
        lb = int(self.lookback_fn(params))
        if spec.min_observations is None:
            return max(1, lb)
        if spec.min_observations < 1:
            raise ValueError("min_observations must be >= 1")
        return int(spec.min_observations)

    def is_point_in_time_safe(self, spec: FeatureSpec) -> bool:
        if self.retrospective_kinds:
            return spec.roll_mode is not RollFeatureMode.RETROSPECTIVE
        return self.point_in_time_safe

    def safety_reason(self, spec: FeatureSpec) -> str | None:
        if self.is_point_in_time_safe(spec):
            return None
        if self.retrospective_kinds:
            return (
                f"{self.kind} in retrospective roll_mode looks forward to a roll that "
                f"has not happened at T"
            )
        return f"{self.kind} is an inherently retrospective (look-ahead) feature"

    def metadata(
        self, spec: FeatureSpec, *, source_safety: SourceSafety | None = None
    ) -> FeatureMetadata:
        params = self.resolve_params(spec.params)
        feat_pit = self.is_point_in_time_safe(spec)
        if source_safety is None:
            # Registry-level (spec-only) query: the feature's own contribution,
            # assuming a fully point-in-time causal source.
            pit, signal = feat_pit, feat_pit
        else:
            eff = combine(
                source_safety, feature=self.canonical_name(spec),
                feature_point_in_time_safe=feat_pit,
                feature_reason=self.safety_reason(spec),
            )
            pit, signal = eff.point_in_time_safe, eff.signal_safe
        return FeatureMetadata(
            feature_name=self.canonical_name(spec),
            version=FEATURE_ENGINE_VERSION,
            family=self.family.value,
            kind=self.kind,
            required_price_domain=[d.value for d in self.effective_domains(spec)],
            lookback=int(self.lookback_fn(params)),
            minimum_observations=self.effective_min_obs(spec),
            session_policy=self.effective_session_policy(spec).value,
            point_in_time_safe=pit,
            signal_safe=signal,
            # A computed feature value is never an execution price -- always False.
            execution_price_safe=False,
            price_field=spec.price_field,
            parameters=params,
            description=self.description,
        )


class FeatureRegistry:
    def __init__(self) -> None:
        self._defs: dict[str, FeatureDef] = {}

    def register(self, d: FeatureDef) -> None:
        if d.kind in self._defs:
            raise ValueError(f"feature kind {d.kind!r} already registered")
        for other in self._defs.values():
            if other.stem() == d.stem() and other.param_order == d.param_order:
                raise ValueError(
                    f"feature {d.kind!r} would produce canonical names colliding with "
                    f"{other.kind!r} (same stem {d.stem()!r} + param order {d.param_order})"
                )
        self._defs[d.kind] = d

    def get(self, kind: str) -> FeatureDef:
        try:
            return self._defs[kind]
        except KeyError:
            raise KeyError(
                f"unknown feature kind {kind!r}; registered: {sorted(self._defs)}"
            ) from None

    def __contains__(self, kind: str) -> bool:
        return kind in self._defs

    def kinds(self) -> list[str]:
        return sorted(self._defs)

    def name_for(self, spec: FeatureSpec) -> str:
        return self.get(spec.kind).canonical_name(spec)

    def metadata_for(
        self, spec: FeatureSpec, *, source_safety: SourceSafety | None = None
    ) -> FeatureMetadata:
        return self.get(spec.kind).metadata(spec, source_safety=source_safety)

    def validate_spec(self, spec: FeatureSpec) -> FeatureMetadata:
        """Full typed parameter-contract check for one spec (unknown kind, unknown
        / missing / mistyped / out-of-range params, cross-parameter constraints,
        name collisions). Returns the resolved metadata. Phase 10's compiler
        calls this before requesting a feature."""
        return self.metadata_for(spec)


REGISTRY = FeatureRegistry()


def require_lt(lo: str, hi: str) -> Callable[[dict], str | None]:
    """Cross-param constraint: ``params[lo] < params[hi]`` (e.g. fast < slow)."""

    def _check(p: dict) -> str | None:
        if p[lo] >= p[hi]:
            return f"{lo} ({p[lo]}) must be strictly less than {hi} ({p[hi]})"
        return None

    return _check


def feature(
    kind: str,
    *,
    family: FeatureFamily,
    param_rules: dict[str, ParamRule] | None = None,
    param_order: tuple[str, ...] = (),
    # Phase 6 (ETF pilot): RAW/SPLIT_ADJUSTED/TOTAL_RETURN added alongside the
    # Futures domains -- purely additive/permissive (a feature that only ever
    # sees a Futures SourceSeries is completely unaffected; this only widens
    # what price domain a feature MAY be computed over, never what it defaults
    # to or requires).
    default_price_domain: tuple[PriceDomain, ...] = (
        PriceDomain.RAW_CONTRACT,
        PriceDomain.RAW_CONTINUOUS,
        PriceDomain.BACK_ADJUSTED,
        PriceDomain.RAW,
        PriceDomain.SPLIT_ADJUSTED,
        PriceDomain.TOTAL_RETURN,
    ),
    default_session_policy: SessionPolicy = SessionPolicy.CONTINUOUS,
    lookback: Callable[[dict], int] | int = 1,
    name_stem: str = "",
    point_in_time_safe: bool = True,
    description: str = "",
    retrospective_kinds: bool = False,
    param_constraints: tuple[Callable[[dict], str | None], ...] = (),
    registry: FeatureRegistry = REGISTRY,
):
    """Decorator: register a compute function as a feature ``kind``."""

    lb_fn = lookback if callable(lookback) else (lambda _p, _v=lookback: _v)

    def wrap(fn: Callable[[FeatureComputeContext], pd.Series]) -> Callable:
        registry.register(
            FeatureDef(
                kind=kind,
                family=family,
                param_rules=param_rules or {},
                param_order=param_order,
                default_price_domain=default_price_domain,
                default_session_policy=default_session_policy,
                lookback_fn=lb_fn,
                compute=fn,
                name_stem=name_stem,
                point_in_time_safe=point_in_time_safe,
                description=description,
                retrospective_kinds=retrospective_kinds,
                param_constraints=param_constraints,
            )
        )
        return fn

    return wrap

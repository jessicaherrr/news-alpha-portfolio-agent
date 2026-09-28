"""Typed feature specification + deterministic metadata.

A :class:`FeatureSpec` is the *only* way to request a feature. It names a
registered ``kind`` and its parameters; it can never carry code. The same spec
always produces the same canonical feature name (section 14) and the same
values (section 19).
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator

from alpha_agent.features.enums import RollFeatureMode, SessionPolicy
from alpha_agent.schemas.market_data import PriceDomain

ParamValue = int | float | str | bool


def _fmt_param(v: ParamValue) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return repr(v).replace("-", "m").replace(".", "p")
    return str(v)


class FeatureSpec(BaseModel):
    """Immutable request for one feature column."""

    model_config = {"frozen": True}

    kind: str
    params: dict[str, ParamValue] = Field(default_factory=dict)
    price_field: str = "close"
    # None -> use the registered FeatureDef default
    session_policy: SessionPolicy | None = None
    required_price_domain: tuple[PriceDomain, ...] | None = None
    min_observations: int | None = None
    roll_mode: RollFeatureMode | None = None

    @field_validator("params")
    @classmethod
    def _no_nested(cls, v: dict) -> dict:
        for key, val in v.items():
            if not isinstance(val, (int, float, str, bool)):
                raise ValueError(  # noqa: TRY004 -- ValueError for parity with the rest of param validation
                    f"param {key!r} must be a scalar (int/float/str/bool), got "
                    f"{type(val).__name__}"
                )
        return v

    def __hash__(self) -> int:
        return hash(self.canonical_json())

    def canonical_json(self) -> str:
        payload = {
            "kind": self.kind,
            "params": {k: self.params[k] for k in sorted(self.params)},
            "price_field": self.price_field,
            "session_policy": self.session_policy.value if self.session_policy else None,
            "required_price_domain": (
                [d.value for d in self.required_price_domain]
                if self.required_price_domain is not None
                else None
            ),
            "min_observations": self.min_observations,
            "roll_mode": self.roll_mode.value if self.roll_mode else None,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class FeatureMetadata(BaseModel):
    """Deterministic, fully-resolved description of a feature (section 13)."""

    feature_name: str
    version: str
    family: str
    kind: str
    required_price_domain: list[str]
    lookback: int
    minimum_observations: int
    session_policy: str
    point_in_time_safe: bool
    signal_safe: bool
    execution_price_safe: bool
    price_field: str
    parameters: dict[str, ParamValue]
    description: str = ""

    def as_dict(self) -> dict:
        return self.model_dump()

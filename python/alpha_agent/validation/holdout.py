"""Two-step holdout lifecycle (section 20).

    research validation (train + validation)  ->  strategy / policy FROZEN
    locked holdout                            ->  exactly one evaluation

The locked holdout is never accessible during research. There is deliberately no
convenience API such as ``select_best(..., holdout_results)``: the only way to
touch the holdout is :meth:`HoldoutRelease.evaluate_once`, which

* requires a :class:`FrozenResearchBundle` produced *before* the holdout is seen;
* verifies the strategy / validation-spec / policy / split / dataset fingerprints
  still match the frozen bundle -- any change makes it a NEW hypothesis needing a
  NEW future holdout (``HoldoutDisciplineError``);
* can be called only once per release object.
"""
from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from alpha_agent.validation.enums import Verdict
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.policy import ReliabilityPolicy
from alpha_agent.validation.report import ValidationReport
from alpha_agent.validation.spec import ValidationSpec


class HoldoutDisciplineError(RuntimeError):
    """Raised on any attempt to misuse the locked holdout."""


class FrozenResearchBundle(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "frozen-research-bundle/1"
    strategy_fingerprint: str
    validation_spec_fingerprint: str
    reliability_policy_fingerprint: str
    split_fingerprint: str
    dataset_fingerprint: str
    trial_family_fingerprint: str
    research_verdict: Verdict
    research_report_fingerprint: str
    frozen_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def bundle_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("frozen_at", None)          # timestamp is non-semantic
        return fingerprint("frozenbundle1", payload)


def freeze_research(
    spec: ValidationSpec, policy: ReliabilityPolicy, research_report: ValidationReport
) -> FrozenResearchBundle:
    if research_report.holdout_evaluated:
        raise HoldoutDisciplineError(
            "cannot freeze a research bundle from a report that already evaluated the holdout"
        )
    if policy.identity() != spec.reliability_policy_fingerprint:
        raise HoldoutDisciplineError("policy does not match the validation spec")
    return FrozenResearchBundle(
        strategy_fingerprint=spec.strategy_fingerprint,
        validation_spec_fingerprint=spec.validation_fingerprint(),
        reliability_policy_fingerprint=policy.identity(),
        split_fingerprint=spec.split_plan.split_fingerprint(),
        dataset_fingerprint=spec.dataset.identity(),
        trial_family_fingerprint=research_report.trial_family_fingerprint,
        research_verdict=research_report.verdict,
        research_report_fingerprint=research_report.report_fingerprint(),
    )


class HoldoutEvaluation(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    bundle_fingerprint: str
    research_verdict: Verdict
    holdout_verdict: Verdict
    holdout_report_fingerprint: str
    holdout_report: ValidationReport


class HoldoutRelease:
    """Single-use gate to the locked holdout. Construct it from a frozen bundle,
    then call :meth:`evaluate_once` exactly once."""

    def __init__(self, bundle: FrozenResearchBundle):
        self._bundle = bundle
        self._used = False

    @property
    def bundle(self) -> FrozenResearchBundle:
        return self._bundle

    def evaluate_once(
        self,
        engine,  # a ValidationEngine; typed loosely to avoid an import cycle
    ) -> HoldoutEvaluation:
        if self._used:
            raise HoldoutDisciplineError(
                "this holdout release has already been evaluated -- the locked holdout "
                "is evaluated exactly once; a re-run needs a new future holdout"
            )
        spec: ValidationSpec = engine.spec
        policy: ReliabilityPolicy = engine.policy

        checks = {
            "strategy_fingerprint": spec.strategy_fingerprint == self._bundle.strategy_fingerprint,
            "validation_spec_fingerprint": (
                spec.validation_fingerprint() == self._bundle.validation_spec_fingerprint
            ),
            "reliability_policy_fingerprint": (
                policy.identity() == self._bundle.reliability_policy_fingerprint
            ),
            "split_fingerprint": (
                spec.split_plan.split_fingerprint() == self._bundle.split_fingerprint
            ),
            "dataset_fingerprint": spec.dataset.identity() == self._bundle.dataset_fingerprint,
        }
        broken = [k for k, ok in checks.items() if not ok]
        if broken:
            raise HoldoutDisciplineError(
                f"strategy / policy / split / dataset changed after the research freeze "
                f"({', '.join(broken)}) -- this is a NEW hypothesis and requires a NEW "
                f"future holdout, not a re-evaluation of this one"
            )

        self._used = True
        report = engine.run_holdout()
        return HoldoutEvaluation(
            bundle_fingerprint=self._bundle.bundle_fingerprint(),
            research_verdict=self._bundle.research_verdict,
            holdout_verdict=report.verdict,
            holdout_report_fingerprint=report.report_fingerprint(),
            holdout_report=report,
        )

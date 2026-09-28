"""The closed, frozen :class:`ValidationSpec` (section 4).

It carries only semantic inputs -- no arbitrary callables, no free-form
expressions. Every semantic choice enters ``validation_fingerprint()``; cosmetic
fields (``label``) never do.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from alpha_agent.validation.bootstrap import BootstrapConfig
from alpha_agent.validation.cost_stress import CostStressPlan
from alpha_agent.validation.dataset import DatasetIdentity
from alpha_agent.validation.fingerprint import VALIDATION_FRAMEWORK_VERSION, fingerprint
from alpha_agent.validation.nulls import NullTestConfig
from alpha_agent.validation.policy import MinimumSampleRequirements
from alpha_agent.validation.splits import SplitPlan
from alpha_agent.validation.stability import ParameterNeighbourhood
from alpha_agent.validation.walkforward import WalkForwardConfig


class ValidationSpec(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "validation-spec/1"
    framework_version: str = VALIDATION_FRAMEWORK_VERSION
    label: str = ""                             # cosmetic -- never in the fingerprint

    strategy_fingerprint: str                   # frozen Phase 10 strategy_fingerprint(spec)
    strategy_key: str                           # family key ("silver_bullet", "tsmom", ...)

    dataset: DatasetIdentity
    split_plan: SplitPlan
    walk_forward: WalkForwardConfig
    capital_base_usd: float = Field(gt=0)

    cost_stress: CostStressPlan
    null_test: NullTestConfig
    bootstrap: BootstrapConfig

    trial_family_id: str
    parameter_neighbourhood: ParameterNeighbourhood | None = None
    minimum_sample: MinimumSampleRequirements = MinimumSampleRequirements()
    reliability_policy_fingerprint: str

    random_seed: int = Field(default=0, ge=0)

    def validation_fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "framework_version": self.framework_version,
            "strategy_fingerprint": self.strategy_fingerprint,
            "strategy_key": self.strategy_key,
            "dataset": self.dataset.identity(),
            "split_plan": self.split_plan.split_fingerprint(),
            "walk_forward": self.walk_forward.identity(),
            "capital_base_usd": self.capital_base_usd,
            "cost_stress": self.cost_stress.identity(),
            "null_test": self.null_test.identity(),
            "bootstrap": self.bootstrap.identity(),
            "trial_family_id": self.trial_family_id,
            "parameter_neighbourhood": (
                self.parameter_neighbourhood.identity()
                if self.parameter_neighbourhood is not None
                else None
            ),
            "minimum_sample": self.minimum_sample.identity(),
            "reliability_policy_fingerprint": self.reliability_policy_fingerprint,
            "random_seed": self.random_seed,
        }
        return fingerprint("validationspec1", payload)


def validation_protocol_fingerprint(
    *,
    dataset: DatasetIdentity,
    split_plan: SplitPlan,
    walk_forward: WalkForwardConfig,
    cost_stress: CostStressPlan,
    null_test: NullTestConfig,
    bootstrap: BootstrapConfig,
    minimum_sample: MinimumSampleRequirements,
    reliability_policy_fingerprint: str,
) -> str:
    """The declared validation PROTOCOL's identity (Agent runtime-integration
    release, section 6/7) -- everything a :class:`ValidationSpec` fixes
    EXCEPT its strategy-specific components (``strategy_fingerprint``,
    ``strategy_key``, ``trial_family_id``, ``parameter_neighbourhood``,
    ``random_seed``).

    A `ValidationSpec.validation_fingerprint()` necessarily varies per
    strategy -- that is by design, since BH/DSR need to know exactly which
    strategy produced which trial. `ResearchOrchestrator`'s
    ``IdentityPlanes.validation_spec_fingerprint``, in contrast, is declared
    ONE frozen value shared by every member of a predeclared family (Phase
    18) and fixed at planning time, before any member's strategy exists -- it
    cannot be keyed on a strategy fingerprint that does not exist yet. This is
    the pre-run identity input a production ``ExecutionValidationService``
    stamps as ``TrialEvidence.produced_under_validation_spec_fingerprint``,
    and it is asserted equal to the frozen manifest's declared value
    (`ResearchOrchestrator._assert_evidence_provenance`).
    """
    payload = {
        "schema": "validation-protocol/1",
        "framework_version": VALIDATION_FRAMEWORK_VERSION,
        "dataset": dataset.identity(),
        "split_plan": split_plan.split_fingerprint(),
        "walk_forward": walk_forward.identity(),
        "cost_stress": cost_stress.identity(),
        "null_test": null_test.identity(),
        "bootstrap": bootstrap.identity(),
        "minimum_sample": minimum_sample.identity(),
        "reliability_policy_fingerprint": reliability_policy_fingerprint,
    }
    return fingerprint("validationprotocol1", payload)

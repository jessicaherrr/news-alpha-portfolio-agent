"""Phase 14 experiment registry + failure memory.

The persistent, deterministic, offline memory a future Research Agent queries
BEFORE proposing or running a hypothesis. Registry truth is computed, never
inferred by a language model.
"""
from __future__ import annotations

from alpha_agent.registry.enums import (
    Authority,
    ExperimentStatus,
    FailureClass,
    FailureScope,
    RegistryVerdict,
    RelationType,
    TrialRole,
)
from alpha_agent.registry.failure_memory import FailureMemory, FailureMemoryResponse
from alpha_agent.registry.holdout_guard import HoldoutAccessError
from alpha_agent.registry.identity import (
    IDENTITY_SCHEMA,
    experiment_identity,
    friendly_experiment_id,
)
from alpha_agent.registry.models import (
    ExperimentRecord,
    FailureRecord,
    ImportBundle,
    LineageEdge,
    MarketWindow,
    ResultRecord,
)
from alpha_agent.registry.schema import SCHEMA_VERSION, UnknownSchemaVersion
from alpha_agent.registry.sqlite_registry import (
    DEFAULT_REGISTRY_PATH,
    ExactDuplicate,
    ExperimentConflict,
    ExperimentRegistry,
    ExperimentView,
    IdentitySchemaMismatch,
    ImmutableResultError,
    RegistrySummary,
    RelatedExperiment,
    ScheduleProvenanceConflict,
    UnknownExperiment,
)

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "IDENTITY_SCHEMA",
    "SCHEMA_VERSION",
    "Authority",
    "ExactDuplicate",
    "ExperimentConflict",
    "ExperimentRecord",
    "ExperimentRegistry",
    "ExperimentStatus",
    "ExperimentView",
    "FailureClass",
    "FailureMemory",
    "FailureMemoryResponse",
    "FailureRecord",
    "FailureScope",
    "HoldoutAccessError",
    "IdentitySchemaMismatch",
    "ImmutableResultError",
    "ImportBundle",
    "LineageEdge",
    "MarketWindow",
    "RegistrySummary",
    "RegistryVerdict",
    "RelatedExperiment",
    "RelationType",
    "ResultRecord",
    "ScheduleProvenanceConflict",
    "TrialRole",
    "UnknownExperiment",
    "UnknownSchemaVersion",
    "experiment_identity",
    "friendly_experiment_id",
]

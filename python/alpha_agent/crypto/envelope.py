"""``SyntheticCryptoValidationArtifact`` -- the typed envelope around a frozen,
untouched Phase 13 :class:`~alpha_agent.validation.report.ValidationReport`
for the Phase 22 synthetic crypto scaffold.

The Phase 13 ``ValidationReport`` schema is FROZEN and is not touched for
Phase 22 (CLAUDE.md: a "ValidationSpec statistical-semantic ... change" is a
FROZEN RESEARCH SEMANTICS stop condition). Every Phase-22-specific fact --
that a result is SYNTHETIC, which series fed it under what provenance, and
the dataset/generator identity that produced it -- lives in this wrapper
instead of leaking into (or being lost around) the real report type.

Presentation semantics (Phase 22.1, prompt-22 review-gate follow-up): the
embedded ``report.verdict`` is the frozen ``Verdict`` enum value VERBATIM
(never renamed, never recomputed) -- it is exactly what the real
``ReliabilityPolicy`` code path would compute for any strategy. What changes
is how it may be PRESENTED: outside this module (CLI, UI), it must always be
labelled a "Synthetic Policy Outcome (NON-AUTHORITATIVE)", never a plain
"Verdict", because the market data and alternative data behind it are
synthetic. ``SyntheticCryptoValidationArtifact`` can never be mistaken for an
authoritative scientific PASS, a real ``ExperimentRegistry`` row, or a paper-
trading candidate: it carries no ``Authority``/``SUPERSEDES`` concept, and the
isolated store it is persisted to
(:mod:`alpha_agent.crypto.synthetic_registry`) is structurally separate from
both.
"""
from __future__ import annotations

from pydantic import BaseModel, model_validator

from alpha_agent.crypto.provenance import (
    SYNTHETIC_BANNER,
    CryptoDataProvenance,
    DataProvenanceRole,
)
from alpha_agent.validation.dataset import DatasetIdentity
from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.report import ValidationReport

ENVELOPE_SCHEMA = "synthetic-crypto-validation-artifact/1"

#: The label every Phase 22 presentation surface (CLI, UI) must use in place
#: of a plain "Verdict" -- see module docstring.
SYNTHETIC_POLICY_OUTCOME_LABEL = "Synthetic Policy Outcome (NON-AUTHORITATIVE)"


class SyntheticCryptoValidationArtifact(BaseModel):
    """Everything one Phase 22 synthetic research run produced, with complete
    per-series provenance carried alongside it -- never collapsed to a single
    record, never separable from the result."""

    model_config = {"frozen": True, "extra": "forbid", "arbitrary_types_allowed": True}

    schema_version: str = ENVELOPE_SCHEMA
    data_role: DataProvenanceRole = DataProvenanceRole.SYNTHETIC
    banner: str = SYNTHETIC_BANNER

    strategy_family: str
    params: dict
    fixture_seed: int

    #: One provenance record per input series that fed this result -- OHLCV,
    #: funding, open interest, on-chain MVRV/SOPR/active addresses, CME basis,
    #: liquidation aggregates, exchange flows.
    provenance_by_series: dict[str, CryptoDataProvenance]

    dataset_identity: DatasetIdentity
    #: Stable, content-based fingerprint of the generator call that produced
    #: this dataset (schema + seed + shape/delay params) --
    #: `alpha_agent.crypto.synthetic_fixtures.synthetic_crypto_generator_identity`.
    generator_identity: str
    strategy_fingerprint: str
    validation_fingerprint: str

    #: The untouched, frozen Phase 13 report -- never modified for Phase 22.
    report: ValidationReport

    @model_validator(mode="after")
    def _synthetic_only(self) -> SyntheticCryptoValidationArtifact:
        if self.data_role is not DataProvenanceRole.SYNTHETIC:
            raise ValueError("SyntheticCryptoValidationArtifact.data_role must be SYNTHETIC")
        if not self.provenance_by_series:
            raise ValueError("provenance_by_series must not be empty")
        for series_name, prov in self.provenance_by_series.items():
            try:
                prov.assert_synthetic()
            except Exception as exc:  # re-raise with the offending series named
                raise ValueError(f"provenance for series {series_name!r} is not SYNTHETIC") from exc
        return self

    @property
    def synthetic_policy_outcome_label(self) -> str:
        return SYNTHETIC_POLICY_OUTCOME_LABEL

    def content_identity(self) -> str:
        """Stable scientific/content identity -- independent of wall-clock
        ``generated_at_utc`` on any provenance record, and independent of when
        this envelope object itself was built. Two runs of byte-identical
        synthetic inputs produce the SAME ``content_identity`` even if run on
        different days (Phase 22.1)."""
        return fingerprint(
            "cryptovalidationartifact1",
            {
                "schema": self.schema_version,
                "data_role": self.data_role.value,
                "strategy_family": self.strategy_family,
                "params": self.params,
                "provenance_by_series": {
                    name: self.provenance_by_series[name].content_identity()
                    for name in sorted(self.provenance_by_series)
                },
                "dataset_identity": self.dataset_identity.identity(),
                "generator_identity": self.generator_identity,
                "strategy_fingerprint": self.strategy_fingerprint,
                "validation_fingerprint": self.validation_fingerprint,
                "report_fingerprint": self.report.report_fingerprint(),
            },
        )

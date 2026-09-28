"""Phase 13.5C -- unique statistical-trial identity for the global
multiple-testing family (Part 1.1 correction).

The ``MultipleTestingFamily`` fed to Benjamini-Hochberg / DSR must contain each
inspected performance HYPOTHESIS exactly once. In Phase 13.5C the 20 baseline
"canonical" trials already ARE the per-root results (4 families x 5 roots), so
:class:`~alpha_agent.validation.crossmarket.CrossMarketEvidence` over those roots
is a **descriptive summary of existing trial results**, not an additional
hypothesis. A canonical strategy-root result must never appear once as
``role="canonical"`` and again as ``role="cross_market"``.

This module:

* :func:`trial_semantic_identity` -- deterministic identity from semantic fields
  (strategy fingerprint + root + schedule hash + dataset identity), role-free;
* :func:`build_global_trial_family` -- de-duplicates a flat list of
  :class:`TrialRecord` by that identity, refuses ``role="cross_market"`` records
  (they are not hypotheses), and raises loudly on a genuine identity conflict;
* :func:`unique_trial_identities` -- the frozen, pre-run enumeration of the
  unique BH/FDR trial family from the candidate manifest, with a uniqueness
  assertion.

Nothing here reads market data or a statistic.
"""
from __future__ import annotations

from pydantic import BaseModel

from alpha_agent.validation.fingerprint import fingerprint
from alpha_agent.validation.multiple_testing import MultipleTestingFamily, TrialRecord

# roles that are genuine inspected hypotheses (each a BH observation)
HYPOTHESIS_ROLES = frozenset({"canonical", "neighbour", "ablation", "variant"})
# descriptive aggregations -- NEVER a BH observation
NON_HYPOTHESIS_ROLES = frozenset({"cross_market"})


class DuplicateTrialIdentity(ValueError):
    """Two trial records share a semantic identity but disagree on the tested
    hypothesis -- the family would double-count (or mis-count) a hypothesis."""


class CrossMarketIsNotATrial(ValueError):
    """A ``role='cross_market'`` record was offered to the BH/FDR trial family.
    Cross-market evidence is a summary of existing per-root trials, not a new
    hypothesis; reference the existing root trial identities instead."""


def trial_semantic_identity(
    *,
    strategy_fingerprint: str,
    root_symbol: str,
    schedule_hash: str | None = None,
    dataset_identity: str | None = None,
) -> str:
    """Deterministic, role-free identity for one inspected performance
    hypothesis. Two records with the same identity are the SAME hypothesis
    regardless of the ``role`` label attached to them."""
    return fingerprint(
        "p135ctrial1",
        {
            "strategy_fingerprint": strategy_fingerprint,
            "root_symbol": root_symbol,
            "schedule_hash": schedule_hash or "",
            "dataset_identity": dataset_identity or "",
        },
    )


def _record_identity(t: TrialRecord, root_of: dict[str, str] | None) -> str:
    root = ""
    if root_of is not None:
        root = root_of.get(t.strategy_fingerprint, "")
    return trial_semantic_identity(
        strategy_fingerprint=t.strategy_fingerprint,
        root_symbol=root,
        schedule_hash=t.schedule_hash,
    )


def build_global_trial_family(
    records: list[TrialRecord],
    *,
    family_id: str = "phase_13_5c.all",
    root_of: dict[str, str] | None = None,
) -> MultipleTestingFamily:
    """De-duplicate ``records`` into the global BH/FDR family.

    * ``role="cross_market"`` records are refused (`CrossMarketIsNotATrial`).
    * records are keyed by :func:`trial_semantic_identity`. A repeated identity
      is collapsed to one trial; a null-tested record beats a non-null-tested
      one; two null-tested records with different p-values raise
      `DuplicateTrialIdentity`.
    * exactly one ``role="canonical"`` record must survive per identity family
      is NOT required here (the global family holds many canonical trials); the
      per-trial verdict step locates each trial's own index.
    """
    kept: dict[str, TrialRecord] = {}
    order: list[str] = []
    for t in records:
        if t.role in NON_HYPOTHESIS_ROLES:
            raise CrossMarketIsNotATrial(
                f"trial {t.label!r} has role {t.role!r}; cross-market evidence is a "
                f"summary of existing per-root trials, not a BH observation"
            )
        key = _record_identity(t, root_of)
        if key not in kept:
            kept[key] = t
            order.append(key)
            continue
        prev = kept[key]
        if prev.null_tested and t.null_tested and abs(prev.p_value - t.p_value) > 1e-12:
            raise DuplicateTrialIdentity(
                f"semantic identity {key[:20]}... appears twice with different "
                f"null-tested p-values ({prev.p_value} vs {t.p_value}); "
                f"labels {prev.label!r} / {t.label!r}"
            )
        if t.null_tested and not prev.null_tested:
            kept[key] = t
    return MultipleTestingFamily(
        family_id=family_id, trials=tuple(kept[k] for k in order)
    )


class UniqueTrialBreakdown(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    unique_trial_count: int
    canonical: int
    neighbour: int
    ablation: int
    cross_market_excluded: int
    identities: tuple[str, ...]

    def identity(self) -> str:
        return fingerprint(
            "p135cbhfamily1",
            {"n": self.unique_trial_count, "identities": sorted(self.identities)},
        )


def unique_trial_identities(manifest) -> UniqueTrialBreakdown:
    """The frozen, pre-run BH/FDR trial family from a
    :class:`~alpha_agent.strategy.candidates_phase_13_5c.CandidateManifest`.

    Unique trials = every canonical strategy fingerprint + every predeclared
    neighbour fingerprint, de-duplicated by ``(strategy_fingerprint, root)``.
    Cross-market roots contribute NO trial. Ablations: none predeclared. Raises
    if two variants collapse to the same identity (they must not).
    """
    seen: dict[str, str] = {}          # identity -> role
    ident_list: list[str] = []
    n_canon = n_neigh = 0
    for tr in manifest.trials:
        canon_id = trial_semantic_identity(
            strategy_fingerprint=tr.strategy_fingerprint, root_symbol=tr.root_symbol
        )
        if canon_id in seen:
            raise DuplicateTrialIdentity(
                f"canonical trial {tr.family_key}/{tr.root_symbol} collides with an "
                f"existing {seen[canon_id]} identity"
            )
        seen[canon_id] = f"canonical:{tr.family_key}/{tr.root_symbol}"
        ident_list.append(canon_id)
        n_canon += 1
        for fp in tr.neighbour_fingerprints:
            nid = trial_semantic_identity(strategy_fingerprint=fp, root_symbol=tr.root_symbol)
            if nid in seen:
                raise DuplicateTrialIdentity(
                    f"neighbour of {tr.family_key}/{tr.root_symbol} ({fp[:16]}...) "
                    f"collides with {seen[nid]}"
                )
            seen[nid] = f"neighbour:{tr.family_key}/{tr.root_symbol}"
            ident_list.append(nid)
            n_neigh += 1
    return UniqueTrialBreakdown(
        unique_trial_count=len(ident_list),
        canonical=n_canon,
        neighbour=n_neigh,
        ablation=0,
        cross_market_excluded=4 * 5,        # 4 baseline families x 5 roots -- summarised, not trials
        identities=tuple(ident_list),
    )

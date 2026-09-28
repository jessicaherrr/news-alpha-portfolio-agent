"""Phase 20 -- read/orchestration boundary for the Streamlit research interface.

This module is deliberately the ONLY place `alpha_agent.ui` touches the rest of
the system. It never imports `streamlit`: every function here is a plain,
testable Python call so the presentation layer (`alpha_agent.ui.pages.*`) stays
a thin rendering shell over it (CLAUDE.md: "keep optional vendor imports
lazy").

Boundary discipline (CLAUDE.md, prompt 20):

* Nothing here computes PnL, fills, risk, validation verdicts, experiment
  identity, or holdout eligibility. Every number surfaced by this module is
  read verbatim from a committed artifact (the experiment registry, a frozen
  Phase 13.5C/15B report, or a committed `trades.csv`) or is the deterministic
  output of an already-approved service (`ResearchAgent`, `StrategyCompilerAgent`,
  `FailureMemory`) -- never re-derived.
* The registry is opened read-only from this module's point of view: nothing
  here calls an `ExperimentRegistry` write method.
* No network call and no paid API use happens unless the caller explicitly
  builds a live `AnthropicClient` (see `llm_demo.build_llm_client`) -- the
  default path is the deterministic `ScriptedLLMClient`.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from alpha_agent.agents import build_family_catalog, build_feature_catalog
from alpha_agent.agents.context import (
    ENGINEERING_FAILURE_CLASSES,
    SCIENTIFIC_FAILURE_CLASSES,
)
from alpha_agent.recommendation import (
    DEFAULT_PROFILE,
    InvestorProfile,
    ProfileStore,
    ResearchCandidateSummary,
    build_candidate_summaries,
    build_candidate_summary,
    promising_candidates,
    validated_strategies,
)
from alpha_agent.registry import (
    DEFAULT_REGISTRY_PATH,
    ExperimentRegistry,
    FailureClass,
    FailureMemory,
    FailureMemoryResponse,
    RegistryVerdict,
    TrialRole,
)
from alpha_agent.registry.enums import AssetDomain
from alpha_agent.registry.holdout_guard import HoldoutAccessError, assert_no_holdout_market_data

#: python/alpha_agent/ui/services.py -> repo root, independent of the
#: process's current working directory (Streamlit is not guaranteed to be
#: launched from the repo root).
REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / DEFAULT_REGISTRY_PATH
#: Phase 8 -- the SEPARATE Community store, never the scientific registry.
COMMUNITY_STORE_PATH = REPO_ROOT / "data" / "community" / "community.sqlite"
CATALOG_JSON = REPO_ROOT / "data" / "catalog" / "real_cme_dataset.json"
OUTPUTS_DIR = REPO_ROOT / "outputs"
PHASE_15B_REPORT = OUTPUTS_DIR / "phase_15" / "PHASE_15B_ATTEMPT_2_CORRECTED_REPORT.json"

#: `st.session_state` key for the Agent page's cached "Refresh Opportunities"
#: snapshot (`{"opportunities": [...], "observed_at": datetime}`) -- the
#: SAME real, current, explicitly-user-triggered Databento evidence
#: `claude_conversation.py` grounds Agent's own live Claude replies with.
#: Defined here (not in `views/agent.py`, which sets/reads it) so any OTHER
#: view can read the identical, already-refreshed snapshot -- e.g. the
#: Research Workflow's Decision Brief step -- without a circular import
#: (`views/agent.py` already imports `views/research.py`, which imports
#: `views/research_workflow.py`). Still only ever WRITTEN by Agent's own
#: explicit "Refresh Opportunities" click; nothing here fetches anything.
OPPORTUNITIES_SNAPSHOT_KEY = "agent_opportunities_snapshot"

DISCLAIMER = (
    "Historical backtest and validation evidence. Past simulated performance "
    "is not a guarantee, projection, or promise of future returns."
)


# ---------------------------------------------------------------------------
# system / execution provenance
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 -- git provenance is best-effort, never fails the page
        return None


def git_provenance() -> dict[str, Any]:
    """Best-effort, offline git provenance. Never raises -- a missing/short git
    binary just yields ``None`` fields."""
    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "last_subject": _git("log", "-1", "--pretty=%s"),
    }


def engine_provenance() -> dict[str, Any]:
    """Whether the optional pybind11 fast boundary (Phase 19) is importable,
    and whether the reference C++ CLIs are built. Presentation only -- this
    never runs a backtest."""
    pybind_available = False
    try:
        import quant_core_py  # noqa: F401

        pybind_available = True
    except ImportError:
        pybind_available = False
    cli = REPO_ROOT / "build" / "cpp" / "cpp" / "quant_backtest_targets_csv"
    return {
        "pybind_fast_boundary_available": pybind_available,
        "reference_cli_built": cli.exists(),
        "reference_cli_path": str(cli.relative_to(REPO_ROOT)) if cli.exists() else None,
    }


def holdout_status() -> dict[str, Any]:
    """The locked-holdout declaration. Re-asserted here (not trusted from the
    label alone) against every artifact this module reads, so a violation
    upstream fails loud in the UI rather than silently rendering a 2025 value."""
    return {
        "boundary": "2025-01-01",
        "declaration": (
            "2025 was NEVER downloaded, queried, cost-fetched, loaded, "
            "featured, selected on, or reported. Any code path touching "
            "ts >= 2025-01-01 fails loudly."
        ),
        "guard": "alpha_agent.registry.holdout_guard.assert_no_holdout_market_data",
    }


def holdout_lifecycle_status(epoch_label: str = "2025") -> dict[str, Any]:
    """Alpha Discovery Part L -- the real, current lifecycle state for one
    holdout epoch (default: the real 2025 final holdout), read straight from
    `alpha_agent.holdout.lifecycle.HoldoutLifecycle` (pure bookkeeping, no
    market-data-loading code at all). No record on disk == SEALED /
    not-accessed, the honest default -- this call never creates one."""
    from alpha_agent.holdout.lifecycle import HoldoutLifecycle

    record = HoldoutLifecycle().get(epoch_label)
    return record.status()


def check_no_holdout_leak(payload: Any, *, path: str) -> str | None:
    """Defensive, belt-and-suspenders re-check of one payload this UI is about
    to render. Returns ``None`` when clean, or a human-readable violation
    string -- this should never trigger given the upstream guarantees, but the
    UI must fail loud, not silently display a locked value, if it ever does."""
    try:
        assert_no_holdout_market_data(payload, path=path)
        return None
    except HoldoutAccessError as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# market / data status (Phase 02-04 catalog, read-only)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_catalog() -> tuple[dict, ...]:
    if not CATALOG_JSON.exists():
        return ()
    rows = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    return tuple(rows)


def catalog_summary() -> dict[str, Any]:
    rows = load_catalog()
    roots = sorted({r["root"] for r in rows})
    by_root: dict[str, dict] = {}
    for root in roots:
        root_rows = [r for r in rows if r["root"] == root]
        by_root[root] = {
            "n_entries": len(root_rows),
            "components": sorted({r["component"] for r in root_rows}),
            "split_roles": sorted({r["split_role"] for r in root_rows}),
            "schemas": sorted({r["schema"] for r in root_rows}),
            "start": min((r["start"] for r in root_rows), default=None),
            "end": max((r["end"] for r in root_rows), default=None),
            "row_count": sum(int(r.get("row_count", 0) or 0) for r in root_rows),
            "vendor": sorted({r["vendor"] for r in root_rows}),
            "dataset": sorted({r["dataset"] for r in root_rows}),
        }
    return {
        "n_rows": len(rows),
        "roots": roots,
        "by_root": by_root,
    }


def approved_universe() -> tuple[str, ...]:
    summary = catalog_summary()
    roots = tuple(summary["roots"])
    return roots or ("ES", "NQ", "CL", "GC", "ZN")


_MARKET_NAMES = {
    "ES": "S&P 500",
    "NQ": "Nasdaq 100",
    "CL": "Crude Oil",
    "GC": "Gold",
    "ZN": "10Y Treasury",
}


def market_name(root: str) -> str:
    """Presentation-only display name for a root symbol -- checks the
    research-universe-only override table first (a couple of names there
    deliberately favor a shorter/more investor-familiar label than the raw
    catalog, e.g. "10Y Treasury" vs. "10-Year T-Note"), then falls back to
    the full 36-product `alpha_agent.marketdata.product_catalog` entry so
    every catalogued Market Intelligence root (CL, RB, SI, ZN, 6E, ...) gets
    a real name instead of just echoing its symbol back."""
    if root in _MARKET_NAMES:
        return _MARKET_NAMES[root]
    from alpha_agent.marketdata.product_catalog import catalog_entry

    entry = catalog_entry(root)
    return entry.display_name if entry else root


#: Release UX (investor-readability pass): presentation-only display names for
#: the Phase 11 baseline `strategy_family` keys -- short, investor-facing
#: labels distinct from `StrategyFamilyCard.name`'s longer descriptive text
#: (`family_catalog()`'s own "Name" column keeps that longer text; this is for
#: compact headers, badges, and selector labels). Never used to change a
#: registry key, an experiment identity, or a query filter value -- only what
#: is painted next to the raw `family_key` a caller already has. An unknown
#: family key (a future baseline, a crypto-lab-only family) safely falls back
#: to the raw key, never a blank or a raised error.
_STRATEGY_NAMES = {
    "tsmom": "Time-Series Momentum",
    "ma_trend": "Moving-Average Trend",
    "breakout": "Breakout",
    "mean_reversion": "Mean Reversion",
    "silver_bullet": "Silver Bullet",
}
_ML_META_LABEL_PREFIX = "ml_meta_label."


def strategy_name(family_key: str | None) -> str:
    """Presentation-only friendly label for a `strategy_family` key. Falls
    back to the raw value (never blank) for anything not in `_STRATEGY_NAMES`,
    including a `ml_meta_label.<base>` composite family, whose base is looked
    up the same way."""
    if not family_key:
        return family_key or "--"
    if family_key in _STRATEGY_NAMES:
        return _STRATEGY_NAMES[family_key]
    if family_key.startswith(_ML_META_LABEL_PREFIX):
        base = family_key[len(_ML_META_LABEL_PREFIX) :]
        return f"ML Meta-Label: {_STRATEGY_NAMES.get(base, base)}"
    return family_key


def research_window() -> dict[str, str]:
    """Research / validation window boundaries, read from the real data catalog
    (`split_role`) rather than hardcoded -- the 2025 holdout boundary is the one
    fixed declarative constant (CLAUDE.md: never downloaded/queried/loaded)."""
    rows = load_catalog()
    research = [r for r in rows if r["split_role"] == "RESEARCH"]
    validation = [r for r in rows if r["split_role"] == "VALIDATION"]
    out = {"research": "unavailable", "validation": "unavailable", "holdout": "2025 -- LOCKED"}
    if research:
        lo = min(r["start"] for r in research)[:4]
        hi = str(int(max(r["end"] for r in research)[:4]) - 1)  # exclusive end -> inclusive year
        out["research"] = f"{lo} - {hi}"
    if validation:
        lo = min(r["start"] for r in validation)[:4]
        hi = str(int(max(r["end"] for r in validation)[:4]) - 1)
        out["validation"] = f"{lo} - {hi}"
    return out


# ---------------------------------------------------------------------------
# experiment registry (Phase 14/14.1/14.2, read-only)
# ---------------------------------------------------------------------------


def open_registry() -> ExperimentRegistry:
    return ExperimentRegistry(REGISTRY_PATH)


def registry_summary() -> dict[str, Any]:
    with open_registry() as reg:
        return reg.summary().model_dump(mode="json")


def list_experiments(
    *,
    strategy_family: str | None = None,
    root_symbol: str | None = None,
    trial_role: TrialRole | None = None,
    verdict: RegistryVerdict | None = None,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    with open_registry() as reg:
        views = reg.experiments(
            strategy_family=strategy_family,
            root_symbol=root_symbol,
            trial_role=trial_role,
            verdict=verdict,
            authoritative_only=not include_superseded,
            include_superseded=include_superseded,
        )
        out = []
        for v in views:
            row = {
                "experiment_id": v.experiment_id,
                "experiment_identity": v.experiment_identity,
                "phase": v.experiment.phase,
                "root_symbol": v.experiment.root_symbol,
                "strategy_family": v.experiment.strategy_family,
                "trial_role": v.experiment.trial_role.value,
                "parameter_variant_label": v.experiment.parameter_variant_label,
                "strategy_fingerprint": v.experiment.strategy_fingerprint,
                "params": dict(v.experiment.strategy_spec_json.get("params", {})),
                "market_window": f"{v.experiment.market_window.start_date}..{v.experiment.market_window.end_date}",
                "authority": v.authority.value,
                "verdict": v.verdict.value if v.verdict else None,
                "reason_codes": list(v.result.reason_codes) if v.result else [],
                "net_pnl_usd": v.result.net_pnl_usd if v.result else None,
                "gross_pnl_usd": v.result.gross_pnl_usd if v.result else None,
                "costs_usd": v.result.costs_usd if v.result else None,
                "daily_sharpe": v.result.daily_sharpe if v.result else None,
                "annualized_sharpe": v.result.annualized_sharpe if v.result else None,
                "bh_q": v.result.bh_q if v.result else None,
                "dsr_probability": v.result.dsr_probability if v.result else None,
                "n_trades": v.result.n_trades if v.result else None,
                "n_valid_attempts": v.n_valid_attempts,
                "n_invalid_attempts": v.n_invalid_attempts,
                "has_valid_authoritative_result": v.has_valid_authoritative_result,
                # Phase B1 (Research Promise / User Fit): the remaining
                # committed `ResultRecord` fields those deterministic scores
                # read. Purely additive -- every key above is unchanged.
                "gating_null_p": v.result.gating_null_p if v.result else None,
                "fold_consistency": v.result.fold_consistency if v.result else None,
                "n_fills": v.result.n_fills if v.result else None,
                "n_oos_days": v.result.n_oos_days if v.result else None,
                "parameter_stability": dict(v.result.parameter_stability) if v.result else {},
                "cost_stress": dict(v.result.cost_stress) if v.result else {},
                "regime_evidence": dict(v.result.regime_evidence) if v.result else {},
                "cross_market_reference": dict(v.result.cross_market_reference) if v.result else {},
                "evidence_completeness": v.result.evidence_completeness if v.result else None,
            }
            out.append(row)
        return out


def get_experiment(experiment_key: str) -> dict[str, Any]:
    """Full detail for one experiment id/identity: the record, authoritative
    result, and every execution attempt (VALID and INVALID_EXECUTION alike)."""
    with open_registry() as reg:
        view = reg.get(experiment_key)
        return {
            "experiment": view.experiment.model_dump(mode="json"),
            "result": view.result.model_dump(mode="json") if view.result else None,
            "authority": view.authority.value,
            "superseded_by": list(view.superseded_by),
            "attempts": [a.model_dump(mode="json") for a in view.attempts],
            "authoritative_attempt_id": view.authoritative_attempt_id,
        }


def latest_authoritative_experiment() -> dict[str, Any] | None:
    """The most recently created AUTHORITATIVE experiment that actually has a
    committed VALID scientific result -- i.e. a real headline verdict, not
    just a registered hypothesis awaiting execution.

    Used ONLY as Research Details' priority-3 fallback (real committed
    Registry evidence, never fabricated) when neither an explicit Agent
    hand-off nor a live Agent-session proposal exists -- so the page opens on
    real evidence instead of an empty state. Returns ``None`` when the
    registry holds no such experiment yet (an honest empty state is still
    correct in that case)."""
    with open_registry() as reg:
        views = [v for v in reg.experiments(authoritative_only=True) if v.result is not None]
        if not views:
            return None
        best = max(views, key=lambda v: (v.experiment.created_at, v.experiment_id))
        return {
            "experiment_id": best.experiment_id,
            "root_symbol": best.experiment.root_symbol,
            "strategy_family": best.experiment.strategy_family,
            "created_at": best.experiment.created_at,
        }


def failure_memory_lookup(
    *,
    strategy_family: str,
    root_symbol: str | None,
    params: dict | None = None,
    top_k_related: int = 8,
) -> dict[str, Any]:
    with open_registry() as reg:
        fm = FailureMemory(reg)
        # this UI surface has no ETF wiring yet (Phase 6 ETF Research Pilot's UI
        # integration is a separate, later phase) -- every real experiment
        # today is Futures, a stated fact, not a new parameter here.
        resp: FailureMemoryResponse = fm.lookup(
            strategy_family=strategy_family,
            root_symbol=root_symbol,
            asset_domain=AssetDomain.FUTURES,
            strategy_spec={"params": params or {}},
            top_k_related=top_k_related,
        )
        return resp.model_dump(mode="json")


def list_failures(
    *,
    failure_class: FailureClass | None = None,
    strategy_family: str | None = None,
    root_symbol: str | None = None,
) -> list[dict[str, Any]]:
    with open_registry() as reg:
        records = reg.failures(
            failure_class=failure_class,
            strategy_family=strategy_family,
            root_symbol=root_symbol,
        )
        out = []
        for r in records:
            d = r.model_dump(mode="json")
            d["is_engineering"] = r.failure_class in ENGINEERING_FAILURE_CLASSES
            d["is_scientific"] = r.failure_class in SCIENTIFIC_FAILURE_CLASSES
            out.append(d)
        return out


def failure_class_split() -> dict[str, list[str]]:
    return {
        "engineering": sorted(c.value for c in ENGINEERING_FAILURE_CLASSES),
        "scientific": sorted(c.value for c in SCIENTIFIC_FAILURE_CLASSES),
    }


# ---------------------------------------------------------------------------
# strategy family / feature catalogs (Phase 09/11/17, read-only)
# ---------------------------------------------------------------------------


def family_catalog() -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in build_family_catalog()]


def feature_catalog() -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in build_feature_catalog()]


def execution_supported_family(family_key: str | None) -> bool:
    """Whether `ProductionExecutionValidationService` can actually run a
    compiled `StrategySpec` in this family (Agent product refactor, section
    2/3: "Execution support: SUPPORTED / UNSUPPORTED" must be the real
    capability, never faked for an arbitrary blueprint). `None` (a blueprint
    build has no Phase 11 family) is always unsupported -- lazily imported
    since `execution_service` pulls in the C++/validation runner stack, which
    other `services.py` callers (e.g. tests that only exercise catalog/
    registry reads) should not have to pay for."""
    from alpha_agent.agents.execution_service import SUPPORTED_FAMILIES

    return family_key in SUPPORTED_FAMILIES


# ---------------------------------------------------------------------------
# Phase 2 -- Personal Alpha Memory (read-only aggregation over the registry;
# `alpha_agent.alpha_memory`'s own module docstrings are the full boundary:
# no registry write, no Claude call, no market-data call, no backtest).
# ---------------------------------------------------------------------------


def list_alpha_research_objects(root_symbol: str | None = None) -> list[dict[str, Any]]:
    """Every real, evidence-grounded AlphaResearchObject across BOTH research
    domains -- "My Alpha" (Phase 2 origin as "My Alpha Library"; the Phase 6
    closure renamed it "Factor Library"; Phase 8's Research UI consolidation
    renamed it back to "My Alpha" to read clearly alongside the new,
    SHARED Community tab). Futures and ETF objects come from two SEPARATE registry
    queries (`alpha_memory.list_alpha_research_objects` /
    `alpha_agent.etf.alpha_memory_bridge.list_etf_alpha_research_objects`),
    never one combined query -- structural separation, not just a shared
    display list. Each dict gains two UI-only fields not part of
    `AlphaResearchObject` itself: `asset_domain` (which query produced it)
    and `factor_status` (`alpha_memory.factor_status.factor_status` --
    VALIDATED / UNDER_RESEARCH / RESEARCH_ARCHIVE, never a boolean)."""
    from alpha_agent.alpha_memory import list_alpha_research_objects as _list_futures
    from alpha_agent.alpha_memory.factor_status import factor_status as _status
    from alpha_agent.etf.alpha_memory_bridge import list_etf_alpha_research_objects as _list_etf

    with open_registry() as reg:
        futures_objs = _list_futures(reg, root_symbol=root_symbol)
        # a Futures root_symbol filter never matches an ETF ticker (disjoint
        # string sets) -- filter the ETF list the SAME way rather than
        # skipping it outright, so filtering by e.g. "XLE" still works.
        etf_objs = tuple(o for o in _list_etf(reg) if root_symbol is None or o.root_symbol == root_symbol)
        out = []
        for domain, objs in (("FUTURES", futures_objs), ("ETF", etf_objs)):
            for o in objs:
                d = o.model_dump(mode="json")
                d["asset_domain"] = domain
                d["factor_status"] = _status(o).value
                out.append(d)
        return out


def get_alpha_research_object(alpha_id: str) -> dict[str, Any] | None:
    """Looks in BOTH domains -- a Factor's `alpha_id` alone doesn't say which
    (it is `root_symbol` + `FactorIdentity.factor_identity`, and while a real
    collision is not possible in practice today, the lookup checks the
    domain the id actually belongs to rather than assuming)."""
    from alpha_agent.alpha_memory import get_alpha_research_object as _get
    from alpha_agent.alpha_memory.factor_status import factor_status as _status
    from alpha_agent.etf.alpha_memory_bridge import list_etf_alpha_research_objects as _list_etf

    with open_registry() as reg:
        obj = _get(reg, alpha_id)
        domain = "FUTURES"
        if obj is None:
            obj = next((o for o in _list_etf(reg) if o.alpha_id == alpha_id), None)
            domain = "ETF"
        if obj is None:
            return None
        d = obj.model_dump(mode="json")
        d["asset_domain"] = domain
        d["factor_status"] = _status(obj).value
        return d


def mechanism_memory_lookup(
    *, mechanism: str, root_symbol: str, strategy_family: str | None = None
) -> dict[str, Any]:
    """Phase 1 integration (section 17; hardened by the identity-hardening
    patch, section 5, then the final Phase 2 semantic fix, section 2):
    "Have I researched something like this before?" for one
    `EconomicMechanism` value on one root. Read-only -- never triggers
    research. Mechanism alone can NEVER establish an exact Factor match
    (Phase 1 does not yet know which `strategy_family` a candidate will
    use), so a mechanism-only call (`strategy_family=None`, today's real
    Phase 1 integration point) returns at most `"RELATED_MECHANISM"`, never
    `"MATCHING_FACTOR"` -- however few real Factor objects exist under the
    mechanism. `"MATCHING_FACTOR"` requires an explicit `strategy_family`
    whose own deterministic Factor identity equals a stored object's
    exactly; `"NONE"` means no real evidence exists at all."""
    from alpha_agent.alpha_memory import mechanism_memory_lookup as _lookup
    from alpha_agent.knowledge.models import EconomicMechanism

    with open_registry() as reg:
        result = _lookup(
            reg, mechanism=EconomicMechanism(mechanism), root_symbol=root_symbol, strategy_family=strategy_family,
        )
        return result.model_dump(mode="json")


def signal_path_memory_for_routes(path_signatures: Sequence[str]):
    """News Alpha Phase H: the recorded memory (registry schema v7) of these
    transmission routes -- every recorded run of each route, from ANY event, so
    a newly described event recalls what is known about its routes. Read-only."""
    from alpha_agent.alpha_memory.signal_path_memory import summarize_path_memory

    with open_registry() as reg:
        records = [r for sig in sorted(set(path_signatures)) for r in reg.signal_path_evidence(path_signature=sig)]
    return list(summarize_path_memory(records))


def signal_path_memory_all():
    """Learn -> My Alpha: every recorded transmission route (registry schema
    v7), summarized per route exactly as `signal_path_memory_for_routes`
    does for one thread's routes. Read-only."""
    from alpha_agent.alpha_memory.signal_path_memory import summarize_path_memory

    with open_registry() as reg:
        records = list(reg.signal_path_evidence())
    return list(summarize_path_memory(records))


def portfolio_validations(source_plan_fingerprint: str) -> list[dict[str, Any]]:
    """News Alpha Phase H: registry experiments that validated the portfolio a
    plan defines (canonical trials). Read-only."""
    out = []
    with open_registry() as reg:
        for view in reg.experiments(strategy_family="news_alpha_portfolio", trial_role=TrialRole.CANONICAL):
            if view.experiment.strategy_spec_json.get("source_plan_fingerprint") == source_plan_fingerprint:
                out.append({"experiment_id": view.experiment_id, "verdict": view.verdict.value if view.verdict else None,
                            "reason_codes": list(view.result.reason_codes) if view.result else [],
                            "net_pnl_usd": view.result.net_pnl_usd if view.result else None})
    return out


def what_changed_for_alpha(
    alpha_id: str,
    *,
    mechanism: str,
    root_symbol: str,
    strategy_family: str | None = None,
    params: dict | None = None,
) -> dict[str, Any] | None:
    """Section 15: "what changed this time?" against one existing
    AlphaResearchObject. `None` when `alpha_id` does not resolve."""
    from alpha_agent.alpha_memory import get_alpha_research_object as _get
    from alpha_agent.alpha_memory import what_changed as _what_changed
    from alpha_agent.knowledge.models import EconomicMechanism

    with open_registry() as reg:
        alpha = _get(reg, alpha_id)
        if alpha is None:
            return None
        wc = _what_changed(
            alpha, mechanism=EconomicMechanism(mechanism), root_symbol=root_symbol,
            strategy_family=strategy_family, params=params,
        )
        return wc.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Phase 7 -- Cross-Asset Alpha Graph (Research Map)
# ---------------------------------------------------------------------------
#
# Thin, read-only wrappers over `alpha_agent.alpha_graph` -- browsing the
# Research Map never runs a backtest, calls Claude, calls a market-data API,
# or writes the registry (mirrors `list_alpha_research_objects` above).


def alpha_graph_mechanisms() -> list[str]:
    """Every Mechanism the graph considers -- the union of Futures' and
    ETF's frozen mechanism->family bridges."""
    from alpha_agent.alpha_graph import MECHANISM_UNIVERSE

    return [m.value for m in MECHANISM_UNIVERSE]


def alpha_graph_summary() -> list[dict[str, Any]]:
    """One compact row per Mechanism for the Research Map landing table --
    counts only, no new score."""
    from alpha_agent.alpha_graph import build_alpha_graph, summarize

    with open_registry() as reg:
        views = build_alpha_graph(reg)
        return [summarize(v).model_dump(mode="json") for v in views]


def alpha_graph_for_mechanism(mechanism: str) -> dict[str, Any]:
    """The full navigable graph view for one Mechanism (Event provenance,
    Factors, per-instrument evidence across both asset domains)."""
    from alpha_agent.alpha_graph import build_mechanism_graph
    from alpha_agent.knowledge.models import EconomicMechanism

    with open_registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism(mechanism))
        return view.model_dump(mode="json")


def alpha_graph_synthesis(mechanism: str) -> dict[str, Any]:
    """The deterministic cross-asset synthesis (what's common / differs /
    repeated failures / underexplored) for one Mechanism -- never a merged
    verdict, never a trade instruction."""
    from alpha_agent.alpha_graph import build_mechanism_graph, cross_asset_synthesis
    from alpha_agent.knowledge.models import EconomicMechanism

    with open_registry() as reg:
        view = build_mechanism_graph(reg, EconomicMechanism(mechanism))
        return cross_asset_synthesis(view).model_dump(mode="json")


def alpha_graph_analogous_research(*, mechanism: str, root_symbol: str) -> list[dict[str, Any]]:
    """The Agent retrieval layer's "where else has this Mechanism been
    researched?" answer -- every OTHER instrument (any asset domain) with
    real evidence, excluding `root_symbol` itself. Never a trade signal --
    see `alpha_agent.alpha_graph.retrieval` module docstring."""
    from alpha_agent.alpha_graph import analogous_research
    from alpha_agent.knowledge.models import EconomicMechanism

    with open_registry() as reg:
        related = analogous_research(reg, mechanism=EconomicMechanism(mechanism), root_symbol=root_symbol)
        return [r.model_dump(mode="json") for r in related]


def alpha_graph_research_gaps() -> list[dict[str, Any]]:
    """Every (Mechanism, Instrument) research gap across the whole graph,
    flattened for a table -- first-class, never a failure."""
    from alpha_agent.alpha_graph import build_alpha_graph, list_research_gaps

    with open_registry() as reg:
        gaps = list_research_gaps(build_alpha_graph(reg))
        return [g.model_dump(mode="json") for g in gaps]


# ---------------------------------------------------------------------------
# Phase 8 -- Community Alpha Network
# ---------------------------------------------------------------------------
#
# A SEPARATE store (`data/community/community.sqlite`), never the scientific
# registry -- mirrors `alpha_agent.paper`'s own "operational store, never
# confused with the registry" boundary. Every write below lands ONLY in that
# store; the registry is opened read-only (to verify a contribution's/
# replication's cited evidence), exactly like every other function in this
# module. `alpha_agent.community`'s own module docstrings are the full
# boundary (threat-model defenses, privacy enforcement).


def open_community_store():
    from alpha_agent.community import CommunityStore

    return CommunityStore(COMMUNITY_STORE_PATH)


def community_contributor_id(display_name: str) -> str:
    """The stable, normalized (self-attested) contributor id for a typed
    display name -- so the UI can consistently address "my contributions"
    without a second identity concept."""
    from alpha_agent.community import contributor_ref

    return contributor_ref(display_name).contributor_id


def community_feed(viewer_display_name: str | None = None) -> list[dict[str, Any]]:
    """PUBLIC (full) + SHARED (redacted unless the viewer is its own
    contributor) contributions -- PRIVATE never appears here for anyone."""
    from alpha_agent.community import community_feed as _feed

    viewer_id = community_contributor_id(viewer_display_name) if viewer_display_name else None
    with open_community_store() as store:
        return [c.model_dump(mode="json") for c in _feed(store, viewer_contributor_id=viewer_id)]


def community_my_contributions(display_name: str) -> list[dict[str, Any]]:
    """Every contribution by this contributor, full detail, any visibility --
    the owner's own personal view (PRIVATE included)."""
    from alpha_agent.community import my_contributions as _mine

    with open_community_store() as store:
        return [c.model_dump(mode="json") for c in _mine(store, community_contributor_id(display_name))]


def community_get_contribution(contribution_id: str) -> dict[str, Any] | None:
    from alpha_agent.community import get_contribution

    with open_community_store() as store:
        c = get_contribution(store, contribution_id)
        return c.model_dump(mode="json") if c else None


def community_replications_for(contribution_id: str) -> list[dict[str, Any]]:
    from alpha_agent.community import list_replications

    with open_community_store() as store:
        return [r.model_dump(mode="json") for r in list_replications(store, contribution_id)]


def community_evidence_state(contribution_id: str) -> dict[str, Any] | None:
    """The full read model a detail view needs: the contribution, its
    replications, and the one deterministic `EvidenceMaturity` computed from
    both -- `None` when the contribution no longer exists."""
    from alpha_agent.community import evidence_maturity, get_contribution, list_replications

    with open_community_store() as store:
        c = get_contribution(store, contribution_id)
        if c is None:
            return None
        reps = list_replications(store, contribution_id)
        maturity, rationale = evidence_maturity(c, reps)
        return {
            "contribution": c.model_dump(mode="json"),
            "replications": [r.model_dump(mode="json") for r in reps],
            "evidence_maturity": maturity.value,
            "maturity_rationale": rationale,
        }


def community_create_contribution(
    *,
    kind: str,
    visibility: str,
    contributor_display_name: str,
    experiment_id: str,
    title: str,
    notes: str = "",
    mechanism: str | None = None,
    novelty_notes: str = "",
) -> dict[str, Any]:
    from alpha_agent.community import ContributionKind, VisibilityLevel, create_contribution
    from alpha_agent.knowledge.models import EconomicMechanism

    with open_community_store() as store, open_registry() as reg:
        c = create_contribution(
            store, reg, kind=ContributionKind(kind), visibility=VisibilityLevel(visibility),
            contributor_display_name=contributor_display_name, experiment_id=experiment_id, title=title,
            notes=notes, mechanism=EconomicMechanism(mechanism) if mechanism else None, novelty_notes=novelty_notes,
        )
        return c.model_dump(mode="json")


def community_create_replication(
    *, contribution_id: str, replicator_display_name: str, experiment_id: str, notes: str = "",
) -> dict[str, Any]:
    from alpha_agent.community import create_replication, get_contribution

    with open_community_store() as store, open_registry() as reg:
        contribution = get_contribution(store, contribution_id)
        if contribution is None:
            raise ValueError(f"unknown contribution {contribution_id!r}")
        r = create_replication(
            store, reg, contribution=contribution, replicator_display_name=replicator_display_name,
            experiment_id=experiment_id, notes=notes,
        )
        return r.model_dump(mode="json")


def community_reputation(display_name: str) -> dict[str, Any]:
    from alpha_agent.community import reputation_profile

    with open_community_store() as store:
        return reputation_profile(store, community_contributor_id(display_name)).model_dump(mode="json")


def community_aggregate(*, mechanism: str, root_symbol: str) -> dict[str, Any] | None:
    """The privacy-safe (SHARED/PUBLIC-only) community aggregate for one
    (mechanism, root) pair -- `None` means no such evidence exists yet, never
    a negative signal."""
    from alpha_agent.community import aggregate_for_mechanism_root
    from alpha_agent.knowledge.models import EconomicMechanism

    with open_community_store() as store:
        agg = aggregate_for_mechanism_root(store, mechanism=EconomicMechanism(mechanism), root_symbol=root_symbol)
        return agg.model_dump(mode="json") if agg else None


def community_memory_lookup(*, mechanism: str, root_symbol: str) -> dict[str, Any]:
    """Event/Context -> Personal + Community Memory -> research
    prioritization (prompt 8 section 8) -- the Agent-facing entry point."""
    from alpha_agent.community import community_memory_lookup as _lookup
    from alpha_agent.knowledge.models import EconomicMechanism

    with open_community_store() as store, open_registry() as reg:
        result = _lookup(store, reg, mechanism=EconomicMechanism(mechanism), root_symbol=root_symbol)
        return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Phase B1 -- investor profile, Research Promise, User Fit, candidate ranking
# ---------------------------------------------------------------------------
#
# Everything below is presentation / research-prioritization only (CLAUDE.md
# / task spec sections 2/6/12): `scientific_verdict` is always copied verbatim
# from the registry's own `RegistryVerdict` (never recomputed here), and
# `InvestorProfile` never reaches `ExperimentRegistry`, `experiment_identity`,
# or `ReliabilityPolicy` -- see `alpha_agent.recommendation` for the scoring
# logic itself; this module only supplies it with real registry rows and a
# small local profile file (never a registry write).

_PROFILE_STORE = ProfileStore()


def load_investor_profile() -> InvestorProfile:
    """The user's saved research-preference profile, or `DEFAULT_PROFILE`
    (never presented as personalized) if none has been saved yet."""
    return _PROFILE_STORE.load()


def save_investor_profile(profile: InvestorProfile) -> None:
    """Persists to the local preference file ONLY -- never the
    ExperimentRegistry, never scientific evidence."""
    _PROFILE_STORE.save(profile)


def has_saved_investor_profile() -> bool:
    """Whether a real preference file has ever been saved -- distinct from
    "the loaded profile happens to equal `DEFAULT_PROFILE`" (a user could
    deliberately choose values that match the defaults). Used only so the UI
    can avoid presenting an untouched default as personalized advice."""
    return _PROFILE_STORE.path.exists()


def _canonical_candidate_rows() -> list[dict[str, Any]]:
    """Every CANONICAL (headline-adjudicated) trial, in the same row shape
    `list_experiments` already returns -- the only trials that carry a real
    `scientific_verdict` (a parameter neighbour is never individually
    adjudicated; see `services.gate_state`'s NOT_ADJUDICATED branch)."""
    return list_experiments(trial_role=TrialRole.CANONICAL)


def research_candidate_summaries(
    profile: InvestorProfile | None = None,
) -> list[ResearchCandidateSummary]:
    """One `ResearchCandidateSummary` per CANONICAL registry trial, scored
    against `profile` (defaults to the saved/default `InvestorProfile`)."""
    rows = _canonical_candidate_rows()
    return build_candidate_summaries(rows, profile=profile or DEFAULT_PROFILE, strategy_name_fn=strategy_name)


def validated_strategy_summaries(
    profile: InvestorProfile | None = None,
) -> list[ResearchCandidateSummary]:
    """PASS-only (task spec section 18) -- real registry state today is 0."""
    return validated_strategies(research_candidate_summaries(profile))


def promising_candidate_summaries(
    profile: InvestorProfile | None = None, *, sort_by: str = "promise"
) -> list[ResearchCandidateSummary]:
    """Every non-PASS candidate, ranked by Research Promise (default) or User
    Fit (`sort_by="fit"`) -- see `alpha_agent.recommendation.candidates.
    promising_candidates` for the tie-break rule."""
    return promising_candidates(research_candidate_summaries(profile), sort_by=sort_by)


def candidate_summary_for_experiment(
    experiment_key: str, profile: InvestorProfile | None = None
) -> ResearchCandidateSummary | None:
    """One `ResearchCandidateSummary` for an ARBITRARY committed experiment
    (Research Details can look up any experiment id, not only a CANONICAL
    trial's) -- `None` if the id/identity is unknown. `get_experiment`'s
    `result` dict already carries every field name
    `alpha_agent.recommendation.promise`/`.fit` read
    (`ResultRecord.model_dump`), so it is passed through unchanged."""
    try:
        detail = get_experiment(experiment_key)
    except Exception:  # noqa: BLE001 -- an unknown/stale id is an honest "not available"
        return None
    exp = detail["experiment"]
    row = {
        **(detail["result"] or {}),
        "experiment_id": exp["experiment_id"],
        "experiment_identity": exp["experiment_identity"],
        "root_symbol": exp["root_symbol"],
        "strategy_family": exp["strategy_family"],
        "verdict": detail["result"]["headline_verdict"] if detail["result"] else None,
    }
    return build_candidate_summary(row, profile=profile or DEFAULT_PROFILE, strategy_name_fn=strategy_name)


# ---------------------------------------------------------------------------
# backtest evidence: committed validation reports + trade ledgers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _phase_13_5c_reports() -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    p135c = OUTPUTS_DIR / "phase_13_5c"
    if not p135c.exists():
        return out
    for f in p135c.glob("*__*__validation_report.json"):
        root, family, _ = f.name.split("__", 2)
        out[(root, family.lower())] = f
    return out


def available_validation_reports() -> list[tuple[str, str]]:
    return sorted(_phase_13_5c_reports().keys())


def load_validation_report(root: str, family: str) -> dict[str, Any] | None:
    path = _phase_13_5c_reports().get((root, family.lower()))
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_source_artifact"] = str(path.relative_to(REPO_ROOT))
    return data


def find_registry_evidence_for_compiled(compiled: dict[str, Any]) -> dict[str, Any] | None:
    """Real, already-committed registry evidence for a just-compiled
    `CompiledStrategyProposal`, for the Agent conversation's evaluation
    stage -- NEVER a new backtest, NEVER a new verdict. Exact-fingerprint
    match first (this literal StrategySpec has already been run); otherwise
    the canonical trial for the same family/root, tagged as family-level
    evidence rather than this exact variant's; otherwise `None` (honestly,
    e.g. the novel blueprint demo scenario has no prior registry evidence).
    """
    fingerprint = compiled.get("strategy_fingerprint")
    if fingerprint:
        for row in list_experiments(trial_role=None):
            if row["strategy_fingerprint"] == fingerprint:
                return {"match_type": "exact_fingerprint", **get_experiment(row["experiment_id"])}

    root = compiled.get("root_symbol")
    family = compiled.get("family_key")
    if root and family:
        rows = list_experiments(strategy_family=family, root_symbol=root, trial_role=TrialRole.CANONICAL)
        if rows:
            return {"match_type": "same_family_root_canonical", **get_experiment(rows[0]["experiment_id"])}
    return None


#: `ExperimentRecord.candidate_manifest_fingerprint`'s own typed prefix names
#: which predeclared, frozen research program (if any) an experiment's family
#: was planned under -- this is not a new classification, it is the SAME
#: fingerprint scheme `CandidateManifest.manifest_fingerprint()` (Phase
#: 13.5C, `p135c-manifest:...`) and the Phase 15B manifest builder
#: (`p15manifest1:...`) already write to every row they create. A row planned
#: by `ResearchOrchestrator.plan_family` (Phase 18) instead carries its own
#: `FamilyManifest.content_fingerprint()`, always `familymanifestcontent1:...`
#: -- structurally a DIFFERENT, later, per-run family, never retroactively
#: mergeable into a frozen program's family/BH-FDR denominator regardless of
#: its own `TrialRole`/`Authority` value (both of those answer a different
#: question -- see the two docstrings below).
_FROZEN_PROGRAM_MANIFEST_PREFIXES = {
    "p135c-manifest:": "Frozen Phase 13.5C research program",
    "p15manifest1:": "Frozen Phase 15B research program",
}
RESEARCH_CLASSIFICATION_FROZEN_PROGRAM = "FROZEN_PROGRAM"
RESEARCH_CLASSIFICATION_EXPLORATORY = "EXPLORATORY"
RESEARCH_CLASSIFICATION_UNKNOWN = "UNKNOWN"


def research_classification(experiment: dict[str, Any]) -> dict[str, Any]:
    """Whether `experiment` belongs to a frozen, predeclared multiple-testing
    research program, or is a later standalone research-session append --
    derived ONLY from `candidate_manifest_fingerprint`'s own already-committed
    typed prefix (see `_FROZEN_PROGRAM_MANIFEST_PREFIXES` above), never from
    `TrialRole`/`Authority`.

    This distinction is real and orthogonal to two easily-conflated registry
    concepts (CLAUDE.md; Research Golden Path V1 acceptance pass, section 1B):

    * `TrialRole.CANONICAL` means "the headline member of the family THIS
      row's own `multiple_testing_family_id` names" -- a structural role
      inside whichever family (frozen or standalone) the row belongs to, NOT
      "belongs to the frozen canonical scientific program".
    * `Authority.AUTHORITATIVE` means "the current, non-superseded result for
      THIS experiment identity" -- execution-result currency, NOT eligibility
      to make a frozen, holdout-backed scientific claim.

    A standalone append is real, real-C++-executed, authoritative evidence --
    it is just never part of the frozen program's own BH/FDR family, and (see
    `ResultRecord.holdout_eligible`, already `False` for every row in this
    registry today, frozen program included -- 2025 has never been unsealed
    for anyone) never entitled to claim untouched-2025-holdout status either
    way. Nothing here computes or infers holdout eligibility; it is read
    verbatim from the committed result, exactly like every other field this
    module surfaces."""
    fp = experiment.get("candidate_manifest_fingerprint") or ""
    for prefix, program_label in _FROZEN_PROGRAM_MANIFEST_PREFIXES.items():
        if fp.startswith(prefix):
            return {
                "classification": RESEARCH_CLASSIFICATION_FROZEN_PROGRAM,
                "label": "Frozen Scientific Program",
                "origin": program_label,
            }
    if fp.startswith("familymanifestcontent1:"):
        return {
            "classification": RESEARCH_CLASSIFICATION_EXPLORATORY,
            "label": "Exploratory",
            "origin": (
                "Standalone research-session append (Agent “Run This Hypothesis” / "
                "Research Workflow “Run Historical Test”) -- its own single-member "
                "research family, never merged into a frozen program's BH/FDR family"
            ),
        }
    return {
        "classification": RESEARCH_CLASSIFICATION_UNKNOWN,
        "label": "Unclassified",
        "origin": f"Unrecognized candidate_manifest_fingerprint scheme: {fp!r}",
    }


def find_experiment_bound_trade_ledger(detail: dict[str, Any]) -> dict[str, Any] | None:
    """A per-trade ledger, IFF it is verifiably bound to this specific
    experiment's own committed authoritative result -- never matched by
    ``root_symbol`` alone.

    Phase 20's first cut attached ``outputs/phase_15/_run_real_work/<ROOT>/
    run_*/trades.csv`` to whatever experiment shared that root. That was
    WRONG: those ``run_000N`` directories are the Phase 15B ML-engine-call
    cache's execution-ORDER scratch output (``CppEngineRunner``'s internal
    counter in ``alpha_agent.ml.engine_io``), not identity-addressed --
    nothing on disk records which trial (of 12 pipelines x 5 cost scenarios)
    produced ``run_0001`` for a given root, and they belong to the Phase 15B
    ML meta-labeling family in any case, never the Phase 13.5C canonical
    baseline trials a user is typically inspecting. Matching by root_symbol
    silently showed one experiment's ledger under a different, unrelated
    experiment.

    The only safe binding: this experiment's own committed
    ``ResultRecord.source_artifact`` names a ``trades.csv`` directly, and (when
    ``source_artifact_sha256`` is recorded) the file's hash matches it exactly.
    No currently-committed ``ResultRecord`` satisfies this -- every Phase
    13.5C / 15B result's ``source_artifact`` is a ``validation_report.json`` /
    JSON report, never a trade ledger -- so this function returns ``None`` for
    every experiment in the registry today. It is written to recognise real
    provenance the moment a future phase commits it, not to reproduce today's
    (incorrect) root-based behaviour.
    """
    result = detail.get("result")
    if not result:
        return None
    source = result.get("source_artifact")
    if not source or not source.endswith("trades.csv"):
        return None
    path = REPO_ROOT / source
    if not path.exists():
        return None
    expected_sha256 = result.get("source_artifact_sha256")
    import csv
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        return None  # committed hash does not match the file on disk -- refuse

    with path.open(newline="", encoding="utf-8") as fh:
        rows: list[dict[str, Any]] = list(csv.DictReader(fh))
    return {
        "source_artifact": source,
        "source_artifact_sha256": digest,
        "trades": rows,
        "n_trades": len(rows),
    }


def trade_ledger_equity_series(ledger: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Cumulative net PnL / running drawdown from a verifiably-bound trade
    ledger's OWN rows (`find_experiment_bound_trade_ledger`), ordered by the
    frozen chronological `trade_index` column (`cpp/include/quant_core/
    trade_export.hpp`'s committed column contract) -- a running sum of real,
    already-committed per-trade `net_pnl_usd` values. This is a display
    transform of real numbers, never an equity curve inferred from a
    headline aggregate (CLAUDE.md / Release UI polish instruction). Returns
    `None` if no ledger is bound or its rows lack the frozen columns."""
    if not ledger or not ledger.get("trades"):
        return None
    trades = ledger["trades"]
    if "net_pnl_usd" not in trades[0] or "trade_index" not in trades[0]:
        return None
    try:
        ordered = sorted(trades, key=lambda r: int(r["trade_index"]))
        cum = 0.0
        peak = 0.0
        out: list[dict[str, Any]] = []
        for r in ordered:
            cum += float(r["net_pnl_usd"])
            peak = max(peak, cum)
            out.append({
                "trade_index": int(r["trade_index"]),
                "cum_net_pnl_usd": cum,
                "drawdown_usd": cum - peak,
            })
        return out
    except (TypeError, ValueError):
        return None


def max_drawdown_usd(series: list[dict[str, Any]] | None) -> float | None:
    """Max drawdown (USD, as a positive magnitude) from an already-computed
    `trade_ledger_equity_series` -- the SAME real, per-trade running
    peak-to-trough the Drawdown chart plots, summarized to one number for a
    metric card (MARKET REALITY pass, mission section 14). `None` when no
    series is bound, never a fabricated value."""
    if not series:
        return None
    return abs(min(r["drawdown_usd"] for r in series))


# ---------------------------------------------------------------------------
# Alpha Discovery Part I/J: verifiably-bound experiment artifact bundles
# (fills/trades/daily-equity) -- same non-negotiable binding discipline as
# find_experiment_bound_trade_ledger above: never matched by root or family,
# only by a committed ResultRecord.source_artifact pointer whose content hash
# is independently re-verified. Every currently committed ResultRecord's
# source_artifact is either None or a Phase 13.5C/15B validation_report.json
# -- this returns None for all of them; it exists for NEW discovery runs that
# opted into alpha_agent.artifacts.store's artifact_dir.
# ---------------------------------------------------------------------------


def find_experiment_bound_artifact_bundle(detail: dict[str, Any]) -> dict[str, Any] | None:
    """The Part I `ArtifactBundle` (as a plain dict), IFF this experiment's
    own committed `ResultRecord.source_artifact` names an artifact-bundle
    manifest directly, the manifest's own content re-hashes to its recorded
    `bundle_sha256`, that matches the committed `source_artifact_sha256`
    (when recorded), AND every individual file the bundle references
    (fills/trades/daily_equity) still hashes to what the bundle says it
    should. Any mismatch anywhere in that chain returns `None` -- never a
    partially-trusted bundle."""
    result = detail.get("result")
    if not result:
        return None
    source = result.get("source_artifact")
    if not source or not source.endswith("manifest.json"):
        return None
    path = Path(source)
    if not path.is_absolute():
        path = REPO_ROOT / source
    if not path.exists():
        return None

    import hashlib

    from alpha_agent.artifacts.store import read_artifact_bundle

    try:
        bundle = read_artifact_bundle(path)
    except Exception:  # noqa: BLE001 -- a malformed manifest is untrusted, never a crash
        return None

    recomputed = hashlib.sha256(bundle.canonical_json().encode("utf-8")).hexdigest()
    if recomputed != bundle.bundle_sha256:
        return None  # the manifest's own content does not match its own recorded hash
    expected = result.get("source_artifact_sha256")
    if expected and expected != bundle.bundle_sha256:
        return None  # committed hash does not match the bundle -- refuse

    for artifact_file in (bundle.fills, bundle.trades, bundle.daily_equity, bundle.price_bars):
        if artifact_file is None:
            continue
        fp = Path(artifact_file.path)
        if not fp.exists() or hashlib.sha256(fp.read_bytes()).hexdigest() != artifact_file.sha256:
            return None  # a referenced file was moved, edited, or is missing -- refuse the whole bundle

    return bundle.model_dump(mode="json")


def daily_equity_rows_from_bundle(bundle: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """The real, official daily-equity trace (`alpha_agent.validation.engine`'s
    `report.oos_daily`, persisted verbatim by `alpha_agent.artifacts.store`)
    from an already hash-verified bundle (`find_experiment_bound_artifact_bundle`).
    `None` if the bundle carries no daily-equity file at all."""
    if not bundle or not bundle.get("daily_equity"):
        return None
    import csv

    path = Path(bundle["daily_equity"]["path"])
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fills_rows_from_bundle(bundle: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """The real per-fill export (`fills.csv`, columns frozen by the C++
    export -- `fill_index, fill_id, order_id, ts_fill_ns, instrument_id,
    raw_symbol, side, quantity, fill_price, commission_usd, slippage_ticks`)
    from an already hash-verified bundle. `None` if the bundle carries no
    fills file."""
    if not bundle or not bundle.get("fills"):
        return None
    import csv

    path = Path(bundle["fills"]["path"])
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def trades_rows_from_bundle(bundle: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """The real per-trade export (`trades.csv`) from an already hash-verified
    bundle. `None` if the bundle carries no trades file."""
    if not bundle or not bundle.get("trades"):
        return None
    import csv

    path = Path(bundle["trades"]["path"])
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def daily_equity_series_from_bundle(bundle: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """The equity-curve/drawdown-chart companion to `trade_ledger_equity_series`,
    sourced from a hash-verified artifact bundle's own real, official daily
    equity trace (`daily_equity_rows_from_bundle` -- the C++ portfolio
    accountant's `oos_daily` mark-to-market series, persisted verbatim) rather
    than a per-trade-summed curve. Re-based to start at zero (subtracting the
    first row's own `equity_usd`) so it plots on the same "cumulative net PnL"
    axis the existing equity/drawdown charts and `max_drawdown_usd` already
    use -- a transparent, documented re-basing of the real committed values,
    never a new computation, smoothing, or invented starting value. Same
    `cum_net_pnl_usd`/`drawdown_usd` key contract as `trade_ledger_equity_series`
    so `max_drawdown_usd` and the existing chart helpers work unchanged.
    `None` if the bundle carries no daily-equity file or its rows lack the
    frozen `equity_usd` column."""
    rows = daily_equity_rows_from_bundle(bundle)
    if not rows or "equity_usd" not in rows[0]:
        return None
    try:
        base = float(rows[0]["equity_usd"])
        peak = 0.0
        out: list[dict[str, Any]] = []
        for r in rows:
            cum = float(r["equity_usd"]) - base
            peak = max(peak, cum)
            out.append({
                "trading_day": r.get("trading_day"),
                "cum_net_pnl_usd": cum,
                "drawdown_usd": cum - peak,
            })
        return out
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Release UX Part G -- Price + Signal + Position (bounded window around one
# REAL trade, from the same hash-verified artifact bundle). Never
# reconstructs a bar, a fill, or a signal timestamp -- see module docstring
# discipline elsewhere in this file: an unavailable artifact renders
# unavailable, it is never synthesized.
# ---------------------------------------------------------------------------


def price_signal_window_from_bundle(
    bundle: dict[str, Any] | None, *, trade_index: int, context_bars: int = 60,
) -> dict[str, Any] | None:
    """A bounded window of REAL `price_bars` rows around one trade's own
    `[ts_open_ns, ts_close_ns]`, padded by `context_bars` real bars each
    side, plus the real fills/trades that fall inside that same window.

    `price_bars.csv` can be very large (a multi-year 1-minute history) --
    this streams the file once and keeps only rows for the trade's own
    `instrument_id`, never loading an unrelated contract's bars. Returns
    `None` if the bundle carries no price_bars/trades file, or the requested
    `trade_index` does not exist."""
    if not bundle or not bundle.get("price_bars") or not bundle.get("trades"):
        return None
    trades = trades_rows_from_bundle(bundle) or []
    trade = next((t for t in trades if int(t["trade_index"]) == trade_index), None)
    if trade is None:
        return None

    instrument_id = str(trade["instrument_id"])
    ts_open = int(trade["ts_open_ns"])
    ts_close = int(trade["ts_close_ns"])

    import csv

    path = Path(bundle["price_bars"]["path"])
    if not path.exists():
        return None
    same_instrument: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("instrument_id") == instrument_id:
                same_instrument.append(row)
    same_instrument.sort(key=lambda r: int(r["ts_event_ns"]))
    if not same_instrument:
        return None

    idx_in_range = [i for i, r in enumerate(same_instrument) if ts_open <= int(r["ts_event_ns"]) <= ts_close]
    if not idx_in_range:
        # The trade's own bars are not in this instrument's slice (e.g. a
        # roll-boundary edge case) -- fall back to the single nearest bar at
        # or after ts_open rather than fabricating a window.
        idx_in_range = [i for i, r in enumerate(same_instrument) if int(r["ts_event_ns"]) >= ts_open][:1]
    if not idx_in_range:
        return None

    lo = max(0, idx_in_range[0] - context_bars)
    hi = min(len(same_instrument), idx_in_range[-1] + context_bars + 1)
    window_bars = same_instrument[lo:hi]
    window_start_ns = int(window_bars[0]["ts_event_ns"])
    window_end_ns = int(window_bars[-1]["ts_event_ns"])

    fills = fills_rows_from_bundle(bundle) or []
    window_fills = [
        f for f in fills
        if f.get("instrument_id") == instrument_id and window_start_ns <= int(f["ts_fill_ns"]) <= window_end_ns
    ]

    return {
        "trade": trade, "bars": window_bars, "fills": window_fills,
        "window_start_ns": window_start_ns, "window_end_ns": window_end_ns,
    }


# ---------------------------------------------------------------------------
# Phase 15B ML meta-labeling adjudication (read-only)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def ml_meta_label_report() -> dict[str, Any] | None:
    if not PHASE_15B_REPORT.exists():
        return None
    data = json.loads(PHASE_15B_REPORT.read_text(encoding="utf-8"))
    data["_source_artifact"] = str(PHASE_15B_REPORT.relative_to(REPO_ROOT))
    return data


# ---------------------------------------------------------------------------
# gate-by-gate validation state -- EXPLICIT committed evidence only
# (Phase 20.1, corrected in Phase 20.1b)
# ---------------------------------------------------------------------------
#
# Phase 20 inferred a gate PASS from the mere ABSENCE of its failure reason
# code. That is wrong: the committed Phase 13.5C report never serialises the
# frozen policy's actual per-gate `gate_results` dict (see
# `alpha_agent.validation.policy.evaluate_policy` / `PolicyOutcome.gate_results`
# -- computed in memory, dropped by `phase_13_5c_report._per_trial_json`), so
# there is no on-disk boolean to read for an individual gate. A reason code
# being present IS explicit, positive, committed evidence that gate failed; its
# absence is NOT proof the gate passed (it is equally consistent with "never
# evaluated" -- e.g. an earlier minimum-sample refusal short-circuits every
# later gate). Phase 20.1 fixed the PASS half of this but still labelled every
# such absence NOT_EVALUATED -- itself an unjustified factual claim requiring
# its own committed proof, which most gates never carry. Phase 20.1b: a gate
# reads PASS, FAIL, NOT_EVALUATED, or REFUSED_BEFORE_GATE only from its own
# matching explicit reason code; every other case -- including a REJECT
# verdict's fail code being absent, or an evidence-family INCONCLUSIVE for a
# gate with no committed "not evaluated" code of its own -- is NOT_AVAILABLE,
# never a positive claim in either direction. Never reconstructs a missing
# historical gate decision and never reruns Phase 13 validation to populate
# this; never a re-derived threshold, never a UI-side adjudication.

#: (display label, ReasonCode.value for an explicit FAIL of this gate).
#: Values are the exact strings of `alpha_agent.validation.enums.ReasonCode`
#: (duplicated, not imported, to keep this module import-light); see
#: `test_gate_definitions_match_the_real_reason_code_enum` for byte-exact proof
#: against the source enum.
GATE_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("Null Hypothesis (Bootstrap)", "null_hypothesis_not_rejected"),
    ("Multiple Testing (BH-FDR)", "fdr_qvalue_above_threshold"),
    ("Deflated Sharpe Ratio", "deflated_sharpe_below_threshold"),
    ("Positive OOS Net PnL", "negative_oos_net_pnl"),
    ("Walk-Forward Consistency", "fold_consistency_below_threshold"),
    ("Cost Stress", "cost_stress_degradation_exceeds_limit"),
    ("Parameter Stability", "parameter_neighbourhood_unstable"),
    ("Regime Robustness", "performance_concentrated_in_one_regime"),
    ("Cross-Market Evidence", "performance_concentrated_in_one_root"),
)

#: ReasonCode values that mean the whole evaluation was refused before any
#: statistical gate ran at all (Section 8's minimum-evidence early return).
_MINIMUM_SAMPLE_CODES = frozenset(
    {
        "insufficient_oos_days",
        "insufficient_trades",
        "insufficient_folds",
        "insufficient_nonzero_observations",
        "no_daily_trace",
    }
)

#: The one reason code the frozen policy emits when EVERY required gate was
#: explicitly satisfied -- the sole committed, positive proof a gate may show PASS.
_ALL_GATES_SATISFIED = "all_required_gates_satisfied"

#: fail_code -> the ReasonCode.value the frozen policy emits when that SAME
#: gate's evidence family was explicitly not evaluated (section 17/18's
#: `not_run` step, extended by the validation-safety fix to also cover
#: parameter stability -- an UNCONDITIONALLY required gate whose missing
#: evidence must never render as PASS just because the headline verdict is
#: PASS). The remaining six gates still have no committed "not evaluated"
#: signal at all -- for them, an absent fail code can NEVER be resolved to
#: NOT_EVALUATED, only NOT_AVAILABLE.
_NOT_EVALUATED_CODES: dict[str, str] = {
    "performance_concentrated_in_one_regime": "regime_not_evaluated",
    "performance_concentrated_in_one_root": "cross_market_not_evaluated",
    "parameter_neighbourhood_unstable": "parameter_stability_not_evaluated",
}

GATE_STATES = (
    "PASS",
    "FAIL",
    "NOT_EVALUATED",
    "NOT_AVAILABLE",
    "REFUSED_BEFORE_GATE",
    "NOT_ADJUDICATED",
)


def gate_state(
    *,
    fail_code: str,
    reason_codes: Sequence[str],
    verdict: str | None,
    trial_role: str,
) -> str:
    """One gate's display state, derived ONLY from what is explicitly
    committed -- never a threshold applied here, never inferred from absence.

    Phase 20.1 fixed inferring PASS from an absent fail code. Phase 20.1b
    fixes the remaining half of the same mistake: NOT_EVALUATED is itself a
    factual claim, and the committed Phase 13.5C artifact does not serialise
    `PolicyOutcome.gate_results`, so an absent fail code is compatible with
    EITHER "evaluated and passed" OR "never evaluated" -- evaluate_policy
    exhaustively computes every gate whose precondition was met (no
    short-circuiting after gate 1) but a precondition simply not being met
    (e.g. no `cost_report`) looks identical from outside to "evaluated,
    passed". Only an explicit committed signal may resolve the ambiguity in
    either direction:

    - this gate's own fail code present                              -> FAIL
    - `all_required_gates_satisfied` present (verdict PASS)           -> PASS
    - this gate's own "not evaluated" code present (`_NOT_EVALUATED_CODES`,
      section 17/18's `not_run` step -- regime / cross-market only)   -> NOT_EVALUATED
    - a minimum-sample code present (section 8's early return, which
      provably short-circuits every later gate)                      -> REFUSED_BEFORE_GATE
    - otherwise: committed evidence cannot distinguish PASS from
      NOT_EVALUATED for this gate                                    -> NOT_AVAILABLE

    Never reconstructs a missing historical gate decision and never reruns
    Phase 13 validation to populate this. Returns one of `GATE_STATES`.
    """
    if trial_role != "CANONICAL":
        # The frozen Phase 13 policy headline-adjudicates only the canonical
        # trial (CLAUDE.md); a neighbour's individual gates are never rendered.
        return "NOT_ADJUDICATED"
    if verdict is None:
        return "NOT_AVAILABLE"
    if fail_code in reason_codes:
        return "FAIL"
    # This gate's own "not evaluated" code is checked BEFORE the generic
    # headline-PASS inference below (validation-safety fix, defense in
    # depth): `evaluate_policy` can no longer emit PASS/ALL_GATES_SATISFIED
    # alongside a "not evaluated" code for a required gate, but this
    # presentation function never assumes that invariant holds elsewhere --
    # an explicit "not evaluated" claim always outranks an inferred PASS.
    not_evaluated_code = _NOT_EVALUATED_CODES.get(fail_code)
    if not_evaluated_code and not_evaluated_code in reason_codes:
        return "NOT_EVALUATED"
    if verdict == "PASS" and _ALL_GATES_SATISFIED in reason_codes:
        return "PASS"
    if any(c in reason_codes for c in _MINIMUM_SAMPLE_CODES):
        return "REFUSED_BEFORE_GATE"
    if verdict == "NOT_ADJUDICATED":
        return "NOT_ADJUDICATED"
    # REJECT, INCONCLUSIVE, or anything else with none of the explicit codes
    # above: this gate's own outcome is not committed evidence one way or the
    # other -- never a positive PASS or NOT_EVALUATED claim.
    return "NOT_AVAILABLE"


def gate_evidence_text(result: dict[str, Any] | None) -> dict[str, str]:
    """A short, factual statistic line per `GATE_DEFINITIONS` label -- every
    number is read directly off the committed `ResultRecord`; nothing here
    judges pass/fail."""
    if not result:
        return {}
    out: dict[str, str] = {}
    if result.get("gating_null_p") is not None:
        out["Null Hypothesis (Bootstrap)"] = f"gating null p = {result['gating_null_p']:.3f}"
    if result.get("bh_q") is not None:
        out["Multiple Testing (BH-FDR)"] = f"q-value = {result['bh_q']:.4f}"
    if result.get("dsr_probability") is not None:
        out["Deflated Sharpe Ratio"] = f"P(DSR) = {result['dsr_probability']:.4f}"
    if result.get("net_pnl_usd") is not None:
        out["Positive OOS Net PnL"] = f"net PnL = ${result['net_pnl_usd']:,.0f}"
    if result.get("fold_consistency") is not None:
        out["Walk-Forward Consistency"] = f"fold consistency = {result['fold_consistency']:.2f}"
    cost = result.get("cost_stress") or {}
    if "max_net_pnl_degradation" in cost:
        out["Cost Stress"] = f"max degradation = {cost['max_net_pnl_degradation']:.2%}"
    ps = result.get("parameter_stability") or {}
    if "fraction_positive_sharpe" in ps:
        out["Parameter Stability"] = (
            f"{ps['fraction_positive_sharpe']:.0%} positive Sharpe across {ps.get('n_evaluated', 0)} neighbours"
        )
    regime = result.get("regime_evidence") or {}
    if regime.get("status") == "evaluated":
        out["Regime Robustness"] = f"max regime PnL share = {regime.get('max_regime_pnl_share', 0):.1%}"
    elif regime.get("status"):
        out["Regime Robustness"] = f"evidence status: {regime['status']}"
    xmkt = result.get("cross_market_reference") or {}
    if xmkt.get("status") == "evaluated":
        out["Cross-Market Evidence"] = f"max single-root PnL share = {xmkt.get('max_single_root_pnl_share', 0):.1%}"
    elif xmkt.get("status"):
        out["Cross-Market Evidence"] = f"evidence status: {xmkt['status']}"
    return out


def plain_language_reason(
    *, result: dict[str, Any] | None, trial_role: str, verdict: str | None
) -> str:
    """A short, investor-readable synthesis of WHY a trial resolved the way it
    did -- built ONLY from `GATE_DEFINITIONS`'s own labels and the same
    `gate_state` resolution the gate-by-gate grid already renders (Release UX,
    investor-readability pass). Never a new judgment, never a threshold this
    function applies itself -- it only restates which already-committed gates
    explicitly failed, in the same words the grid uses. The exact machine
    reason codes remain available wherever the caller already shows them
    (`result['reason_codes']`)."""
    if verdict == "PASS":
        return "All required validation gates were explicitly satisfied."
    reason_codes = tuple((result or {}).get("reason_codes") or ())
    states = {
        label: gate_state(fail_code=code, reason_codes=reason_codes, verdict=verdict, trial_role=trial_role)
        for label, code in GATE_DEFINITIONS
    }
    failed = [label for label, state in states.items() if state == "FAIL"]
    if failed:
        if len(failed) == 1:
            return f"{failed[0]} was not met."
        return ", ".join(failed[:-1]) + f", and {failed[-1]} requirements were not met."
    if any(state == "REFUSED_BEFORE_GATE" for state in states.values()):
        return "Evaluation was refused before any statistical gate ran -- an insufficient sample (trades/days/folds)."
    if verdict == "INCONCLUSIVE":
        return "Committed evidence was insufficient to reach a PASS or REJECT verdict."
    if verdict is None:
        return "This experiment has no committed scientific verdict yet."
    return "See the gate-by-gate detail below for the exact committed reason codes."


def gate_table(*, reason_codes: Sequence[str], verdict: str | None, trial_role: str) -> list[dict[str, str]]:
    """The full `GATE_DEFINITIONS` table with each gate's explicit-only state."""
    return [
        {
            "label": label,
            "state": gate_state(
                fail_code=code, reason_codes=reason_codes, verdict=verdict, trial_role=trial_role
            ),
        }
        for label, code in GATE_DEFINITIONS
    ]


# ---------------------------------------------------------------------------
# Release UI polish -- registry-level (portfolio-wide) evidence aggregation.
#
# Every function below is a pure tally over rows `list_experiments()` /
# `registry_summary()` already return -- no new statistic, threshold, or
# verdict is computed here. This exists so the Overview page can fill its
# main visual area with REAL committed evidence instead of reserving empty
# "NOT AVAILABLE" chart panels for the (today, always absent) per-experiment
# equity/trade series. Never used to infer or imply a headline verdict for
# any individual experiment beyond what the registry itself already says.
# ---------------------------------------------------------------------------


def canonical_verdict_distribution() -> dict[str, int]:
    """The 61 canonical (headline-adjudicated) trials' verdict counts, exactly
    as `RegistrySummary.canonical_verdict_counts` already reports them."""
    s = registry_summary()
    counts = dict(s.get("canonical_verdict_counts", {}))
    for k in ("PASS", "REJECT", "INCONCLUSIVE", "NOT_ADJUDICATED"):
        counts.setdefault(k, 0)
    return counts


def strategy_market_matrix() -> list[dict[str, Any]]:
    """One aggregated cell per (root, strategy_family) among the CANONICAL
    (headline-adjudicated) trials -- the only trials a verdict can be shown
    for. `count` is the real number of canonical experiment identities in
    that cell (a family/root combination the frozen candidate manifest and
    Phase 15B both may declare more than one canonical variant for, e.g. a
    regime-baseline + ablation pairing); `verdict` is that verdict if every
    experiment in the cell agrees, else the literal string "MIXED" -- never a
    single value picked arbitrarily from several disagreeing ones."""
    rows = list_experiments(trial_role=TrialRole.CANONICAL)
    cells: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        key = (r["root_symbol"], r["strategy_family"])
        cells.setdefault(key, []).append(r["verdict"] or "NOT_ADJUDICATED")
    out: list[dict[str, Any]] = []
    for (root, family), verdicts in cells.items():
        unique = set(verdicts)
        out.append({
            "root_symbol": root,
            "strategy_family": family,
            "count": len(verdicts),
            "verdict": next(iter(unique)) if len(unique) == 1 else "MIXED",
        })
    return out


def canonical_sharpe_vs_fdr() -> list[dict[str, Any]]:
    """One point per CANONICAL trial that carries both a committed
    `annualized_sharpe` and `bh_q` -- the exact fields `list_experiments()`
    already exposes from the authoritative `ResultRecord`."""
    rows = list_experiments(trial_role=TrialRole.CANONICAL)
    return [
        {
            "root_symbol": r["root_symbol"],
            "strategy_family": r["strategy_family"],
            "experiment_id": r["experiment_id"],
            "annualized_sharpe": r["annualized_sharpe"],
            "bh_q": r["bh_q"],
            "verdict": r["verdict"] or "NOT_ADJUDICATED",
        }
        for r in rows
        if r["annualized_sharpe"] is not None and r["bh_q"] is not None
    ]


def failure_reason_distribution(*, top_n: int = 10) -> list[tuple[str, int]]:
    """The registry's own `failure_class_counts`, sorted descending -- exactly
    the typed, permanently-preserved failure-memory tally `RegistrySummary`
    already reports."""
    s = registry_summary()
    items = sorted(s.get("failure_class_counts", {}).items(), key=lambda kv: kv[1], reverse=True)
    return items[:top_n]


def gate_failure_rates() -> list[dict[str, Any]]:
    """Per-gate state tally across every CANONICAL trial (the only trials any
    gate is individually adjudicated for), using the same `gate_state`
    resolution the Validation page renders per-experiment -- aggregated here,
    never re-derived with different logic."""
    rows = list_experiments(trial_role=TrialRole.CANONICAL)
    tally: dict[str, dict[str, int]] = {label: {} for label, _ in GATE_DEFINITIONS}
    for r in rows:
        table = gate_table(reason_codes=r["reason_codes"], verdict=r["verdict"], trial_role=r["trial_role"])
        for entry in table:
            bucket = tally[entry["label"]]
            bucket[entry["state"]] = bucket.get(entry["state"], 0) + 1
    return [{"label": label, "states": tally[label]} for label, _ in GATE_DEFINITIONS]


# ---------------------------------------------------------------------------
# Phase 21 -- paper trading (read-only)
# ---------------------------------------------------------------------------
#
# Nothing here computes PnL, a fill, or a risk decision, and nothing here
# starts, steps, or stops a paper run -- that requires
# `scripts/phase_21_paper_trading.py` (or a direct `PaperTradingEngine` call),
# never this page's read path. `paper_eligible_experiments` is the
# deterministic output of the already-approved
# `alpha_agent.paper.eligibility.assert_paper_trading_eligible` service; the
# ledger reads below are plain SELECTs over the already-committed persistent
# paper ledger (`alpha_agent.paper.ledger.PaperLedger`), exactly like this
# module's registry reads.

PAPER_LEDGER_PATH = REPO_ROOT / "data" / "paper_trading" / "paper_ledger.sqlite"


def paper_ledger_exists() -> bool:
    return PAPER_LEDGER_PATH.exists()


def paper_eligible_experiments() -> list[dict[str, Any]]:
    """Every AUTHORITATIVE registry experiment that
    `alpha_agent.paper.eligibility.assert_paper_trading_eligible` currently
    admits -- i.e. has passed frozen validation AND whose StrategySpec this
    phase can deterministically rebuild and fingerprint-verify. Real registry
    state today is 0/N (see `docs/EXPERIMENT_REGISTRY.md` / phase-status notes);
    this is not hidden or special-cased -- it is exactly what the frozen
    evidence says."""
    from alpha_agent.paper.eligibility import (
        PaperTradingEligibilityError,
        assert_paper_trading_eligible,
    )

    out: list[dict[str, Any]] = []
    with open_registry() as reg:
        for view in reg.experiments(authoritative_only=True):
            try:
                elig = assert_paper_trading_eligible(reg, view.experiment_identity)
            except PaperTradingEligibilityError:
                continue
            out.append(
                {
                    "experiment_id": elig.experiment_id,
                    "experiment_identity": elig.experiment_identity,
                    "root_symbol": elig.root_symbol,
                    "strategy_family": elig.strategy_family,
                    "params": dict(elig.params),
                    "backtest_daily_sharpe": elig.backtest_daily_sharpe,
                    "backtest_net_pnl_usd": elig.backtest_net_pnl_usd,
                    "backtest_n_trades": elig.backtest_n_trades,
                }
            )
    return out


def paper_eligible_experiment(experiment_key: str) -> bool:
    """Whether ONE experiment (by id or identity) currently passes
    `assert_paper_trading_eligible` -- the same real, deterministic check
    `paper_eligible_experiments` sweeps the whole registry with, just for a
    single already-known id (e.g. the Agent page's own "Run This Hypothesis"
    result summary). Never raises; a missing/ineligible experiment is simply
    `False`, exactly matching CLAUDE.md's paper-eligibility gate."""
    from alpha_agent.paper.eligibility import (
        PaperTradingEligibilityError,
        assert_paper_trading_eligible,
    )

    try:
        with open_registry() as reg:
            assert_paper_trading_eligible(reg, experiment_key)
        return True
    except PaperTradingEligibilityError:
        return False


def list_paper_runs() -> list[dict[str, Any]]:
    if not PAPER_LEDGER_PATH.exists():
        return []
    from alpha_agent.paper.ledger import PaperLedger

    with PaperLedger(PAPER_LEDGER_PATH) as ledger:
        return [r.model_dump(mode="json") for r in ledger.list_runs()]


def paper_run_report(run_id: str) -> dict[str, Any] | None:
    """Full committed state for one paper run (run + every step / fill /
    alert row), or `None` if the ledger or the run does not exist."""
    if not PAPER_LEDGER_PATH.exists():
        return None
    from alpha_agent.paper.ledger import PaperLedger, UnknownPaperRun
    from alpha_agent.paper.report import build_paper_run_report

    with PaperLedger(PAPER_LEDGER_PATH) as ledger:
        try:
            return build_paper_run_report(ledger, run_id)
        except UnknownPaperRun:
            return None


# ---------------------------------------------------------------------------
# Phase 22 -- crypto / on-chain extension (SYNTHETIC SCAFFOLD ONLY)
# ---------------------------------------------------------------------------
# Reads ONLY the isolated `data/crypto_synthetic/registry.sqlite` store
# (`alpha_agent.crypto.synthetic_registry`), never the real Phase 14 registry
# above. Every row it returns is stamped `data_role == "SYNTHETIC"` at the
# schema level (a CHECK constraint) -- this module renders that stamp, it
# never infers or overrides it. Writing a new synthetic experiment is
# CLI-only (`scripts/phase_22_crypto_research.py`), matching Phase 21's
# paper-trading convention: nothing below ever calls
# `alpha_agent.crypto.synthetic_registry.record_synthetic_experiment`.

CRYPTO_SYNTHETIC_DB_PATH = REPO_ROOT / "data" / "crypto_synthetic" / "registry.sqlite"


def crypto_synthetic_db_exists() -> bool:
    return CRYPTO_SYNTHETIC_DB_PATH.exists()


def crypto_synthetic_banner() -> str:
    from alpha_agent.crypto.provenance import SYNTHETIC_BANNER

    return SYNTHETIC_BANNER


def list_crypto_synthetic_experiments() -> list[dict[str, Any]]:
    """Every Phase 22 synthetic crypto research result recorded so far. Never
    the real registry, never a paper-trading candidate, never a real BH/FDR
    trial -- see `alpha_agent.crypto.synthetic_registry` module docstring."""
    if not CRYPTO_SYNTHETIC_DB_PATH.exists():
        return []
    from alpha_agent.crypto.synthetic_registry import list_synthetic_experiments

    return list_synthetic_experiments(db_path=CRYPTO_SYNTHETIC_DB_PATH)


def get_crypto_synthetic_experiment(experiment_id: str) -> dict[str, Any] | None:
    if not CRYPTO_SYNTHETIC_DB_PATH.exists():
        return None
    from alpha_agent.crypto.synthetic_registry import get_synthetic_experiment

    return get_synthetic_experiment(experiment_id, db_path=CRYPTO_SYNTHETIC_DB_PATH)


# ---------------------------------------------------------------------------
# Phase 4 -- Learn (read-only; `alpha_agent.learn`'s own module docstrings are
# the full boundary: no registry write, no market-data call, no execution).
# Concepts/Strategies/Learning Paths are pure, fixed content with no registry
# dependency; the Failure Autopsy reuses this module's OWN already-tested
# `gate_table`/`gate_evidence_text` (never a second gate-state resolution).
# ---------------------------------------------------------------------------


def learn_concepts() -> list[dict[str, Any]]:
    from alpha_agent.learn.concepts import list_concepts

    return [c.model_dump(mode="json") for c in list_concepts()]


def learn_concept(concept_id: str) -> dict[str, Any] | None:
    from alpha_agent.learn.concepts import get_concept

    c = get_concept(concept_id)
    return c.model_dump(mode="json") if c else None


def learn_strategy_explainers() -> list[dict[str, Any]]:
    from alpha_agent.learn.strategy_explainers import list_strategy_explainers

    return [s.model_dump(mode="json") for s in list_strategy_explainers()]


def learn_strategy_explainer(family_key: str) -> dict[str, Any] | None:
    from alpha_agent.learn.strategy_explainers import build_strategy_explainer

    try:
        return build_strategy_explainer(family_key).model_dump(mode="json")
    except KeyError:
        return None


def learn_learning_paths() -> list[dict[str, Any]]:
    from alpha_agent.learn.learning_paths import list_learning_paths

    return [p.model_dump(mode="json") for p in list_learning_paths()]


def learn_pipeline_walkthrough(experiment_id: str | None = None) -> dict[str, Any]:
    """A generic pipeline walkthrough, or -- when ``experiment_id`` names an
    experiment with a verifiably-bound trade ledger -- one attached real
    example value per applicable stage (never a fabricated one)."""
    from alpha_agent.learn.pipeline_explainer import build_pipeline_walkthrough

    ledger = None
    if experiment_id:
        detail = get_experiment(experiment_id)
        ledger = find_experiment_bound_trade_ledger(detail)
    return build_pipeline_walkthrough(trade_ledger=ledger).model_dump(mode="json")


def learn_failure_autopsy(experiment_id: str) -> dict[str, Any]:
    from alpha_agent.learn.autopsy import build_failure_autopsy

    with open_registry() as reg:
        view = reg.get(experiment_id)
        result = view.result.model_dump(mode="json") if view.result else None
        verdict = view.verdict.value if view.verdict else None
        reason_codes = list(view.result.reason_codes) if view.result else []
        rows = gate_table(reason_codes=reason_codes, verdict=verdict, trial_role=view.experiment.trial_role.value)
        evidence = gate_evidence_text(result)
        autopsy = build_failure_autopsy(reg, experiment_id, gate_table=rows, gate_evidence=evidence)
    return autopsy.model_dump(mode="json")


def learn_failure_autopsy_narrative(experiment_id: str, *, client) -> str | None:
    """Optional live-Claude enrichment on top of `learn_failure_autopsy` --
    NEVER called by a default page render (see
    `test_phase4_learn_ui.py::test_learn_never_calls_claude_or_market_data_by_default`).
    Returns `None` on any schema/rejection/transport failure rather than
    raising -- mirrors `decision_brief.generate_decision_brief`'s own
    defensive UI-boundary handling -- so a beginner reading the page never
    sees a raw LLM stack trace, just an honest "no explanation available"."""
    from alpha_agent.learn.autopsy import FailureAutopsy
    from alpha_agent.learn.narration_agent import FailureAutopsyNarrator, NarrationRejected

    autopsy = FailureAutopsy.model_validate(learn_failure_autopsy(experiment_id))
    try:
        return FailureAutopsyNarrator(client).narrate(autopsy)
    except NarrationRejected:
        return None
    except Exception:  # noqa: BLE001 -- defensive UI boundary, never a raw stack trace or key material
        return None

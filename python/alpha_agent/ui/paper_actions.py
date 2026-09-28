"""The ONE write boundary from the Streamlit UI into paper trading (Product
UI Polish pass, section 11: "Paper Start Workflow").

Deliberately separate from `alpha_agent.ui.services` -- the read-only
registry/report/catalog boundary, statically grep-tested
(`tests/python/test_phase_21_streamlit_ui.py`) to never call a
`PaperLedger` write method. This module exists ONLY to start a new paper run
through the EXISTING, unchanged `alpha_agent.paper.engine.PaperTradingEngine`
-- the exact same call path `scripts/phase_21_paper_trading.py start` already
uses -- never a second, Streamlit-side reimplementation of paper execution
(CLAUDE.md architecture boundary: C++ owns fills/PnL/risk; this module only
wires the SAME Python orchestration the CLI already wires).

Eligibility is decided once more, INSIDE `PaperTradingEngine.start_run`
itself (`alpha_agent.paper.eligibility.assert_paper_trading_eligible`), so a
UI action can never start a run for a strategy that is not genuinely eligible
even if the page's own cached eligibility list were somehow stale. The replay
window is hardcoded to the same historical bounds the CLI uses -- the
already-acquired 2018-2024 research + validation data -- and can never reach
the 2025 holdout (CLAUDE.md: "Any code path touching ts >= 2025-01-01 must
fail loudly"; enforced independently inside `RealMarketWindowProvider` /
`reconstitute_root`, not merely by this module choosing a safe constant).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from alpha_agent.data.real_market_dataset import HOLDOUT_START_NS, RESEARCH_WINDOW
from alpha_agent.paper.engine import DEFAULT_PAPER_TRADING_CLI, PaperTradingEngine
from alpha_agent.paper.errors import PaperTradingError
from alpha_agent.paper.ledger import PaperLedger
from alpha_agent.paper.risk_policy import PaperRiskPolicy
from alpha_agent.paper.schedule_window import RealMarketWindowProvider
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.ui import services

#: Same historical replay bounds `scripts/phase_21_paper_trading.py` uses --
#: never the 2025 holdout, never a live feed.
WINDOW_START_NS: int = int(datetime.fromisoformat(RESEARCH_WINDOW[0]).replace(tzinfo=UTC).timestamp() * 1e9)
WINDOW_END_NS: int = HOLDOUT_START_NS  # 2025-01-01, exclusive -- never crossed


@dataclass(frozen=True)
class StartPaperRunResult:
    """The outcome of one `start_paper_run` call -- never a fabricated
    `run_id` on failure."""

    ok: bool
    run_id: str | None = None
    error: str | None = None


def start_paper_run(
    experiment_id: str,
    *,
    capital_usd: float = 100_000.0,
    max_drawdown_pct: float = 0.10,
    max_daily_loss_usd: float = 1500.0,
    commission_per_contract_usd: float = 2.0,
) -> StartPaperRunResult:
    """Start one new paper run for `experiment_id`, via the exact same
    `PaperTradingEngine.start_run` call `phase_21_paper_trading.py start`
    uses -- same registry path, same ledger path, same compiled C++
    executable, same risk-policy defaults. A refusal from
    `assert_paper_trading_eligible` (re-checked inside `start_run`) or the
    C++ boundary comes back as `StartPaperRunResult(ok=False, error=...)`,
    never a silently-swallowed exception and never a fabricated run id."""
    try:
        with ExperimentRegistry(services.REGISTRY_PATH) as reg, PaperLedger(services.PAPER_LEDGER_PATH) as ledger:
            provider = RealMarketWindowProvider()
            engine = PaperTradingEngine(
                registry=reg, ledger=ledger, provider=provider,
                executable=services.REPO_ROOT / DEFAULT_PAPER_TRADING_CLI,
            )
            risk_policy = PaperRiskPolicy(
                max_drawdown_pct=max_drawdown_pct, max_daily_loss_usd=max_daily_loss_usd,
                starting_capital_usd=capital_usd,
            )
            run_id = engine.start_run(
                experiment_key=experiment_id,
                window_start_ns=WINDOW_START_NS,
                window_end_ns=WINDOW_END_NS,
                risk_policy=risk_policy,
                commission_per_contract_usd=commission_per_contract_usd,
            )
    except PaperTradingError as exc:
        return StartPaperRunResult(ok=False, error=str(exc))
    return StartPaperRunResult(ok=True, run_id=run_id)

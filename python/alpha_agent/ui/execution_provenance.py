"""Execution provenance panel (Research Golden Path, section 8A): "Display
Engine / Execution timing / Contract economics source / Risk manager /
Commission-costs / Roll methodology... Read these from the real run/
configuration."

Every fact below is either a literal frozen constant from the same module the
real Phase 13.5C execution driver imports its own assumptions from
(`alpha_agent.strategy.candidates_phase_13_5c.EXECUTION_ASSUMPTIONS` /
`RISK_ASSUMPTIONS` -- these are not guessed here, they are the actual values
the reference CLI is invoked with) or a value read verbatim off the given
experiment's own committed record. Nothing here computes a number.
"""
from __future__ import annotations

from typing import Any

from alpha_agent.strategy.candidates_phase_13_5c import EXECUTION_ASSUMPTIONS, RISK_ASSUMPTIONS

#: `cpp/include/quant_core/engine.hpp`'s `RollExecutionPolicy` -- the frozen
#: reference path's actual default. Duplicated as a factual description (not
#: imported, since this is a Python-side presentation module and the enum
#: lives in C++) -- see CLAUDE.md "Auxiliary roll-close marks" section.
ROLL_METHODOLOGY_TEXT = (
    "RejectDefer (frozen default): a roll that cannot be priced at its nominal instant is deferred, "
    "never silently forward-filled. An optional StaleObservedClose fallback and additive same-timestamp "
    "auxiliary roll-close marks may resolve a thin-market roll a bounded number of bars later -- still a "
    "real observed close, never fabricated."
)

CONTRACT_ECONOMICS_TEXT = (
    "Derived per-contract from the CME definition record "
    "(`alpha_agent.data.contract_economics.derive_contract_economics`) -- point value / tick value is "
    "computed from the published contract specification, never guessed and never taken from the "
    "unreliable `contract_multiplier` sentinel field."
)

EXECUTION_TIMING_TEXT = (
    "No look-ahead: a signal decided at bar T executes no earlier than the next eligible native "
    "1-minute raw-contract bar (T+1+latency); official PnL is the C++ portfolio accountant's own daily "
    "equity trace, never recomputed in Python."
)


def execution_provenance_rows(detail: dict[str, Any] | None) -> list[tuple[str, str]]:
    """`(label, text)` rows for `components.provenance_row`. `detail` is a
    `services.get_experiment(...)` record (or `None` before any run exists)
    -- when present, its own commission/slippage/spread are read verbatim
    from `ResultRecord`; otherwise the frozen reference default is shown,
    explicitly labelled as the default rather than an observed value."""
    result = (detail or {}).get("result") or {}
    commission = result.get("costs_usd")
    rows = [
        ("Engine", "C++ Quant Core (`quant_backtest_targets_csv` -- the frozen reference CLI)"),
        ("Execution timing", EXECUTION_TIMING_TEXT),
        ("Contract economics source", CONTRACT_ECONOMICS_TEXT),
        (
            "Risk manager",
            f"{RISK_ASSUMPTIONS['risk_manager']} -- {RISK_ASSUMPTIONS['limitation']}",
        ),
        (
            "Commission / costs (this run)" if commission is not None else "Commission / costs (frozen default)",
            f"${commission:,.0f} total committed cost" if commission is not None
            else f"${EXECUTION_ASSUMPTIONS['commission_per_contract_usd']:.2f}/contract, "
                 f"{EXECUTION_ASSUMPTIONS['slippage_ticks']:.1f} slippage ticks, "
                 f"{EXECUTION_ASSUMPTIONS['spread_ticks']:.1f} spread ticks",
        ),
        ("Roll methodology", ROLL_METHODOLOGY_TEXT),
    ]
    return rows

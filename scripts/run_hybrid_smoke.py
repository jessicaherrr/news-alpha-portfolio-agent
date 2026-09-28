from __future__ import annotations

import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401  # adds repo python/ to sys.path; must precede alpha_agent
import numpy as np
import pandas as pd
from alpha_agent.adapters.cpp_cli import run_momentum_backtest_cli

_MIN_NS = 60_000_000_000  # one minute in nanoseconds


def main() -> None:
    exe = Path('build/cpp/cpp/quant_backtest_csv')
    if not exe.exists():
        raise SystemExit('Build C++ first: bash scripts/build_cpp.sh')

    n = 300
    close = 100 + np.r_[np.arange(170) * 0.15, 170 * 0.15 - np.arange(n - 170) * 0.12]
    open_ = np.r_[close[0], close[:-1]]
    ts = (np.arange(n, dtype=np.int64) + 1) * _MIN_NS

    bars = pd.DataFrame({
        'ts_event_ns': ts,
        'instrument_id': np.full(n, 1001, dtype=np.int64),
        'open': open_,
        'high': np.maximum(open_, close) + 0.1,
        'low': np.minimum(open_, close) - 0.1,
        'close': close,
        'volume': np.full(n, 1000, dtype=np.int64),
    })
    contracts = pd.DataFrame([{
        'instrument_id': 1001,
        'raw_symbol': 'NQH6',
        'root_symbol': 'NQ',
        'exchange': 'XCME',
        'tick_size': 0.25,
        'multiplier': 20.0,
        'activation_ns': 1,
        'expiration_ns': int(ts[-1]) + _MIN_NS,
        'first_notice_ns': '',
        'last_trade_ns': '',
    }])

    with tempfile.TemporaryDirectory() as tmp:
        result = run_momentum_backtest_cli(
            bars,
            contracts,
            executable=exe,
            lookback=20,
            threshold_return=0.01,
            work_dir=Path(tmp),
        )
    assert result['trades'] > 0
    assert result['contracts_resolved'] == 1
    print('HYBRID_SMOKE_OK', result)


if __name__ == '__main__':
    main()

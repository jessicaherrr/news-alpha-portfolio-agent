from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    out["ret_1"] = out["close"].pct_change()
    out["momentum_20"] = out["close"].pct_change(20)
    out["realized_vol_20"] = out["ret_1"].rolling(20).std() * np.sqrt(20)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    return out

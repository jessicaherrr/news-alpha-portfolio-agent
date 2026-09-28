import pandas as pd
from alpha_agent.features.price import add_price_features


def test_add_price_features():
    df = pd.DataFrame({
        "close": list(range(100, 140)),
        "high": list(range(101, 141)),
        "low": list(range(99, 139)),
    })
    out = add_price_features(df)
    assert "ret_1" in out
    assert out["momentum_20"].notna().sum() > 0

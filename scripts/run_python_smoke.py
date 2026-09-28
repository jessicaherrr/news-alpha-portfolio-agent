import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401  # adds repo python/ to sys.path; must precede alpha_agent
import numpy as np
import pandas as pd
from alpha_agent.features.price import add_price_features
from alpha_agent.registry.sqlite_registry import ExperimentRegistry
from alpha_agent.schemas.hypothesis import HypothesisSpec
from alpha_agent.validation.fdr import benjamini_hochberg


def main() -> None:
    df = pd.DataFrame({
        "close": np.linspace(100, 110, 60),
        "high": np.linspace(100.2, 110.2, 60),
        "low": np.linspace(99.8, 109.8, 60),
    })
    out = add_price_features(df)
    assert "momentum_20" in out
    assert benjamini_hochberg([0.001, 0.2, 0.9], 0.10)[0]

    HypothesisSpec(
        hypothesis_id="H-SMOKE",
        title="Smoke",
        economic_mechanism="test",
        universe=["NQ"],
        horizon="1h",
        required_features=["ret_1"],
        signal_description="test",
        expected_regime="any",
        failure_regime="none",
        falsification_test="test",
    )

    with tempfile.TemporaryDirectory() as tmp:
        reg = ExperimentRegistry(Path(tmp) / "exp.sqlite")
        reg.record(
            experiment_id="E-SMOKE",
            strategy_family="time_series_momentum",
            strategy_spec={},
            dataset_manifest={},
            result={"ok": True},
            status="SMOKE",
        )
        assert reg.count() == 1
    print("PYTHON_SMOKE_OK")


if __name__ == "__main__":
    main()

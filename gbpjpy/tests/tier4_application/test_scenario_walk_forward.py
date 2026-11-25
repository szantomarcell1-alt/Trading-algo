"""Tier 4 Application Scenario 3: Walk-Forward Analysis execution across rolling IS/OOS windows."""

import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.backtest.walk_forward_validator import WalkForwardValidator, WalkForwardResult
except ImportError:
    WalkForwardValidator, WalkForwardResult = None, None

try:
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYScalper
except ImportError:
    GBPJPYScalper = None


class TestScenarioWalkForward(unittest.TestCase):
    """Anti-overfitting validation using Walk-Forward Efficiency (WFE >= 60%) across rolling windows."""

    def _get_instance(self, cls, **kwargs):
        if cls is None:
            return None
        try:
            return cls(**kwargs)
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return cls(config=cfg, **kwargs)
            except Exception:
                try:
                    return cls()
                except Exception:
                    return None

    def _generate_synthetic_multi_window_data(self, total_days=180):
        """Generate 6-month synthetic M5 rates partitioned into sequential periods."""
        np.random.seed(42)
        start_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        total_bars = total_days * 288
        timestamps = [start_time + timedelta(minutes=5 * i) for i in range(total_bars)]

        t = np.linspace(0, 10, total_bars)
        price = 190.0 + np.sin(t * 5) * 2.0 + np.cumsum(np.random.normal(0, 0.02, total_bars))

        df = pd.DataFrame({
            "time": timestamps,
            "open": np.round(price, 3),
            "high": np.round(price + 0.05, 3),
            "low": np.round(price - 0.05, 3),
            "close": np.round(price + np.random.normal(0, 0.01, total_bars), 3),
            "volume": 200
        })
        return df

    def test_walk_forward_window_generation_and_leakage_check(self):
        """Rolling IS/OOS windows must be strictly chronological with zero lookahead overlap."""
        self.assertIsNotNone(WalkForwardValidator, "gbpjpy.backtest.walk_forward_validator not implemented")

        wfv = self._get_instance(
            WalkForwardValidator,
            in_sample_bars=1000,
            out_of_sample_bars=250,
            n_splits=10
        )
        self.assertIsNotNone(wfv, "Unable to instantiate WalkForwardValidator")

        df = self._generate_synthetic_multi_window_data(total_days=60)

        generate_splits = getattr(wfv, "generate_splits", None) or getattr(wfv, "create_windows", None)
        self.assertIsNotNone(generate_splits, "WalkForwardValidator missing window splitting method")

        splits = generate_splits(df)
        self.assertGreaterEqual(len(splits), 5, "Expected at least 5 rolling walk-forward windows")

        for is_data, oos_data in splits:
            is_max_time = is_data["time"].max()
            oos_min_time = oos_data["time"].min()
            self.assertGreaterEqual(
                oos_min_time,
                is_max_time,
                f"Data leakage detected! IS end ({is_max_time}) overlaps OOS start ({oos_min_time})"
            )

    def test_walk_forward_efficiency_ratio_threshold(self):
        """Walk-Forward Efficiency (WFE = Annualized OOS / Annualized IS) must exceed 60%."""
        self.assertIsNotNone(WalkForwardValidator, "gbpjpy.backtest.walk_forward_validator not implemented")

        wfv = self._get_instance(WalkForwardValidator)
        self.assertIsNotNone(wfv, "Unable to instantiate WalkForwardValidator")

        if hasattr(wfv, "calculate_wfe"):
            wfe = wfv.calculate_wfe(is_annual_return=0.12, oos_annual_return=0.085)
            self.assertGreaterEqual(wfe, 0.60, "WFE must meet or exceed 60% anti-overfitting standard")

            wfe_overfit = wfv.calculate_wfe(is_annual_return=0.30, oos_annual_return=0.06)
            self.assertLess(wfe_overfit, 0.60, "Overfitted strategy WFE should fall below 60%")
        elif hasattr(wfv, "run"):
            df = self._generate_synthetic_multi_window_data(total_days=90)
            result = wfv.run(df=df)
            self.assertIsNotNone(result)
            wfe_score = getattr(result, "wfe", None) or (result.get("wfe") if isinstance(result, dict) else None)
            if wfe_score is not None:
                self.assertGreaterEqual(wfe_score, 0.60)


if __name__ == "__main__":
    unittest.main()

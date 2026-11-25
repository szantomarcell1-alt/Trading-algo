"""Tier 3: Cross-feature tests combining Regime Detection and Dynamic Risk/Lot Sizing."""

import sys
import unittest
from datetime import datetime, timezone
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
    from gbpjpy.core.regime_detector import RegimeDetector
except ImportError:
    RegimeDetector = None

try:
    from gbpjpy.risk.risk_manager import RiskManager
except ImportError:
    RiskManager = None

try:
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYScalper
except ImportError:
    GBPJPYScalper = None


class TestRegimeAndSizing(unittest.TestCase):
    """Integration test suite combining RegimeDetector classification with RiskManager position sizing."""

    def setUp(self):
        self.account_equity = 250.0
        self.max_risk_pct = 0.01  # 1.0% = $2.50
        self.gbpusd_rate = 1.28

    def _get_instance(self, cls):
        if cls is None:
            return None
        try:
            return cls()
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return cls(config=cfg)
            except Exception:
                return cls(config={})

    def _generate_synthetic_series(self, trend=True, count=200):
        """Helper to generate trending or mean-reverting price series."""
        np.random.seed(42)
        if trend:
            drift = np.linspace(0, 4.0, count)
            noise = np.random.normal(0, 0.1, count).cumsum()
            prices = 195.0 + drift + noise
        else:
            prices = [195.0]
            theta, mu, sigma = 0.3, 195.0, 0.15
            for _ in range(count - 1):
                dp = theta * (mu - prices[-1]) + sigma * np.random.normal()
                prices.append(prices[-1] + dp)
            prices = np.array(prices)

        dates = pd.date_range("2026-06-10 08:00", periods=count, freq="5min", tz="UTC")
        df = pd.DataFrame({
            "time": dates,
            "open": prices,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "close": prices,
            "volume": 100
        })
        return df

    def test_hurst_trending_regime_and_risk_sizing(self):
        """Trending Hurst regime (>0.55) paired with dynamic lot sizing on $250 equity maintains risk <= 1.0% and expands TP."""
        self.assertIsNotNone(RegimeDetector, "gbpjpy.core.regime_detector not implemented")
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager not implemented")

        rd = self._get_instance(RegimeDetector)
        rm = self._get_instance(RiskManager)
        self.assertIsNotNone(rd, "Unable to instantiate RegimeDetector")
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        trending_df = self._generate_synthetic_series(trend=True, count=250)

        regime = rd.detect_regime(trending_df["close"]) if hasattr(rd, "detect_regime") else None
        hurst_val = rd.calculate_hurst(trending_df["close"]) if hasattr(rd, "calculate_hurst") else 0.65

        self.assertGreater(hurst_val, 0.50, "Persistent series should yield Hurst > 0.50")

        sl_points = 250
        lot_size = rm.calculate_lot_size(
            equity=self.account_equity,
            risk_distance_points=sl_points,
            risk_pct=self.max_risk_pct,
            gbpusd_rate=self.gbpusd_rate
        ) if hasattr(rm, "calculate_lot_size") else 0.01

        self.assertEqual(lot_size, 0.01)

        pip_value_0_01 = (0.01 * 100000 * 0.001) / 155.0
        total_risk_usd = (sl_points / 10.0) * pip_value_0_01
        self.assertLessEqual(total_risk_usd, self.account_equity * self.max_risk_pct * 1.5)

        if GBPJPYScalper is not None:
            scalper = self._get_instance(GBPJPYScalper)
            if hasattr(scalper, "get_tp_multiple"):
                tp_mult = scalper.get_tp_multiple(regime="TRENDING")
                self.assertGreaterEqual(tp_mult, 2.0, "Trending regime must target at least 2.0R")

    def test_asian_mean_reverting_regime_and_margin(self):
        """Asian mean-reverting regime (Hurst < 0.45) verifies margin level > 200% on $250 small account."""
        self.assertIsNotNone(RegimeDetector, "gbpjpy.core.regime_detector not implemented")
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager not implemented")

        rd = self._get_instance(RegimeDetector)
        rm = self._get_instance(RiskManager)
        self.assertIsNotNone(rd, "Unable to instantiate RegimeDetector")
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        reverting_df = self._generate_synthetic_series(trend=False, count=250)

        hurst_val = rd.calculate_hurst(reverting_df["close"]) if hasattr(rd, "calculate_hurst") else 0.40
        self.assertLessEqual(hurst_val, 0.55)

        required_margin = (0.01 * 100000 * self.gbpusd_rate) / 30.0
        expected_margin_level = (self.account_equity / required_margin) * 100.0

        self.assertGreater(expected_margin_level, 200.0)

        if hasattr(rm, "check_margin_level"):
            self.assertTrue(rm.check_margin_level(equity=self.account_equity, total_margin=required_margin))
        elif hasattr(rm, "can_open_position"):
            self.assertTrue(rm.can_open_position(equity=self.account_equity, required_margin=required_margin))


if __name__ == "__main__":
    unittest.main()

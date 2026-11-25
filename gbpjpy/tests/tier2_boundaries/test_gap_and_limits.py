"""Tier 2: Boundary test cases for lot size lower bound (0.01), Hurst divide-by-zero, and corrupt data handling."""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.risk.risk_manager import RiskManager
except ImportError:
    RiskManager = None

try:
    from gbpjpy.core.regime_detector import RegimeDetector
except ImportError:
    RegimeDetector = None

try:
    from gbpjpy.core.data_engine import DataEngine
except ImportError:
    DataEngine = None


class TestGapAndLimits(unittest.TestCase):
    """Boundary tests for min lot clamping (0.01), Hurst stddev=0 divide-by-zero, and corrupt tick filtering."""

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

    def test_min_lot_enforcement(self):
        """Calculated fractional lot size < 0.01 must be clamped to the 0.01 lot minimum parity."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager module not implemented")

        rm = self._get_instance(RiskManager)
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        if hasattr(rm, "calculate_lot_size"):
            lot_size = rm.calculate_lot_size(
                equity=250.0,
                risk_distance_points=500,  # 50 pips
                risk_pct=0.005             # 0.5% risk = $1.25
            )
            self.assertGreaterEqual(lot_size, 0.01, "Lot size must not be less than 0.01 minimum lot")
            self.assertEqual(lot_size, 0.01)
        elif hasattr(rm, "clamp_lot_size"):
            clamped = rm.clamp_lot_size(0.004)
            self.assertEqual(clamped, 0.01)
        else:
            self.fail("RiskManager missing lot size calculation or clamping method")

    def test_flat_price_series_hurst_zero_division(self):
        """Flat price series (stddev=0 in returns) must not trigger ZeroDivisionError in Hurst calculation."""
        self.assertIsNotNone(RegimeDetector, "gbpjpy.core.regime_detector module not implemented")

        rd = self._get_instance(RegimeDetector)
        self.assertIsNotNone(rd, "Unable to instantiate RegimeDetector")

        flat_prices = pd.Series([195.500] * 200)

        calculate_hurst = getattr(rd, "calculate_hurst", None) or getattr(rd, "compute_hurst", None)
        if calculate_hurst is not None:
            try:
                hurst_value = calculate_hurst(flat_prices)
                self.assertIsNotNone(hurst_value)
                self.assertFalse(np.isnan(hurst_value), "Hurst exponent should not be NaN on flat series")
                self.assertFalse(np.isinf(hurst_value), "Hurst exponent should not be Inf on flat series")
                self.assertTrue(0.0 <= hurst_value <= 1.0, f"Hurst value {hurst_value} outside [0.0, 1.0]")
            except ZeroDivisionError:
                self.fail("RegimeDetector crashed with ZeroDivisionError on flat price series")
        else:
            detect_regime = getattr(rd, "detect_regime", None)
            self.assertIsNotNone(detect_regime, "RegimeDetector missing calculate_hurst or detect_regime")
            try:
                regime = detect_regime(flat_prices)
                self.assertIsNotNone(regime)
            except ZeroDivisionError:
                self.fail("RegimeDetector detect_regime crashed with ZeroDivisionError on flat price series")

    def test_corrupt_tick_data_rejection(self):
        """Corrupt market data (negative price, zero price, inverted spread) must be rejected/filtered."""
        self.assertIsNotNone(DataEngine, "gbpjpy.core.data_engine module not implemented")

        engine = self._get_instance(DataEngine)
        self.assertIsNotNone(engine, "Unable to instantiate DataEngine")

        corrupt_samples = [
            {"bid": -195.200, "ask": 195.220},
            {"bid": 0.0, "ask": 195.220},
            {"bid": 195.500, "ask": 195.200},
            {"bid": float("nan"), "ask": 195.2},
        ]

        if hasattr(engine, "validate_tick"):
            for sample in corrupt_samples:
                is_valid = engine.validate_tick(sample["bid"], sample["ask"])
                self.assertFalse(is_valid, f"Corrupt sample {sample} was not rejected")

            self.assertTrue(engine.validate_tick(195.200, 195.220))
        elif hasattr(engine, "clean_data"):
            df_dirty = pd.DataFrame([
                {"open": 195.0, "high": 195.5, "low": 194.8, "close": 195.2, "volume": 100},
                {"open": -195.0, "high": 195.5, "low": 194.8, "close": 195.2, "volume": 100},
                {"open": 195.0, "high": 194.0, "low": 196.0, "close": 195.2, "volume": 100},
            ])
            cleaned = engine.clean_data(df_dirty)
            self.assertEqual(len(cleaned), 1, "Dirty rows should be filtered out")
        else:
            validate_method = getattr(engine, "validate_data", None) or getattr(engine, "validate_tick", None)
            self.assertIsNotNone(validate_method, "DataEngine missing data validation / cleaning method")


if __name__ == "__main__":
    unittest.main()

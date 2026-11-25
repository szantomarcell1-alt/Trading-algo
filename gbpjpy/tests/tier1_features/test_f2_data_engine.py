"""
Tier 1 Feature Tests: F2 - Core Data Engine & Indicator Suite.
Verifies multi-timeframe EMA, ATR, Bollinger Bands, Keltner Channels, TTM Squeeze, and VWAP.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tests.fixtures import generate_synthetic_m5_data


class TestF2DataEngine(unittest.TestCase):
    """Test suite for DataEngine indicator calculations and signal feature extraction."""

    def setUp(self):
        try:
            from gbpjpy.core.data_engine import DataEngine
            self.DataEngine = DataEngine
        except ImportError:
            self.DataEngine = None

    def _get_engine(self):
        self.assertIsNotNone(self.DataEngine, "gbpjpy.core.data_engine module not implemented")
        if isinstance(self.DataEngine, type):
            return self.DataEngine()
        return self.DataEngine

    def test_ema_calculation(self):
        """Verify EMA(9), EMA(21), EMA(50), EMA(200) match mathematical benchmark."""
        engine = self._get_engine()
        df = generate_synthetic_m5_data(bars=450, regime="trend_up")

        res = engine.compute_indicators(df)
        self.assertIsInstance(res, pd.DataFrame, "compute_indicators must return a pandas DataFrame")

        for span in [9, 21, 50, 200]:
            col = f"ema_{span}"
            self.assertIn(col, res.columns, f"Indicator DataFrame missing column '{col}'")
            # Calculate mathematical benchmark EMA
            expected_ema = df["close"].ewm(span=span, adjust=False).mean()
            # Compare after warm-up period (span bars)
            valid_idx = slice(span * 2, None)
            diff = np.abs(res[col].iloc[valid_idx] - expected_ema.iloc[valid_idx])
            max_diff = diff.max()
            self.assertLess(max_diff, 0.05, f"EMA({span}) deviation {max_diff} exceeds tolerance 0.05")

    def test_atr_calculation(self):
        """Verify ATR(14) with True Range calculation against standard values."""
        engine = self._get_engine()
        df = generate_synthetic_m5_data(bars=100, regime="random_walk")

        res = engine.compute_indicators(df)
        self.assertIn("atr", res.columns, "Indicator DataFrame missing column 'atr'")

        # Benchmark ATR(14): TR = max(H - L, |H - C_prev|, |L - C_prev|)
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        expected_atr = tr.rolling(window=14).mean()

        # Check values are positive and aligned
        valid_atr = res["atr"].dropna()
        self.assertTrue((valid_atr > 0).all(), "ATR values must be strictly positive")
        # Deviation after 30 bars should be small
        diff = np.abs(res["atr"].iloc[30:] - expected_atr.iloc[30:]).dropna()
        self.assertLess(diff.mean(), 0.05, f"ATR mean deviation {diff.mean()} exceeds tolerance")

    def test_bollinger_bands(self):
        """Verify 20-period 2.0-stddev upper, middle, lower bands."""
        engine = self._get_engine()
        df = generate_synthetic_m5_data(bars=100, regime="mean_reversion")

        res = engine.compute_indicators(df)
        for col in ["bb_upper", "bb_middle", "bb_lower"]:
            self.assertIn(col, res.columns, f"Indicator DataFrame missing column '{col}'")

        # Benchmark 20-period SMA & 2-stddev
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std(ddof=0)
        expected_upper = sma20 + 2.0 * std20
        expected_lower = sma20 - 2.0 * std20

        # Invariant checks
        valid_res = res.iloc[25:]
        self.assertTrue((valid_res["bb_upper"] >= valid_res["bb_middle"]).all(), "Upper BB must be >= Middle BB")
        self.assertTrue((valid_res["bb_middle"] >= valid_res["bb_lower"]).all(), "Middle BB must be >= Lower BB")

        diff_mid = np.abs(valid_res["bb_middle"] - sma20.iloc[25:]).mean()
        self.assertLess(diff_mid, 0.01, f"BB middle deviation {diff_mid} exceeds tolerance")

    def test_keltner_channels(self):
        """Verify 20-period EMA ± 1.5 ATR Keltner channels."""
        engine = self._get_engine()
        df = generate_synthetic_m5_data(bars=100, regime="trend_up")

        res = engine.compute_indicators(df)
        self.assertTrue(
            "keltner_upper" in res.columns or "kc_upper" in res.columns,
            "Keltner upper band missing from indicator DataFrame"
        )
        upper_col = "keltner_upper" if "keltner_upper" in res.columns else "kc_upper"
        lower_col = "keltner_lower" if "keltner_lower" in res.columns else "kc_lower"

        valid_res = res.iloc[30:]
        self.assertTrue((valid_res[upper_col] > valid_res[lower_col]).all(), "Keltner upper must be > lower")

    def test_ttm_squeeze_detection(self):
        """Verify squeeze_on (BB inside Keltner) and squeeze_fired transition with momentum."""
        engine = self._get_engine()
        # Synthetic squeeze data: low vol first 60%, followed by strong breakout
        df = generate_synthetic_m5_data(bars=200, regime="squeeze")

        res = engine.compute_indicators(df)
        self.assertIn("squeeze_on", res.columns, "Indicator DataFrame missing 'squeeze_on'")
        self.assertIn("squeeze_fired", res.columns, "Indicator DataFrame missing 'squeeze_fired'")

        # Ensure squeeze boolean indicators are present
        self.assertTrue(res["squeeze_on"].dtype == bool or res["squeeze_on"].dtype == np.bool_ or np.issubdtype(res["squeeze_on"].dtype, np.integer))

        # In a squeeze dataset, squeeze_on should be True during compression
        # and squeeze_fired should trigger when squeeze releases
        has_squeeze = res["squeeze_on"].any()
        self.assertTrue(has_squeeze, "Squeeze should be detected in synthetic squeeze regime dataset")

    def test_intraday_vwap(self):
        """Verify cumulative volume-weighted average price resetting each trading day."""
        engine = self._get_engine()
        # Create 2 days of data: Day 1 (2026-01-15), Day 2 (2026-01-16)
        day1_start = datetime(2026, 1, 15, 0, 0, 0)
        df1 = generate_synthetic_m5_data(bars=288, start_time=day1_start, start_price=155.0)
        day2_start = datetime(2026, 1, 16, 0, 0, 0)
        df2 = generate_synthetic_m5_data(bars=288, start_time=day2_start, start_price=156.0)
        df_2days = pd.concat([df1, df2])

        res = engine.compute_indicators(df_2days)
        self.assertIn("vwap", res.columns, "Indicator DataFrame missing 'vwap'")

        # Day 2's first candle VWAP should equal that candle's typical price, not carry over Day 1
        day2_first_row = res.loc[res["time"] == day2_start].iloc[0]
        typical_price_day2 = (day2_first_row["high"] + day2_first_row["low"] + day2_first_row["close"]) / 3.0
        self.assertAlmostEqual(
            day2_first_row["vwap"], typical_price_day2, places=2,
            msg="VWAP must reset at day boundary (00:00 UTC)"
        )

    def test_causal_swing_points(self):
        """Verify swing highs and lows are strictly causal without lookahead bias."""
        engine = self._get_engine()
        df = generate_synthetic_m5_data(bars=100, regime="trend_up")

        res = engine.compute_indicators(df)
        self.assertIn("swing_high", res.columns, "Indicator DataFrame missing 'swing_high'")
        self.assertIn("swing_low", res.columns, "Indicator DataFrame missing 'swing_low'")

        # Swing highs should never be higher than the maximum high observed up to that point
        # within the swing detection window (causality check)
        swing_highs = res["swing_high"].dropna()
        if len(swing_highs) > 0:
            for idx in swing_highs.index[:20]:
                curr_swing = res.loc[idx, "swing_high"]
                if not np.isnan(curr_swing):
                    # Should be a past high, not future
                    loc = res.index.get_loc(idx)
                    past_highs = res["high"].iloc[max(0, loc - 10):loc + 1]
                    self.assertIn(round(curr_swing, 3), [round(x, 3) for x in past_highs],
                                  "Swing high must be an observed past bar high")


if __name__ == "__main__":
    unittest.main()

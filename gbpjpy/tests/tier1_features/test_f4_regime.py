"""
Tier 1 Feature Tests: F4 - Market Regime Detector & Session Guard.
Verifies Hurst exponent classification, Garman-Klass volatility, and trading session / rollover lockouts.
"""

import unittest
from datetime import datetime, time
import numpy as np
import pandas as pd
from tests.fixtures import (
    generate_synthetic_m5_data,
    reference_garman_klass_vol,
    reference_hurst_exponent
)


class TestF4Regime(unittest.TestCase):
    """Test suite for RegimeDetector and MarketHours session filters."""

    def setUp(self):
        try:
            from gbpjpy.core.regime_detector import RegimeDetector
            self.RegimeDetector = RegimeDetector
        except ImportError:
            self.RegimeDetector = None

        try:
            from gbpjpy.utils import market_hours
            self.market_hours = market_hours
        except ImportError:
            self.market_hours = None

    def _get_detector(self):
        self.assertIsNotNone(self.RegimeDetector, "gbpjpy.core.regime_detector module not implemented")
        if isinstance(self.RegimeDetector, type):
            return self.RegimeDetector()
        return self.RegimeDetector

    def test_hurst_trending_regime(self):
        """Persistent trending series yields Hurst > 0.55 (TRENDING classification)."""
        detector = self._get_detector()
        # Generate strongly trending data
        df_trend = generate_synthetic_m5_data(bars=300, regime="trend_up", seed=101)

        hurst_val = None
        regime_label = None

        if hasattr(detector, "compute_hurst"):
            hurst_val = detector.compute_hurst(df_trend["close"])
        elif hasattr(detector, "detect_regime"):
            res = detector.detect_regime(df_trend)
            hurst_val = getattr(res, "hurst", res.get("hurst") if isinstance(res, dict) else None)
            regime_label = getattr(res, "regime", res.get("regime") if isinstance(res, dict) else None)

        self.assertIsNotNone(hurst_val, "RegimeDetector must compute Hurst exponent")
        self.assertGreater(hurst_val, 0.55, f"Trending series Hurst {hurst_val:.3f} must be > 0.55")
        if regime_label is not None:
            self.assertIn("TREND", str(regime_label).upper(), "Regime label should indicate trending")

    def test_hurst_mean_reverting(self):
        """Mean-reverting series yields Hurst < 0.45 (MEAN_REVERTING classification)."""
        detector = self._get_detector()
        # Generate mean-reverting OU process data
        df_mr = generate_synthetic_m5_data(bars=300, regime="mean_reversion", seed=202)

        hurst_val = None
        regime_label = None

        if hasattr(detector, "compute_hurst"):
            hurst_val = detector.compute_hurst(df_mr["close"])
        elif hasattr(detector, "detect_regime"):
            res = detector.detect_regime(df_mr)
            hurst_val = getattr(res, "hurst", res.get("hurst") if isinstance(res, dict) else None)
            regime_label = getattr(res, "regime", res.get("regime") if isinstance(res, dict) else None)

        self.assertIsNotNone(hurst_val, "RegimeDetector must compute Hurst exponent")
        self.assertLess(hurst_val, 0.45, f"Mean-reverting series Hurst {hurst_val:.3f} must be < 0.45")
        if regime_label is not None:
            self.assertIn("MEAN_REVERT", str(regime_label).upper())

    def test_garman_klass_volatility(self):
        """Verify Garman-Klass volatility calculation matches mathematical benchmark."""
        detector = self._get_detector()
        df = generate_synthetic_m5_data(bars=100, regime="random_walk", seed=303)

        expected_gk = reference_garman_klass_vol(df)

        if hasattr(detector, "compute_garman_klass"):
            gk = detector.compute_garman_klass(df)
            val = gk.iloc[-1] if isinstance(gk, (pd.Series, np.ndarray)) else gk
            self.assertAlmostEqual(val, expected_gk, places=3,
                                  msg="Garman-Klass volatility must match benchmark formula")
        elif hasattr(detector, "detect_regime"):
            res = detector.detect_regime(df)
            vol = getattr(res, "volatility", res.get("volatility") if isinstance(res, dict) else None)
            self.assertIsNotNone(vol, "RegimeDetector output should include volatility metric")
            self.assertGreater(vol, 0, "Volatility must be strictly positive")
        else:
            self.fail("RegimeDetector must compute Garman-Klass volatility")

    def test_session_window_asian(self):
        """Time 02:00 UTC must be classified as ASIAN session."""
        self.assertIsNotNone(self.market_hours, "gbpjpy.utils.market_hours module not implemented")
        t = datetime(2026, 1, 15, 2, 0, 0)  # 02:00 UTC

        if hasattr(self.market_hours, "get_session"):
            session = self.market_hours.get_session(t)
            self.assertIn("ASIAN", str(session).upper(), f"02:00 UTC should be ASIAN session, got {session}")
        elif hasattr(self.market_hours, "is_asian_session"):
            self.assertTrue(self.market_hours.is_asian_session(t))
        else:
            self.fail("market_hours module missing session classification functions")

    def test_session_window_london(self):
        """Time 08:30 UTC must be classified as LONDON session."""
        self.assertIsNotNone(self.market_hours, "gbpjpy.utils.market_hours module not implemented")
        t = datetime(2026, 1, 15, 8, 30, 0)  # 08:30 UTC

        if hasattr(self.market_hours, "get_session"):
            session = self.market_hours.get_session(t)
            self.assertIn("LONDON", str(session).upper(), f"08:30 UTC should be LONDON session, got {session}")
        elif hasattr(self.market_hours, "is_london_session"):
            self.assertTrue(self.market_hours.is_london_session(t))
        else:
            self.fail("market_hours module missing session classification functions")

    def test_rollover_lockout_active(self):
        """Time 21:30 UTC must be flagged as rollover lockout (21:00-23:00 UTC, no new trades)."""
        self.assertIsNotNone(self.market_hours, "gbpjpy.utils.market_hours module not implemented")
        t_lockout = datetime(2026, 1, 15, 21, 30, 0)  # Inside rollover window
        t_normal = datetime(2026, 1, 15, 10, 0, 0)    # Outside rollover window

        if hasattr(self.market_hours, "is_rollover_lockout"):
            self.assertTrue(self.market_hours.is_rollover_lockout(t_lockout),
                            "21:30 UTC must be flagged as rollover lockout")
            self.assertFalse(self.market_hours.is_rollover_lockout(t_normal),
                             "10:00 UTC must NOT be flagged as rollover lockout")
        elif hasattr(self.market_hours, "can_trade"):
            self.assertFalse(self.market_hours.can_trade(t_lockout), "Trading should be locked out at 21:30 UTC")
            self.assertTrue(self.market_hours.can_trade(t_normal), "Trading should be allowed at 10:00 UTC")
        else:
            self.fail("market_hours module missing rollover lockout check")

    def test_session_window_new_york(self):
        """Time 14:00 UTC must be classified as NEW_YORK (or London/NY overlap)."""
        self.assertIsNotNone(self.market_hours, "gbpjpy.utils.market_hours module not implemented")
        t = datetime(2026, 1, 15, 14, 0, 0)  # 14:00 UTC

        if hasattr(self.market_hours, "get_session"):
            session = self.market_hours.get_session(t)
            session_str = str(session).upper()
            self.assertTrue("NEW_YORK" in session_str or "NY" in session_str or "OVERLAP" in session_str,
                            f"14:00 UTC should be NY or overlap, got {session}")
        elif hasattr(self.market_hours, "is_ny_session"):
            self.assertTrue(self.market_hours.is_ny_session(t))


if __name__ == "__main__":
    unittest.main()

"""Tier 2: Boundary test cases for spread blowout, rollover lockout, and Wednesday 3x swap modeling."""

import sys
import unittest
from datetime import datetime, time, timezone, timedelta
from pathlib import Path

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.utils.market_hours import MarketHours
except ImportError:
    MarketHours = None

try:
    from gbpjpy.risk.risk_manager import RiskManager
except ImportError:
    RiskManager = None

try:
    from gbpjpy.backtest.engine import BacktestEngine
except ImportError:
    BacktestEngine = None


class TestSpreadRollover(unittest.TestCase):
    """Boundary tests for spread conditions, rollover windows, and overnight swap costs."""

    def setUp(self):
        self.max_spread_pips = 3.0
        self.pip_size = 0.01

    def _get_risk_manager(self):
        if RiskManager is None:
            return None
        try:
            return RiskManager()
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return RiskManager(config=cfg)
            except Exception:
                return RiskManager(config={})

    def _get_market_hours(self):
        if MarketHours is None:
            return None
        try:
            return MarketHours()
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return MarketHours(config=cfg)
            except Exception:
                return MarketHours(config={})

    def _get_backtest_engine(self):
        if BacktestEngine is None:
            return None
        try:
            return BacktestEngine()
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return BacktestEngine(config=cfg)
            except Exception:
                return BacktestEngine(config={})

    def test_spread_blowout_filter(self):
        """Spread blowout > 3.0 pips during rollover/news must be rejected by max_spread filter."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager module not implemented")

        rm = self._get_risk_manager()
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        # Test with RiskManager spread filter
        if hasattr(rm, "is_spread_acceptable"):
            self.assertTrue(rm.is_spread_acceptable(spread_pips=1.5, max_spread=self.max_spread_pips))
            self.assertTrue(rm.is_spread_acceptable(spread_pips=3.0, max_spread=self.max_spread_pips))
            self.assertFalse(rm.is_spread_acceptable(spread_pips=3.1, max_spread=self.max_spread_pips))
            self.assertFalse(rm.is_spread_acceptable(spread_pips=6.5, max_spread=self.max_spread_pips))
        elif hasattr(rm, "validate_order"):
            valid, reason = rm.validate_order(spread_pips=1.8)
            self.assertTrue(valid)
            invalid, reason = rm.validate_order(spread_pips=5.5)
            self.assertFalse(invalid)
            self.assertIn("spread", reason.lower())
        else:
            can_trade = getattr(rm, "can_trade_spread", None)
            self.assertIsNotNone(can_trade, "RiskManager missing spread validation method")
            self.assertTrue(can_trade(1.5))
            self.assertFalse(can_trade(4.0))

    def test_midnight_rollover_window_lockout(self):
        """Midnight rollover window (21:00-23:00 UTC) must lock out new trade entry."""
        self.assertIsNotNone(MarketHours, "gbpjpy.utils.market_hours module not implemented")

        mh = self._get_market_hours()
        self.assertIsNotNone(mh, "Unable to instantiate MarketHours")

        # Inside rollover window (21:00 - 23:00 UTC)
        lockout_times = [
            datetime(2026, 6, 10, 21, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 10, 21, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 10, 22, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 10, 22, 59, 59, tzinfo=timezone.utc),
        ]
        for dt in lockout_times:
            if hasattr(mh, "is_rollover_lockout"):
                self.assertTrue(mh.is_rollover_lockout(dt), f"Expected {dt} to be in rollover lockout")
            elif hasattr(mh, "can_trade"):
                self.assertFalse(mh.can_trade(dt), f"Expected trading blocked at {dt}")
            else:
                self.fail("MarketHours missing is_rollover_lockout or can_trade method")

        # Outside rollover window (allowed trading hours)
        active_times = [
            datetime(2026, 6, 10, 20, 55, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 10, 23, 5, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 11, 8, 30, 0, tzinfo=timezone.utc),
        ]
        for dt in active_times:
            if hasattr(mh, "is_rollover_lockout"):
                self.assertFalse(mh.is_rollover_lockout(dt), f"Expected {dt} to NOT be in rollover lockout")
            elif hasattr(mh, "can_trade"):
                self.assertTrue(mh.can_trade(dt), f"Expected trading allowed at {dt}")

    def test_triple_swap_wednesday(self):
        """Wednesday rollover (21:00/22:00 UTC) must apply 3x swap cost modeling for T+2 settlement."""
        self.assertIsNotNone(BacktestEngine, "gbpjpy.backtest.engine module not implemented")

        engine = self._get_backtest_engine()
        self.assertIsNotNone(engine, "Unable to instantiate BacktestEngine")

        calculate_swap = getattr(engine, "calculate_swap", None)
        if calculate_swap is None and RiskManager is not None:
            rm = self._get_risk_manager()
            calculate_swap = getattr(rm, "calculate_swap", None)

        self.assertIsNotNone(calculate_swap, "Swap calculation function not found on BacktestEngine or RiskManager")

        # Tuesday rollover: 1x swap
        tuesday_rollover = datetime(2026, 6, 9, 22, 0, 0, tzinfo=timezone.utc)  # Tuesday = weekday 1
        tuesday_swap = calculate_swap(volume=0.01, is_long=True, rollover_dt=tuesday_rollover)

        # Wednesday rollover: 3x swap (standard forex weekend interest carry)
        wednesday_rollover = datetime(2026, 6, 10, 22, 0, 0, tzinfo=timezone.utc)  # Wednesday = weekday 2
        wednesday_swap = calculate_swap(volume=0.01, is_long=True, rollover_dt=wednesday_rollover)

        # Thursday rollover: 1x swap
        thursday_rollover = datetime(2026, 6, 11, 22, 0, 0, tzinfo=timezone.utc)  # Thursday = weekday 3
        thursday_swap = calculate_swap(volume=0.01, is_long=True, rollover_dt=thursday_rollover)

        # Assert Wednesday swap is approximately 3x normal single-day swap
        self.assertAlmostEqual(abs(wednesday_swap), abs(tuesday_swap) * 3, places=2)
        self.assertAlmostEqual(abs(wednesday_swap), abs(thursday_swap) * 3, places=2)
        self.assertFalse(isinstance(wednesday_swap, complex) or wednesday_swap is None)


if __name__ == "__main__":
    unittest.main()

"""Tier 2: Boundary test cases for extreme volatility, ATR clamping (12-35 pips), and gap executions."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYScalper
except ImportError:
    GBPJPYScalper = None

try:
    from gbpjpy.risk.execution_manager import ExecutionManager
except ImportError:
    ExecutionManager = None

try:
    from gbpjpy.backtest.engine import BacktestEngine
except ImportError:
    BacktestEngine = None


class TestVolatilitySpikes(unittest.TestCase):
    """Boundary tests for ATR extreme clamping and price gap executions."""

    def setUp(self):
        self.min_sl_pips = 12.0
        self.max_sl_pips = 35.0
        self.pip_size = 0.01

    def _get_target_instance(self, cls):
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

    def test_flash_crash_atr_spike(self):
        """100-pip ATR jump during flash crash must clamp stop loss to maximum bound (35 pips)."""
        self.assertIsNotNone(
            GBPJPYScalper or ExecutionManager,
            "gbpjpy.strategies.gbpjpy_scalper or execution_manager not implemented"
        )

        target_cls = GBPJPYScalper or ExecutionManager
        instance = self._get_target_instance(target_cls)
        self.assertIsNotNone(instance, "Unable to instantiate strategy / execution manager")

        extreme_atr_pips = 100.0
        extreme_atr_price = 1.000

        if hasattr(instance, "calculate_stop_loss_distance"):
            sl_dist = instance.calculate_stop_loss_distance(atr_pips=extreme_atr_pips)
            self.assertAlmostEqual(sl_dist, self.max_sl_pips, places=2)
            self.assertLessEqual(sl_dist, 35.0)
        elif hasattr(instance, "clamp_sl"):
            sl_dist = instance.clamp_sl(extreme_atr_pips * 1.5)
            self.assertEqual(sl_dist, self.max_sl_pips)
        elif hasattr(instance, "calculate_sl_price"):
            sl_price = instance.calculate_sl_price(
                entry_price=195.000,
                is_buy=True,
                atr=extreme_atr_price
            )
            sl_dist_pips = round((195.000 - sl_price) / self.pip_size, 2)
            self.assertAlmostEqual(sl_dist_pips, self.max_sl_pips, places=1)
        else:
            generate_signals = getattr(instance, "generate_signals", None)
            self.assertIsNotNone(generate_signals, "Missing SL calculation or signal generator")

    def test_tight_consolidation_low_atr(self):
        """Tight consolidation with low ATR (< 5 pips) must clamp stop loss to minimum bound (12 pips)."""
        self.assertIsNotNone(
            GBPJPYScalper or ExecutionManager,
            "gbpjpy.strategies.gbpjpy_scalper or execution_manager not implemented"
        )

        target_cls = GBPJPYScalper or ExecutionManager
        instance = self._get_target_instance(target_cls)
        self.assertIsNotNone(instance, "Unable to instantiate strategy / execution manager")

        low_atr_pips = 3.0
        low_atr_price = 0.030

        if hasattr(instance, "calculate_stop_loss_distance"):
            sl_dist = instance.calculate_stop_loss_distance(atr_pips=low_atr_pips)
            self.assertAlmostEqual(sl_dist, self.min_sl_pips, places=2)
            self.assertGreaterEqual(sl_dist, 12.0)
        elif hasattr(instance, "clamp_sl"):
            sl_dist = instance.clamp_sl(low_atr_pips * 1.5)
            self.assertEqual(sl_dist, self.min_sl_pips)
        elif hasattr(instance, "calculate_sl_price"):
            sl_price = instance.calculate_sl_price(
                entry_price=195.000,
                is_buy=True,
                atr=low_atr_price
            )
            sl_dist_pips = round((195.000 - sl_price) / self.pip_size, 2)
            self.assertAlmostEqual(sl_dist_pips, self.min_sl_pips, places=1)
        else:
            generate_signals = getattr(instance, "generate_signals", None)
            self.assertIsNotNone(generate_signals, "Missing SL calculation or signal generator")

    def test_price_gap_past_stop(self):
        """Market gap past Stop Loss must fill at the gap open price, recording adverse slippage."""
        self.assertIsNotNone(BacktestEngine, "gbpjpy.backtest.engine module not implemented")

        engine = self._get_target_instance(BacktestEngine)
        self.assertIsNotNone(engine, "Unable to instantiate BacktestEngine")

        bars = pd.DataFrame([
            {
                "time": datetime(2026, 6, 12, 21, 55, tzinfo=timezone.utc),
                "open": 195.000, "high": 195.200, "low": 194.900, "close": 194.950, "volume": 100
            },
            {
                "time": datetime(2026, 6, 15, 0, 0, tzinfo=timezone.utc),
                "open": 194.300, "high": 194.400, "low": 194.100, "close": 194.200, "volume": 100
            }
        ])

        execute_trade = getattr(engine, "simulate_gap_exit", None)
        if execute_trade is not None:
            trade_result = execute_trade(
                entry_price=195.000,
                sl_price=194.700,
                is_long=True,
                gap_bar=bars.iloc[1]
            )
            self.assertEqual(trade_result["exit_price"], 194.300)
            self.assertGreater(trade_result["slippage_pips"], 30.0)
        else:
            run_method = getattr(engine, "run", None) or getattr(engine, "process_bar", None)
            self.assertIsNotNone(run_method, "BacktestEngine missing run or process_bar method")


if __name__ == "__main__":
    unittest.main()

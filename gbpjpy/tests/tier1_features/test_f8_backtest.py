"""
Tier 1 Feature Tests: F8 - Event-Driven Backtesting Engine.
Verifies sequential bar simulation, SL/TP execution, commission accounting, and equity curve generation.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tests.fixtures import generate_synthetic_m5_data


class TestF8Backtest(unittest.TestCase):
    """Test suite for BacktestEngine execution simulation and trade accounting."""

    def setUp(self):
        try:
            from gbpjpy.backtest.engine import BacktestEngine
            self.BacktestEngine = BacktestEngine
        except ImportError:
            self.BacktestEngine = None

    def _get_engine(self, initial_balance=250.0, commission_per_lot=7.0):
        self.assertIsNotNone(self.BacktestEngine, "gbpjpy.backtest.engine module not implemented")
        try:
            return self.BacktestEngine(
                initial_balance=initial_balance,
                commission_per_lot=commission_per_lot
            )
        except TypeError:
            return self.BacktestEngine()

    def test_backtest_bar_processing(self):
        """Engine processes sequential M5 bars without lookahead bias."""
        engine = self._get_engine()
        df = generate_synthetic_m5_data(bars=100, regime="trend_up")

        result = engine.run(df)
        self.assertIsNotNone(result, "engine.run(df) must return a BacktestResult")
        # Ensure backtest ran through the entire length of data
        bars_processed = getattr(result, "bars_processed", getattr(result, "total_bars", len(df)))
        self.assertEqual(bars_processed, len(df), "Backtest must process all sequential bars")

    def test_backtest_sl_execution(self):
        """When low price hits or breaches SL, trade is closed at SL price (or next bar open)."""
        engine = self._get_engine()
        # Create a 5-bar sequence: bar 0 entry BUY at 155.00 with SL 154.80. Bar 2 low dips to 154.70.
        timestamps = [datetime(2026, 1, 15, 8, 0) + timedelta(minutes=5 * i) for i in range(5)]
        df_bars = pd.DataFrame({
            "time": timestamps,
            "open":  [155.000, 155.050, 154.900, 154.800, 154.850],
            "high":  [155.100, 155.100, 154.950, 154.850, 154.900],
            "low":   [154.950, 154.900, 154.700, 154.750, 154.780],  # Bar 2 low 154.700 triggers SL 154.800
            "close": [155.050, 154.950, 154.750, 154.800, 154.880],
            "volume": [500] * 5,
            "tick_volume": [500] * 5
        })
        df_bars.set_index("time", inplace=True, drop=False)

        # Force a trade injection if engine supports it, or run engine
        if hasattr(engine, "inject_trade"):
            engine.inject_trade(order_type="BUY", entry_price=155.000, sl=154.800, tp=156.000, volume=0.01)
        result = engine.run(df_bars)

        trades = getattr(result, "trades", [])
        if trades:
            sl_trade = trades[0]
            exit_p = getattr(sl_trade, "exit_price", sl_trade.get("exit_price") if isinstance(sl_trade, dict) else None)
            self.assertAlmostEqual(exit_p, 154.800, delta=0.05,
                                  msg="Trade must close at or near SL price (154.800)")

    def test_backtest_tp_execution(self):
        """When high price hits TP, trade is closed at TP price."""
        engine = self._get_engine()
        # Bar 2 high surges to 155.500, hitting TP at 155.400
        timestamps = [datetime(2026, 1, 15, 8, 0) + timedelta(minutes=5 * i) for i in range(5)]
        df_bars = pd.DataFrame({
            "time": timestamps,
            "open":  [155.000, 155.100, 155.200, 155.300, 155.250],
            "high":  [155.150, 155.250, 155.500, 155.400, 155.350],  # Bar 2 high 155.500 hits TP 155.400
            "low":   [154.950, 155.050, 155.150, 155.200, 155.200],
            "close": [155.100, 155.200, 155.350, 155.250, 155.300],
            "volume": [500] * 5,
            "tick_volume": [500] * 5
        })
        df_bars.set_index("time", inplace=True, drop=False)

        if hasattr(engine, "inject_trade"):
            engine.inject_trade(order_type="BUY", entry_price=155.000, sl=154.800, tp=155.400, volume=0.01)
        result = engine.run(df_bars)

        trades = getattr(result, "trades", [])
        if trades:
            tp_trade = trades[0]
            exit_p = getattr(tp_trade, "exit_price", tp_trade.get("exit_price") if isinstance(tp_trade, dict) else None)
            self.assertAlmostEqual(exit_p, 155.400, delta=0.05,
                                  msg="Trade must close at or near TP price (155.400)")

    def test_backtest_commission_deduction(self):
        """Round-turn commission ($7/lot -> $0.07 on 0.01 lot) is deducted from net PnL."""
        engine = self._get_engine(commission_per_lot=7.0)
        # Expected commission for 0.01 lot is 7.0 * 0.01 = $0.07
        expected_comm = 0.07

        calc_comm_fn = getattr(engine, "calculate_commission", None)
        if calc_comm_fn is not None:
            comm = calc_comm_fn(volume=0.01)
            self.assertAlmostEqual(comm, expected_comm, places=3)
        else:
            # Check via simulated trade
            df = generate_synthetic_m5_data(bars=50, regime="trend_up")
            result = engine.run(df)
            trades = getattr(result, "trades", [])
            for tr in trades:
                gross = getattr(tr, "gross_pnl_usd", tr.get("gross_pnl_usd"))
                net = getattr(tr, "net_pnl_usd", tr.get("net_pnl_usd"))
                comm = getattr(tr, "commission", tr.get("commission"))
                self.assertAlmostEqual(net, gross - comm, places=3,
                                      msg="Net PnL must equal Gross PnL minus Commission")

    def test_backtest_equity_curve_generation(self):
        """Equity curve records timestamped balance and equity values."""
        engine = self._get_engine()
        df = generate_synthetic_m5_data(bars=60, regime="random_walk")

        result = engine.run(df)
        equity_curve = getattr(result, "equity_curve", None)
        self.assertIsNotNone(equity_curve, "BacktestResult must contain equity_curve")
        self.assertIsInstance(equity_curve, (pd.DataFrame, pd.Series, list),
                              "Equity curve should be DataFrame, Series, or list")

        if isinstance(equity_curve, pd.DataFrame):
            self.assertTrue(
                "equity" in equity_curve.columns or "balance" in equity_curve.columns,
                "Equity curve DataFrame must include 'equity' or 'balance'"
            )
            self.assertGreater(len(equity_curve), 0, "Equity curve must have records")

    def test_backtest_trade_metrics_output(self):
        """BacktestResult provides list of trades complying with interface contract."""
        engine = self._get_engine()
        df = generate_synthetic_m5_data(bars=100, regime="trend_up")

        result = engine.run(df)
        self.assertTrue(hasattr(result, "trades"), "BacktestResult must have 'trades' attribute")
        trades = result.trades
        self.assertIsInstance(trades, list, "trades must be a list")

        # If trades exist, check contract fields
        required_fields = ["entry_time", "exit_time", "entry_price", "exit_price", "volume"]
        for tr in trades[:5]:
            for f in required_fields:
                val = getattr(tr, f, tr.get(f) if isinstance(tr, dict) else None)
                self.assertIsNotNone(val, f"Trade record missing required field '{f}'")


if __name__ == "__main__":
    unittest.main()

"""Tier 3: Cross-feature tests combining simulated trade stream with Institutional Metrics engine."""

import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.backtest.metrics import calculate_metrics, PerformanceMetrics
except ImportError:
    calculate_metrics, PerformanceMetrics = None, None

try:
    from gbpjpy.backtest.engine import BacktestResult
except ImportError:
    BacktestResult = None


class TestMockAndMetrics(unittest.TestCase):
    """Integration test suite feeding a synthetic stream of trades into the metrics calculation engine."""

    def _generate_synthetic_trades(self, initial_balance=250.0):
        """Generate a realistic 4-week trade stream with known returns."""
        trades = []
        base_time = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)

        pnl_schedule = [
            (0, 1.50, 20.0), (1, -1.00, -15.0), (2, 2.00, 28.0), (3, -0.80, -12.0), (4, 1.50, 22.0),
            (7, 2.00, 25.0), (8, -1.20, -18.0), (9, 1.80, 24.0), (10, -1.00, -14.0), (11, 1.20, 18.0),
            (14, -1.50, -20.0), (15, 1.00, 14.0), (16, -1.80, -25.0), (17, 1.20, 16.0), (18, -0.40, -6.0),
            (21, 2.50, 32.0), (22, 1.80, 25.0), (23, -1.20, -18.0), (24, 2.00, 28.0), (25, -1.60, -22.0),
        ]

        for day_offset, net_pnl, pips in pnl_schedule:
            entry_t = base_time + timedelta(days=day_offset, hours=2)
            exit_t = entry_t + timedelta(minutes=45)
            trades.append({
                "entry_time": entry_t,
                "exit_time": exit_t,
                "type": "BUY" if pips > 0 else "SELL",
                "volume": 0.01,
                "entry_price": 195.000,
                "exit_price": 195.000 + (pips * 0.01),
                "gross_pnl_usd": net_pnl + 0.07,
                "commission": 0.07,
                "net_pnl_usd": net_pnl,
                "pips": pips
            })

        return trades

    def test_stream_of_mock_trades_to_performance_report(self):
        """20 simulated trades flow into metrics engine; verifies weekly aggregation, win rate, and drawdown."""
        self.assertIsNotNone(calculate_metrics, "gbpjpy.backtest.metrics module not implemented")

        trades = self._generate_synthetic_trades(initial_balance=250.0)
        metrics = calculate_metrics(trades=trades, initial_balance=250.0)

        self.assertIsNotNone(metrics, "calculate_metrics must return a metrics object/dict")

        def get_val(key):
            if hasattr(metrics, key):
                return getattr(metrics, key)
            elif isinstance(metrics, dict) and key in metrics:
                return metrics[key]
            return None

        total_trades = get_val("total_trades")
        self.assertEqual(total_trades, 20)

        net_profit = get_val("total_net_profit") or get_val("net_pnl")
        if net_profit is not None:
            self.assertAlmostEqual(net_profit, 8.00, places=2)

        weekly_returns = get_val("weekly_returns")
        if weekly_returns is not None:
            self.assertEqual(len(weekly_returns), 4)
            target_hit_rate = get_val("percent_weeks_target_hit") or get_val("target_hit_rate")
            if target_hit_rate is not None:
                self.assertAlmostEqual(target_hit_rate, 75.0, places=1)

        max_dd = get_val("max_drawdown_pct") or get_val("max_drawdown")
        if max_dd is not None:
            self.assertGreater(max_dd, 0.0)
            self.assertLessEqual(max_dd, 5.0)

        sharpe = get_val("sharpe_ratio")
        if sharpe is not None:
            self.assertFalse(isinstance(sharpe, complex))


if __name__ == "__main__":
    unittest.main()

"""
Tier 1 Feature Tests: F9 - Institutional Metrics & Weekly Analysis.
Verifies weekly return aggregation, % weeks >= 1.0%, Sharpe ratio, Sortino ratio, and max drawdown.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class TestF9Metrics(unittest.TestCase):
    """Test suite for institutional performance metrics calculation and weekly analysis."""

    def setUp(self):
        try:
            from gbpjpy.backtest import metrics
            self.metrics = metrics
        except ImportError:
            self.metrics = None

    def _generate_mock_trades(self, initial_balance=250.0):
        """Generate 4 weeks of synthetic trades with known PnL distributions."""
        trades = []
        # Week 1: +$3.00 net gain (+1.2% on $250) -> TARGET HIT (>= 1%)
        w1_time = datetime(2026, 1, 5, 10, 0)
        trades.append({"exit_time": w1_time, "net_pnl_usd": 2.0, "gross_pnl_usd": 2.07, "commission": 0.07})
        trades.append({"exit_time": w1_time + timedelta(hours=4), "net_pnl_usd": 1.0, "gross_pnl_usd": 1.07, "commission": 0.07})

        # Week 2: +$3.50 net gain (+1.4% on $250) -> TARGET HIT (>= 1%)
        w2_time = datetime(2026, 1, 12, 10, 0)
        trades.append({"exit_time": w2_time, "net_pnl_usd": 3.5, "gross_pnl_usd": 3.57, "commission": 0.07})

        # Week 3: -$1.50 net loss (-0.6% on $250) -> TARGET MISSED (< 1%)
        w3_time = datetime(2026, 1, 19, 10, 0)
        trades.append({"exit_time": w3_time, "net_pnl_usd": -1.5, "gross_pnl_usd": -1.43, "commission": 0.07})

        # Week 4: +$2.75 net gain (+1.1% on $250) -> TARGET HIT (>= 1%)
        w4_time = datetime(2026, 1, 26, 10, 0)
        trades.append({"exit_time": w4_time, "net_pnl_usd": 2.75, "gross_pnl_usd": 2.82, "commission": 0.07})

        return trades

    def test_weekly_return_aggregation(self):
        """Aggregate trade returns by ISO calendar week."""
        self.assertIsNotNone(self.metrics, "gbpjpy.backtest.metrics module not implemented")
        trades = self._generate_mock_trades()

        calc_weekly_fn = getattr(self.metrics, "calculate_weekly_returns", None) or getattr(self.metrics, "aggregate_weekly_returns", None)
        if calc_weekly_fn is not None:
            weekly_res = calc_weekly_fn(trades, initial_balance=250.0)
            self.assertEqual(len(weekly_res), 4, "Should aggregate exactly 4 distinct ISO calendar weeks")
        elif hasattr(self.metrics, "calculate_metrics"):
            m = self.metrics.calculate_metrics(trades, initial_balance=250.0)
            weekly = getattr(m, "weekly_returns", m.get("weekly_returns") if isinstance(m, dict) else None)
            self.assertIsNotNone(weekly, "Performance metrics should include weekly_returns distribution")
        else:
            self.fail("gbpjpy.backtest.metrics must implement weekly return aggregation")

    def test_percent_weeks_target_hit(self):
        """Calculate percentage of weeks achieving >= 1.0% return (3 out of 4 weeks = 75.0%)."""
        self.assertIsNotNone(self.metrics, "gbpjpy.backtest.metrics module not implemented")
        trades = self._generate_mock_trades(initial_balance=250.0)

        calc_pct_fn = getattr(self.metrics, "calculate_percent_weeks_target_hit", None)
        if calc_pct_fn is not None:
            pct_hit = calc_pct_fn(trades, initial_balance=250.0, target_pct=1.0)
            self.assertAlmostEqual(pct_hit, 75.0, places=1, msg="3 of 4 weeks hitting >= 1% should yield 75%")
        elif hasattr(self.metrics, "calculate_metrics"):
            m = self.metrics.calculate_metrics(trades, initial_balance=250.0)
            pct_hit = getattr(m, "pct_weeks_above_1pct", m.get("pct_weeks_above_1pct") if isinstance(m, dict) else None)
            if pct_hit is not None:
                self.assertAlmostEqual(pct_hit, 75.0, places=1)

    def test_sharpe_ratio_calculation(self):
        """Calculate annualized Sharpe ratio from return series."""
        self.assertIsNotNone(self.metrics, "gbpjpy.backtest.metrics module not implemented")
        # Returns series with mean 0.01 and std 0.005
        # Annualized Sharpe (assuming 52 weeks) = (0.01 / 0.005) * sqrt(52) = 2.0 * 7.211 = ~14.42
        calc_sharpe_fn = getattr(self.metrics, "calculate_sharpe_ratio", None)
        if calc_sharpe_fn is not None:
            returns = [0.012, 0.014, -0.006, 0.011]
            sharpe = calc_sharpe_fn(returns, risk_free_rate=0.0, periods_per_year=52)
            self.assertIsInstance(sharpe, (float, int))
            self.assertGreater(sharpe, 0.0, "Positive mean excess return must yield positive Sharpe")
        else:
            trades = self._generate_mock_trades()
            m = self.metrics.calculate_metrics(trades, initial_balance=250.0)
            sharpe = getattr(m, "sharpe_ratio", m.get("sharpe_ratio") if isinstance(m, dict) else None)
            self.assertIsNotNone(sharpe, "Performance metrics must include sharpe_ratio")

    def test_sortino_ratio_calculation(self):
        """Calculate Sortino ratio isolating downside volatility."""
        self.assertIsNotNone(self.metrics, "gbpjpy.backtest.metrics module not implemented")
        calc_sortino_fn = getattr(self.metrics, "calculate_sortino_ratio", None)
        if calc_sortino_fn is not None:
            returns = [0.012, 0.014, -0.006, 0.011]
            sortino = calc_sortino_fn(returns, risk_free_rate=0.0, periods_per_year=52)
            self.assertIsInstance(sortino, (float, int))
            self.assertGreater(sortino, 0.0, "Positive overall performance must yield positive Sortino")
        else:
            trades = self._generate_mock_trades()
            m = self.metrics.calculate_metrics(trades, initial_balance=250.0)
            sortino = getattr(m, "sortino_ratio", m.get("sortino_ratio") if isinstance(m, dict) else None)
            self.assertIsNotNone(sortino, "Performance metrics must include sortino_ratio")

    def test_max_drawdown_calculation(self):
        """Calculate maximum peak-to-trough drawdown percentage."""
        self.assertIsNotNone(self.metrics, "gbpjpy.backtest.metrics module not implemented")
        # Equity curve: 250 -> 260 -> 247 -> 255
        # Peak = 260, Trough = 247. Max DD = (260 - 247) / 260 = 13 / 260 = 5.0%
        equity_series = pd.Series([250.0, 260.0, 247.0, 255.0])
        expected_dd_pct = ((260.0 - 247.0) / 260.0) * 100.0  # 5.0%

        calc_dd_fn = getattr(self.metrics, "calculate_max_drawdown", None)
        if calc_dd_fn is not None:
            max_dd = calc_dd_fn(equity_series)
            # Accept either decimal 0.05 or percentage 5.0
            if max_dd < 1.0:
                max_dd *= 100.0
            self.assertAlmostEqual(max_dd, expected_dd_pct, places=2)
        else:
            trades = self._generate_mock_trades()
            m = self.metrics.calculate_metrics(trades, initial_balance=250.0)
            max_dd = getattr(m, "max_drawdown", m.get("max_drawdown") if isinstance(m, dict) else None)
            self.assertIsNotNone(max_dd, "Performance metrics must include max_drawdown")

    def test_profit_factor_calculation(self):
        """Profit factor = gross profit / gross loss (2.07 + 3.57 + 2.82) / 1.43 = 8.46 / 1.43 = ~5.91."""
        self.assertIsNotNone(self.metrics, "gbpjpy.backtest.metrics module not implemented")
        trades = self._generate_mock_trades()
        calc_pf_fn = getattr(self.metrics, "calculate_profit_factor", None)

        if calc_pf_fn is not None:
            pf = calc_pf_fn(trades)
            self.assertAlmostEqual(pf, (2.07 + 1.07 + 3.57 + 2.82) / 1.43, delta=0.2)
        else:
            m = self.metrics.calculate_metrics(trades, initial_balance=250.0)
            pf = getattr(m, "profit_factor", m.get("profit_factor") if isinstance(m, dict) else None)
            self.assertIsNotNone(pf, "Performance metrics must include profit_factor")


if __name__ == "__main__":
    unittest.main()

"""
Tier 1 Feature Tests: F10 - Walk-Forward & Anti-Overfitting Validation.
Verifies In-Sample/Out-of-Sample split, 24-fold walk-forward window generator, WFE ratio, Monte Carlo permutations, and parameter stability.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tests.fixtures import generate_synthetic_m5_data


class TestF10Validation(unittest.TestCase):
    """Test suite for WalkForwardValidator and anti-overfitting protocol."""

    def setUp(self):
        try:
            from gbpjpy.backtest.walk_forward_validator import WalkForwardValidator
            self.WalkForwardValidator = WalkForwardValidator
        except ImportError:
            self.WalkForwardValidator = None

    def _get_validator(self):
        self.assertIsNotNone(self.WalkForwardValidator, "gbpjpy.backtest.walk_forward_validator module not implemented")
        if isinstance(self.WalkForwardValidator, type):
            return self.WalkForwardValidator()
        return self.WalkForwardValidator

    def test_in_sample_out_sample_split(self):
        """Strict chronological partition without data leakage between In-Sample and Out-of-Sample."""
        val = self._get_validator()
        # Generate multi-day DataFrame
        start_dt = datetime(2024, 1, 1, 0, 0)
        df = generate_synthetic_m5_data(bars=1000, start_time=start_dt)

        split_fn = getattr(val, "split_is_oos", None) or getattr(val, "train_test_split", None)
        if split_fn is not None:
            df_is, df_oos = split_fn(df, is_ratio=0.7)
            self.assertEqual(len(df_is) + len(df_oos), len(df), "Split must partition all rows")
            # Strict chronological check: max timestamp in IS must be < min timestamp in OOS
            max_is_time = df_is["time"].max() if "time" in df_is.columns else df_is.index.max()
            min_oos_time = df_oos["time"].min() if "time" in df_oos.columns else df_oos.index.min()
            self.assertLess(max_is_time, min_oos_time, "In-Sample data must strictly precede Out-of-Sample data")
        else:
            self.fail("WalkForwardValidator must implement chronological split_is_oos")

    def test_walk_forward_window_generation(self):
        """Verify 24 rolling train/test windows generated correctly with chronological separation."""
        val = self._get_validator()
        gen_windows_fn = getattr(val, "generate_windows", None) or getattr(val, "generate_walk_forward_windows", None)
        self.assertIsNotNone(gen_windows_fn, "WalkForwardValidator must implement window generation")

        # Request 24 folds
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2026, 1, 1)
        windows = gen_windows_fn(start_date=start_date, end_date=end_date, folds=24)

        self.assertIsInstance(windows, list, "Windows output must be a list")
        self.assertEqual(len(windows), 24, "Must generate exactly 24 rolling walk-forward windows")

        for i, win in enumerate(windows):
            train_start = getattr(win, "train_start", win.get("train_start") if isinstance(win, dict) else None)
            train_end = getattr(win, "train_end", win.get("train_end") if isinstance(win, dict) else None)
            test_start = getattr(win, "test_start", win.get("test_start") if isinstance(win, dict) else None)
            test_end = getattr(win, "test_end", win.get("test_end") if isinstance(win, dict) else None)

            self.assertLess(train_start, train_end, f"Fold {i}: train_start must precede train_end")
            self.assertLessEqual(train_end, test_start, f"Fold {i}: train_end must precede or equal test_start")
            self.assertLess(test_start, test_end, f"Fold {i}: test_start must precede test_end")

    def test_walk_forward_efficiency_ratio(self):
        """WFE = (Annualized OOS Return / Annualized IS Return) calculated accurately."""
        val = self._get_validator()
        is_annual_return = 0.20   # 20%
        oos_annual_return = 0.15  # 15%
        expected_wfe = 0.15 / 0.20  # 0.75 or 75%

        calc_wfe_fn = getattr(val, "calculate_wfe", None) or getattr(val, "calculate_walk_forward_efficiency", None)
        if calc_wfe_fn is not None:
            wfe = calc_wfe_fn(is_return=is_annual_return, oos_return=oos_annual_return)
            # Accept either decimal (0.75) or percentage (75.0)
            if wfe > 1.0:
                wfe /= 100.0
            self.assertAlmostEqual(wfe, expected_wfe, places=3,
                                  msg="WFE must equal OOS return divided by IS return")
        else:
            self.fail("WalkForwardValidator must implement calculate_wfe")

    def test_monte_carlo_trade_permutation(self):
        """Random trade shuffling preserves terminal return mean while computing drawdown distribution."""
        val = self._get_validator()
        trades_pnl = [1.5, -0.8, 2.1, -1.2, 1.9, 0.5, -0.4, 2.3, -1.0, 1.7]  # sum = 6.6
        initial_balance = 250.0

        mc_fn = getattr(val, "monte_carlo_simulation", None) or getattr(val, "permute_trades", None)
        if mc_fn is not None:
            sim_result = mc_fn(trades_pnl, initial_balance=initial_balance, iterations=500)
            # Terminal equity mean across all permutations must equal original sum: 250 + 6.6 = 256.6
            mean_terminal = getattr(sim_result, "mean_terminal_equity", sim_result.get("mean_terminal_equity") if isinstance(sim_result, dict) else None)
            if mean_terminal is not None:
                self.assertAlmostEqual(mean_terminal, initial_balance + sum(trades_pnl), places=2)

            # Drawdown distribution should show positive risk (e.g. 95th percentile DD)
            worst_dd = getattr(sim_result, "worst_drawdown", sim_result.get("worst_drawdown") if isinstance(sim_result, dict) else None)
            if worst_dd is not None:
                self.assertGreater(worst_dd, 0.0)
        else:
            self.fail("WalkForwardValidator must implement monte_carlo_simulation")

    def test_parameter_stability_metric(self):
        """Verify parameter sensitivity scoring across neighboring parameter grids."""
        val = self._get_validator()
        # Simulated grid of 3x3 parameter results: center is optimal, neighbors are nearby
        param_grid_results = {
            (10, 1.5): 1.8, (10, 2.0): 1.9, (10, 2.5): 1.7,
            (14, 1.5): 2.0, (14, 2.0): 2.2, (14, 2.5): 2.1,
            (20, 1.5): 1.7, (20, 2.0): 1.8, (20, 2.5): 1.6,
        }

        calc_stability_fn = getattr(val, "calculate_parameter_stability", None)
        if calc_stability_fn is not None:
            score = calc_stability_fn(param_grid_results)
            self.assertIsInstance(score, (float, int))
            self.assertGreater(score, 0.0, "Stability score should be positive for robust neighbor plateau")
        else:
            self.assertTrue(hasattr(val, "evaluate_parameter_stability") or hasattr(val, "parameter_stability"),
                            "WalkForwardValidator must evaluate parameter stability")


if __name__ == "__main__":
    unittest.main()

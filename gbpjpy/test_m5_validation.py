"""
Milestone 5 Comprehensive Test Suite: Walk-Forward Validation, Anti-Overfitting & Main CLI
Verifies 29 unit and integration tests across:
  - WalkForwardValidator: IS/OOS chronological split, 24-fold window generator, WFE calculation,
    Monte Carlo trade permutation, parameter stability scoring, and full walk-forward run.
  - Application Entry Point (main.py): CLI argument parsing across all 4 modes, handler dispatching,
    ASCII summary generation, export output, and error handling.
  - DataEngine Validation Hooks: validate_tick corrupt data rejection and clean_data filtering.
  - Final E2E Gate: Multi-fold walk-forward pipeline execution and zero lookahead data leakage.
"""

from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_this_dir = Path(__file__).resolve().parent
_parent_root = _this_dir.parent
for _p in [str(_parent_root), str(_this_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gbpjpy.backtest.walk_forward_validator import WalkForwardValidator, WalkForwardResult, MonteCarloResult, WalkForwardWindow
from gbpjpy.core.data_engine import DataEngine
from gbpjpy.main import parse_arguments, main, format_walk_forward_ascii, format_monte_carlo_ascii

try:
    from tests.fixtures import generate_synthetic_m5_data, MockBenchmarkStrategy
except ImportError:
    from gbpjpy.tests.fixtures import generate_synthetic_m5_data, MockBenchmarkStrategy


class TestWalkForwardValidator(unittest.TestCase):
    def setUp(self):
        self.val = WalkForwardValidator()

    def test_01_initialization_defaults_and_custom(self):
        v = WalkForwardValidator(folds=12, in_sample_bars=500, out_of_sample_bars=100)
        self.assertEqual(v.folds, 12)
        self.assertEqual(v.in_sample_bars, 500)
        self.assertEqual(v.out_of_sample_bars, 100)
        self.assertEqual(v.wfe_threshold, 0.60)

    def test_02_split_is_oos_chronological_partition(self):
        df = generate_synthetic_m5_data(bars=1000)
        df_is, df_oos = self.val.split_is_oos(df, is_ratio=0.7)
        self.assertEqual(len(df_is) + len(df_oos), 1000)
        self.assertEqual(len(df_is), 700)
        self.assertEqual(len(df_oos), 300)
        self.assertLess(df_is["time"].max(), df_oos["time"].min())

    def test_03_generate_windows_24_folds_order(self):
        windows = self.val.generate_windows(datetime(2022, 1, 1), datetime(2026, 1, 1), folds=24)
        self.assertEqual(len(windows), 24)
        for w in windows:
            self.assertLess(w["train_start"], w["train_end"])
            self.assertLessEqual(w["train_end"], w["test_start"])
            self.assertLess(w["test_start"], w["test_end"])

    def test_04_generate_splits_zero_lookahead_overlap(self):
        df = generate_synthetic_m5_data(bars=3000)
        splits = self.val.generate_splits(df)
        self.assertGreaterEqual(len(splits), 5)
        for is_df, oos_df in splits:
            self.assertGreaterEqual(oos_df["time"].min(), is_df["time"].max())

    def test_05_calculate_wfe_standard_ratio(self):
        wfe = self.val.calculate_wfe(is_return=0.20, oos_return=0.15)
        self.assertAlmostEqual(wfe, 0.75, places=3)

    def test_06_calculate_wfe_annual_params(self):
        wfe = self.val.calculate_wfe(is_annual_return=0.12, oos_annual_return=0.085)
        self.assertAlmostEqual(wfe, 0.085 / 0.12, places=3)
        self.assertGreaterEqual(wfe, 0.60)

    def test_07_calculate_wfe_zero_division_guard(self):
        wfe = self.val.calculate_wfe(is_return=0.0, oos_return=0.10)
        self.assertEqual(wfe, 0.0)

    def test_08_monte_carlo_preserves_mean_equity(self):
        pnls = [1.5, -0.8, 2.1, -1.2, 1.9, 0.5, -0.4, 2.3, -1.0, 1.7]
        res = self.val.monte_carlo_simulation(pnls, initial_balance=250.0, iterations=500)
        self.assertAlmostEqual(res["mean_terminal_equity"], 250.0 + sum(pnls), places=2)
        self.assertGreater(res["worst_drawdown"], 0.0)
        self.assertGreaterEqual(res["p95_drawdown"], 0.0)

    def test_09_parameter_stability_scoring(self):
        grid = {(10, 1.5): 1.8, (10, 2.0): 1.9, (14, 1.5): 2.0, (14, 2.0): 2.2}
        score = self.val.calculate_parameter_stability(grid)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_10_full_walk_forward_run(self):
        df = generate_synthetic_m5_data(bars=2500, regime="mean_reversion")
        res = self.val.run(df=df)
        self.assertIsInstance(res, WalkForwardResult)
        self.assertGreaterEqual(res.wfe, 0.60)
        self.assertTrue(res.passed)
        # Verify result is authentic and not rigged
        expected_wfe = self.val.calculate_wfe(res.mean_is_return, res.mean_oos_return)
        self.assertAlmostEqual(res.wfe, expected_wfe, places=1)
        self.assertEqual(res.folds_passed, sum(1 for f in res.fold_details if f["passed"]))

    def test_10b_walk_forward_honestly_fails_unprofitable_data(self):
        df = generate_synthetic_m5_data(bars=1500, regime="random_walk")
        res = self.val.run(df=df)
        self.assertIsInstance(res, WalkForwardResult)
        if res.mean_is_return <= 0 or res.mean_oos_return <= 0:
            self.assertEqual(res.wfe, 0.0)
            self.assertFalse(res.passed)


class TestMainEntrypoint(unittest.TestCase):
    def test_11_cli_defaults_backtest_mode(self):
        args = parse_arguments([])
        self.assertEqual(args.mode, "backtest")
        self.assertEqual(args.initial_balance, 250.0)
        self.assertEqual(args.folds, 24)

    def test_12_cli_walk_forward_mode(self):
        args = parse_arguments(["--mode", "walk-forward", "--folds", "12", "--bars", "1500"])
        self.assertEqual(args.mode, "walk-forward")
        self.assertEqual(args.folds, 12)
        self.assertEqual(args.bars, 1500)

    def test_12b_cli_aliases_and_symbol_flag(self):
        args = parse_arguments(["--mode", "walk-forward", "--symbol", "GBPJPY", "--windows", "12"])
        self.assertEqual(args.symbol, "GBPJPY")
        self.assertEqual(args.folds, 12)
        self.assertEqual(args.windows, 12)

        args_mc = parse_arguments(["--mode", "monte-carlo", "--symbol", "GBPJPY", "--simulations", "250"])
        self.assertEqual(args_mc.symbol, "GBPJPY")
        self.assertEqual(args_mc.iterations, 250)
        self.assertEqual(args_mc.simulations, 250)

    def test_13_cli_monte_carlo_mode(self):
        args = parse_arguments(["--mode", "monte-carlo", "--iterations", "500"])
        self.assertEqual(args.mode, "monte-carlo")
        self.assertEqual(args.iterations, 500)

    def test_14_cli_stream_mode(self):
        args = parse_arguments(["--mode", "stream", "--cycles", "5"])
        self.assertEqual(args.mode, "stream")
        self.assertEqual(args.cycles, 5)

    def test_15_main_backtest_execution(self):
        exit_code = main(["--mode", "backtest", "--bars", "150"])
        self.assertEqual(exit_code, 0)

    def test_16_main_walk_forward_execution(self):
        exit_code = main(["--mode", "walk-forward", "--bars", "2000", "--folds", "5"])
        self.assertEqual(exit_code, 0)

    def test_17_main_monte_carlo_execution(self):
        exit_code = main(["--mode", "monte-carlo", "--bars", "150", "--iterations", "100"])
        self.assertEqual(exit_code, 0)

    def test_18_main_stream_execution(self):
        exit_code = main(["--mode", "stream", "--cycles", "3"])
        self.assertEqual(exit_code, 0)

    def test_19_main_export_json_output(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            exit_code = main(["--mode", "backtest", "--bars", "150", "--output", tmp_path])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("metrics", data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_20_ascii_walk_forward_formatter(self):
        res = WalkForwardResult(wfe=0.75, passed=True, folds_total=24, folds_passed=22,
                                mean_is_return=0.15, mean_oos_return=0.11)
        text = format_walk_forward_ascii(res)
        self.assertIn("WALK-FORWARD VALIDATION REPORT", text)
        self.assertIn("75.00%", text)

    def test_21_ascii_monte_carlo_formatter(self):
        mc = {"mean_terminal_equity": 256.6, "worst_drawdown": 5.2, "median_drawdown": 2.1,
              "p95_drawdown": 4.0, "p99_drawdown": 4.8, "ruin_probability": 0.0}
        text = format_monte_carlo_ascii(mc)
        self.assertIn("MONTE CARLO STRESS REPORT", text)
        self.assertIn("$256.60", text)


class TestDataEngineHooksAndE2EGate(unittest.TestCase):
    def setUp(self):
        self.engine = DataEngine()

    def test_22_validate_tick_valid_quotes(self):
        self.assertTrue(self.engine.validate_tick(195.200, 195.220))

    def test_23_validate_tick_corrupt_negative(self):
        self.assertFalse(self.engine.validate_tick(-195.200, 195.220))
        self.assertFalse(self.engine.validate_tick(195.200, -195.220))

    def test_24_validate_tick_corrupt_zero(self):
        self.assertFalse(self.engine.validate_tick(0.0, 195.220))
        self.assertFalse(self.engine.validate_tick(195.200, 0.0))

    def test_25_validate_tick_inverted_or_zero_spread(self):
        self.assertFalse(self.engine.validate_tick(195.500, 195.200))
        self.assertFalse(self.engine.validate_tick(195.200, 195.200))

    def test_26_validate_tick_nan_values(self):
        self.assertFalse(self.engine.validate_tick(float("nan"), 195.220))
        self.assertFalse(self.engine.validate_tick(195.200, float("nan")))

    def test_27_clean_data_filters_corrupt_rows(self):
        df_dirty = pd.DataFrame([
            {"open": 195.0, "high": 195.5, "low": 194.8, "close": 195.2, "volume": 100},
            {"open": -195.0, "high": 195.5, "low": 194.8, "close": 195.2, "volume": 100},
            {"open": 195.0, "high": 194.0, "low": 196.0, "close": 195.2, "volume": 100},
        ])
        cleaned = self.engine.clean_data(df_dirty)
        self.assertEqual(len(cleaned), 1)

    def test_28_tier2_boundary_test_corrupt_tick_parity(self):
        corrupt_samples = [
            {"bid": -195.200, "ask": 195.220},
            {"bid": 0.0, "ask": 195.220},
            {"bid": 195.500, "ask": 195.200},
            {"bid": float("nan"), "ask": 195.2},
        ]
        for s in corrupt_samples:
            self.assertFalse(self.engine.validate_tick(s["bid"], s["ask"]))
        self.assertTrue(self.engine.validate_tick(195.200, 195.220))

    def test_29_e2e_full_pipeline_walk_forward_gate(self):
        df = generate_synthetic_m5_data(bars=2000, regime="trend_up")
        df = self.engine.compute_indicators(df)
        val = WalkForwardValidator(folds=5, in_sample_bars=800, out_of_sample_bars=200)
        strat = MockBenchmarkStrategy()
        res = val.run(df=df, strategy=strat)
        self.assertIsInstance(res, WalkForwardResult)
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.wfe, 0.60)
        self.assertGreater(res.folds_passed, 0)


if __name__ == "__main__":
    unittest.main()

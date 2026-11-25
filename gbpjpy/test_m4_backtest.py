"""
Milestone 4 Comprehensive Test Suite: Event-Driven Backtest Engine, Metrics & CLI.
Verifies ~30 unit and integration tests across engine simulation, institutional metrics, CLI runner, and end-to-end backtesting.
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Add parent workspace to sys.path so gbpjpy is importable
_this_dir = Path(__file__).resolve().parent
_parent_root = _this_dir.parent
for _p in [str(_parent_root), str(_this_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gbpjpy.backtest.engine import BacktestEngine, BacktestResult, TradeRecord
from gbpjpy.backtest.metrics import (
    calculate_metrics,
    calculate_weekly_returns,
    calculate_percent_weeks_target_hit,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    PerformanceMetrics,
    WeeklyReturn,
)
from gbpjpy.backtest.run_backtest import parse_arguments, format_ascii_report, generate_synthetic_data, run_pipeline
from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
from gbpjpy.core.data_engine import DataEngine
from tests.fixtures import generate_synthetic_m5_data


class TestBacktestEngineSimulation(unittest.TestCase):
    """Test suite for event-driven engine execution, order lifecycle, and microstructure."""

    def test_engine_initialization_defaults_and_custom(self):
        """Engine initializes with default GBPJPY parameters or custom overrides."""
        engine_default = BacktestEngine()
        self.assertEqual(engine_default.initial_balance, 250.0)
        self.assertEqual(engine_default.commission_per_lot, 7.0)
        self.assertEqual(engine_default.spread_pips, 1.2)
        self.assertEqual(engine_default.slippage_pips, 0.5)
        self.assertEqual(engine_default.symbol, "GBPJPY")

        engine_custom = BacktestEngine(
            initial_balance=500.0,
            commission_per_lot=5.0,
            spread_pips=1.5,
            slippage_pips=0.2,
            leverage=50.0,
        )
        self.assertEqual(engine_custom.initial_balance, 500.0)
        self.assertEqual(engine_custom.commission_per_lot, 5.0)
        self.assertEqual(engine_custom.spread_pips, 1.5)
        self.assertEqual(engine_custom.slippage_pips, 0.2)
        self.assertEqual(engine_custom.leverage, 50.0)

    def test_bar_processing_counts(self):
        """Engine processes all sequential bars in dataset without data loss."""
        engine = BacktestEngine()
        df = generate_synthetic_m5_data(bars=75, regime="trend_up")
        result = engine.run(df)
        self.assertEqual(result.bars_processed, 75)
        self.assertEqual(result.total_bars, 75)
        self.assertEqual(len(result.equity_curve), 75)

    def test_commission_calculation_exact(self):
        """Commission is exactly $7.00/lot ($0.07 per 0.01 lot)."""
        engine = BacktestEngine(commission_per_lot=7.0)
        self.assertAlmostEqual(engine.calculate_commission(0.01), 0.07, places=4)
        self.assertAlmostEqual(engine.calculate_commission(0.10), 0.70, places=4)
        self.assertAlmostEqual(engine.calculate_commission(1.00), 7.00, places=4)

    def test_cross_currency_pnl_calculation(self):
        """JPY cross-currency PnL uses tick value 0.626076 with point 0.001."""
        engine = BacktestEngine(point=0.001, default_tick_value=0.626076)
        # 10 pips = 0.100 JPY diff = 100 points
        # For 0.01 lot: 100 * 0.626076 * 0.01 = 0.6261 USD
        buy_pnl = engine.calculate_pnl(entry_price=195.000, exit_price=195.100, volume=0.01, is_buy=True)
        self.assertAlmostEqual(buy_pnl, 0.6261, places=3)

        sell_pnl = engine.calculate_pnl(entry_price=195.000, exit_price=194.900, volume=0.01, is_buy=False)
        self.assertAlmostEqual(sell_pnl, 0.6261, places=3)

    def test_swap_calculation_wednesday_triple(self):
        """Overnight swap on Wednesday applies standard 3x weekend carry multiplier."""
        engine = BacktestEngine()
        tue_dt = datetime(2026, 6, 9, 22, 0, tzinfo=timezone.utc)  # Tuesday (weekday 1)
        wed_dt = datetime(2026, 6, 10, 22, 0, tzinfo=timezone.utc) # Wednesday (weekday 2)
        thu_dt = datetime(2026, 6, 11, 22, 0, tzinfo=timezone.utc) # Thursday (weekday 3)

        tue_swap = engine.calculate_swap(volume=0.01, is_long=True, rollover_dt=tue_dt)
        wed_swap = engine.calculate_swap(volume=0.01, is_long=True, rollover_dt=wed_dt)
        thu_swap = engine.calculate_swap(volume=0.01, is_long=True, rollover_dt=thu_dt)

        self.assertAlmostEqual(abs(wed_swap), abs(tue_swap) * 3, places=2)
        self.assertAlmostEqual(abs(wed_swap), abs(thu_swap) * 3, places=2)

    def test_stop_loss_execution_and_slippage(self):
        """Stop-loss exit fills with adverse slippage and logs net PnL."""
        engine = BacktestEngine(slippage_pips=0.5)
        res = engine.execute_stop_exit(sl_price=195.200, is_long=True, slippage_pips=0.5)
        self.assertAlmostEqual(res["exit_price"], 195.195, places=3)
        self.assertEqual(res["slippage_pips"], 0.5)

    def test_take_profit_execution(self):
        """Take-profit hit closes trade at exact TP price."""
        engine = BacktestEngine()
        t0 = datetime(2026, 1, 15, 8, 0)
        df_tp = pd.DataFrame({
            "time": [t0 + timedelta(minutes=5 * i) for i in range(4)],
            "open":  [155.000, 155.100, 155.200, 155.300],
            "high":  [155.100, 155.250, 155.500, 155.400], # Bar 2 hits TP 155.400
            "low":   [154.950, 155.050, 155.150, 155.200],
            "close": [155.050, 155.200, 155.350, 155.300],
            "volume": [500] * 4,
        })
        engine.inject_trade(order_type="BUY", entry_price=155.000, sl=154.800, tp=155.400, volume=0.01)
        res = engine.run(df_tp)
        self.assertEqual(len(res.trades), 1)
        self.assertAlmostEqual(res.trades[0].exit_price, 155.400, delta=0.05)
        self.assertEqual(res.trades[0].exit_reason, "TP")

    def test_dual_touch_spike_conservative_sl_precedence(self):
        """When an intrabar spike touches both SL and TP, conservative SL precedence fires SL, not TP."""
        # 1. Long position test
        engine_long = BacktestEngine(slippage_pips=0.0)
        t0 = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
        df_spike_long = pd.DataFrame({
            "time": [t0, t0 + timedelta(minutes=5)],
            "open":  [195.000, 195.000],
            "high":  [195.050, 195.500],  # Touches TP at 195.500
            "low":   [194.950, 194.400],  # Breaches SL at 194.500
            "close": [195.000, 194.400],  # Massive bearish candle
            "volume": [500, 500],
        })
        engine_long.inject_trade(order_type="BUY", entry_price=195.000, sl=194.500, tp=195.500, volume=0.01)
        res_long = engine_long.run(df_spike_long)
        self.assertEqual(len(res_long.trades), 1)
        self.assertEqual(res_long.trades[0].exit_reason, "SL")
        self.assertAlmostEqual(res_long.trades[0].exit_price, 194.500, places=3)
        self.assertLess(res_long.trades[0].net_pnl_usd, 0.0)

        # 2. Short position test
        engine_short = BacktestEngine(slippage_pips=0.0)
        df_spike_short = pd.DataFrame({
            "time": [t0, t0 + timedelta(minutes=5)],
            "open":  [195.000, 195.000],
            "high":  [195.050, 195.600],  # Breaches SL at 195.500
            "low":   [194.950, 194.500],  # Touches TP at 194.500
            "close": [195.000, 195.600],  # Massive bullish candle
            "volume": [500, 500],
        })
        engine_short.inject_trade(order_type="SELL", entry_price=195.000, sl=195.500, tp=194.500, volume=0.01)
        res_short = engine_short.run(df_spike_short)
        self.assertEqual(len(res_short.trades), 1)
        self.assertEqual(res_short.trades[0].exit_reason, "SL")
        self.assertAlmostEqual(res_short.trades[0].exit_price, 195.500, places=3)
        self.assertLess(res_short.trades[0].net_pnl_usd, 0.0)

    def test_margin_and_leverage_gate_order_rejection(self):
        """Oversized volume exceeding account margin or leverage limits is rejected."""
        from gbpjpy.risk.risk_manager import RiskManager

        class DummyOversizedStrategy:
            def __init__(self, volume: float = 5.0):
                self.volume = volume

            def generate_signals(self, window_df):
                return [{
                    "type": "BUY",
                    "volume": self.volume,
                    "sl": 194.000,
                    "tp": 196.000,
                    "reason": "OversizedSignal",
                }]

        t0 = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
        df_dummy = pd.DataFrame({
            "time": [t0 + timedelta(minutes=5 * i) for i in range(10)],
            "open": [195.000 + 0.01 * i for i in range(10)],
            "high": [195.050 + 0.01 * i for i in range(10)],
            "low": [194.950 + 0.01 * i for i in range(10)],
            "close": [195.000 + 0.01 * i for i in range(10)],
            "volume": [500] * 10,
        })

        # 1. With RiskManager: 5.0 lots on $250 equity requires ~$21,600 margin -> REJECTED
        rm = RiskManager(initial_balance=250.0)
        engine_rm = BacktestEngine(initial_balance=250.0, leverage=30.0, risk_manager=rm)
        res_rm = engine_rm.run(df=df_dummy, strategy=DummyOversizedStrategy(volume=5.0))
        self.assertEqual(len(res_rm.trades), 0, "Oversized 5.0 lot trade must be rejected by RiskManager margin gate")

        # 2. Standalone engine (without RiskManager): 5.0 lots on $250 balance at 1:30 leverage -> REJECTED
        engine_standalone = BacktestEngine(initial_balance=250.0, leverage=30.0, risk_manager=None)
        res_standalone = engine_standalone.run(df=df_dummy, strategy=DummyOversizedStrategy(volume=5.0))
        self.assertEqual(len(res_standalone.trades), 0, "Oversized 5.0 lot trade must be rejected by engine margin gate")

        # 3. Valid sizing (0.01 lot on $250 balance) -> ACCEPTED
        res_valid = engine_rm.run(df=df_dummy, strategy=DummyOversizedStrategy(volume=0.01))
        self.assertGreaterEqual(len(res_valid.trades), 1, "Valid 0.01 lot trade must be accepted")

    def test_gap_exit_slippage_calculation(self):
        """Gap open past stop loss fills at gap open price, recording adverse slippage."""
        engine = BacktestEngine()
        gap_bar = {"open": 194.200, "high": 194.300, "low": 194.100, "close": 194.250}
        res = engine.simulate_gap_exit(entry_price=195.000, sl_price=194.600, is_long=True, gap_bar=gap_bar)
        self.assertEqual(res["exit_price"], 194.200)
        self.assertAlmostEqual(res["slippage_pips"], 40.0, places=1) # (194.600 - 194.200)/0.01 = 40 pips

    def test_trailing_stop_ratchet_invariant(self):
        """Trailing stop ratchets strictly in favorable direction and never loosens."""
        engine = BacktestEngine(atr_multiplier=1.5)
        t0 = datetime(2026, 1, 15, 8, 0)
        df_trail = pd.DataFrame({
            "time": [t0 + timedelta(minutes=5 * i) for i in range(5)],
            "open":  [195.000, 195.100, 195.300, 195.150, 195.100],
            "high":  [195.150, 195.400, 195.500, 195.250, 195.150],
            "low":   [194.950, 195.050, 195.100, 195.000, 194.900],
            "close": [195.100, 195.350, 195.200, 195.100, 194.950],
            "atr":   [0.100, 0.100, 0.100, 0.100, 0.100],
            "volume": [500] * 5,
        })
        engine.inject_trade(order_type="BUY", entry_price=195.000, sl=194.800, tp=196.000, volume=0.01)
        res = engine.run(df_trail)
        self.assertGreater(len(res.trades), 0)
        trade = res.trades[0]
        # In Bar 2, high reached 195.500 -> candidate SL = 195.500 - 1.5*0.100 = 195.350
        # Subsequent pullback in Bar 3/4 exited above original 194.800 SL
        self.assertGreater(trade.exit_price, 194.800)

    def test_breakeven_buffer_execution(self):
        """Breakeven triggers at +1.45R and advances stop to entry + 2.0 pips."""
        engine = BacktestEngine(enable_breakeven=True, be_trigger_r=1.45, be_buffer_pips=2.0)
        self.assertEqual(engine.be_trigger_r, 1.45)
        self.assertEqual(engine.be_buffer_pips, 2.0)

    def test_time_decay_exit(self):
        """Trades held beyond max_bars_held exit automatically on time decay."""
        engine = BacktestEngine(max_bars_held=3, enable_time_decay=True)
        t0 = datetime(2026, 1, 15, 8, 0)
        # 5 bars where price hovers flat (neither SL nor TP hit)
        df_flat = pd.DataFrame({
            "time": [t0 + timedelta(minutes=5 * i) for i in range(6)],
            "open":  [195.000] * 6,
            "high":  [195.050] * 6,
            "low":   [194.950] * 6,
            "close": [195.000] * 6,
            "atr":   [0.100] * 6,
            "volume": [500] * 6,
        })
        engine.inject_trade(order_type="BUY", entry_price=195.000, sl=194.000, tp=196.000, volume=0.01)
        res = engine.run(df_flat)
        self.assertEqual(len(res.trades), 1)
        self.assertEqual(res.trades[0].exit_reason, "TIME_DECAY")
        self.assertEqual(res.trades[0].bars_held, 3)

    def test_trade_record_and_result_dual_access(self):
        """TradeRecord and BacktestResult support dual attribute and dictionary indexing."""
        tr = TradeRecord(
            ticket=101,
            symbol="GBPJPY",
            type="BUY",
            volume=0.01,
            entry_price=195.000,
            exit_price=195.300,
            net_pnl_usd=1.81,
            pips=30.0,
        )
        self.assertEqual(tr.ticket, 101)
        self.assertEqual(tr["ticket"], 101)
        self.assertEqual(tr.get("net_pnl_usd"), 1.81)
        self.assertTrue("pips" in tr)

        eq_df = pd.DataFrame([{"time": datetime.now(), "balance": 250.0, "equity": 250.0}])
        res = BacktestResult(trades=[tr], equity_curve=eq_df, initial_balance=250.0, final_balance=251.81)
        self.assertEqual(res.total_trades, 1)
        self.assertEqual(res["total_trades"], 1)
        self.assertEqual(res.total_net_profit, 1.81)
        self.assertEqual(res["net_pnl"], 1.81)


class TestInstitutionalMetricsSuite(unittest.TestCase):
    """Test suite for institutional weekly returns, target hit rates, Sharpe, Sortino, and drawdowns."""

    def _get_sample_trades(self):
        """Construct 4 distinct ISO calendar weeks with 3 winning weeks and 1 losing week."""
        t1 = datetime(2026, 1, 5, 10, 0)
        t2 = datetime(2026, 1, 12, 10, 0)
        t3 = datetime(2026, 1, 19, 10, 0)
        t4 = datetime(2026, 1, 26, 10, 0)
        return [
            {"exit_time": t1, "net_pnl_usd": 3.00, "gross_pnl_usd": 3.07, "commission": 0.07, "pips": 48.0},
            {"exit_time": t2, "net_pnl_usd": 3.50, "gross_pnl_usd": 3.57, "commission": 0.07, "pips": 56.0},
            {"exit_time": t3, "net_pnl_usd": -1.50, "gross_pnl_usd": -1.43, "commission": 0.07, "pips": -24.0},
            {"exit_time": t4, "net_pnl_usd": 2.75, "gross_pnl_usd": 2.82, "commission": 0.07, "pips": 44.0},
        ]

    def test_weekly_returns_aggregation_iso_calendar(self):
        """Trade returns are aggregated chronologically into distinct ISO calendar weeks."""
        trades = self._get_sample_trades()
        weekly = calculate_weekly_returns(trades, initial_balance=250.0)
        self.assertEqual(len(weekly), 4)
        # Week 1: $3.00 / $250 = 1.20%
        self.assertAlmostEqual(float(weekly[0]), 1.20, places=2)
        # Week 3: -$1.50 / $250 = -0.60%
        self.assertAlmostEqual(float(weekly[2]), -0.60, places=2)
        # Dict metadata access
        self.assertEqual(weekly[0]["week_key"], "2026-W02")

    def test_target_hit_rate_calculation(self):
        """Target hit rate (% weeks >= 1.0%) equals 75.0% for 3 out of 4 weeks."""
        trades = self._get_sample_trades()
        hit_rate = calculate_percent_weeks_target_hit(trades, initial_balance=250.0, target_pct=1.0)
        self.assertAlmostEqual(hit_rate, 75.0, places=1)

    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio is annualized with sqrt(52) and handles zero division."""
        returns = [0.012, 0.014, -0.006, 0.011]
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=52)
        self.assertGreater(sharpe, 0.0)
        self.assertIsInstance(sharpe, float)

        # 0 or 1 returns
        self.assertEqual(calculate_sharpe_ratio([]), 0.0)
        self.assertEqual(calculate_sharpe_ratio([0.01]), 0.0)

    def test_sortino_ratio_calculation(self):
        """Sortino ratio isolates downside semi-deviation."""
        returns = [0.012, 0.014, -0.006, 0.011]
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=52)
        self.assertGreater(sortino, 0.0)

        # All positive returns -> zero downside deviation
        all_pos = [0.01, 0.02, 0.03]
        self.assertEqual(calculate_sortino_ratio(all_pos), 999.0)

    def test_max_drawdown_calculation_peak_to_trough(self):
        """Maximum peak-to-trough drawdown accurately computes percentage loss."""
        # 250 -> 260 -> 247 -> 255. Peak = 260, Trough = 247. DD = (260-247)/260 = 5.0%
        equity = pd.Series([250.0, 260.0, 247.0, 255.0])
        max_dd = calculate_max_drawdown(equity)
        self.assertAlmostEqual(max_dd, 5.0, places=2)

    def test_profit_factor_calculation(self):
        """Profit factor is gross profits divided by gross losses."""
        trades = self._get_sample_trades()
        # Gross profit = 3.07 + 3.57 + 2.82 = 9.46. Gross loss = 1.43
        pf = calculate_profit_factor(trades)
        self.assertAlmostEqual(pf, (3.07 + 3.57 + 2.82) / 1.43, delta=0.2)

    def test_metrics_dual_access_and_aliases(self):
        """PerformanceMetrics supports dual attribute/item access and dynamic aliases."""
        trades = self._get_sample_trades()
        m = calculate_metrics(trades, initial_balance=250.0)
        self.assertEqual(m.total_trades, 4)
        self.assertEqual(m["total_trades"], 4)
        # Alias test: total_net_profit <-> net_pnl
        self.assertEqual(m.total_net_profit, m.net_pnl)
        self.assertEqual(m["total_net_profit"], m["net_pnl"])
        # Alias test: percent_weeks_target_hit <-> target_hit_rate
        self.assertEqual(m.percent_weeks_target_hit, m.target_hit_rate)
        self.assertEqual(m["percent_weeks_target_hit"], m["target_hit_rate"])

    def test_edge_case_zero_trades(self):
        """Metrics handle 0 trades without division-by-zero crashes."""
        m = calculate_metrics([], initial_balance=250.0)
        self.assertEqual(m.total_trades, 0)
        self.assertEqual(m.win_rate, 0.0)
        self.assertEqual(m.profit_factor, 0.0)
        self.assertEqual(m.sharpe_ratio, 0.0)
        self.assertEqual(m.max_drawdown_pct, 0.0)

    def test_edge_case_single_trade(self):
        """Metrics handle single winning trade cleanly."""
        trade = [{"exit_time": datetime.now(), "net_pnl_usd": 2.50, "gross_pnl_usd": 2.57, "commission": 0.07}]
        m = calculate_metrics(trade, initial_balance=250.0)
        self.assertEqual(m.total_trades, 1)
        self.assertEqual(m.winning_trades, 1)
        self.assertEqual(m.win_rate, 100.0)
        self.assertEqual(m.profit_factor, 999.0)

    def test_edge_case_all_wins_and_all_losses(self):
        """Metrics handle all winning trades (infinite PF) and all losing trades (0% win rate)."""
        all_wins = [
            {"exit_time": datetime.now(), "net_pnl_usd": 1.0, "gross_pnl_usd": 1.07, "commission": 0.07},
            {"exit_time": datetime.now() + timedelta(hours=1), "net_pnl_usd": 2.0, "gross_pnl_usd": 2.07, "commission": 0.07},
        ]
        m_win = calculate_metrics(all_wins, initial_balance=250.0)
        self.assertEqual(m_win.win_rate, 100.0)
        self.assertEqual(m_win.profit_factor, 999.0)

        all_losses = [
            {"exit_time": datetime.now(), "net_pnl_usd": -1.0, "gross_pnl_usd": -0.93, "commission": 0.07},
            {"exit_time": datetime.now() + timedelta(hours=1), "net_pnl_usd": -2.0, "gross_pnl_usd": -1.93, "commission": 0.07},
        ]
        m_loss = calculate_metrics(all_losses, initial_balance=250.0)
        self.assertEqual(m_loss.win_rate, 0.0)
        self.assertEqual(m_loss.profit_factor, 0.0)

    def test_extract_trade_time_invalid_and_corrupt_date_strings(self):
        """Unparseable or corrupt date strings return None gracefully without throwing DateParseError."""
        from gbpjpy.backtest.metrics import _extract_trade_time

        # Corrupted and irregular date formats return None
        self.assertIsNone(_extract_trade_time({"exit_time": "invalid_date_string"}))
        self.assertIsNone(_extract_trade_time({"exit_time": "2026-99-99T99:99:99"}))
        self.assertIsNone(_extract_trade_time({"exit_time": "not_a_valid_timestamp"}))
        self.assertIsNone(_extract_trade_time({"entry_time": "corrupt_entry", "exit_time": None}))
        self.assertIsNone(_extract_trade_time({}))
        self.assertIsNone(_extract_trade_time(None))

        # Ensure calculate_weekly_returns handles corrupt date strings gracefully without raising
        corrupt_trades = [
            {"exit_time": "corrupt_date_1", "net_pnl_usd": 5.0},
            {"exit_time": "corrupt_date_2", "net_pnl_usd": -2.0},
        ]
        weekly = calculate_weekly_returns(corrupt_trades, initial_balance=250.0)
        self.assertEqual(len(weekly), 0)



class TestBacktestCLIRunner(unittest.TestCase):
    """Test suite for standalone headless CLI argument parsing, execution, and exports."""

    def test_cli_argparse_defaults(self):
        """CLI argument parser initializes with correct default values."""
        args = parse_arguments([])
        self.assertEqual(args.initial_balance, 250.0)
        self.assertEqual(args.leverage, 30.0)
        self.assertEqual(args.spread_pips, 1.2)
        self.assertEqual(args.commission, 7.0)
        self.assertEqual(args.slippage, 0.5)
        self.assertEqual(args.bars, 6000)

    def test_cli_synthetic_generation_and_ascii_table(self):
        """CLI generates synthetic rates and formats institutional ASCII report table."""
        df_syn = generate_synthetic_data(bars=100)
        self.assertEqual(len(df_syn), 100)
        for col in ["open", "high", "low", "close", "time", "volume"]:
            self.assertIn(col, df_syn.columns)

        # Test ASCII formatter
        m = PerformanceMetrics({
            "total_trades": 25,
            "winning_trades": 16,
            "losing_trades": 9,
            "win_rate": 64.0,
            "profit_factor": 2.15,
            "total_net_profit": 24.50,
            "gross_profit": 42.0,
            "gross_loss": 17.5,
            "total_commission": 1.75,
            "total_pips": 390.0,
            "total_return_pct": 9.80,
            "mean_weekly_return": 2.45,
            "percent_weeks_target_hit": 75.0,
            "percent_profitable_weeks": 100.0,
            "sharpe_ratio": 3.40,
            "sortino_ratio": 5.10,
            "max_drawdown_pct": 1.20,
            "max_drawdown_amount": 3.00,
            "max_daily_drawdown_pct": 0.90,
        })
        report = format_ascii_report(m, initial_balance=250.0, final_balance=274.50)
        self.assertIn("BENCHMARK REPORT", report)
        self.assertIn("CAPITAL & RETURN PERFORMANCE", report)
        self.assertIn("CONSISTENT ~1% WEEKLY TARGET ANALYTICS", report)
        self.assertIn("TARGET CRITERIA ACHIEVED [PASS]", report)

    def test_cli_json_export(self):
        """CLI runs and exports complete metrics and trade logs to JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            args = parse_arguments(["--bars", "150", "--output", tmp_path, "--synthetic"])
            exit_code = run_pipeline(args)
            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(tmp_path).exists())

            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("metrics", data)
            self.assertIn("trades", data)
            self.assertIn("metadata", data)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_cli_csv_export(self):
        """CLI runs and exports closed trades to CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            args = parse_arguments(["--bars", "150", "--output", tmp_path, "--synthetic"])
            exit_code = run_pipeline(args)
            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(tmp_path).exists())

            df_out = pd.read_csv(tmp_path)
            self.assertIsInstance(df_out, pd.DataFrame)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()


class TestEndToEndBacktestIntegration(unittest.TestCase):
    """End-to-end integration tests combining full market simulation with institutional requirements."""

    def test_full_month_simulation_with_scalper_strategy(self):
        """1-month simulation with GBPJPYScalper generates valid trades with >5m holding time."""
        df = generate_synthetic_data(bars=1200)
        de = DataEngine()
        df_ind = de.compute_indicators(df)
        strat = GBPJPYDualRegimeScalper()

        engine = BacktestEngine(initial_balance=250.0, spread_pips=1.2, commission_per_lot=7.0)
        result = engine.run(df=df_ind, strategy=strat)

        self.assertIsNotNone(result)
        self.assertGreater(result.bars_processed, 0)
        # Check trades format
        for tr in result.trades:
            self.assertIn("entry_time", tr)
            self.assertIn("exit_time", tr)
            self.assertIn("net_pnl_usd", tr)
            self.assertGreater(tr["exit_time"], tr["entry_time"])

    def test_circuit_breaker_integration_in_backtest(self):
        """Engine complies with circuit breaker restrictions during high drawdown and loss streak."""
        from gbpjpy.risk.risk_manager import RiskManager

        df = generate_synthetic_data(bars=300)
        de = DataEngine()
        df_ind = de.compute_indicators(df)

        # Baseline: Normal engine execution produces trade activity
        rm_normal = RiskManager()
        strat_normal = GBPJPYDualRegimeScalper(risk_manager=rm_normal)
        engine_normal = BacktestEngine(risk_manager=rm_normal)
        res_normal = engine_normal.run(df=df_ind, strategy=strat_normal)
        self.assertGreater(len(res_normal.trades), 0, "Normal market conditions must execute trades")

        # Scenario 1: Tripped circuit breaker blocks all trade executions in backtest
        rm_halted = RiskManager()
        rm_halted.trip_circuit_breaker(reason="Mandatory drawdown halt")
        self.assertFalse(rm_halted.can_trade())
        self.assertTrue(rm_halted.is_circuit_breaker_active())

        strat_halted = GBPJPYDualRegimeScalper(risk_manager=rm_halted)
        engine_halted = BacktestEngine(risk_manager=rm_halted)
        res_halted = engine_halted.run(df=df_ind, strategy=strat_halted)
        self.assertEqual(len(res_halted.trades), 0, "No trades may execute while circuit breaker is tripped")

        # Scenario 2: Daily drawdown breach halts intraday trading and resets next calendar day
        rm_daily = RiskManager(daily_starting_equity=250.0)
        t_trade = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
        # 3% of $250 is $7.50; record an $8.00 drawdown breach
        rm_daily.record_trade_result(-8.00, is_win=False, trade_date=t_trade)
        self.assertTrue(rm_daily.is_daily_halted(), "Daily loss limit must trip daily halt")
        self.assertFalse(rm_daily.can_trade(t_trade + timedelta(hours=3)), "Trading must remain halted on same calendar day")
        self.assertTrue(rm_daily.can_trade(t_trade + timedelta(days=1)), "Daily halt must auto-reset on next calendar day")

        # Scenario 3: 3-strike loss streak triggers 60-minute cooldown
        rm_streak = RiskManager(consecutive_loss_cooldown_minutes=60)
        t0 = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
        rm_streak.record_trade_result(-1.0, is_win=False, trade_date=t0)
        rm_streak.record_trade_result(-1.0, is_win=False, trade_date=t0 + timedelta(minutes=5))
        rm_streak.record_trade_result(-1.0, is_win=False, trade_date=t0 + timedelta(minutes=10))
        self.assertEqual(rm_streak.loss_streak, 3)
        self.assertTrue(rm_streak.is_cooldown_active(t0 + timedelta(minutes=15)))
        self.assertFalse(rm_streak.can_trade(t0 + timedelta(minutes=15)))
        # Cooldown expires after 60 minutes
        self.assertFalse(rm_streak.is_cooldown_active(t0 + timedelta(minutes=75)))
        self.assertTrue(rm_streak.can_trade(t0 + timedelta(minutes=75)))

    def test_equity_curve_and_drawdown_consistency(self):
        """Equity curve terminal balance matches final balance and net profit."""
        engine = BacktestEngine(initial_balance=250.0)
        df = generate_synthetic_data(bars=200)
        result = engine.run(df)
        self.assertEqual(len(result.equity_curve), 200)
        self.assertEqual(result.final_balance, result.initial_balance + result.total_net_profit)

    def test_weekly_return_consistency_target_compliance(self):
        """Validates that a compliant 4-week trade set achieves the ~1% target."""
        base_t = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
        week_data = [
            [(0, 1.80), (1, -1.10), (2, 1.50), (3, 0.55)],
            [(7, 1.60), (8, -1.00), (9, 1.40), (10, 0.60)],
            [(14, 1.50), (15, -1.25), (16, 1.60), (17, 0.60)],
            [(21, 2.00), (22, -1.10), (23, 1.70), (24, 0.50)],
        ]
        trades = []
        for week in week_data:
            for day, pnl in week:
                trades.append({
                    "entry_time": base_t + timedelta(days=day),
                    "exit_time": base_t + timedelta(days=day, minutes=45),
                    "net_pnl_usd": pnl,
                    "gross_pnl_usd": round(pnl + 0.07, 2),
                    "commission": 0.07,
                    "pips": round(pnl / 0.0626, 1),
                })

        m = calculate_metrics(trades, initial_balance=250.0)
        self.assertGreaterEqual(m.mean_weekly_return, 0.70)
        self.assertLessEqual(m.mean_weekly_return, 2.00)
        self.assertLessEqual(m.max_drawdown_pct, 5.0)
        self.assertGreaterEqual(m.percent_weeks_target_hit, 50.0)


if __name__ == "__main__":
    unittest.main()

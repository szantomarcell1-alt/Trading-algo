"""
Comprehensive Test Suite for Titan Quant Scalper
Verifies all modules, mathematical models, risk calculations, and mock execution.
"""

import unittest
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from config.settings import load_config, AppConfig, RiskManagementConfig
from utils.market_hours import MarketHoursGuard
from core.regime_detector import RegimeDetector, MarketRegime
from core.data_engine import DataEngine
from core.mt5_client import SymbolSpecs, AccountSnapshot
from core.mock_client import MockMT5Client
from strategies.titan_scalper import TitanScalperStrategy
from strategies.signals import SignalType
from risk.risk_manager import RiskManager
from risk.execution_manager import ExecutionManager
from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics, BacktestMetrics
from backtest.run_backtest import generate_synthetic_forex_data


class TestTitanQuantScalper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_01_config_loading(self):
        self.assertIsNotNone(self.config)
        self.assertEqual(self.config.general.bot_name, "Titan Quant Scalper")
        self.assertGreater(len(self.config.instruments), 0)
        self.assertGreater(self.config.risk_management.risk_per_trade_percent, 0)
        eurusd = self.config.get_symbol_config("EURUSD")
        self.assertIsNotNone(eurusd)
        self.assertTrue(eurusd.enabled)

    def test_02_market_hours_guard(self):
        guard = MarketHoursGuard(allowed_start_utc_hour=7, allowed_end_utc_hour=20)
        # 10:00 UTC Tuesday -> Allowed
        dt_tuesday_10 = datetime(2026, 11, 3, 10, 0, 0, tzinfo=timezone.utc)
        allowed, msg = guard.is_trading_allowed(dt_tuesday_10)
        self.assertTrue(allowed)

        # 22:00 UTC -> Rollover window (Blocked)
        dt_rollover = datetime(2026, 11, 3, 22, 15, 0, tzinfo=timezone.utc)
        allowed, msg = guard.is_trading_allowed(dt_rollover)
        self.assertFalse(allowed)
        self.assertIn("Rollover", msg)

        # Saturday -> Blocked
        dt_saturday = datetime(2026, 11, 7, 12, 0, 0, tzinfo=timezone.utc)
        allowed, msg = guard.is_trading_allowed(dt_saturday)
        self.assertFalse(allowed)
        self.assertIn("Szombat", msg)

    def test_03_regime_detector_hurst(self):
        detector = RegimeDetector()

        # Pure trend series (persistent)
        t = np.linspace(1, 10, 200)
        trending_series = 1.0 + 0.05 * t + np.random.normal(0, 0.002, 200)
        hurst_trend = detector.calculate_hurst_exponent(trending_series)
        self.assertGreater(hurst_trend, 0.50)

        # Mean-reverting series (anti-persistent / Ornstein-Uhlenbeck)
        mr_series = [1.0]
        for _ in range(200):
            mr_series.append(mr_series[-1] - 0.5 * (mr_series[-1] - 1.0) + np.random.normal(0, 0.01))
        hurst_mr = detector.calculate_hurst_exponent(np.array(mr_series))
        self.assertLess(hurst_mr, 0.55)

    def test_04_data_engine_indicators(self):
        df = generate_synthetic_forex_data(n_bars=150)
        proc = DataEngine.compute_indicators(df)

        self.assertIn("ema_fast", proc.columns)
        self.assertIn("atr", proc.columns)
        self.assertIn("bb_upper", proc.columns)
        self.assertIn("kc_upper", proc.columns)
        self.assertIn("squeeze_on", proc.columns)
        self.assertIn("squeeze_momentum", proc.columns)
        self.assertIn("rsi", proc.columns)
        self.assertIn("vwap", proc.columns)

        # Ensure no all-NaNs in calculated features
        self.assertFalse(proc["atr"].iloc[-1] != proc["atr"].iloc[-1])  # not NaN
        self.assertFalse(proc["rsi"].iloc[-1] != proc["rsi"].iloc[-1])

    def test_05_risk_manager_lot_sizing(self):
        rm = RiskManager(self.config)
        account = AccountSnapshot(
            login=10001,
            balance=1000.0,
            equity=1000.0,
            margin=0.0,
            margin_free=1000.0,
            margin_level=9999.0,
            profit=0.0,
            currency="USD",
        )
        specs = SymbolSpecs(
            name="EURUSD",
            digits=5,
            point=0.00001,
            tick_value=1.0,
            tick_size=0.00001,
            contract_size=100000.0,
            min_volume=0.01,
            max_volume=100.0,
            volume_step=0.01,
            pip_size=0.0001,
        )

        # Risk 1.0% of $1,000 = $10 risk
        # Stop loss distance = 20 pips = 200 points
        # 1 lot loses $200 on 200 points.
        # $10 risk / ($200 SL + $7 commission) = 0.0483 -> floored to 0.04 lots
        lots = rm.calculate_lot_size(account, specs, risk_distance_points=200.0)
        self.assertAlmostEqual(lots, 0.04, delta=0.01)

    def test_06_circuit_breakers(self):
        rm = RiskManager(self.config)
        dt_today = datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)

        acc_good = AccountSnapshot(
            login=10001, balance=1000.0, equity=1000.0, margin=0.0,
            margin_free=1000.0, margin_level=9999.0, profit=0.0, currency="USD"
        )
        can_trade, _ = rm.can_open_new_trade(acc_good, 0, 0, now=dt_today)
        self.assertTrue(can_trade)

        # Simulate 3 consecutive losses
        rm.register_trade_result(-10.0)
        rm.register_trade_result(-10.0)
        rm.register_trade_result(-10.0)

        # Now circuit breaker should trigger cooldown
        can_trade_after_losses, reason = rm.can_open_new_trade(acc_good, 0, 0, now=dt_today)
        self.assertFalse(can_trade_after_losses)
        self.assertIn("pihenő", reason.lower())

    def test_07_mock_client_execution(self):
        client = MockMT5Client(initial_balance=1000.0)
        client.connect()

        # Place simulated buy order relative to current price
        quote = client.get_quote("EURUSD")
        curr_p = quote[0] if quote else 1.0850
        ticket = client.send_market_order("EURUSD", "BUY", volume=0.05, sl=curr_p - 0.0050, tp=curr_p + 0.0050)
        self.assertIsNotNone(ticket)
        self.assertGreater(ticket, 0)

        positions = client.get_open_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].ticket, ticket)

        # Modify SL/TP
        mod_ok = client.modify_position(ticket, "EURUSD", sl=curr_p - 0.0030, tp=curr_p + 0.0060)
        self.assertTrue(mod_ok)

        # Close position
        close_ok = client.close_position(ticket, "EURUSD", volume=0.05, is_buy=True)
        self.assertTrue(close_ok)
        self.assertEqual(len(client.get_open_positions()), 0)

    def test_08_end_to_end_backtest_pipeline(self):
        engine = BacktestEngine(config=self.config, initial_balance=1000.0)
        df_m1 = generate_synthetic_forex_data(n_bars=600, start_price=1.0850)
        metrics = engine.run(df_m1=df_m1, symbol="EURUSD")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.initial_balance, 1000.0)

    def test_09_fixed_250_baseline_unification(self):
        """Verifies $250.0 initial balance across config/settings/engine, multipliers, and enabled symbols."""
        self.assertEqual(self.config.risk_management.initial_balance, 250.0)
        self.assertEqual(RiskManagementConfig().initial_balance, 250.0)
        self.assertEqual(self.config.risk_management.max_risk_multiplier_for_min_lot, 1.80)
        self.assertEqual(RiskManagementConfig().max_risk_multiplier_for_min_lot, 1.80)
        self.assertEqual(self.config.risk_management.max_consecutive_losses, 3)
        self.assertEqual(RiskManagementConfig().max_consecutive_losses, 3)

        mock = MockMT5Client(initial_balance=self.config.risk_management.initial_balance)
        self.assertEqual(mock.balance, 250.0)

        engine = BacktestEngine(config=self.config, initial_balance=self.config.risk_management.initial_balance)
        self.assertEqual(engine.initial_balance, 250.0)

        enabled_symbols = [inst.name for inst in self.config.get_enabled_symbols()]
        self.assertIn("EURUSD", enabled_symbols)
        self.assertIn("XAUUSD", enabled_symbols)
        self.assertNotIn("GBPUSD", enabled_symbols)
        gbpusd = self.config.get_symbol_config("GBPUSD")
        self.assertFalse(gbpusd.enabled)

    def test_10_risk_per_trade_1pct_eurusd(self):
        """Verifies 1.0% ($2.50) trade risk calculation on $250.00 account."""
        rm = RiskManager(self.config)
        acc_250 = AccountSnapshot(
            login=10001,
            balance=250.0,
            equity=250.0,
            margin=0.0,
            margin_free=250.0,
            margin_level=9999.0,
            profit=0.0,
            currency="USD",
        )
        specs_eur = SymbolSpecs(
            name="EURUSD",
            digits=5,
            point=0.00001,
            tick_value=1.0,
            tick_size=0.00001,
            contract_size=100000.0,
            min_volume=0.01,
            max_volume=100.0,
            volume_step=0.01,
            pip_size=0.0001,
        )
        # 1% of $250 is $2.50
        # Stop loss = 15 pips = 150 points
        # loss_per_lot = 150 * 1.0 + $7 comm = $157.00
        # raw_lot = 2.50 / 157.0 = 0.0159 -> bounded to 0.01 lot
        lot = rm.calculate_lot_size(acc_250, specs_eur, risk_distance_points=150.0)
        self.assertEqual(lot, 0.01)
        # Monetary risk of 0.01 lot is $1.57 <= $2.50 (1% of $250)
        monetary_risk = (150 * specs_eur.tick_value * 0.01) + (7.0 * 0.01)
        self.assertLessEqual(monetary_risk, 2.50)

    def test_11_gold_min_lot_180_multiplier_acceptance_and_rejection(self):
        """Verifies 1.80x min lot multiplier accommodates 0.01 lot Gold CFD without rejection and rejects >1.80x."""
        rm = RiskManager(self.config)
        specs_gold = SymbolSpecs(
            name="XAUUSD",
            digits=2,
            point=0.01,
            tick_value=1.0,
            tick_size=0.01,
            contract_size=100.0,
            min_volume=0.01,
            max_volume=100.0,
            volume_step=0.01,
            pip_size=0.10,
        )
        acc_250 = AccountSnapshot(
            login=10001,
            balance=250.0,
            equity=250.0,
            margin=0.0,
            margin_free=250.0,
            margin_level=9999.0,
            profit=0.0,
            currency="USD",
        )
        # Gold SL at 43.5 pips = 435 points ($4.35 price move)
        # loss_per_lot = 435 * 1.0 + 7.0 = $442.00
        # min_lot_loss = 0.01 * 442.00 = $4.42
        # Max tolerable risk at 1.80x = 250 * 1% * 1.80 = $4.50
        # $4.42 <= $4.50 -> Should be APPROVED (0.01 lot)
        lot_ok = rm.calculate_lot_size(acc_250, specs_gold, risk_distance_points=435.0)
        self.assertEqual(lot_ok, 0.01)

        # Temporary drawdown: equity drops to $245.00
        # base_equity = max(245.0, 250.0) = 250.0 -> max_tolerable_risk remains $4.50
        acc_drawdown = AccountSnapshot(
            login=10001,
            balance=245.0,
            equity=245.0,
            margin=0.0,
            margin_free=245.0,
            margin_level=9999.0,
            profit=0.0,
            currency="USD",
        )
        lot_drawdown = rm.calculate_lot_size(acc_drawdown, specs_gold, risk_distance_points=435.0)
        self.assertEqual(lot_drawdown, 0.01)

        # Excessive SL rejection: 50 pips = 500 points ($5.00 price move)
        # loss_per_lot = 500 * 1.0 + 7.0 = $507.00
        # min_lot_loss = 0.01 * 507.00 = $5.07 > $4.50 -> REJECTED (0.0)
        lot_rejected = rm.calculate_lot_size(acc_250, specs_gold, risk_distance_points=500.0)
        self.assertEqual(lot_rejected, 0.0)

    def test_12_daily_3pct_drawdown_ceiling_start_and_peak(self):
        """Verifies 3% daily drawdown ceiling trips against start equity and peak equity, and resets next day."""
        rm = RiskManager(self.config)
        now_day1 = datetime(2026, 11, 4, 8, 0, 0, tzinfo=timezone.utc)

        # Baseline: $250.00
        acc_start = AccountSnapshot(
            login=10001, balance=250.0, equity=250.0, margin=0.0,
            margin_free=250.0, margin_level=9999.0, profit=0.0, currency="USD"
        )
        can_trade, msg = rm.can_open_new_trade(acc_start, 0, 0, now=now_day1)
        self.assertTrue(can_trade)
        self.assertEqual(rm.daily_start_equity, 250.0)
        self.assertEqual(rm.daily_peak_equity, 250.0)

        # Scenario A: Equity drops to $242.40 (Loss = $7.60 = 3.04% >= 3.0%)
        acc_loss = AccountSnapshot(
            login=10001, balance=242.40, equity=242.40, margin=0.0,
            margin_free=242.40, margin_level=9999.0, profit=0.0, currency="USD"
        )
        now_day1_later = datetime(2026, 11, 4, 11, 0, 0, tzinfo=timezone.utc)
        can_trade_dd, reason = rm.can_open_new_trade(acc_loss, 0, 0, now=now_day1_later)
        self.assertFalse(can_trade_dd)
        self.assertTrue(rm.daily_circuit_breaker_tripped)
        self.assertIn("Drawdown elérve", reason)

        # Advance to Day 2: Circuit breaker should reset
        now_day2 = datetime(2026, 11, 5, 8, 0, 0, tzinfo=timezone.utc)
        can_trade_d2, _ = rm.can_open_new_trade(acc_loss, 0, 0, now=now_day2)
        self.assertTrue(can_trade_d2)
        self.assertFalse(rm.daily_circuit_breaker_tripped)
        self.assertEqual(rm.daily_start_equity, 242.40)
        self.assertEqual(rm.daily_peak_equity, 242.40)

        # Scenario B: Peak equity tracking
        # Day 2 equity grows to $260.00
        acc_profit = AccountSnapshot(
            login=10001, balance=260.0, equity=260.0, margin=0.0,
            margin_free=260.0, margin_level=9999.0, profit=0.0, currency="USD"
        )
        now_day2_peak = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)
        rm.can_open_new_trade(acc_profit, 0, 0, now=now_day2_peak)
        self.assertEqual(rm.daily_peak_equity, 260.0)

        # Equity pulls back to $252.00 (Drop from peak = $8.00 / 260.0 = 3.0769% >= 3.0%)
        acc_peak_dd = AccountSnapshot(
            login=10001, balance=252.0, equity=252.0, margin=0.0,
            margin_free=252.0, margin_level=9999.0, profit=0.0, currency="USD"
        )
        now_day2_pullback = datetime(2026, 11, 5, 14, 0, 0, tzinfo=timezone.utc)
        can_trade_peak_dd, reason_peak = rm.can_open_new_trade(acc_peak_dd, 0, 0, now=now_day2_pullback)
        self.assertFalse(can_trade_peak_dd)
        self.assertTrue(rm.daily_circuit_breaker_tripped)
        self.assertIn("Drawdown elérve", reason_peak)

    def test_13_three_strike_rule_60min_candle_cooldown(self):
        """Verifies 3 consecutive losses activate 60-min candle time cooldown, and clearing after 60 min."""
        rm = RiskManager(self.config)
        acc_good = AccountSnapshot(
            login=10001, balance=250.0, equity=250.0, margin=0.0,
            margin_free=250.0, margin_level=9999.0, profit=0.0, currency="USD"
        )
        t0 = datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 11, 4, 10, 5, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 4, 10, 10, 0, tzinfo=timezone.utc)

        # Trade 1 loss
        rm.register_trade_result(-2.0, now=t0)
        self.assertEqual(rm.consecutive_losses, 1)
        self.assertIsNone(rm.cooldown_until)

        # Trade 2 loss
        rm.register_trade_result(-2.0, now=t1)
        self.assertEqual(rm.consecutive_losses, 2)
        self.assertIsNone(rm.cooldown_until)

        # Trade 3 loss -> Triggers 3-strike rule
        rm.register_trade_result(-2.0, now=t2)
        self.assertEqual(rm.consecutive_losses, 3)
        self.assertIsNotNone(rm.cooldown_until)
        expected_cooldown_end = t2.timestamp() + 3600
        self.assertEqual(rm.cooldown_until, expected_cooldown_end)

        # 30 minutes after 3rd loss (still in cooldown)
        t_mid = datetime(2026, 11, 4, 10, 40, 0, tzinfo=timezone.utc)
        can_trade_mid, reason_mid = rm.can_open_new_trade(acc_good, 0, 0, now=t_mid)
        self.assertFalse(can_trade_mid)
        self.assertIn("kényszerpihenő", reason_mid.lower())

        # 61 minutes after 3rd loss (cooldown elapsed)
        t_after = datetime(2026, 11, 4, 11, 11, 0, tzinfo=timezone.utc)
        can_trade_after, _ = rm.can_open_new_trade(acc_good, 0, 0, now=t_after)
        self.assertTrue(can_trade_after)
        self.assertEqual(rm.consecutive_losses, 0)
        self.assertIsNone(rm.cooldown_until)

    def test_14_portfolio_concurrency_and_combined_risk(self):
        """Verifies max 2 open positions, max 1 per symbol, and combined portfolio risk safety."""
        rm = RiskManager(self.config)
        acc = AccountSnapshot(
            login=10001, balance=250.0, equity=250.0, margin=0.0,
            margin_free=250.0, margin_level=9999.0, profit=0.0, currency="USD"
        )
        now = datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)

        # 1st trade on EURUSD allowed
        can_eur, _ = rm.can_open_new_trade(acc, open_positions_count=0, symbol_positions_count=0, now=now)
        self.assertTrue(can_eur)

        # 2nd trade on same symbol EURUSD rejected (max 1 per symbol)
        can_eur2, reason_eur2 = rm.can_open_new_trade(acc, open_positions_count=1, symbol_positions_count=1, now=now)
        self.assertFalse(can_eur2)
        self.assertIn("nyitott pozíció", reason_eur2)

        # 2nd trade on different symbol (XAUUSD) allowed
        can_gold, _ = rm.can_open_new_trade(acc, open_positions_count=1, symbol_positions_count=0, now=now)
        self.assertTrue(can_gold)

        # 3rd trade on any symbol rejected (max 2 total)
        can_third, reason_third = rm.can_open_new_trade(acc, open_positions_count=2, symbol_positions_count=0, now=now)
        self.assertFalse(can_third)
        self.assertIn("maximális egyidejű pozíciószámot", reason_third)

        # Combined portfolio risk verification:
        # EURUSD max SL 18 pips = 180 points: 0.01 * (180 * 1.0 + 7.0) = $1.87
        # XAUUSD max SL 43.5 pips = 435 points: 0.01 * (435 * 1.0 + 7.0) = $4.42
        worst_case_combined_loss = 1.87 + 4.42  # $6.29
        max_daily_loss_limit = 250.0 * 0.03     # $7.50
        self.assertLess(worst_case_combined_loss, max_daily_loss_limit)
        self.assertLessEqual(worst_case_combined_loss / 250.0 * 100.0, 2.55)

    def test_15_daily_drawdown_metric_calculation(self):
        """Verifies BacktestMetrics accurately computes max_daily_drawdown_pct across multi-day backtests."""
        trades_multi_day = [
            # Day 1: Starting balance $250.00
            # Trade 1: -$3.00 -> balance $247.00
            # Trade 2: -$3.00 -> balance $244.00 (Day 1 DD = 2.40%)
            {
                "entry_time": datetime(2026, 11, 4, 8, 0, 0),
                "exit_time": datetime(2026, 11, 4, 9, 0, 0),
                "profit": -3.0,
            },
            {
                "entry_time": datetime(2026, 11, 4, 10, 0, 0),
                "exit_time": datetime(2026, 11, 4, 11, 0, 0),
                "profit": -3.0,
            },
            # Day 2: Starting balance $244.00
            # Trade 3: +$10.00 -> balance $254.00 (Day 2 peak = $254.00)
            # Trade 4: -$8.00 -> balance $246.00 (Drop from peak = $8.00 / $254.00 = 3.1496% ~ 3.15%)
            {
                "entry_time": datetime(2026, 11, 5, 8, 0, 0),
                "exit_time": datetime(2026, 11, 5, 9, 0, 0),
                "profit": 10.0,
            },
            {
                "entry_time": datetime(2026, 11, 5, 10, 0, 0),
                "exit_time": datetime(2026, 11, 5, 11, 0, 0),
                "profit": -8.0,
            },
        ]
        metrics = calculate_metrics(trades_multi_day, initial_balance=250.0)
        self.assertAlmostEqual(metrics.max_daily_drawdown_pct, 3.15, places=2)
        self.assertEqual(metrics.initial_balance, 250.0)
        self.assertEqual(metrics.final_balance, 246.0)

        # Empty trades test
        empty_metrics = calculate_metrics([], initial_balance=250.0)
        self.assertEqual(empty_metrics.max_daily_drawdown_pct, 0.0)

    def _build_test_breakout_data(self, is_bull: bool = True, is_gold: bool = False):
        """Helper generating synthetic dataframe that triggers pure squeeze breakout logic."""
        dates = pd.date_range("2026-11-04 08:00", periods=120, freq="5min" if is_gold else "1min", tz="UTC")
        if is_gold:
            close = np.linspace(2650.0, 2660.0, 120) if is_bull else np.linspace(2660.0, 2650.0, 120)
            df = pd.DataFrame({
                "open": close - (0.5 if is_bull else -0.5),
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": [100.0] * 120,
            }, index=dates)
        else:
            close = np.linspace(1.0800, 1.0850, 120) if is_bull else np.linspace(1.0850, 1.0800, 120)
            df = pd.DataFrame({
                "open": close - (0.0001 if is_bull else -0.0001),
                "high": close + 0.0002,
                "low": close - 0.0002,
                "close": close,
                "volume": [100.0] * 120,
            }, index=dates)

        df_proc = DataEngine.compute_indicators(df)
        df_proc.loc[df_proc.index[-6:-1], "squeeze_on"] = True

        idx_prev = df_proc.index[-3]
        idx_last = df_proc.index[-2]

        if is_bull and not is_gold:
            df_proc.loc[idx_prev, "close"] = 1.0850
            df_proc.loc[idx_prev, "high"] = 1.0852
            df_proc.loc[idx_prev, "low"] = 1.0845
            df_proc.loc[idx_prev, "ema_fast"] = 1.0848
            df_proc.loc[idx_prev, "ema_med"] = 1.0846
            df_proc.loc[idx_prev, "ema_slow"] = 1.0840
            df_proc.loc[idx_prev, "squeeze_momentum"] = 0.0001

            df_proc.loc[idx_last, "open"] = 1.0850
            df_proc.loc[idx_last, "low"] = 1.0847
            df_proc.loc[idx_last, "high"] = 1.0858
            df_proc.loc[idx_last, "close"] = 1.0856
            df_proc.loc[idx_last, "ema_fast"] = 1.0851
            df_proc.loc[idx_last, "ema_med"] = 1.0848
            df_proc.loc[idx_last, "ema_slow"] = 1.0842
            df_proc.loc[idx_last, "vwap"] = 1.0850
            df_proc.loc[idx_last, "rsi"] = 60.0
            df_proc.loc[idx_last, "atr"] = 0.0005
            df_proc.loc[idx_last, "squeeze_momentum"] = 0.0003
            df_proc.loc[idx_last, "volume"] = 130.0

        elif not is_bull and not is_gold:
            df_proc.loc[idx_prev, "close"] = 1.0805
            df_proc.loc[idx_prev, "high"] = 1.0810
            df_proc.loc[idx_prev, "low"] = 1.0803
            df_proc.loc[idx_prev, "ema_fast"] = 1.0807
            df_proc.loc[idx_prev, "ema_med"] = 1.0810
            df_proc.loc[idx_prev, "ema_slow"] = 1.0815
            df_proc.loc[idx_prev, "squeeze_momentum"] = -0.0001

            df_proc.loc[idx_last, "open"] = 1.0805
            df_proc.loc[idx_last, "low"] = 1.0795
            df_proc.loc[idx_last, "high"] = 1.0806
            df_proc.loc[idx_last, "close"] = 1.0797
            df_proc.loc[idx_last, "ema_fast"] = 1.0803
            df_proc.loc[idx_last, "ema_med"] = 1.0808
            df_proc.loc[idx_last, "ema_slow"] = 1.0814
            df_proc.loc[idx_last, "vwap"] = 1.0805
            df_proc.loc[idx_last, "rsi"] = 40.0
            df_proc.loc[idx_last, "atr"] = 0.0005
            df_proc.loc[idx_last, "squeeze_momentum"] = -0.0003
            df_proc.loc[idx_last, "volume"] = 130.0

        elif is_bull and is_gold:
            df_proc.loc[idx_prev, "close"] = 2659.0
            df_proc.loc[idx_prev, "high"] = 2660.0
            df_proc.loc[idx_prev, "low"] = 2657.0
            df_proc.loc[idx_prev, "ema_fast"] = 2658.0
            df_proc.loc[idx_prev, "ema_med"] = 2656.0
            df_proc.loc[idx_prev, "ema_slow"] = 2652.0
            df_proc.loc[idx_prev, "squeeze_momentum"] = 0.5

            df_proc.loc[idx_last, "open"] = 2659.0
            df_proc.loc[idx_last, "low"] = 2657.5
            df_proc.loc[idx_last, "high"] = 2662.0
            df_proc.loc[idx_last, "close"] = 2661.5
            df_proc.loc[idx_last, "ema_fast"] = 2659.0
            df_proc.loc[idx_last, "ema_med"] = 2657.0
            df_proc.loc[idx_last, "ema_slow"] = 2653.0
            df_proc.loc[idx_last, "vwap"] = 2658.0
            df_proc.loc[idx_last, "rsi"] = 62.0
            df_proc.loc[idx_last, "atr"] = 2.0
            df_proc.loc[idx_last, "squeeze_momentum"] = 1.0
            df_proc.loc[idx_last, "volume"] = 130.0

        return df_proc

    def test_16_m15_h1_ema200_macro_trend_filter(self):
        """Feature 7: Macro Trend Filter requires close > ema_200 on BOTH M15 and H1 for BUY, and close < ema_200 on BOTH for SELL."""
        strat = TitanScalperStrategy(self.config)
        specs_eur = SymbolSpecs(
            name="EURUSD", digits=5, point=0.00001, tick_value=1.0, tick_size=0.00001,
            contract_size=100000.0, min_volume=0.01, max_volume=100.0, volume_step=0.01, pip_size=0.0001
        )

        df_exec = self._build_test_breakout_data(is_bull=True)
        df_m15 = df_exec.copy()
        df_h1 = df_exec.copy()

        # 1. Price above EMA 200 on BOTH M15 and H1 -> BUY accepted
        df_m15["ema_200"] = 1.0830  # close 1.0856 > 1.0830
        df_h1["ema_200"] = 1.0820   # close 1.0856 > 1.0820
        sig_buy = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec, "structure": df_exec, "m15": df_m15, "h1": df_h1},
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNotNone(sig_buy)
        self.assertEqual(sig_buy.signal_type, SignalType.BUY)
        self.assertEqual(sig_buy.strategy_mode, "SQUEEZE_BREAKOUT_BULL")

        # 2. M15 close < ema_200, H1 close > ema_200 -> BUY rejected
        df_m15["ema_200"] = 1.0870  # close 1.0856 < 1.0870
        df_h1["ema_200"] = 1.0820
        sig_m15_fail = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec, "structure": df_exec, "m15": df_m15, "h1": df_h1},
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNone(sig_m15_fail)

        # 3. M15 close > ema_200, H1 close < ema_200 -> BUY rejected
        df_m15["ema_200"] = 1.0830
        df_h1["ema_200"] = 1.0870
        sig_h1_fail = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec, "structure": df_exec, "m15": df_m15, "h1": df_h1},
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNone(sig_h1_fail)

        # 4. Hierarchical key resolution: "structure" (M15) and "macro" (H1)
        df_m15["ema_200"] = 1.0830
        df_h1["ema_200"] = 1.0820
        sig_macro_key = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec, "structure": df_m15, "macro": df_h1},
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNotNone(sig_macro_key)
        self.assertEqual(sig_macro_key.signal_type, SignalType.BUY)

        # 5. Bearish Macro Trend Filter
        df_exec_bear = self._build_test_breakout_data(is_bull=False)
        df_m15_bear = df_exec_bear.copy()
        df_h1_bear = df_exec_bear.copy()

        # Both below EMA 200 -> SELL accepted
        df_m15_bear["ema_200"] = 1.0820  # close 1.0797 < 1.0820
        df_h1_bear["ema_200"] = 1.0830   # close 1.0797 < 1.0830
        sig_sell = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec_bear, "structure": df_exec_bear, "m15": df_m15_bear, "h1": df_h1_bear},
            symbol_specs=specs_eur,
            current_bid=1.0797,
            current_ask=1.0798,
            current_spread_pips=1.0,
        )
        self.assertIsNotNone(sig_sell)
        self.assertEqual(sig_sell.signal_type, SignalType.SELL)
        self.assertEqual(sig_sell.strategy_mode, "SQUEEZE_BREAKOUT_BEAR")

        # Bearish rejected when price is above EMA 200 on M15
        df_m15_bear["ema_200"] = 1.0780  # close 1.0797 > 1.0780
        sig_sell_rej = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec_bear, "structure": df_exec_bear, "m15": df_m15_bear, "h1": df_h1_bear},
            symbol_specs=specs_eur,
            current_bid=1.0797,
            current_ask=1.0798,
            current_spread_pips=1.0,
        )
        self.assertIsNone(sig_sell_rej)

        # 6. Graceful fallback when higher timeframes are absent in isolated unit tests
        sig_isolated = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec},
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNotNone(sig_isolated)

    def test_17_breakout_volume_surge_confirmation(self):
        """Feature 8: Volume Surge requires Volume > 1.15x SMA20(Volume) over prior 20 closed candles [-22:-2]."""
        strat = TitanScalperStrategy(self.config)
        specs_eur = SymbolSpecs(
            name="EURUSD", digits=5, point=0.00001, tick_value=1.0, tick_size=0.00001,
            contract_size=100000.0, min_volume=0.01, max_volume=100.0, volume_step=0.01, pip_size=0.0001
        )

        df_exec = self._build_test_breakout_data(is_bull=True)
        # Prior 20 completed candles [-22:-2] have volume 100.0 -> vol_20 = 100.0
        df_m15 = df_exec.copy()
        df_m15["ema_200"] = 1.0830
        df_h1 = df_exec.copy()
        df_h1["ema_200"] = 1.0820

        # Scenario A: Breakout bar volume at 1.10x SMA20 (110.0 <= 115.0) -> Rejected
        df_exec.loc[df_exec.index[-2], "volume"] = 110.0
        sig_110 = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec, "structure": df_exec, "m15": df_m15, "h1": df_h1},
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNone(sig_110)

        # Scenario B: Breakout bar volume at 1.15x SMA20 (115.0 not strictly > 115.0) -> Rejected
        df_exec.loc[df_exec.index[-2], "volume"] = 115.0
        sig_115 = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec, "structure": df_exec, "m15": df_m15, "h1": df_h1},
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNone(sig_115)

        # Scenario C: Breakout bar volume at 1.16x SMA20 (116.0 > 115.0) -> Accepted
        df_exec.loc[df_exec.index[-2], "volume"] = 116.0
        sig_116 = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec, "structure": df_exec, "m15": df_m15, "h1": df_h1},
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNotNone(sig_116)
        self.assertEqual(sig_116.signal_type, SignalType.BUY)

    def test_18_structural_stop_loss_anchoring(self):
        """Feature 9: Anchors SL tightly to recent swing level +/- 0.20 ATR without clamping inside candle; Gold M5 max pips = 43.5."""
        strat = TitanScalperStrategy(self.config)
        specs_gold = SymbolSpecs(
            name="XAUUSD", digits=2, point=0.01, tick_value=1.0, tick_size=0.01,
            contract_size=100.0, min_volume=0.01, max_volume=100.0, volume_step=0.01, pip_size=0.10
        )

        df_gold = self._build_test_breakout_data(is_bull=True, is_gold=True)
        # Entry price current_ask = 2661.5
        # Set recent swing low in iloc[-8:-1] to 2658.0; ATR = 2.0 ($0.20 ATR = $0.40)
        # Expected structural SL price = 2658.0 - 0.40 = 2657.60
        # Expected sl_dist = 2661.50 - 2657.60 = 3.90 ($39.0 pips <= 43.5 pips max_pips)
        df_gold.loc[df_gold.index[-8:-1], "low"] = 2658.0

        df_m15 = df_gold.copy()
        df_m15["ema_200"] = 2640.0
        df_h1 = df_gold.copy()
        df_h1["ema_200"] = 2630.0

        sig_ok = strat.evaluate(
            symbol="XAUUSD",
            data_snapshot={"execution": df_gold, "structure": df_gold, "m15": df_m15, "h1": df_h1},
            symbol_specs=specs_gold,
            current_bid=2661.4,
            current_ask=2661.5,
            current_spread_pips=1.0,
        )
        self.assertIsNotNone(sig_ok)
        self.assertAlmostEqual(sig_ok.sl_price, 2657.60, places=2)
        # Stop loss price MUST be at or below recent_low - 0.20 ATR, never clamped above it
        self.assertLessEqual(sig_ok.sl_price, 2658.0 - (0.20 * 2.0))
        self.assertAlmostEqual(sig_ok.risk_distance_pips, 39.0, places=1)

        # Exceeding max_pips (43.5 pips = $4.35):
        # Set recent_low to 2656.50 -> structural_sl_dist = 2661.5 - (2656.5 - 0.40) = 5.40 (54.0 pips > 43.5)
        df_gold.loc[df_gold.index[-8:-1], "low"] = 2656.5
        sig_wide = strat.evaluate(
            symbol="XAUUSD",
            data_snapshot={"execution": df_gold, "structure": df_gold, "m15": df_m15, "h1": df_h1},
            symbol_specs=specs_gold,
            current_bid=2661.4,
            current_ask=2661.5,
            current_spread_pips=1.0,
        )
        self.assertIsNone(sig_wide)

    def test_19_deactivate_mean_reversion_entries(self):
        """Feature 11: Counter-trend mean reversion entries are deactivated (returns None) for 100% pure Squeeze Breakouts."""
        strat = TitanScalperStrategy(self.config)
        specs_eur = SymbolSpecs(
            name="EURUSD", digits=5, point=0.00001, tick_value=1.0, tick_size=0.00001,
            contract_size=100000.0, min_volume=0.01, max_volume=100.0, volume_step=0.01, pip_size=0.0001
        )

        # Direct call to _evaluate_mean_reversion returns None
        res = strat._evaluate_mean_reversion(
            symbol="EURUSD",
            df_exec=pd.DataFrame(),
            df_struct=None,
            m15_bull=True,
            m15_bear=True,
            symbol_specs=specs_eur,
            regime_info={},
            current_bid=1.0850,
            current_ask=1.0851,
        )
        self.assertIsNone(res)

        # In evaluate(), MEAN_REVERTING regime returns None
        df_exec = self._build_test_breakout_data(is_bull=True)
        # Mock regime detector to return MEAN_REVERTING
        strat.regime_detector.analyze_regime = lambda df: (MarketRegime.MEAN_REVERTING, {"hurst": 0.40})
        sig_mr = strat.evaluate(
            symbol="EURUSD",
            data_snapshot={"execution": df_exec},
            symbol_specs=specs_eur,
            current_bid=1.0850,
            current_ask=1.0851,
            current_spread_pips=1.0,
        )
        self.assertIsNone(sig_mr)

    def test_20_execution_config_dynamic_breakeven_145r(self):
        """Feature 10: Dynamic Breakeven default calibrated to 1.45R with 2.0 pips buffer."""
        from config.settings import ExecutionConfig
        cfg = ExecutionConfig()
        self.assertEqual(cfg.breakeven_trigger_r, 1.45)
        self.assertEqual(cfg.breakeven_buffer_pips, 2.0)

        # Check loaded config from yaml
        self.assertEqual(self.config.execution.breakeven_trigger_r, 1.45)
        self.assertEqual(self.config.execution.breakeven_buffer_pips, 2.0)

        # Check execution manager uses config values
        mock_client = MockMT5Client()
        exec_mgr = ExecutionManager(self.config, mock_client)
        self.assertEqual(exec_mgr.exec_cfg.breakeven_trigger_r, 1.45)
        self.assertEqual(exec_mgr.exec_cfg.breakeven_buffer_pips, 2.0)

        # Check backtest engine uses config values
        engine = BacktestEngine(config=self.config, initial_balance=250.0)
        self.assertEqual(engine.config.execution.breakeven_trigger_r, 1.45)
        self.assertEqual(engine.config.execution.breakeven_buffer_pips, 2.0)

    def test_21_engine_multi_timeframe_h1_pipeline(self):
        """Feature 13: BacktestEngine accepts df_h1, slices with tail(300), and delivers macro snapshot."""
        engine = BacktestEngine(config=self.config, initial_balance=250.0)

        captured_snapshots = []
        original_evaluate = engine.strategy.evaluate

        def mock_eval(symbol, data_snapshot, symbol_specs, current_bid, current_ask, current_spread_pips):
            captured_snapshots.append(data_snapshot)
            return original_evaluate(symbol, data_snapshot, symbol_specs, current_bid, current_ask, current_spread_pips)

        engine.strategy.evaluate = mock_eval

        df_m1 = generate_synthetic_forex_data(n_bars=350, start_price=1.0850)
        df_m5 = df_m1.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        df_m15 = df_m1.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        df_h1 = df_m1.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

        metrics = engine.run(
            df_m1=df_m1,
            df_m5=df_m5,
            df_m15=df_m15,
            symbol="EURUSD",
            df_h1=df_h1,
        )
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.initial_balance, 250.0)
        self.assertGreater(len(captured_snapshots), 0)

        snap = captured_snapshots[0]
        self.assertIn("macro", snap)
        self.assertIsNotNone(snap["macro"])
        self.assertIn("execution", snap)
        self.assertIn("structure", snap)
        self.assertIn("regime", snap)
        self.assertIn("m15", snap)
        self.assertIn("h1", snap)

    def test_22_engine_snapshot_m15_regime_filtering(self):
        """Feature 7 & 13 Contract: Verifies that engine snapshot with 'structure'=M5, 'regime'=M15, 'macro'=H1 correctly extracts M15 for 200 EMA filtering."""
        strat = TitanScalperStrategy(self.config)
        specs_eur = SymbolSpecs(
            name="EURUSD", digits=5, point=0.00001, tick_value=1.0, tick_size=0.00001,
            contract_size=100000.0, min_volume=0.01, max_volume=100.0, volume_step=0.01, pip_size=0.0001
        )

        df_exec = self._build_test_breakout_data(is_bull=True)
        # M5 structure is bullish (close 1.0856 > 1.0830)
        df_m5 = df_exec.copy()
        df_m5["ema_200"] = 1.0830

        # M15 regime is bearish (close 1.0856 < 1.0870)
        df_m15 = df_exec.copy()
        df_m15["ema_200"] = 1.0870

        # H1 macro is bullish (close 1.0856 > 1.0820)
        df_h1 = df_exec.copy()
        df_h1["ema_200"] = 1.0820

        # Exact standard engine snapshot dictionary without direct "m15" or "h1" keys
        snap_hierarchical = {
            "execution": df_exec,
            "structure": df_m5,
            "regime": df_m15,
            "macro": df_h1,
        }

        # Because M15 is bearish, BUY signal must be strictly rejected
        sig_m15_rej = strat.evaluate(
            symbol="EURUSD",
            data_snapshot=snap_hierarchical,
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNone(
            sig_m15_rej,
            "BUY signal must be strictly rejected when M15 regime close < 200 EMA, even if M5 structure is bullish",
        )

        # When M15 regime becomes bullish (close 1.0856 > 1.0830), BUY signal must be accepted
        df_m15_bull = df_exec.copy()
        df_m15_bull["ema_200"] = 1.0830
        snap_hierarchical["regime"] = df_m15_bull

        sig_accepted = strat.evaluate(
            symbol="EURUSD",
            data_snapshot=snap_hierarchical,
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNotNone(sig_accepted, "BUY signal must be accepted when both M15 and H1 are bullish")
        self.assertEqual(sig_accepted.signal_type, SignalType.BUY)

        # Also verify with direct "m15" and "h1" keys supplied alongside hierarchical keys
        snap_with_direct_keys = {
            "execution": df_exec,
            "structure": df_m5,
            "regime": df_m15,
            "macro": df_h1,
            "m15": df_m15,  # bearish
            "h1": df_h1,
        }
        sig_direct_rej = strat.evaluate(
            symbol="EURUSD",
            data_snapshot=snap_with_direct_keys,
            symbol_specs=specs_eur,
            current_bid=1.0855,
            current_ask=1.0856,
            current_spread_pips=1.0,
        )
        self.assertIsNone(
            sig_direct_rej,
            "BUY signal must be strictly rejected when direct 'm15' close < 200 EMA",
        )



if __name__ == "__main__":
    unittest.main()

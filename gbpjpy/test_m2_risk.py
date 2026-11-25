"""
Milestone 2 Risk & Execution Management Test Suite for GBPJPY Framework.
Verifies Market Regime Detection, Session Filters, Small-Capital Risk Management ($250, 1:30),
Execution Lifecycle (Breakeven, Trailing Stop Ratchet, Time-Decay Exit), and End-to-End Integration.

Run via:
    python "c:\\Trading algo\\gbpjpy\\test_m2_risk.py"
"""

import os
import sys
import unittest
from datetime import datetime, date, timezone, timedelta
import numpy as np
import pandas as pd

# Guarantee gbpjpy package is available on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [PARENT_DIR, CURRENT_DIR]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

try:
    from gbpjpy.config.settings import load_config, GBPJPY_SPECS
    from gbpjpy.core.regime_detector import RegimeDetector
    from gbpjpy.utils.market_hours import MarketHours
    from gbpjpy.risk.risk_manager import RiskManager
    from gbpjpy.risk.execution_manager import ExecutionManager
    from gbpjpy.core.mock_client import MockMT5Client, SymbolSpecs
except ImportError:
    from config.settings import load_config, GBPJPY_SPECS
    from core.regime_detector import RegimeDetector
    from utils.market_hours import MarketHours
    from risk.risk_manager import RiskManager
    from risk.execution_manager import ExecutionManager
    from core.mock_client import MockMT5Client, SymbolSpecs
from tests.fixtures import (
    generate_synthetic_m5_data,
    reference_garman_klass_vol,
    reference_hurst_exponent,
    reference_required_margin_usd,
    MockAccountInfo,
)


class TestM2RegimeAndSessions(unittest.TestCase):
    """Verifies Market Regime Detection (Hurst, Garman-Klass) and Market Hours Session Filters."""

    def setUp(self):
        self.detector = RegimeDetector()
        self.hours = MarketHours()

    def test_hurst_trending_classification(self):
        """Persistent trending series must yield Hurst > 0.55."""
        df_trend = generate_synthetic_m5_data(bars=300, regime="trend_up", seed=101)
        hurst = self.detector.compute_hurst(df_trend["close"])
        self.assertGreater(hurst, 0.55, f"Expected Hurst > 0.55, got {hurst}")

        res = self.detector.detect_regime(df_trend)
        self.assertIn("TREND", str(res.regime).upper())

    def test_hurst_mean_reverting_classification(self):
        """Mean-reverting OU series must yield Hurst < 0.45."""
        df_mr = generate_synthetic_m5_data(bars=300, regime="mean_reversion", seed=202)
        hurst = self.detector.compute_hurst(df_mr["close"])
        self.assertLess(hurst, 0.45, f"Expected Hurst < 0.45, got {hurst}")

        res = self.detector.detect_regime(df_mr)
        self.assertIn("MEAN_REVERT", str(res.regime).upper())

    def test_garman_klass_volatility_matches_oracle(self):
        """Garman-Klass volatility calculation must match benchmark formula."""
        df = generate_synthetic_m5_data(bars=100, regime="random_walk", seed=303)
        expected_gk = reference_garman_klass_vol(df)
        actual_gk = self.detector.compute_garman_klass(df)
        val = actual_gk.iloc[-1] if isinstance(actual_gk, (pd.Series, np.ndarray)) else actual_gk
        self.assertAlmostEqual(val, expected_gk, places=3)

    def test_hurst_flat_series_resilience(self):
        """Flat price series must not trigger ZeroDivisionError."""
        flat_prices = pd.Series([195.500] * 200)
        hurst = self.detector.compute_hurst(flat_prices)
        self.assertFalse(np.isnan(hurst))
        self.assertFalse(np.isinf(hurst))
        self.assertTrue(0.0 <= hurst <= 1.0)

    def test_session_window_classifications(self):
        """Verifies Asian, London, Tokyo overlap, and NY overlap session detection."""
        # 02:00 UTC -> Asian
        t_asian = datetime(2026, 6, 10, 2, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(self.hours.is_asian_session(t_asian))
        self.assertIn("ASIAN", str(self.hours.get_session(t_asian)).upper())

        # 08:30 UTC -> London
        t_london = datetime(2026, 6, 10, 8, 30, 0, tzinfo=timezone.utc)
        self.assertTrue(self.hours.is_london_session(t_london))
        self.assertIn("LONDON", str(self.hours.get_session(t_london)).upper())

        # 07:30 UTC -> Tokyo overlap
        t_tokyo_overlap = datetime(2026, 6, 10, 7, 30, 0, tzinfo=timezone.utc)
        if hasattr(self.hours, "is_tokyo_overlap"):
            self.assertTrue(self.hours.is_tokyo_overlap(t_tokyo_overlap))

        # 14:00 UTC -> NY overlap
        t_ny = datetime(2026, 6, 10, 14, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(self.hours.is_ny_session(t_ny))

    def test_rollover_lockout_and_cutoffs(self):
        """Verifies 21:00-23:00 UTC rollover lockout and Friday 19:00 close cutoff."""
        t_lockout = datetime(2026, 6, 10, 21, 30, 0, tzinfo=timezone.utc)
        t_normal = datetime(2026, 6, 10, 10, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(self.hours.is_rollover_lockout(t_lockout))
        self.assertFalse(self.hours.is_rollover_lockout(t_normal))
        self.assertFalse(self.hours.can_trade(t_lockout))
        self.assertTrue(self.hours.can_trade(t_normal))

        # Friday 19:30 UTC -> blocked
        t_friday = datetime(2026, 6, 12, 19, 30, 0, tzinfo=timezone.utc)
        self.assertFalse(self.hours.can_trade(t_friday))


class TestM2RiskManager(unittest.TestCase):
    """Verifies Small-Capital Risk Manager ($250, 1:30), Margin Guards, Circuit Breakers, and Sizing."""

    def setUp(self):
        self.rm = RiskManager()

    def test_margin_requirement_calculation(self):
        """0.01 lot GBPJPY at 1:30 leverage on GBPUSD=1.28 requires ~$42.67 margin."""
        volume = 0.01
        leverage = 30.0
        gbpusd = 1.28
        expected = reference_required_margin_usd(volume, leverage, 100000.0, gbpusd)
        actual = self.rm.calculate_required_margin(volume=volume, leverage=leverage, gbpusd_rate=gbpusd)
        self.assertAlmostEqual(actual, expected, places=2)

    def test_margin_level_guard_200pct(self):
        """Equity $60 gives margin level <200% and must block new orders."""
        min_margin = (0.01 * 100000.0 * 1.28) / 30.0  # $42.67
        # $60 / 42.67 = 140.6% (< 200%)
        can_open = self.rm.can_open_position(equity=60.0, required_margin=min_margin, current_margin=0.0) if hasattr(self.rm, "can_open_position") else self.rm.check_margin_level(equity=60.0, total_margin=min_margin)
        self.assertFalse(can_open)

    def test_equity_depletion_blocking(self):
        """Equity $35 below absolute min margin ($42.67) must block all trading."""
        lots = self.rm.calculate_lot_size(equity=35.0, risk_distance_points=200, gbpusd_rate=1.28)
        self.assertEqual(lots, 0.0)

    def test_dynamic_lot_sizing_1pct(self):
        """1% risk on $250 ($2.50) with 20 pip SL clamps to 0.01 lot minimum parity."""
        lot = self.rm.calculate_lot_size(equity=250.0, sl_pips=20.0)
        self.assertEqual(lot, 0.01)

    def test_single_position_limit_enforcement(self):
        """Second concurrent trade request is rejected under max_open_positions: 1."""
        can_1 = self.rm.can_open_trade(open_positions_count=0)
        self.assertTrue(can_1)
        can_2 = self.rm.can_open_trade(open_positions_count=1)
        self.assertFalse(can_2)

    def test_three_strike_circuit_breaker(self):
        """3 consecutive losses activate circuit breaker cooldown."""
        cb = getattr(self.rm, "circuit_breaker", self.rm)
        cb.record_trade_result(pnl=-2.50, is_win=False)
        self.assertTrue(cb.can_trade())
        cb.record_trade_result(pnl=-2.50, is_win=False)
        self.assertTrue(cb.can_trade())
        cb.record_trade_result(pnl=-2.50, is_win=False)
        self.assertFalse(cb.can_trade())
        self.assertTrue(cb.is_circuit_breaker_active() if hasattr(cb, "is_circuit_breaker_active") else cb.is_cooldown_active())

    def test_daily_drawdown_limit_3pct(self):
        """Daily loss hitting $7.50 (3.0% of $250) immediately halts trading."""
        cb = getattr(self.rm, "circuit_breaker", self.rm)
        if hasattr(cb, "set_daily_starting_equity"):
            cb.set_daily_starting_equity(250.0)
        cb.record_trade_result(pnl=-4.00, is_win=False)
        self.assertTrue(cb.can_trade())
        cb.record_trade_result(pnl=-3.50, is_win=False)
        self.assertFalse(cb.can_trade())

    def test_circuit_breaker_resets_next_day(self):
        """Circuit breaker lockout clears automatically on the next calendar day."""
        cb = getattr(self.rm, "circuit_breaker", self.rm)
        day1 = date(2026, 6, 10)
        day2 = date(2026, 6, 11)
        cb.record_trade_result(pnl=-3.00, is_win=False, trade_date=day1)
        cb.record_trade_result(pnl=-3.00, is_win=False, trade_date=day1)
        cb.record_trade_result(pnl=-3.00, is_win=False, trade_date=day1)
        self.assertFalse(cb.can_trade(current_date=day1))

        if hasattr(cb, "reset_daily_limits"):
            cb.reset_daily_limits(new_equity=241.0, current_date=day2)
        elif hasattr(cb, "on_new_day"):
            cb.on_new_day(current_date=day2)
        self.assertTrue(cb.can_trade(current_date=day2))

    def test_max_spread_filter(self):
        """Spread > 3.0 pips is rejected."""
        self.assertTrue(self.rm.is_spread_acceptable(spread_pips=1.8, max_spread=3.0))
        self.assertFalse(self.rm.is_spread_acceptable(spread_pips=3.2, max_spread=3.0))

    def test_swap_cost_wednesday_3x(self):
        """Wednesday rollover applies 3x swap cost."""
        if hasattr(self.rm, "calculate_swap"):
            tue = datetime(2026, 6, 9, 22, 0, 0, tzinfo=timezone.utc)
            wed = datetime(2026, 6, 10, 22, 0, 0, tzinfo=timezone.utc)
            swap_tue = self.rm.calculate_swap(volume=0.01, is_long=True, rollover_dt=tue)
            swap_wed = self.rm.calculate_swap(volume=0.01, is_long=True, rollover_dt=wed)
            self.assertAlmostEqual(abs(swap_wed), abs(swap_tue) * 3, places=2)

    def test_circuit_breaker_cooldown_expiration_historical_and_wall_clock(self):
        """Historical backtest and wall-clock cooldown expiration at T+30m (False) and T+65m (True)."""
        cb = getattr(self.rm, "circuit_breaker", self.rm)
        t0 = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
        cb.record_trade_result(pnl=-1.0, is_win=False, trade_date=t0)
        cb.record_trade_result(pnl=-1.0, is_win=False, trade_date=t0)
        cb.record_trade_result(pnl=-1.0, is_win=False, trade_date=t0)

        # Check at T+30m (must still be in cooldown)
        t_30m = t0 + timedelta(minutes=30)
        self.assertFalse(cb.can_trade(current_date=t_30m))
        self.assertTrue(cb.is_cooldown_active(now=t_30m))
        self.assertFalse(self.rm.can_trade(current_date=t_30m))
        self.assertTrue(self.rm.is_cooldown_active(now=t_30m))

        # Check at T+65m (must be cleared)
        t_65m = t0 + timedelta(minutes=65)
        self.assertTrue(cb.can_trade(current_date=t_65m))
        self.assertFalse(cb.is_cooldown_active(now=t_65m))
        self.assertTrue(self.rm.can_trade(current_date=t_65m))
        self.assertFalse(self.rm.is_cooldown_active(now=t_65m))

        # Wall-clock live execution verification (now is None)
        cb.consecutive_losses = 3
        cb.cooldown_active = True
        cb.cooldown_until = datetime.now(timezone.utc).timestamp() - 5
        self.assertFalse(cb.is_cooldown_active())
        self.assertTrue(cb.can_trade())
        self.assertFalse(self.rm.is_cooldown_active())
        self.assertTrue(self.rm.can_trade())


class TestM2ExecutionManager(unittest.TestCase):
    """Verifies Breakeven, ATR Trailing Stop, Ratchet Invariant, Time Decay Exit, and Signal Suppression."""

    def setUp(self):
        self.em = ExecutionManager()

    def test_breakeven_trigger_and_buffer(self):
        """Breakeven triggers at +1.45R with +2.0 pip buffer."""
        entry = 155.000
        initial_sl = 154.800
        # 1.4R = 155.280 -> False
        self.assertFalse(self.em.check_breakeven(order_type="BUY", entry_price=entry, current_price=155.280, initial_sl=initial_sl, trigger_r=1.45))
        # 1.5R = 155.300 -> True
        self.assertTrue(self.em.check_breakeven(order_type="BUY", entry_price=entry, current_price=155.300, initial_sl=initial_sl, trigger_r=1.45))

        # Check buffer prices
        be_buy = self.em.calculate_breakeven_sl(order_type="BUY", entry_price=entry, buffer_pips=2.0)
        self.assertAlmostEqual(be_buy, 155.020, places=3)
        be_sell = self.em.calculate_breakeven_sl(order_type="SELL", entry_price=entry, buffer_pips=2.0)
        self.assertAlmostEqual(be_sell, 154.980, places=3)

    def test_atr_trailing_stop_calculation(self):
        """ATR trailing stop calculates highest - (1.5 * atr) for BUY."""
        new_sl = self.em.calculate_trailing_stop(order_type="BUY", highest_price=155.600, current_sl=155.020, atr=0.150, multiplier=1.5)
        self.assertAlmostEqual(new_sl, 155.375, places=3)

    def test_trailing_stop_ratchet_invariant(self):
        """Trailing stop must never retreat / loosen on price pullbacks."""
        # BUY: price pullbacks from 155.600 to 155.450. SL must stay at 155.400, never drop to 155.150.
        sl_buy = self.em.calculate_trailing_stop(order_type="BUY", highest_price=155.450, current_sl=155.400, atr=0.200, multiplier=1.5)
        self.assertGreaterEqual(sl_buy, 155.400)

        # SELL: price bounces from 154.550 upwards. SL must stay at 154.600, never rise to 154.850.
        sl_sell = self.em.calculate_trailing_stop(order_type="SELL", lowest_price=154.550, current_sl=154.600, atr=0.200, multiplier=1.5)
        self.assertLessEqual(sl_sell, 154.600)

    def test_time_decay_stale_trade_exit(self):
        """Stale trades held >= 48 bars trigger exit."""
        self.assertFalse(self.em.check_time_decay_exit(bars_held=10, max_bars=48))
        self.assertTrue(self.em.check_time_decay_exit(bars_held=50, max_bars=48))

    def test_circuit_breaker_suppresses_execution(self):
        """execute_signal must suppress order execution when circuit breaker is active."""
        rm = RiskManager()
        cb = getattr(rm, "circuit_breaker", rm)
        cb.record_trade_result(pnl=-2.50, is_win=False)
        cb.record_trade_result(pnl=-2.50, is_win=False)
        cb.record_trade_result(pnl=-2.50, is_win=False)

        signal = {
            "symbol": "GBPJPY", "signal_type": "BUY", "entry_price": 195.500,
            "sl": 195.300, "tp": 196.000, "volume": 0.01
        }
        res = self.em.execute_signal(signal, risk_manager=rm)
        self.assertFalse(res.get("executed", False))
        self.assertIn("circuit", res.get("reason", "").lower())


class TestM2FullIntegrationLifecycle(unittest.TestCase):
    """End-to-End integration test uniting MockMT5Client, MarketHours, RegimeDetector, RiskManager, and ExecutionManager."""

    def test_complete_m2_trade_lifecycle(self):
        client = MockMT5Client(initial_balance=250.0, leverage=30.0)
        client.connect()
        hours = MarketHours()
        detector = RegimeDetector()
        rm = RiskManager()
        em = ExecutionManager(mt5_client=client, risk_manager=rm)

        # 1. Rollover check
        t_rollover = datetime(2026, 6, 10, 21, 30, 0, tzinfo=timezone.utc)
        self.assertFalse(hours.can_trade(t_rollover), "Rollover lockout must block trades")

        # 2. London session open
        t_london = datetime(2026, 6, 10, 8, 30, 0, tzinfo=timezone.utc)
        self.assertTrue(hours.can_trade(t_london), "London session must allow trades")

        # 3. Regime detection
        df_trend = generate_synthetic_m5_data(bars=300, regime="trend_up", seed=42)
        hurst = detector.compute_hurst(df_trend["close"])
        self.assertGreater(hurst, 0.55, "Trending regime detected")

        # 4. Sizing and margin validation
        lot = rm.calculate_lot_size(equity=client.equity, sl_pips=20.0)
        self.assertEqual(lot, 0.01)
        self.assertTrue(rm.can_open_trade(open_positions_count=0))

        # 5. Execute Signal
        client.set_sim_price("GBPJPY", 195.000, 195.015)
        signal = {
            "symbol": "GBPJPY", "signal_type": "BUY", "entry_price": 195.015,
            "sl": 194.815, "tp": 195.615, "volume": lot
        }
        exec_res = em.execute_signal(signal, risk_manager=rm)
        self.assertTrue(exec_res["executed"])
        self.assertEqual(len(client.get_open_positions()), 1)

        # 6. Single position concurrency rejection
        self.assertFalse(rm.can_open_trade(open_positions_count=len(client.get_open_positions())))

        # 7. Price advances to 1.45R (profit = 29 pips -> price = 195.305)
        client.set_sim_price("GBPJPY", 195.305, 195.320)
        new_sl = em.update_trailing_stop(
            is_long=True, entry_price=195.015, current_sl=194.815,
            current_price=195.305, atr=0.150
        )
        self.assertGreaterEqual(new_sl, 195.015 + 0.020, "Moved to breakeven + buffer")

        # 8. Price ratchets higher to 195.600
        client.set_sim_price("GBPJPY", 195.600, 195.615)
        trailed_sl = em.update_trailing_stop(
            is_long=True, entry_price=195.015, current_sl=new_sl,
            current_price=195.600, atr=0.150
        )
        self.assertGreater(trailed_sl, new_sl)

        # 9. Retracement preserves ratchet invariant
        pullback_sl = em.update_trailing_stop(
            is_long=True, entry_price=195.015, current_sl=trailed_sl,
            current_price=195.450, atr=0.150
        )
        self.assertEqual(pullback_sl, trailed_sl)


if __name__ == "__main__":
    unittest.main()

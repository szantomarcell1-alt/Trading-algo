"""
Unified Portfolio Risk & Multi-Symbol Orchestration Test Suite
Comprehensive verification of PortfolioConfig, UnifiedPortfolioRiskManager,
PortfolioDispatcher, and PortfolioMetrics under strict Small-Account constraints.
"""

import unittest
from datetime import datetime, timezone, timedelta, date
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from portfolio.portfolio_config import (
    PortfolioConfig,
    SymbolRiskProfile,
    load_portfolio_config,
)
from portfolio.portfolio_risk_manager import UnifiedPortfolioRiskManager
from portfolio.portfolio_dispatcher import PortfolioDispatcher
from portfolio.portfolio_metrics import (
    evaluate_portfolio_performance,
    calculate_weekly_returns,
    PortfolioVerificationReport,
    PortfolioWeeklyReturn,
)


@dataclass
class DummySignal:
    symbol: str
    signal_type: str = "BUY"
    is_buy: bool = True
    is_sell: bool = False
    entry_price: float = 1.0850
    sl: float = 1.0830
    tp: float = 1.0890
    volume: float = 0.01
    confidence: float = 0.85
    reason: str = "TestSignal"


class TestPortfolioConfig(unittest.TestCase):
    """Verifies default portfolio configuration values and small-account bounds."""

    def setUp(self):
        self.config = load_portfolio_config()

    def test_default_account_parameters(self):
        self.assertEqual(self.config.initial_balance, 250.0)
        self.assertEqual(self.config.leverage, 30.0)
        self.assertEqual(self.config.max_open_positions, 2)
        self.assertEqual(self.config.max_open_positions_per_symbol, 1)

    def test_risk_guardrail_parameters(self):
        self.assertEqual(self.config.min_margin_level_percent, 200.0)
        self.assertEqual(self.config.max_daily_drawdown_percent, 3.0)
        self.assertEqual(self.config.max_total_drawdown_percent, 5.5)
        self.assertEqual(self.config.max_consecutive_losses, 3)
        self.assertEqual(self.config.consecutive_loss_cooldown_minutes, 60)

    def test_symbol_profiles_presence(self):
        for sym in ["EURUSD", "XAUUSD", "GBPJPY"]:
            profile = self.config.get_symbol_profile(sym)
            self.assertIsNotNone(profile, f"Missing profile for {sym}")
            self.assertTrue(profile.enabled)
            self.assertGreater(profile.point, 0.0)
            self.assertGreater(profile.pip_size, 0.0)


class TestPortfolioRiskManagerMargin(unittest.TestCase):
    """Verifies margin calculation and projected margin level guard (>= 200%)."""

    def setUp(self):
        self.config = load_portfolio_config()
        self.rm = UnifiedPortfolioRiskManager(self.config)
        self.rm.balance = 250.0
        self.rm.equity = 250.0

    def test_margin_calculation_per_symbol(self):
        # EURUSD: 0.01 lot * 100,000 * 1.0850 / 30 = $36.17 USD
        m_eur = self.rm.calculate_required_margin("EURUSD", 0.01, 1.0850)
        self.assertAlmostEqual(m_eur, 36.17, places=1)

        # XAUUSD: 0.01 lot * 100 oz * 2650.00 / 30 = $88.33 USD
        m_gold = self.rm.calculate_required_margin("XAUUSD", 0.01, 2650.00)
        self.assertAlmostEqual(m_gold, 88.33, places=1)

        # GBPJPY: 0.01 lot * 100,000 * 1.28 GBPUSD / 30 = ~$42.67 USD
        m_gbp = self.rm.calculate_required_margin("GBPJPY", 0.01, 195.00)
        self.assertGreater(m_gbp, 30.0)
        self.assertLess(m_gbp, 50.0)

    def test_projected_margin_guard_approval(self):
        # 1 micro-lot EURUSD requires ~$36.17 margin. With $250 equity, margin level is ~691% >= 200%
        now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
        sig = DummySignal(symbol="EURUSD", entry_price=1.0850, sl=1.0835, tp=1.0880)
        approved, reason, vol = self.rm.can_open_trade("EURUSD", sig, 1.0850, now)
        self.assertTrue(approved, f"Expected approval, got rejected: {reason}")
        self.assertEqual(vol, 0.01)

    def test_projected_margin_guard_rejection_low_equity(self):
        # Lower equity so projected margin level drops below 200%
        # Margin needed for gold is ~$88.33. If equity is $150, projected margin level is (150/88.33)*100 = 169.8% < 200%
        self.rm.all_time_peak_equity = 150.0
        self.rm.equity = 150.0
        now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
        sig = DummySignal(symbol="XAUUSD", entry_price=2650.00, sl=2646.00, tp=2660.00)
        approved, reason, vol = self.rm.can_open_trade("XAUUSD", sig, 2650.00, now)
        self.assertFalse(approved)
        self.assertIn("Projected margin level", reason)


class TestPortfolioRiskManagerConcurrency(unittest.TestCase):
    """Verifies max 2 concurrent positions across portfolio and max 1 per symbol."""

    def setUp(self):
        self.config = load_portfolio_config()
        self.rm = UnifiedPortfolioRiskManager(self.config)
        self.rm.balance = 250.0
        self.rm.equity = 250.0
        self.now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

    def test_max_portfolio_concurrency_cap(self):
        # Open 1st position (EURUSD)
        self.rm.register_position_opened("EURUSD", 0.01, 1.0850, self.now, 1.0835, 1.0880, ticket=101)
        self.assertEqual(len(self.rm.open_positions), 1)

        # Open 2nd position (GBPJPY)
        self.rm.register_position_opened("GBPJPY", 0.01, 195.00, self.now, 194.80, 195.40, ticket=102)
        self.assertEqual(len(self.rm.open_positions), 2)

        # Attempt to open 3rd position (XAUUSD) -> Must be rejected
        sig = DummySignal(symbol="XAUUSD", entry_price=2650.00, sl=2646.00, tp=2660.00)
        approved, reason, vol = self.rm.can_open_trade("XAUUSD", sig, 2650.00, self.now)
        self.assertFalse(approved)
        self.assertIn("concurrency cap reached", reason)

    def test_max_single_symbol_concurrency_cap(self):
        # Open EURUSD
        self.rm.register_position_opened("EURUSD", 0.01, 1.0850, self.now, 1.0835, 1.0880, ticket=101)

        # Attempt to open a 2nd EURUSD position -> Must be rejected
        sig = DummySignal(symbol="EURUSD", entry_price=1.0855, sl=1.0840, tp=1.0885)
        approved, reason, vol = self.rm.can_open_trade("EURUSD", sig, 1.0855, self.now)
        self.assertFalse(approved)
        self.assertIn("already has an active position", reason)


class TestPortfolioRiskManagerGoldCap(unittest.TestCase):
    """Verifies strict $4.50 absolute small-account risk cap on Gold CFD (0.01 lot)."""

    def setUp(self):
        self.config = load_portfolio_config()
        self.rm = UnifiedPortfolioRiskManager(self.config)
        self.rm.balance = 250.0
        self.rm.equity = 250.0
        self.now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

    def test_gold_cap_approval_under_budget(self):
        # 0.01 lot XAUUSD with 43.5 pips stop ($4.35 risk <= $4.50 cap)
        sig = DummySignal(symbol="XAUUSD", entry_price=2650.00, sl=2645.65, tp=2660.00)
        approved, reason, vol = self.rm.can_open_trade("XAUUSD", sig, 2650.00, self.now)
        self.assertTrue(approved, f"Expected approval, got: {reason}")
        self.assertEqual(vol, 0.01)

    def test_gold_cap_rejection_over_budget(self):
        # 0.01 lot XAUUSD with 50.0 pips stop ($5.00 risk > $4.50 cap)
        sig = DummySignal(symbol="XAUUSD", entry_price=2650.00, sl=2645.00, tp=2660.00)
        approved, reason, vol = self.rm.can_open_trade("XAUUSD", sig, 2650.00, self.now)
        self.assertFalse(approved)
        self.assertTrue("4.50 USD" in reason or "plafont" in reason)


class TestPortfolioRiskManagerCircuitBreaker(unittest.TestCase):
    """Verifies shared 3-strike circuit breaker with 60-minute candle timestamp cooldown."""

    def setUp(self):
        self.config = load_portfolio_config()
        self.rm = UnifiedPortfolioRiskManager(self.config)
        self.rm.balance = 250.0
        self.rm.equity = 250.0
        self.base_time = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

    def test_three_consecutive_losses_trip_circuit_breaker(self):
        # 1st loss on EURUSD (-$1.50)
        t1 = self.base_time
        self.rm.register_trade_close("EURUSD", -1.50, t1, ticket=101)
        self.assertEqual(self.rm.consecutive_losses, 1)
        self.assertIsNone(self.rm.cooldown_until)

        # 2nd loss on GBPJPY (-$1.50)
        t2 = self.base_time + timedelta(minutes=15)
        self.rm.register_trade_close("GBPJPY", -1.50, t2, ticket=102)
        self.assertEqual(self.rm.consecutive_losses, 2)
        self.assertIsNone(self.rm.cooldown_until)

        # 3rd loss on XAUUSD (-$1.50) -> Trips Circuit Breaker!
        t3 = self.base_time + timedelta(minutes=30)
        self.rm.register_trade_close("XAUUSD", -1.50, t3, ticket=103)
        self.assertEqual(self.rm.consecutive_losses, 3)
        self.assertIsNotNone(self.rm.cooldown_until)
        self.assertEqual(self.rm.cooldown_until, t3 + timedelta(minutes=60))

        # Attempt trade during cooldown (+30 mins after trip) -> REJECT
        during_cooldown = t3 + timedelta(minutes=30)
        sig = DummySignal(symbol="EURUSD", entry_price=1.0850, sl=1.0835, tp=1.0880)
        approved, reason, _ = self.rm.can_open_trade("EURUSD", sig, 1.0850, during_cooldown)
        self.assertFalse(approved)
        self.assertIn("circuit breaker active", reason.lower())

        # Attempt trade after cooldown (+65 mins after trip) -> APPROVED and resets losses
        after_cooldown = t3 + timedelta(minutes=65)
        approved, reason, vol = self.rm.can_open_trade("EURUSD", sig, 1.0850, after_cooldown)
        self.assertTrue(approved, f"Expected approval after cooldown, got: {reason}")
        self.assertIsNone(self.rm.cooldown_until)
        self.assertEqual(self.rm.consecutive_losses, 0)

    def test_win_breaks_loss_streak_before_trip(self):
        # 2 consecutive losses
        self.rm.register_trade_close("EURUSD", -2.00, self.base_time, ticket=101)
        self.rm.register_trade_close("GBPJPY", -2.00, self.base_time + timedelta(minutes=15), ticket=102)
        self.assertEqual(self.rm.consecutive_losses, 2)

        # 3rd trade is a WIN -> breaks streak
        self.rm.register_trade_close("EURUSD", +3.50, self.base_time + timedelta(minutes=30), ticket=103)
        self.assertEqual(self.rm.consecutive_losses, 0)
        self.assertIsNone(self.rm.cooldown_until)


class TestPortfolioRiskManagerDrawdown(unittest.TestCase):
    """Verifies combined daily drawdown ceiling (3.0%), budgeting, and daily reset."""

    def setUp(self):
        self.config = load_portfolio_config()
        self.rm = UnifiedPortfolioRiskManager(self.config)
        self.rm.balance = 250.0
        self.rm.equity = 250.0
        self.day1 = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

    def test_daily_drawdown_ceiling_enforcement(self):
        # 3.0% of $250.00 is $7.50. Register losses totaling $7.60
        self.rm.register_trade_close("EURUSD", -4.00, self.day1, ticket=101)
        self.rm.register_trade_close("GBPJPY", -3.60, self.day1 + timedelta(minutes=30), ticket=102)
        self.assertAlmostEqual(self.rm.daily_realized_pnl, -7.60)

        # Next trade must be rejected by daily drawdown limit
        sig = DummySignal(symbol="EURUSD", entry_price=1.0850, sl=1.0835, tp=1.0880)
        approved, reason, _ = self.rm.can_open_trade("EURUSD", sig, 1.0850, self.day1 + timedelta(minutes=45))
        self.assertFalse(approved)
        self.assertIn("daily drawdown limit reached", reason.lower())

    def test_daily_drawdown_reset_on_new_utc_day(self):
        # Register a loss on Day 1
        self.rm.register_trade_close("EURUSD", -5.00, self.day1, ticket=101)
        self.assertEqual(self.rm.daily_realized_pnl, -5.00)

        # Advance to Day 2 (02:00 UTC)
        day2 = datetime(2026, 3, 3, 2, 0, tzinfo=timezone.utc)
        sig = DummySignal(symbol="EURUSD", entry_price=1.0850, sl=1.0835, tp=1.0880)
        approved, reason, vol = self.rm.can_open_trade("EURUSD", sig, 1.0850, day2)
        self.assertTrue(approved, f"Expected reset and approval on new day, got: {reason}")
        self.assertEqual(self.rm.daily_realized_pnl, 0.0)
        self.assertEqual(self.rm.daily_start_equity, 245.0)

    def test_master_portfolio_kill_switch(self):
        # If total drawdown reaches 5.5% ($13.75), kill-switch halts all trading
        self.rm.all_time_peak_equity = 250.0
        self.rm.equity = 236.0  # $14.00 drop = 5.6% DD
        sig = DummySignal(symbol="EURUSD", entry_price=1.0850, sl=1.0835, tp=1.0880)
        approved, reason, _ = self.rm.can_open_trade("EURUSD", sig, 1.0850, self.day1)
        self.assertFalse(approved)
        self.assertIn("kill-switch", reason.lower())


class TestPortfolioDispatcherConflictResolution(unittest.TestCase):
    """Verifies multi-symbol conflict resolution and priority ranking."""

    def setUp(self):
        self.config = load_portfolio_config()
        self.rm = UnifiedPortfolioRiskManager(self.config)
        self.rm.balance = 250.0
        self.rm.equity = 250.0
        self.dispatcher = PortfolioDispatcher(self.config, self.rm)
        self.now = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

    def test_conflict_resolution_prioritizes_higher_confidence(self):
        # 1 slot available with EURUSD already open
        self.rm.register_position_opened("EURUSD", 0.01, 1.0850, self.now, 1.0835, 1.0880, ticket=101)
        self.assertEqual(len(self.rm.open_positions), 1)

        # 2 signals arrive concurrently: GBPJPY (conf 0.88) and XAUUSD (conf 0.92)
        sig_gbp = DummySignal(symbol="GBPJPY", entry_price=195.00, sl=194.80, tp=195.40, confidence=0.88)
        sig_gold = DummySignal(symbol="XAUUSD", entry_price=2650.00, sl=2646.00, tp=2660.00, confidence=0.92)

        candidates = [
            {"symbol": "GBPJPY", "signal": sig_gbp, "price": 195.00, "spread_pips": 1.8, "timestamp": self.now},
            {"symbol": "XAUUSD", "signal": sig_gold, "price": 2650.00, "spread_pips": 2.0, "timestamp": self.now},
        ]

        approved = self.rm.resolve_conflicts(candidates, self.now)
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["symbol"], "XAUUSD")


class TestPortfolioMetricsEvaluation(unittest.TestCase):
    """Verifies weekly returns, winning periods, drawdown calculations and report format."""

    def test_evaluate_portfolio_performance_gates(self):
        # Synthetic clean trade series over 4 calendar weeks
        t0 = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
        trades = [
            # Week 1 (+ $3.00)
            {"symbol": "EURUSD", "net_pnl_usd": 3.00, "exit_time": t0 + timedelta(days=1)},
            # Week 2 (+ $2.50)
            {"symbol": "GBPJPY", "net_pnl_usd": 2.50, "exit_time": t0 + timedelta(days=8)},
            # Week 3 (+ $2.80)
            {"symbol": "XAUUSD", "net_pnl_usd": 2.80, "exit_time": t0 + timedelta(days=15)},
            # Week 4 (+ $2.20)
            {"symbol": "EURUSD", "net_pnl_usd": 2.20, "exit_time": t0 + timedelta(days=22)},
        ]

        equity_curve = [
            (t0, 250.0),
            (t0 + timedelta(days=1), 253.0),
            (t0 + timedelta(days=8), 255.5),
            (t0 + timedelta(days=15), 258.3),
            (t0 + timedelta(days=22), 260.5),
        ]

        daily_dds = {
            (t0 + timedelta(days=1)).date(): 0.5,
            (t0 + timedelta(days=8)).date(): 0.8,
            (t0 + timedelta(days=15)).date(): 1.2,
            (t0 + timedelta(days=22)).date(): 0.6,
        }

        report = evaluate_portfolio_performance(
            trades=trades,
            equity_curve=equity_curve,
            daily_drawdowns=daily_dds,
            initial_balance=250.0,
        )

        self.assertEqual(report.total_trades, 4)
        self.assertEqual(report.winning_periods_pct, 100.0)
        self.assertTrue(report.winning_periods_met)
        self.assertTrue(report.daily_max_dd_met)
        self.assertIn("UNIFIED PORTFOLIO PERFORMANCE", report.format_summary_table())


if __name__ == "__main__":
    unittest.main()

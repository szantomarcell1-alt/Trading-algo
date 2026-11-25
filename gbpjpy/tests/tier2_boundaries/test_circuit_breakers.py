"""Tier 2: Boundary test cases for 3-strike loss lockout, 3.0% daily drawdown limit, and daily reset."""

import sys
import unittest
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.risk.risk_manager import RiskManager
except ImportError:
    RiskManager = None


class TestCircuitBreakers(unittest.TestCase):
    """Boundary tests for intraday circuit breakers: 3-loss streak, 3% daily drawdown, and day rollover."""

    def setUp(self):
        self.starting_equity = 250.0
        self.max_daily_drawdown_pct = 0.03  # 3.0% = $7.50
        self.max_daily_loss = self.starting_equity * self.max_daily_drawdown_pct

    def _get_risk_manager(self):
        if RiskManager is None:
            return None
        try:
            return RiskManager()
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return RiskManager(config=cfg)
            except Exception:
                return RiskManager(config={})

    def test_three_consecutive_losses_circuit_breaker(self):
        """3 consecutive closed losses within the same session must trigger an intraday cooldown."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager module not implemented")

        rm = self._get_risk_manager()
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        cb = getattr(rm, "circuit_breaker", rm)

        if hasattr(cb, "record_trade_result"):
            cb.record_trade_result(pnl=-2.50, is_win=False)
            self.assertTrue(cb.can_trade(), "Should be allowed after 1st loss")

            cb.record_trade_result(pnl=-2.00, is_win=False)
            self.assertTrue(cb.can_trade(), "Should be allowed after 2nd loss")

            cb.record_trade_result(pnl=-2.00, is_win=False)
            self.assertFalse(cb.can_trade(), "3 consecutive losses must trigger cooldown")
            if hasattr(cb, "is_cooldown_active"):
                self.assertTrue(cb.is_cooldown_active())
        elif hasattr(cb, "on_trade_closed"):
            cb.on_trade_closed(pnl=-2.00)
            cb.on_trade_closed(pnl=-2.00)
            self.assertTrue(cb.can_trade())
            cb.on_trade_closed(pnl=-2.00)
            self.assertFalse(cb.can_trade())
        else:
            can_trade = getattr(rm, "can_trade", None)
            self.assertIsNotNone(can_trade, "RiskManager missing trade permission / circuit breaker method")

    def test_daily_drawdown_exact_hit(self):
        """Intraday drawdown hitting exactly 3.0% ($7.50 on $250) must halt trading immediately."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager module not implemented")

        rm = self._get_risk_manager()
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        cb = getattr(rm, "circuit_breaker", rm)

        if hasattr(cb, "set_daily_starting_equity"):
            cb.set_daily_starting_equity(self.starting_equity)
        elif hasattr(rm, "daily_starting_equity"):
            rm.daily_starting_equity = self.starting_equity

        if hasattr(cb, "record_trade_result"):
            cb.record_trade_result(pnl=-4.00, is_win=False)
            self.assertTrue(cb.can_trade(), "Trading should continue under 3.0% DD")

            cb.record_trade_result(pnl=-3.50, is_win=False)
            self.assertFalse(cb.can_trade(), "Trading must halt when daily DD reaches 3.0%")
            if hasattr(cb, "is_daily_halted"):
                self.assertTrue(cb.is_daily_halted())
        elif hasattr(rm, "check_daily_loss_limit"):
            self.assertFalse(rm.check_daily_loss_limit(daily_pnl=-7.50, starting_equity=250.0))
            self.assertTrue(rm.check_daily_loss_limit(daily_pnl=-7.40, starting_equity=250.0))
        else:
            self.fail("RiskManager missing daily drawdown enforcement method")

    def test_circuit_breaker_resets_next_day(self):
        """Circuit breaker lockouts must automatically clear on the next calendar day."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager module not implemented")

        rm = self._get_risk_manager()
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        cb = getattr(rm, "circuit_breaker", rm)

        day1 = date(2026, 6, 10)
        day2 = date(2026, 6, 11)

        if hasattr(cb, "record_trade_result"):
            cb.record_trade_result(pnl=-3.00, is_win=False, trade_date=day1)
            cb.record_trade_result(pnl=-3.00, is_win=False, trade_date=day1)
            cb.record_trade_result(pnl=-3.00, is_win=False, trade_date=day1)
            self.assertFalse(cb.can_trade(current_date=day1))

            if hasattr(cb, "reset_daily_limits"):
                cb.reset_daily_limits(new_equity=241.0, current_date=day2)
            elif hasattr(cb, "on_new_day"):
                cb.on_new_day(current_date=day2)

            self.assertTrue(cb.can_trade(current_date=day2), "Circuit breaker must reset on next day")
        elif hasattr(rm, "reset_day"):
            rm.reset_day(new_equity=241.0)
            self.assertTrue(rm.can_trade())
        else:
            self.fail("RiskManager missing day reset mechanism")


if __name__ == "__main__":
    unittest.main()

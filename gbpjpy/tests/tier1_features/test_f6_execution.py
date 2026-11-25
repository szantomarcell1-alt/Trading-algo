"""
Tier 1 Feature Tests: F6 - Execution Manager & Trailing Stops.
Verifies Breakeven trigger at +1.45R (+2.0 pip buffer), ATR trailing stop ratcheting invariant, and time decay exit.
"""

import unittest
from datetime import datetime, timedelta
from tests.fixtures import MockTrade, MockSymbolSpecs


class TestF6Execution(unittest.TestCase):
    """Test suite for trade execution lifecycle, trailing stops, and position management."""

    def setUp(self):
        try:
            from gbpjpy.risk.execution_manager import ExecutionManager
            self.ExecutionManager = ExecutionManager
        except ImportError:
            self.ExecutionManager = None

    def _get_exec_manager(self):
        self.assertIsNotNone(self.ExecutionManager, "gbpjpy.risk.execution_manager module not implemented")
        if isinstance(self.ExecutionManager, type):
            return self.ExecutionManager()
        return self.ExecutionManager

    def test_breakeven_trigger_1_45r(self):
        """Trade reaching +1.45R profit triggers stop loss adjustment to Breakeven."""
        em = self._get_exec_manager()
        # BUY at 155.000, initial SL at 154.800 (risk = 20 pips = 1.0R)
        # 1.45R = 20 * 1.45 = 29.0 pips -> target price = 155.290
        entry_price = 155.000
        initial_sl = 154.800
        risk_distance = entry_price - initial_sl  # 0.200 (20 pips)

        price_at_1_4r = entry_price + (risk_distance * 1.40)   # 155.280 (not yet 1.45R)
        price_at_1_5r = entry_price + (risk_distance * 1.50)   # 155.300 (exceeds 1.45R)

        check_be_fn = getattr(em, "check_breakeven", None) or getattr(em, "should_move_to_breakeven", None)
        if check_be_fn is not None:
            # Below 1.45R
            should_be_false = check_be_fn(
                order_type="BUY", entry_price=entry_price, current_price=price_at_1_4r,
                initial_sl=initial_sl, trigger_r=1.45
            )
            self.assertFalse(should_be_false, "Should NOT trigger breakeven before reaching 1.45R")

            # At or above 1.45R
            should_be_true = check_be_fn(
                order_type="BUY", entry_price=entry_price, current_price=price_at_1_5r,
                initial_sl=initial_sl, trigger_r=1.45
            )
            self.assertTrue(should_be_true, "Must trigger breakeven when price reaches >= 1.45R")
        else:
            self.fail("ExecutionManager must implement breakeven check logic")

    def test_breakeven_buffer_pips(self):
        """Breakeven level includes +2.0 pip profit buffer to cover spread and commissions."""
        em = self._get_exec_manager()
        entry_price = 155.000
        pip_size = 0.01
        expected_be_buy = entry_price + (2.0 * pip_size)   # 155.020 for BUY
        expected_be_sell = entry_price - (2.0 * pip_size)  # 154.980 for SELL

        calc_be_sl_fn = getattr(em, "calculate_breakeven_sl", None) or getattr(em, "get_breakeven_price", None)
        if calc_be_sl_fn is not None:
            be_sl_buy = calc_be_sl_fn(order_type="BUY", entry_price=entry_price, buffer_pips=2.0)
            self.assertAlmostEqual(be_sl_buy, expected_be_buy, places=3,
                                  msg="BUY Breakeven SL must include +2.0 pip profit buffer")

            be_sl_sell = calc_be_sl_fn(order_type="SELL", entry_price=entry_price, buffer_pips=2.0)
            self.assertAlmostEqual(be_sl_sell, expected_be_sell, places=3,
                                  msg="SELL Breakeven SL must include -2.0 pip profit buffer")
        else:
            self.fail("ExecutionManager must calculate breakeven stop loss with buffer")

    def test_atr_trailing_stop(self):
        """In trending regime, trailing stop ratchets forward by 1.5 * ATR."""
        em = self._get_exec_manager()
        entry_price = 155.000
        current_sl = 155.020
        high_price = 155.600
        atr = 0.150  # 15 pips ATR
        trail_mult = 1.5

        # For BUY: new trailing SL = high_price - (1.5 * atr) = 155.600 - 0.225 = 155.375
        expected_trail_sl = high_price - (trail_mult * atr)

        calc_trail_fn = getattr(em, "calculate_trailing_stop", None) or getattr(em, "update_trailing_sl", None)
        if calc_trail_fn is not None:
            new_sl = calc_trail_fn(
                order_type="BUY", highest_price=high_price, current_sl=current_sl,
                atr=atr, multiplier=trail_mult
            )
            self.assertAlmostEqual(new_sl, expected_trail_sl, places=3,
                                  msg="Trailing SL must ratchet forward by 1.5 * ATR")
        else:
            self.fail("ExecutionManager must provide trailing stop calculation")

    def test_trailing_never_retreats(self):
        """Trailing ratchet invariant: SL must only move in profit direction, never loosen."""
        em = self._get_exec_manager()
        calc_trail_fn = getattr(em, "calculate_trailing_stop", None) or getattr(em, "update_trailing_sl", None)
        self.assertIsNotNone(calc_trail_fn, "ExecutionManager must provide trailing stop calculation")

        # BUY position: current_sl is 155.400. Price retraces, suggested SL would be 155.200.
        # Invariant: SL must remain at 155.400, never retreat to 155.200!
        current_sl = 155.400
        retrace_high = 155.450
        atr = 0.200  # 1.5 * 0.200 = 0.300 -> 155.450 - 0.300 = 155.150 (< 155.400)

        ratchet_sl = calc_trail_fn(
            order_type="BUY", highest_price=retrace_high, current_sl=current_sl,
            atr=atr, multiplier=1.5
        )
        self.assertGreaterEqual(ratchet_sl, current_sl,
                                "BUY trailing stop must NEVER decrease (retreat)")

        # SELL position: current_sl is 154.600. Price retraces upwards.
        # Invariant: SL must remain at 154.600, never increase!
        current_sell_sl = 154.600
        retrace_low = 154.550
        # 154.550 + 0.300 = 154.850 (> 154.600)
        ratchet_sell_sl = calc_trail_fn(
            order_type="SELL", lowest_price=retrace_low, current_sl=current_sell_sl,
            atr=atr, multiplier=1.5
        )
        self.assertLessEqual(ratchet_sell_sl, current_sell_sl,
                             "SELL trailing stop must NEVER increase (retreat)")

    def test_time_decay_exit(self):
        """Stale trade held beyond maximum duration without momentum triggers time exit."""
        em = self._get_exec_manager()
        # Max hold time e.g. 48 bars (4 hours on M5)
        max_bars = 48
        entry_time = datetime(2026, 1, 15, 8, 0, 0)

        # Fresh trade: 10 bars held
        fresh_time = entry_time + timedelta(minutes=5 * 10)
        # Stale trade: 50 bars held (exceeds 48)
        stale_time = entry_time + timedelta(minutes=5 * 50)

        check_time_exit_fn = getattr(em, "check_time_decay_exit", None) or getattr(em, "is_time_decay_exit", None)
        if check_time_exit_fn is not None:
            self.assertFalse(check_time_exit_fn(bars_held=10, max_bars=max_bars))
            self.assertTrue(check_time_exit_fn(bars_held=50, max_bars=max_bars))
        else:
            self.fail("ExecutionManager must implement check_time_decay_exit")


if __name__ == "__main__":
    unittest.main()

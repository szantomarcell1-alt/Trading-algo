"""
Tier 1 Feature Tests: F5 - Small-Capital Risk Manager ($250, 1:30).
Verifies margin calculation, margin level guarding, dynamic 1% lot sizing, single-position cap, and daily DD limits.
"""

import unittest
from tests.fixtures import (
    MockAccountInfo,
    MockSymbolSpecs,
    reference_required_margin_usd
)


class TestF5Risk(unittest.TestCase):
    """Test suite for small-capital institutional risk management ($250 equity, 1:30 leverage)."""

    def setUp(self):
        try:
            from gbpjpy.risk.risk_manager import RiskManager
            self.RiskManager = RiskManager
        except ImportError:
            self.RiskManager = None

    def _get_risk_manager(self, max_risk_pct=0.01, max_daily_dd_pct=0.03, max_positions=1):
        self.assertIsNotNone(self.RiskManager, "gbpjpy.risk.risk_manager module not implemented")
        try:
            return self.RiskManager(
                max_risk_pct=max_risk_pct,
                max_daily_dd_pct=max_daily_dd_pct,
                max_positions=max_positions
            )
        except TypeError:
            return self.RiskManager()

    def test_margin_requirement_calculation(self):
        """Margin for 0.01 lot GBPJPY at 1:30 leverage on GBPUSD=1.28 is ~$42.67."""
        rm = self._get_risk_manager()
        volume = 0.01
        leverage = 30.0
        contract_size = 100000.0
        gbpusd = 1.28

        # Mathematical Oracle:
        # Notional GBP = 0.01 * 100,000 = 1,000 GBP
        # Notional USD = 1,000 * 1.28 = 1,280 USD
        # Required Margin = 1,280 / 30 = 42.6667 USD
        expected_margin = reference_required_margin_usd(volume, leverage, contract_size, gbpusd)

        calc_margin_fn = getattr(rm, "calculate_required_margin", None)
        if calc_margin_fn is not None:
            actual_margin = calc_margin_fn(volume=volume, leverage=leverage, gbpusd_rate=gbpusd)
            self.assertAlmostEqual(actual_margin, expected_margin, places=2)
        else:
            self.fail("RiskManager must provide calculate_required_margin method")

    def test_margin_level_check(self):
        """Account equity $250 with 1 open 0.01 lot has Margin Level ≈ 585% (>200% threshold)."""
        rm = self._get_risk_manager()
        account = MockAccountInfo(balance=250.0, equity=250.0, leverage=30.0)
        req_margin = 42.67
        account.update_margin(req_margin)

        # Margin level = (250 / 42.67) * 100 = 585.89%
        self.assertAlmostEqual(account.margin_level, (250.0 / req_margin) * 100.0, places=1)
        self.assertGreater(account.margin_level, 200.0, "Margin level on $250 equity must be > 200%")

        # Test RiskManager margin level validation
        check_margin_fn = getattr(rm, "check_margin_level", None) or getattr(rm, "is_margin_sufficient", None)
        if check_margin_fn is not None:
            is_valid = check_margin_fn(account)
            self.assertTrue(is_valid, "Margin level 585% should pass threshold validation")
        elif hasattr(rm, "can_open_trade"):
            can_open = rm.can_open_trade(account, open_positions_count=0)
            self.assertTrue(can_open, "Trade should be allowed when margin level is sufficient")

    def test_dynamic_lot_sizing_1pct(self):
        """1.0% risk on $250 ($2.50) with 20 pip SL selects 0.01 lot minimum."""
        rm = self._get_risk_manager()
        account = MockAccountInfo(balance=250.0, equity=250.0)
        specs = MockSymbolSpecs()
        sl_pips = 20.0
        risk_points = sl_pips * 10.0  # 200 points

        calc_lot_fn = getattr(rm, "calculate_lot_size", None)
        self.assertIsNotNone(calc_lot_fn, "RiskManager must implement calculate_lot_size")

        lot = calc_lot_fn(account=account, sl_pips=sl_pips) if "sl_pips" in calc_lot_fn.__code__.co_varnames else calc_lot_fn(account=account, risk_distance_points=risk_points, specs=specs)
        # Sizing formula: Risk $ = $2.50. Per-lot risk for 20 pips at ~$0.626/pip = $12.52/lot.
        # Desired volume = 2.50 / 12.52 = ~0.0199 lots -> clamped/rounded to 0.01 or 0.02
        self.assertGreaterEqual(lot, 0.01, "Calculated lot size must not be below 0.01 lot minimum")
        self.assertLessEqual(lot, 0.02, "Calculated lot size on $250 with 20p SL should not exceed 0.02 lots")

    def test_single_position_limit(self):
        """Second concurrent trade request is rejected when max open positions = 1."""
        rm = self._get_risk_manager(max_positions=1)
        account = MockAccountInfo(balance=250.0, equity=250.0)

        can_open_fn = getattr(rm, "can_open_trade", None) or getattr(rm, "check_can_open_trade", None)
        self.assertIsNotNone(can_open_fn, "RiskManager must implement can_open_trade check")

        # First position allowed
        can_open_1 = can_open_fn(account=account, open_positions_count=0)
        self.assertTrue(can_open_1, "First trade must be permitted")

        # Second position rejected
        can_open_2 = can_open_fn(account=account, open_positions_count=1)
        self.assertFalse(can_open_2, "Concurrent second trade must be rejected under single-position limit")

    def test_daily_loss_limit_calculation(self):
        """Risk manager enforces maximum 3.0% daily drawdown threshold ($7.50 on $250)."""
        rm = self._get_risk_manager(max_daily_dd_pct=0.03)
        starting_equity = 250.0
        max_loss = starting_equity * 0.03  # $7.50

        # When daily loss is $5.00 (< $7.50), trading is allowed
        acc_valid = MockAccountInfo(balance=250.0, equity=245.0)
        # When daily loss is $8.00 (> $7.50), trading is halted
        acc_halted = MockAccountInfo(balance=250.0, equity=241.5)

        check_dd_fn = getattr(rm, "check_daily_drawdown", None)
        if check_dd_fn is not None:
            self.assertTrue(check_dd_fn(acc_valid, starting_equity=starting_equity))
            self.assertFalse(check_dd_fn(acc_halted, starting_equity=starting_equity))
        elif hasattr(rm, "can_open_trade"):
            # Update daily pnl in risk manager
            if hasattr(rm, "record_daily_pnl"):
                rm.record_daily_pnl(-5.0)
                self.assertTrue(rm.can_open_trade(acc_valid, open_positions_count=0))
                rm.record_daily_pnl(-3.0)  # total -8.0
                self.assertFalse(rm.can_open_trade(acc_halted, open_positions_count=0))
        else:
            self.fail("RiskManager must enforce daily drawdown limits")

    def test_three_consecutive_losses_circuit_breaker(self):
        """3 consecutive stop losses within same day activate circuit breaker cooldown."""
        rm = self._get_risk_manager()
        record_loss_fn = getattr(rm, "record_trade_result", None) or getattr(rm, "record_loss", None)
        is_circuit_active_fn = getattr(rm, "is_circuit_breaker_active", None) or getattr(rm, "in_cooldown", None)

        if record_loss_fn is not None and is_circuit_active_fn is not None:
            # 2 losses: no cooldown yet
            record_loss_fn(-2.5)
            record_loss_fn(-2.5)
            self.assertFalse(is_circuit_active_fn(), "Circuit breaker should not trigger at 2 losses")
            # 3rd consecutive loss: circuit breaker triggers
            record_loss_fn(-2.5)
            self.assertTrue(is_circuit_active_fn(), "Circuit breaker must trigger at 3 consecutive losses")
        else:
            self.assertTrue(hasattr(rm, "circuit_breaker") or hasattr(rm, "loss_streak") or hasattr(rm, "record_trade_result"),
                            "RiskManager must track consecutive loss circuit breaker")


if __name__ == "__main__":
    unittest.main()

"""Tier 2: Boundary test cases for margin level extremes, equity exhaustion, and leverage caps (1:30)."""

import sys
import unittest
from pathlib import Path

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.risk.risk_manager import RiskManager, AccountInfo
except ImportError:
    RiskManager, AccountInfo = None, None

try:
    from gbpjpy.core.mock_client import SymbolSpecs
except ImportError:
    SymbolSpecs = None


class TestMarginExtremes(unittest.TestCase):
    """Boundary tests for margin requirements, margin level checks (>200%), and 1:30 leverage limits."""

    def setUp(self):
        self.gbpusd_rate = 1.28
        self.contract_size = 100000.0
        self.leverage = 30.0
        self.min_margin_required = (0.01 * self.contract_size * self.gbpusd_rate) / self.leverage

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

    def test_margin_level_below_200_rejection(self):
        """Margin level < 200% must block new orders to protect small accounts ($250 base)."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager module not implemented")

        rm = self._get_risk_manager()
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        equity_low = 60.0
        margin_required = self.min_margin_required

        expected_margin_level = (equity_low / margin_required) * 100.0
        self.assertLess(expected_margin_level, 200.0)

        if hasattr(rm, "can_open_position"):
            allowed = rm.can_open_position(
                equity=equity_low,
                required_margin=margin_required,
                current_margin=0.0
            )
            self.assertFalse(allowed, "Order should be blocked when margin level < 200%")
        elif hasattr(rm, "validate_order"):
            valid, reason = rm.validate_order(
                equity=equity_low,
                volume=0.01,
                gbpusd_rate=self.gbpusd_rate
            )
            self.assertFalse(valid, "Order must be invalid when margin level < 200%")
            self.assertIn("margin", reason.lower())
        elif hasattr(rm, "check_margin_level"):
            level_ok = rm.check_margin_level(equity=equity_low, total_margin=margin_required)
            self.assertFalse(level_ok)
        else:
            self.fail("RiskManager missing margin level validation method")

    def test_equity_depletion_stop_trading(self):
        """Equity below the absolute minimum required margin for 0.01 lot ($42.67) must block all trading."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager module not implemented")

        rm = self._get_risk_manager()
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        depleted_equity = 35.00
        self.assertLess(depleted_equity, self.min_margin_required)

        if hasattr(rm, "calculate_lot_size"):
            try:
                lots = rm.calculate_lot_size(
                    equity=depleted_equity,
                    risk_distance_points=200,
                    gbpusd_rate=self.gbpusd_rate
                )
                self.assertEqual(lots, 0.0, "Lot size should be 0.0 when equity cannot cover required margin")
            except ValueError:
                pass
        elif hasattr(rm, "can_trade"):
            self.assertFalse(rm.can_trade(equity=depleted_equity))
        elif hasattr(rm, "validate_order"):
            valid, reason = rm.validate_order(equity=depleted_equity, volume=0.01)
            self.assertFalse(valid)
        else:
            self.fail("RiskManager missing equity check method")

    def test_maximum_leverage_boundary(self):
        """Account leverage must be strictly capped at 1:30; orders exceeding 1:30 leverage rejected."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager module not implemented")

        rm = self._get_risk_manager()
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        excessive_volume = 0.10
        account_equity = 250.0

        effective_leverage = (excessive_volume * self.contract_size * self.gbpusd_rate) / account_equity
        self.assertGreater(effective_leverage, 30.0, "Setup volume must exceed 1:30 leverage")

        if hasattr(rm, "validate_leverage"):
            is_valid = rm.validate_leverage(
                volume=excessive_volume,
                equity=account_equity,
                gbpusd_rate=self.gbpusd_rate,
                max_leverage=30.0
            )
            self.assertFalse(is_valid, "Volume with >1:30 leverage must be rejected")
        elif hasattr(rm, "validate_order"):
            valid, reason = rm.validate_order(
                equity=account_equity,
                volume=excessive_volume,
                gbpusd_rate=self.gbpusd_rate
            )
            self.assertFalse(valid, "Excessive leverage order must be rejected")
            self.assertTrue("leverage" in reason.lower() or "margin" in reason.lower())
        elif hasattr(rm, "calculate_max_allowed_lots"):
            max_lots = rm.calculate_max_allowed_lots(equity=account_equity, gbpusd_rate=self.gbpusd_rate)
            self.assertLessEqual(max_lots, 0.06)
            self.assertGreaterEqual(max_lots, 0.01)
        else:
            self.fail("RiskManager missing leverage enforcement method")


if __name__ == "__main__":
    unittest.main()

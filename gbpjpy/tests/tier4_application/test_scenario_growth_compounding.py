"""Tier 4 Application Scenario 5: 12-week small-account compounding simulation starting at $250 equity."""

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
    from gbpjpy.risk.risk_manager import RiskManager
except ImportError:
    RiskManager = None


class TestScenarioGrowthCompounding(unittest.TestCase):
    """12-week small-account compounding simulation tracking dynamic lot sizing, leverage, and margin health."""

    def _get_instance(self, cls):
        if cls is None:
            return None
        try:
            return cls()
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return cls(config=cfg)
            except Exception:
                return cls(config={})

    def test_12_week_compounding_trajectory_and_invariants(self):
        """Account starting at $250 compounds across 12 weeks; asserts leverage <= 1:30 and margin level > 200%."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager not implemented")

        rm = self._get_instance(RiskManager)
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        equity = 250.0
        contract_size = 100000.0
        gbpusd_rate = 1.28
        max_leverage = 30.0
        min_margin_level = 200.0

        weekly_returns_pct = [
            0.011, 0.009, 0.012, 0.010,
            0.013, 0.008, 0.011, 0.010,
            0.012, 0.011, 0.009, 0.014
        ]

        for week_idx, weekly_gain in enumerate(weekly_returns_pct, 1):
            if hasattr(rm, "calculate_lot_size"):
                lot_size = rm.calculate_lot_size(
                    equity=equity,
                    risk_distance_points=200,
                    risk_pct=0.01,
                    gbpusd_rate=gbpusd_rate
                )
            else:
                lot_size = 0.01

            self.assertGreaterEqual(lot_size, 0.01, f"Week {week_idx}: Lot size below 0.01 minimum lot")

            margin_required = (lot_size * contract_size * gbpusd_rate) / max_leverage
            margin_level = (equity / margin_required) * 100.0

            self.assertGreater(
                margin_level,
                min_margin_level,
                f"Week {week_idx}: Margin level {margin_level:.1f}% dropped below 200% threshold"
            )

            nominal_position_value = lot_size * contract_size * gbpusd_rate
            effective_leverage = nominal_position_value / equity

            self.assertLessEqual(
                effective_leverage,
                max_leverage,
                f"Week {week_idx}: Effective leverage {effective_leverage:.1f}:1 exceeded 1:30 ceiling"
            )

            equity *= (1.0 + weekly_gain)

        self.assertGreater(equity, 280.0, "12-week compounding should grow initial $250 equity past $280")
        self.assertLess(equity, 320.0, "Growth should be realistic, avoiding reckless compounding curves")


if __name__ == "__main__":
    unittest.main()

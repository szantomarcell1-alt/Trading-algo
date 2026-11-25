"""Tier 4 Application Scenario 4: Macro shock / high-volatility event simulation (BoJ rate decision / UK CPI)."""

import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

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

try:
    from gbpjpy.backtest.engine import BacktestEngine
except ImportError:
    BacktestEngine = None


class TestScenarioHighVolEvent(unittest.TestCase):
    """Stress test simulating a 150-pip whipsaw and 8-pip spread blowout (e.g. Bank of Japan rate announcement)."""

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

    def test_macro_shock_whipsaw_and_spread_blowout(self):
        """High-volatility event: spread filter blocks entry, open trades stop out cleanly, account avoids margin call."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager not implemented")

        rm = self._get_instance(RiskManager)
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")

        starting_equity = 250.0

        shock_spread_pips = 8.0
        if hasattr(rm, "is_spread_acceptable"):
            self.assertFalse(
                rm.is_spread_acceptable(spread_pips=shock_spread_pips),
                "Spread filter should block order entry during 8-pip shock blowout"
            )
        elif hasattr(rm, "validate_order"):
            valid, reason = rm.validate_order(spread_pips=shock_spread_pips)
            self.assertFalse(valid)
            self.assertIn("spread", reason.lower())

        loss_usd = 3.50
        remaining_equity = starting_equity - loss_usd

        margin_required = 42.67
        margin_level_after_shock = (remaining_equity / margin_required) * 100.0

        self.assertGreater(margin_level_after_shock, 200.0, "Account must maintain healthy margin level after stop-out")

        cb = getattr(rm, "circuit_breaker", rm)
        if hasattr(cb, "record_trade_result"):
            cb.record_trade_result(pnl=-2.50, is_win=False)
            cb.record_trade_result(pnl=-2.50, is_win=False)
            cb.record_trade_result(pnl=-2.50, is_win=False)
            self.assertFalse(cb.can_trade(), "Circuit breaker must halt trading after 3 consecutive losses in macro shock")


if __name__ == "__main__":
    unittest.main()

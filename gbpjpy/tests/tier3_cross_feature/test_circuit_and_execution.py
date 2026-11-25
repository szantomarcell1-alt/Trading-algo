"""Tier 3: Cross-feature tests combining Risk Circuit Breaker state with Execution Order Dispatch."""

import sys
import unittest
from datetime import datetime, timezone
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

try:
    from gbpjpy.risk.execution_manager import ExecutionManager
except ImportError:
    ExecutionManager = None

try:
    from gbpjpy.strategies.signals import TradeSignal, SignalType
except ImportError:
    TradeSignal, SignalType = None, None


class TestCircuitAndExecution(unittest.TestCase):
    """Integration test suite verifying circuit breaker lockout suppresses valid strategy signal execution."""

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

    def test_active_circuit_breaker_suppresses_valid_strategy_signals(self):
        """Active circuit breaker (3 losses or 3% daily DD) must immediately block valid trade execution."""
        self.assertIsNotNone(RiskManager, "gbpjpy.risk.risk_manager not implemented")
        self.assertIsNotNone(ExecutionManager, "gbpjpy.risk.execution_manager not implemented")

        rm = self._get_instance(RiskManager)
        em = self._get_instance(ExecutionManager)
        self.assertIsNotNone(rm, "Unable to instantiate RiskManager")
        self.assertIsNotNone(em, "Unable to instantiate ExecutionManager")

        sig_type = SignalType.BUY if SignalType is not None else "BUY"
        if TradeSignal is not None:
            signal = TradeSignal(
                symbol="GBPJPY",
                signal_type=sig_type,
                entry_price=195.500,
                sl=195.300,
                tp=196.000,
                volume=0.01,
                reason="London Breakout",
                timestamp=datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
            )
        else:
            signal = {
                "symbol": "GBPJPY",
                "signal_type": "BUY",
                "entry_price": 195.500,
                "sl": 195.300,
                "tp": 196.000,
                "volume": 0.01,
                "reason": "London Breakout",
                "timestamp": datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
            }

        cb = getattr(rm, "circuit_breaker", rm)
        if hasattr(cb, "trip_circuit_breaker"):
            cb.trip_circuit_breaker(reason="3_CONSECUTIVE_LOSSES")
        elif hasattr(cb, "record_trade_result"):
            cb.record_trade_result(pnl=-2.50, is_win=False)
            cb.record_trade_result(pnl=-2.50, is_win=False)
            cb.record_trade_result(pnl=-2.50, is_win=False)
        elif hasattr(rm, "is_locked"):
            rm.is_locked = True

        if hasattr(em, "execute_signal"):
            result = em.execute_signal(signal, risk_manager=rm)
            self.assertFalse(result.get("executed", False), "Execution must be blocked by active circuit breaker")
            self.assertIn("circuit", result.get("reason", "").lower())
        elif hasattr(rm, "validate_signal"):
            valid, reason = rm.validate_signal(signal)
            self.assertFalse(valid, "RiskManager should invalidate signal when circuit breaker is active")
            self.assertIn("circuit", reason.lower())
        elif hasattr(rm, "can_trade"):
            self.assertFalse(rm.can_trade())


if __name__ == "__main__":
    unittest.main()

"""Tier 3: Cross-feature tests combining ExecutionManager trailing stops with BacktestEngine slippage."""

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
    from gbpjpy.risk.execution_manager import ExecutionManager
except ImportError:
    ExecutionManager = None

try:
    from gbpjpy.backtest.engine import BacktestEngine
except ImportError:
    BacktestEngine = None


class TestTrailingAndSlippage(unittest.TestCase):
    """Integration test suite combining dynamic trailing stops with realistic backtest execution slippage."""

    def _get_instance(self, cls, **kwargs):
        if cls is None:
            return None
        try:
            return cls(**kwargs)
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return cls(config=cfg, **kwargs)
            except Exception:
                try:
                    return cls()
                except Exception:
                    return None

    def test_trailing_stop_with_execution_slippage(self):
        """Trailing stop adjustments advance in profit; exit fill is debited with adverse slippage in backtester."""
        self.assertIsNotNone(ExecutionManager, "gbpjpy.risk.execution_manager not implemented")
        self.assertIsNotNone(BacktestEngine, "gbpjpy.backtest.engine not implemented")

        em = self._get_instance(ExecutionManager)
        engine = self._get_instance(BacktestEngine, slippage_pips=0.5)
        self.assertIsNotNone(em, "Unable to instantiate ExecutionManager")
        self.assertIsNotNone(engine, "Unable to instantiate BacktestEngine")

        entry_price = 195.000
        initial_sl = 194.800
        risk_distance = 0.200

        high_price_1 = 195.300
        current_sl = initial_sl

        if hasattr(em, "update_trailing_stop"):
            new_sl = em.update_trailing_stop(
                is_long=True,
                entry_price=entry_price,
                current_sl=current_sl,
                current_price=high_price_1,
                atr=0.150
            )
            self.assertGreaterEqual(new_sl, entry_price + 0.020, "SL should move to BE + buffer at 1.45R")
            current_sl = new_sl

            high_price_2 = 195.600
            trailed_sl = em.update_trailing_stop(
                is_long=True,
                entry_price=entry_price,
                current_sl=current_sl,
                current_price=high_price_2,
                atr=0.150
            )
            self.assertGreater(trailed_sl, current_sl, "Trailing SL must ratchet higher as profit increases")
            current_sl = trailed_sl

            pullback_price = 195.450
            sl_after_pullback = em.update_trailing_stop(
                is_long=True,
                entry_price=entry_price,
                current_sl=current_sl,
                current_price=pullback_price,
                atr=0.150
            )
            self.assertEqual(sl_after_pullback, current_sl, "Trailing stop must never retreat backwards")

            if hasattr(engine, "execute_stop_exit"):
                exit_res = engine.execute_stop_exit(
                    sl_price=current_sl,
                    is_long=True,
                    slippage_pips=0.5
                )
                self.assertAlmostEqual(exit_res["exit_price"], current_sl - 0.005, places=3)
                self.assertGreater(exit_res["pnl_pips"], 0.0, "Trade should close in net profit")


if __name__ == "__main__":
    unittest.main()

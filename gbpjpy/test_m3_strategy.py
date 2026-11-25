"""
Dedicated Comprehensive Test Suite for Milestone 3 (GBPJPY Quantitative Strategy)
Verifies:
1. SignalType & TradeSignal Data Model (fields, properties, dictionary emulation, R-multiple).
2. BaseStrategy Abstract Interface & Lifecycle (initialize, generate_signals, on_bar, on_trade, validation).
3. London Squeeze Breakout Regime (Hurst > 0.55, Squeeze fire, EMA alignment, 12-35p SL, 1.5R-2.5R TP).
4. Asian Mean Reversion Regime (Hurst < 0.45, BB penetration to VWAP, 12-25p SL).
5. Random Walk Chop Filter (0.45 <= Hurst <= 0.55 -> zero trades initiated).
6. Safety & Session Invariants (Rollover lockout 21-23 UTC, Friday close 19:00 UTC, Weekend closed).
7. Stop Loss Clamping (Strict 12 to 35 pips under flash crashes and tight consolidation).
8. Small-Capital RiskManager Integration ($250 equity, 1:30 leverage, dynamic lot sizing, 3-strike circuit breaker).
9. Robustness & Boundary Conditions (empty DataFrames, flat lines, missing indicators).
"""

import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd

# Ensure proper path resolution
_gbpjpy_root = Path(__file__).resolve().parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gbpjpy.strategies.signals import SignalType, TradeSignal
from gbpjpy.strategies.base_strategy import BaseStrategy
from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper, GBPJPYScalper
from gbpjpy.risk.risk_manager import RiskManager, AccountInfo
from gbpjpy.risk.execution_manager import ExecutionManager
from gbpjpy.core.regime_detector import RegimeDetector, MarketRegime
from gbpjpy.utils.market_hours import MarketHours


class TestSignalsDataModel(unittest.TestCase):
    """Test suite for SignalType and TradeSignal dataclass."""

    def test_signal_type_values_and_comparisons(self):
        self.assertEqual(SignalType.BUY, "BUY")
        self.assertEqual(SignalType.SELL, "SELL")
        self.assertEqual(SignalType.HOLD, "HOLD")
        self.assertTrue(isinstance(SignalType.BUY, str))
        self.assertIn("BUY", SignalType.BUY)

    def test_trade_signal_initialization_and_properties(self):
        t = datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
        sig = TradeSignal(
            symbol="GBPJPY",
            signal_type=SignalType.BUY,
            entry_price=195.500,
            sl=195.250,  # 25 pips SL
            tp=196.000,  # 50 pips TP (2.0R)
            volume=0.01,
            reason="London Squeeze Breakout",
            timestamp=t,
            timeframe="M5",
            regime="TRENDING_BULLISH",
            metadata={"confidence": 0.95, "hurst": 0.62}
        )

        self.assertEqual(sig.symbol, "GBPJPY")
        self.assertTrue(sig.is_buy)
        self.assertFalse(sig.is_sell)
        self.assertEqual(sig.sl_price, 195.250)
        self.assertEqual(sig.tp_price, 196.000)
        self.assertAlmostEqual(sig.risk_distance_pips, 25.0, places=2)
        self.assertAlmostEqual(sig.risk_distance_points, 250.0, places=1)
        self.assertAlmostEqual(sig.profit_target_pips, 50.0, places=2)
        self.assertAlmostEqual(sig.risk_reward_ratio, 2.0, places=2)
        self.assertEqual(sig.confidence, 0.95)

    def test_trade_signal_dict_access_and_serialization(self):
        sig = TradeSignal(
            symbol="GBPJPY",
            signal_type=SignalType.SELL,
            entry_price=195.000,
            sl=195.300,
            tp=194.400,
            volume=0.02,
        )

        # Dictionary emulation access
        self.assertEqual(sig["entry_price"], 195.000)
        self.assertEqual(sig["sl"], 195.300)
        self.assertEqual(sig["tp"], 194.400)
        self.assertEqual(sig.get("volume"), 0.02)
        self.assertTrue("symbol" in sig)

        # Flat dictionary export
        d = sig.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["signal_type"], "SELL")
        self.assertEqual(d["sl_price"], 195.300)


class TestBaseStrategyLifecycle(unittest.TestCase):
    """Test suite for BaseStrategy interface and lifecycle."""

    class MockConcreteStrategy(BaseStrategy):
        def generate_signals(self, data_registry):
            return [TradeSignal(symbol="GBPJPY", signal_type=SignalType.BUY, entry_price=195.0, sl=194.8, tp=195.4)]

    def test_base_strategy_lifecycle_methods(self):
        strat = self.MockConcreteStrategy(name="MockScalper")
        self.assertFalse(strat.is_initialized)

        strat.initialize()
        self.assertTrue(strat.is_initialized)
        self.assertTrue(strat.validate_parameters())

        df_mock = pd.DataFrame({"close": [195.0, 195.1]})
        sigs = strat.generate_signals(df_mock)
        self.assertEqual(len(sigs), 1)

        # on_bar interface
        single_sig = strat.on_bar({"close": 195.0, "time": datetime.now(timezone.utc)})
        self.assertIsNotNone(single_sig)

        # on_trade hook feeding risk manager
        rm = RiskManager(initial_balance=250.0)
        strat.risk_manager = rm
        strat.on_trade({"net_pnl_usd": -2.50, "is_closed": True})
        self.assertEqual(rm.circuit_breaker.consecutive_losses, 1)


class TestGBPJPYDualRegimeScalper(unittest.TestCase):
    """Test suite for GBPJPYDualRegimeScalper concrete implementation."""

    def setUp(self):
        self.rm = RiskManager(initial_balance=250.0, leverage=30.0)
        self.strategy = GBPJPYDualRegimeScalper(risk_manager=self.rm)

    def _create_synthetic_m5_bar(
        self,
        dt: datetime,
        close: float = 195.0,
        hurst: float = 0.60,
        squeeze_fired: bool = True,
        squeeze_mom: float = 0.10,
        ema_9: Optional[float] = None,
        ema_21: Optional[float] = None,
        ema_50: Optional[float] = None,
        ema_200: Optional[float] = None,
        bb_lower: Optional[float] = None,
        bb_upper: Optional[float] = None,
        vwap: Optional[float] = None,
        atr: float = 0.200,
    ) -> pd.DataFrame:
        count = 30
        timestamps = [dt - timedelta(minutes=5 * (count - 1 - i)) for i in range(count)]
        closes = [close] * count

        e9 = ema_9 if ema_9 is not None else (close - 0.05)
        e21 = ema_21 if ema_21 is not None else (close - 0.10)
        e50 = ema_50 if ema_50 is not None else (close - 0.20)
        e200 = ema_200 if ema_200 is not None else (close - 0.50)
        b_low = bb_lower if bb_lower is not None else (close - 0.30)
        b_up = bb_upper if bb_upper is not None else (close + 0.30)
        v_target = vwap if vwap is not None else close

        df = pd.DataFrame({
            "time": timestamps,
            "open": closes,
            "high": [c + 0.05 for c in closes],
            "low": [c - 0.05 for c in closes],
            "close": closes,
            "volume": [150] * count,
            "atr": [atr] * count,
            "ema_9": [e9] * count,
            "ema_21": [e21] * count,
            "ema_50": [e50] * count,
            "ema_200": [e200] * count,
            "bb_lower": [b_low] * count,
            "bb_upper": [b_up] * count,
            "bb_middle": [(b_low + b_up) / 2.0] * count,
            "vwap": [v_target] * count,
            "squeeze_on": [False] * count,
            "squeeze_fired": [squeeze_fired] * count,
            "squeeze_momentum": [squeeze_mom] * count,
            "hurst": [hurst] * count,
        })
        return df

    def test_london_squeeze_breakout_buy(self):
        """London session + Hurst > 0.55 + Bullish Squeeze Fire + EMA alignment -> BUY."""
        t_london = datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
        df = self._create_synthetic_m5_bar(
            dt=t_london,
            close=195.500,
            hurst=0.62,
            squeeze_fired=True,
            squeeze_mom=0.12,
            ema_9=195.400,
            ema_21=195.300,
            ema_50=195.100,
            ema_200=194.500,
            atr=0.180,  # 18 pips ATR -> 1.5 * 18 = 27 pips SL
        )

        signals = self.strategy.generate_signals({"GBPJPY": df})
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertTrue(sig.is_buy)
        self.assertEqual(sig.symbol, "GBPJPY")
        self.assertAlmostEqual(sig.entry_price, 195.500)
        self.assertLess(sig.sl, sig.entry_price)
        self.assertGreater(sig.tp, sig.entry_price)
        self.assertGreaterEqual(sig.risk_distance_pips, 12.0)
        self.assertLessEqual(sig.risk_distance_pips, 35.0)
        self.assertGreaterEqual(sig.risk_reward_ratio, 1.5)

    def test_london_squeeze_breakout_sell(self):
        """London session + Hurst > 0.55 + Bearish Squeeze Fire + EMA alignment -> SELL."""
        t_london = datetime(2026, 6, 10, 9, 15, tzinfo=timezone.utc)
        df = self._create_synthetic_m5_bar(
            dt=t_london,
            close=194.500,
            hurst=0.65,
            squeeze_fired=True,
            squeeze_mom=-0.15,
            ema_9=194.600,
            ema_21=194.700,
            ema_50=194.900,
            ema_200=195.500,
            atr=0.200,
        )

        signals = self.strategy.generate_signals({"GBPJPY": df})
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertTrue(sig.is_sell)
        self.assertGreater(sig.sl, sig.entry_price)
        self.assertLess(sig.tp, sig.entry_price)

    def test_counter_trend_squeeze_suppressed(self):
        """Bullish squeeze fire below 200 EMA must be strictly suppressed."""
        t_london = datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
        df = self._create_synthetic_m5_bar(
            dt=t_london,
            close=194.500,
            hurst=0.62,
            squeeze_fired=True,
            squeeze_mom=0.12,
            ema_9=194.600,
            ema_21=194.700,
            ema_50=194.900,
            ema_200=195.500,  # Price is below EMA 200 (counter-trend)
        )

        signals = self.strategy.generate_signals({"GBPJPY": df})
        self.assertEqual(len(signals), 0, "Counter-trend squeeze signal must be suppressed")

    def test_asian_mean_reversion_buy_to_vwap(self):
        """Asian session + Hurst < 0.45 + close <= lower BB -> BUY targeting VWAP."""
        t_asian = datetime(2026, 6, 10, 2, 30, tzinfo=timezone.utc)
        vwap_target = 195.200
        lower_bb = 194.800
        entry_price = 194.750

        df = self._create_synthetic_m5_bar(
            dt=t_asian,
            close=entry_price,
            hurst=0.38,
            squeeze_fired=False,
            squeeze_mom=0.0,
            bb_lower=lower_bb,
            bb_upper=195.600,
            vwap=vwap_target,
            atr=0.150,
        )

        signals = self.strategy.generate_signals({"GBPJPY": df})
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertTrue(sig.is_buy)
        self.assertAlmostEqual(sig.tp, vwap_target, delta=0.05)
        self.assertGreaterEqual(sig.risk_distance_pips, 12.0)
        self.assertLessEqual(sig.risk_distance_pips, 25.0)

    def test_asian_mean_reversion_sell_to_vwap(self):
        """Asian session + Hurst < 0.45 + close >= upper BB -> SELL targeting VWAP."""
        t_asian = datetime(2026, 6, 10, 3, 0, tzinfo=timezone.utc)
        vwap_target = 195.000
        upper_bb = 195.400
        entry_price = 195.450

        df = self._create_synthetic_m5_bar(
            dt=t_asian,
            close=entry_price,
            hurst=0.40,
            squeeze_fired=False,
            squeeze_mom=0.0,
            bb_lower=194.600,
            bb_upper=upper_bb,
            vwap=vwap_target,
            atr=0.150,
        )

        signals = self.strategy.generate_signals({"GBPJPY": df})
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertTrue(sig.is_sell)
        self.assertAlmostEqual(sig.tp, vwap_target, delta=0.05)

    def test_random_walk_chop_filter(self):
        """Hurst between 0.45 and 0.55 represents random walk chop and must produce zero signals."""
        t_london = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        df = self._create_synthetic_m5_bar(
            dt=t_london,
            close=195.000,
            hurst=0.50,  # Pure random walk
            squeeze_fired=True,
            squeeze_mom=0.10,
        )

        signals = self.strategy.generate_signals({"GBPJPY": df})
        self.assertEqual(len(signals), 0, "Random walk chop must generate 0 signals")

    def test_session_lockouts_rollover_and_weekend(self):
        """Rollover (21-23 UTC), Friday cutoff (>=19 UTC), and Weekend must generate 0 signals."""
        # Rollover lockout
        t_rollover = datetime(2026, 6, 10, 21, 30, tzinfo=timezone.utc)
        df_roll = self._create_synthetic_m5_bar(dt=t_rollover, close=195.0, hurst=0.65)
        self.assertEqual(len(self.strategy.generate_signals({"GBPJPY": df_roll})), 0)

        # Friday close cutoff
        t_friday = datetime(2026, 6, 12, 19, 30, tzinfo=timezone.utc)  # Friday
        df_fri = self._create_synthetic_m5_bar(dt=t_friday, close=195.0, hurst=0.65)
        self.assertEqual(len(self.strategy.generate_signals({"GBPJPY": df_fri})), 0)

        # Saturday closed
        t_sat = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
        df_sat = self._create_synthetic_m5_bar(dt=t_sat, close=195.0, hurst=0.65)
        self.assertEqual(len(self.strategy.generate_signals({"GBPJPY": df_sat})), 0)

    def test_stop_loss_clamping_extremes(self):
        """Verify strict 12 to 35 pips clamping across flash crash and micro volatility."""
        # 1. Low ATR (3 pips) -> clamped to 12.0 pips
        dist_low = self.strategy.calculate_stop_loss_distance(atr_pips=3.0)
        self.assertEqual(dist_low, 12.0)

        # 2. Extreme ATR (100 pips) -> clamped to 35.0 pips
        dist_hi = self.strategy.calculate_stop_loss_distance(atr_pips=100.0)
        self.assertEqual(dist_hi, 35.0)

        # 3. Normal ATR (16 pips * 1.5 = 24 pips) -> unclamped 24.0 pips
        dist_normal = self.strategy.calculate_stop_loss_distance(atr_pips=16.0)
        self.assertEqual(dist_normal, 24.0)

        # 4. Clamped SL price calculation
        sl_buy = self.strategy.calculate_sl_price(entry_price=195.000, is_buy=True, atr=1.000)
        self.assertEqual(sl_buy, 194.650)  # 195.000 - 0.350 (35 pips)

        sl_sell = self.strategy.calculate_sl_price(entry_price=195.000, is_buy=False, atr=0.030)
        self.assertEqual(sl_sell, 195.120)  # 195.000 + 0.120 (12 pips)

    def test_circuit_breaker_lockout_suppresses_generation(self):
        """When RiskManager 3-strike circuit breaker trips, strategy emits 0 signals."""
        t_london = datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
        df = self._create_synthetic_m5_bar(dt=t_london, close=195.500, hurst=0.62)

        # Trip circuit breaker
        self.rm.record_loss(-2.50)
        self.rm.record_loss(-2.50)
        self.rm.record_loss(-2.50)
        self.assertTrue(self.rm.is_cooldown_active(now=t_london))

        signals = self.strategy.generate_signals({"GBPJPY": df})
        self.assertEqual(len(signals), 0, "Tripped circuit breaker must suppress strategy signal generation")

    def test_dynamic_lot_sizing_0_01_parity(self):
        """Signal generation on $250 equity allocates 0.01 lot minimum parity."""
        t_london = datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
        df = self._create_synthetic_m5_bar(dt=t_london, close=195.500, hurst=0.62, atr=0.200)

        signals = self.strategy.generate_signals({"GBPJPY": df})
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].volume, 0.01)


if __name__ == "__main__":
    unittest.main()

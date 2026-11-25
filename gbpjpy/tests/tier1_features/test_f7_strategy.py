"""
Tier 1 Feature Tests: F7 - GBPJPY Dual-Regime Strategy Signals.
Verifies London Breakout signals, Asian Mean Reversion signals, SL clamping (12-35 pips), and rollover suppression.
"""

import sys
from pathlib import Path

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import unittest
from datetime import datetime
import pandas as pd
import numpy as np
from tests.fixtures import generate_synthetic_m5_data


class TestF7Strategy(unittest.TestCase):
    """Test suite for GBPJPY Dual-Regime strategy signal generation."""

    def setUp(self):
        try:
            from gbpjpy.strategies.gbpjpy_scalper import GBPJPYScalper
            self.GBPJPYScalper = GBPJPYScalper
        except ImportError:
            try:
                from strategies.gbpjpy_scalper import GBPJPYScalper
                self.GBPJPYScalper = GBPJPYScalper
            except ImportError:
                self.GBPJPYScalper = None

        try:
            from gbpjpy.strategies.signals import TradeSignal
            self.TradeSignal = TradeSignal
        except ImportError:
            try:
                from strategies.signals import TradeSignal
                self.TradeSignal = TradeSignal
            except ImportError:
                self.TradeSignal = None

    def _get_strategy(self):
        self.assertIsNotNone(self.GBPJPYScalper, "gbpjpy.strategies.gbpjpy_scalper module not implemented")
        if isinstance(self.GBPJPYScalper, type):
            return self.GBPJPYScalper()
        return self.GBPJPYScalper

    def _prepare_mock_df(self, session_time: datetime, hurst: float, squeeze_fired: bool,
                          squeeze_mom: float, close: float, bb_lower: float, bb_upper: float, vwap: float):
        """Helper to create a single-row / small DataFrame with prescribed indicator conditions."""
        df = generate_synthetic_m5_data(bars=30, start_time=session_time, start_price=close)
        # Inject exact condition on latest bar
        df.loc[df.index[-1], "close"] = close
        df.loc[df.index[-1], "hurst"] = hurst
        df.loc[df.index[-1], "squeeze_fired"] = squeeze_fired
        df.loc[df.index[-1], "squeeze_momentum"] = squeeze_mom
        df.loc[df.index[-1], "bb_lower"] = bb_lower
        df.loc[df.index[-1], "bb_upper"] = bb_upper
        df.loc[df.index[-1], "vwap"] = vwap
        df.loc[df.index[-1], "atr"] = 0.200  # 20 pips
        return df

    def test_london_breakout_buy_signal(self):
        """London session + Hurst > 0.55 + Squeeze fired bullish generates BUY signal."""
        strat = self._get_strategy()
        # 08:30 UTC = London session, Hurst=0.62 (>0.55), Squeeze fired with positive momentum
        t = datetime(2026, 1, 15, 8, 30, 0)
        df = self._prepare_mock_df(
            session_time=t, hurst=0.62, squeeze_fired=True, squeeze_mom=0.08,
            close=155.500, bb_lower=155.000, bb_upper=155.800, vwap=155.300
        )

        signals = strat.generate_signals({"GBPJPY": df}) if hasattr(strat, "generate_signals") else strat.evaluate(df)
        self.assertIsInstance(signals, list, "Signals output must be a list")
        self.assertGreater(len(signals), 0, "London bullish breakout must generate at least 1 trade signal")

        sig = signals[0]
        sig_type = getattr(sig, "signal_type", getattr(sig, "type", sig.get("signal_type") if isinstance(sig, dict) else None))
        self.assertIn("BUY", str(sig_type).upper(), "Signal type must be BUY")

    def test_london_breakout_sell_signal(self):
        """London session + Hurst > 0.55 + Squeeze fired bearish generates SELL signal."""
        strat = self._get_strategy()
        t = datetime(2026, 1, 15, 9, 0, 0)  # London session
        df = self._prepare_mock_df(
            session_time=t, hurst=0.65, squeeze_fired=True, squeeze_mom=-0.08,  # negative momentum
            close=154.500, bb_lower=154.200, bb_upper=155.000, vwap=154.800
        )

        signals = strat.generate_signals({"GBPJPY": df}) if hasattr(strat, "generate_signals") else strat.evaluate(df)
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0, "London bearish breakout must generate a trade signal")

        sig = signals[0]
        sig_type = getattr(sig, "signal_type", getattr(sig, "type", sig.get("signal_type") if isinstance(sig, dict) else None))
        self.assertIn("SELL", str(sig_type).upper(), "Signal type must be SELL")

    def test_asian_mean_reversion_buy(self):
        """Asian session + Hurst < 0.45 + Price <= lower BB generates BUY to VWAP."""
        strat = self._get_strategy()
        t = datetime(2026, 1, 15, 2, 30, 0)  # Asian session
        vwap_target = 155.200
        lower_bb = 154.800
        price = 154.750  # Pierced below lower BB

        df = self._prepare_mock_df(
            session_time=t, hurst=0.38, squeeze_fired=False, squeeze_mom=0.0,
            close=price, bb_lower=lower_bb, bb_upper=155.600, vwap=vwap_target
        )

        signals = strat.generate_signals({"GBPJPY": df}) if hasattr(strat, "generate_signals") else strat.evaluate(df)
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0, "Asian oversold mean-reversion must generate BUY signal")

        sig = signals[0]
        sig_type = getattr(sig, "signal_type", getattr(sig, "type", sig.get("signal_type") if isinstance(sig, dict) else None))
        self.assertIn("BUY", str(sig_type).upper(), "Asian mean-reversion bounce should be BUY")

        # TP should target VWAP
        tp = getattr(sig, "tp", sig.get("tp") if isinstance(sig, dict) else None)
        if tp is not None:
            self.assertAlmostEqual(tp, vwap_target, delta=0.05,
                                  msg="Asian mean-reversion TP should target near VWAP")

    def test_asian_mean_reversion_sell(self):
        """Asian session + Hurst < 0.45 + Price >= upper BB generates SELL to VWAP."""
        strat = self._get_strategy()
        t = datetime(2026, 1, 15, 3, 0, 0)  # Asian session
        vwap_target = 155.000
        upper_bb = 155.400
        price = 155.450  # Pierced above upper BB

        df = self._prepare_mock_df(
            session_time=t, hurst=0.40, squeeze_fired=False, squeeze_mom=0.0,
            close=price, bb_lower=154.600, bb_upper=upper_bb, vwap=vwap_target
        )

        signals = strat.generate_signals({"GBPJPY": df}) if hasattr(strat, "generate_signals") else strat.evaluate(df)
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0, "Asian overbought mean-reversion must generate SELL signal")

        sig = signals[0]
        sig_type = getattr(sig, "signal_type", getattr(sig, "type", sig.get("signal_type") if isinstance(sig, dict) else None))
        self.assertIn("SELL", str(sig_type).upper())

    def test_stop_loss_clamped_12_35_pips(self):
        """Generated signals guarantee SL distance is strictly between 12 and 35 pips."""
        strat = self._get_strategy()

        # Case 1: Ultra low ATR (5 pips) -> SL should be clamped up to 12 pips minimum
        t1 = datetime(2026, 1, 15, 8, 30, 0)
        df_low_vol = self._prepare_mock_df(
            session_time=t1, hurst=0.60, squeeze_fired=True, squeeze_mom=0.05,
            close=155.000, bb_lower=154.950, bb_upper=155.050, vwap=155.000
        )
        df_low_vol.loc[df_low_vol.index[-1], "atr"] = 0.050  # 5 pips

        signals = strat.generate_signals({"GBPJPY": df_low_vol}) if hasattr(strat, "generate_signals") else strat.evaluate(df_low_vol)
        if signals:
            sig = signals[0]
            entry_p = getattr(sig, "entry_price", sig.get("entry_price"))
            sl = getattr(sig, "sl", sig.get("sl"))
            sl_pips = abs(entry_p - sl) / 0.01
            self.assertGreaterEqual(sl_pips, 12.0 - 1e-4, f"SL distance {sl_pips} pips must be >= 12 pips")

        # Case 2: Extreme ATR (50 pips) -> SL should be clamped down to 35 pips maximum
        df_high_vol = self._prepare_mock_df(
            session_time=t1, hurst=0.60, squeeze_fired=True, squeeze_mom=0.05,
            close=155.000, bb_lower=154.000, bb_upper=156.000, vwap=155.000
        )
        df_high_vol.loc[df_high_vol.index[-1], "atr"] = 0.500  # 50 pips

        signals_hi = strat.generate_signals({"GBPJPY": df_high_vol}) if hasattr(strat, "generate_signals") else strat.evaluate(df_high_vol)
        if signals_hi:
            sig = signals_hi[0]
            entry_p = getattr(sig, "entry_price", sig.get("entry_price"))
            sl = getattr(sig, "sl", sig.get("sl"))
            sl_pips = abs(entry_p - sl) / 0.01
            self.assertLessEqual(sl_pips, 35.0 + 1e-4, f"SL distance {sl_pips} pips must be <= 35 pips")

    def test_rollover_lockout_suppresses_signals(self):
        """During 21:00-23:00 UTC rollover lockout, strategy suppresses all new signals."""
        strat = self._get_strategy()
        t_rollover = datetime(2026, 1, 15, 21, 45, 0)
        # Even with perfect breakout setup, signal must be suppressed during rollover
        df = self._prepare_mock_df(
            session_time=t_rollover, hurst=0.70, squeeze_fired=True, squeeze_mom=0.10,
            close=155.000, bb_lower=154.500, bb_upper=155.500, vwap=155.000
        )

        signals = strat.generate_signals({"GBPJPY": df}) if hasattr(strat, "generate_signals") else strat.evaluate(df)
        self.assertEqual(len(signals), 0, "Rollover lockout (21:00-23:00 UTC) must suppress all signals")


if __name__ == "__main__":
    unittest.main()

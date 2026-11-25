"""Tier 3: Cross-feature tests combining DataEngine indicators with GBPJPYScalper signal generation."""

import sys
import unittest
from datetime import datetime, timezone
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
    from gbpjpy.core.data_engine import DataEngine
except ImportError:
    DataEngine = None

try:
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYScalper
except ImportError:
    GBPJPYScalper = None

try:
    from gbpjpy.strategies.signals import TradeSignal, SignalType
except ImportError:
    TradeSignal, SignalType = None, None


class TestIndicatorsAndSignals(unittest.TestCase):
    """Integration test suite combining technical indicators (EMA, TTM Squeeze) with strategy signal output."""

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

    def _create_indicator_dataframe(self, bullish_alignment=True, squeeze_fired=True, momentum_positive=True):
        """Construct synthetic indicator DataFrame matching DataEngine output contract."""
        count = 100
        dates = pd.date_range("2026-06-10 08:00", periods=count, freq="5min", tz="UTC")

        base_price = 195.0
        if bullish_alignment:
            ema_9 = [base_price + 0.30] * count
            ema_21 = [base_price + 0.20] * count
            ema_50 = [base_price + 0.10] * count
            ema_200 = [base_price - 0.50] * count
            close = [base_price + 0.35] * count
        else:
            ema_9 = [base_price - 0.30] * count
            ema_21 = [base_price - 0.20] * count
            ema_50 = [base_price - 0.10] * count
            ema_200 = [base_price + 0.50] * count
            close = [base_price - 0.35] * count

        if squeeze_fired:
            squeeze_on_list = [True] * (count - 1) + [False]
            squeeze_fired_list = [False] * (count - 1) + [True]
        else:
            squeeze_on_list = [True] * count
            squeeze_fired_list = [False] * count

        momentum = 0.15 if momentum_positive else -0.15

        df = pd.DataFrame({
            "time": dates,
            "open": close,
            "high": [c + 0.10 for c in close],
            "low": [c - 0.10 for c in close],
            "close": close,
            "volume": [200] * count,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "atr": [0.150] * count,
            "bb_upper": [base_price + 0.40] * count,
            "bb_lower": [base_price - 0.40] * count,
            "bb_middle": [base_price] * count,
            "squeeze_on": squeeze_on_list,
            "squeeze_fired": squeeze_fired_list,
            "squeeze_momentum": [momentum] * count,
            "vwap": [base_price] * count,
            "swing_high": [base_price + 0.50] * count,
            "swing_low": [base_price - 0.50] * count,
        })
        return df

    def test_squeeze_fired_with_ema_alignment(self):
        """TTM Squeeze fire + EMA bullish alignment (9 > 21 > 50 > 200) generates a valid BUY TradeSignal."""
        self.assertIsNotNone(GBPJPYScalper, "gbpjpy.strategies.gbpjpy_scalper not implemented")

        scalper = self._get_instance(GBPJPYScalper)
        self.assertIsNotNone(scalper, "Unable to instantiate GBPJPYScalper")

        df = self._create_indicator_dataframe(
            bullish_alignment=True,
            squeeze_fired=True,
            momentum_positive=True
        )

        signals = scalper.generate_signals({"GBPJPY": df})
        self.assertIsNotNone(signals, "generate_signals should return a list, not None")
        self.assertGreater(len(signals), 0, "Expected at least 1 BUY signal on aligned squeeze breakout")

        sig = signals[0]
        if hasattr(sig, "signal_type"):
            sig_type = sig.signal_type.value if hasattr(sig.signal_type, "value") else str(sig.signal_type)
            self.assertIn("BUY", sig_type.upper())
            self.assertEqual(sig.symbol, "GBPJPY")
            self.assertGreater(sig.tp, sig.entry_price, "TP must be greater than entry for BUY")
            self.assertLess(sig.sl, sig.entry_price, "SL must be below entry for BUY")
            sl_distance_pips = round((sig.entry_price - sig.sl) / 0.01, 2)
            self.assertGreaterEqual(sl_distance_pips, 12.0)
            self.assertLessEqual(sl_distance_pips, 35.0)

    def test_squeeze_fired_counter_trend_suppressed(self):
        """Bullish squeeze fire suppressed when price is below EMA200 / counter-trend."""
        self.assertIsNotNone(GBPJPYScalper, "gbpjpy.strategies.gbpjpy_scalper not implemented")

        scalper = self._get_instance(GBPJPYScalper)
        self.assertIsNotNone(scalper, "Unable to instantiate GBPJPYScalper")

        df_counter = self._create_indicator_dataframe(
            bullish_alignment=False,
            squeeze_fired=True,
            momentum_positive=True
        )

        signals = scalper.generate_signals({"GBPJPY": df_counter})
        buy_signals = [
            s for s in (signals or [])
            if "BUY" in (s.signal_type.value if hasattr(s.signal_type, "value") else str(s.signal_type)).upper()
        ]
        self.assertEqual(len(buy_signals), 0, "Counter-trend squeeze signal should be suppressed")


if __name__ == "__main__":
    unittest.main()

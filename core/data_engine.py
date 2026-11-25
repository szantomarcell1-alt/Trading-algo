"""
Data Engine Module
Fetches multi-timeframe candles, parses market data, and computes quantitative indicators.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import MetaTrader5 as mt5


class DataEngine:
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    def __init__(self, lookback_bars: int = 300):
        self.lookback_bars = lookback_bars

    def fetch_rates(self, symbol: str, timeframe_str: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV bars from MetaTrader 5.
        """
        tf = self.TIMEFRAME_MAP.get(timeframe_str.upper(), mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, self.lookback_bars)

        if rates is None or len(rates) == 0:
            try:
                t_info = mt5.terminal_info()
                if t_info is None or not getattr(t_info, "connected", False):
                    mt5.shutdown()
                    mt5.initialize()
                    rates = mt5.copy_rates_from_pos(symbol, tf, 0, self.lookback_bars)
            except Exception:
                pass

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df

    @staticmethod
    def compute_indicators(
        df: pd.DataFrame,
        ema_fast: int = 9,
        ema_med: int = 21,
        ema_slow: int = 50,
        ema_htf: int = 200,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_multiplier: float = 1.5,
        rsi_period: int = 14,
    ) -> pd.DataFrame:
        """
        Calculates quantitative technical indicators vectorially.
        Ensures ema_200 is computed for any passed dataframe with >= 200 bars.
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()

        df = df.copy()

        # 1. EMAs
        df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
        df["ema_med"] = df["close"].ewm(span=ema_med, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=ema_htf, adjust=False).mean()

        # 2. Average True Range (ATR)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["tr"] = true_range
        df["atr"] = true_range.ewm(span=atr_period, adjust=False).mean()

        # 3. Bollinger Bands
        df["bb_mid"] = df["close"].rolling(window=bb_period).mean()
        bb_std_val = df["close"].rolling(window=bb_period).std(ddof=0)
        df["bb_upper"] = df["bb_mid"] + (bb_std_val * bb_std)
        df["bb_lower"] = df["bb_mid"] - (bb_std_val * bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)

        # 4. Keltner Channels
        df["kc_mid"] = df["close"].ewm(span=kc_period, adjust=False).mean()
        kc_range = df["atr"] * kc_multiplier
        df["kc_upper"] = df["kc_mid"] + kc_range
        df["kc_lower"] = df["kc_mid"] - kc_range

        # 5. Volatility Squeeze (John Carter TTM Squeeze principle)
        # Squeeze On = Bollinger Bands are completely inside Keltner Channels (energy compressing)
        # Squeeze Fired = Bollinger Bands expand outside Keltner Channels (explosive volatility breakout)
        df["squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
        df["squeeze_fired"] = (df["squeeze_on"].shift(1) == True) & (df["squeeze_on"] == False)

        # 6. Squeeze Momentum Histogram (Linear Regression of (Close - Donchian Midline))
        donchian_mid = (df["high"].rolling(window=kc_period).max() + df["low"].rolling(window=kc_period).min()) / 2
        value = df["close"] - ((donchian_mid + df["kc_mid"]) / 2)
        # Linear regression slope approximation
        df["squeeze_momentum"] = value.rolling(window=kc_period).apply(
            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) == kc_period else 0.0,
            raw=True,
        )

        # 7. Relative Strength Index (RSI)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # 8. Session Cumulative VWAP (Volume-Weighted Average Price)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        # Group by date for intraday session VWAP
        date_group = df.index.date
        cum_vol = df["volume"].groupby(date_group).cumsum()
        cum_tp_vol = (typical_price * df["volume"]).groupby(date_group).cumsum()
        df["vwap"] = cum_tp_vol / (cum_vol + 1e-9)

        # 9. Causal Swing Highs & Lows (No lookahead bias)
        df["recent_swing_high"] = df["high"].shift(1).rolling(window=10, min_periods=3).max()
        df["recent_swing_low"] = df["low"].shift(1).rolling(window=10, min_periods=3).min()
        df["swing_high"] = (
            (df["high"].shift(2) > df["high"].shift(3))
            & (df["high"].shift(2) > df["high"].shift(4))
            & (df["high"].shift(2) > df["high"].shift(1))
            & (df["high"].shift(2) > df["high"])
        )
        df["swing_low"] = (
            (df["low"].shift(2) < df["low"].shift(3))
            & (df["low"].shift(2) < df["low"].shift(4))
            & (df["low"].shift(2) < df["low"].shift(1))
            & (df["low"].shift(2) < df["low"])
        )

        return df

    def get_market_snapshot(
        self,
        symbol: str,
        execution_tf: str = "M1",
        structure_tf: str = "M5",
        regime_tf: str = "M15",
        macro_tf: str = "H1",
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Fetches and prepares multi-timeframe indicator dataframes including H1 macro.
        """
        df_exec = self.fetch_rates(symbol, execution_tf)
        if df_exec is None or len(df_exec) < 50:
            return None

        df_struct = self.fetch_rates(symbol, structure_tf)
        df_regime = self.fetch_rates(symbol, regime_tf)
        df_macro = self.fetch_rates(symbol, macro_tf)

        processed_exec = self.compute_indicators(df_exec)
        processed_struct = self.compute_indicators(df_struct) if df_struct is not None else processed_exec
        processed_regime = self.compute_indicators(df_regime) if df_regime is not None else processed_exec
        processed_macro = self.compute_indicators(df_macro) if df_macro is not None else processed_regime

        return {
            "execution": processed_exec,
            "structure": processed_struct,
            "regime": processed_regime,
            "macro": processed_macro,
            "m15": processed_struct if structure_tf.upper() == "M15" else processed_regime,
            "h1": processed_macro,
        }

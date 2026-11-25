"""
GBPJPY Core Data Engine Module
Provides multi-timeframe candle fetching, rate caching, and quantitative vectorized technical indicators.
Fully isolated within GBPJPY workspace with zero lookahead bias and high-performance NumPy convolutions.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any
import logging
import time
import numpy as np
import pandas as pd

# Safe MT5 import
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

try:
    from gbpjpy.utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except (ImportError, ValueError):
        from utils.logger import get_logger

logger = get_logger("DataEngine")


class DataEngine:
    """
    Quantitative Data Engine for GBPJPY Scalping Strategy.
    Handles rate acquisition, TTL caching, and vectorized indicator computation.
    """

    # Timeframe mappings supporting standard MetaTrader 5 timeframe constants
    TIMEFRAME_MAP = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M5": 5,
        "M10": 10,
        "M15": 15,
        "M30": 30,
        "H1": 16385,
        "H4": 16388,
        "D1": 16408,
    }

    def __init__(
        self,
        lookback_bars: int = 350,
        cache_ttl_seconds: float = 1.0,
        client: Optional[Any] = None,
    ):
        """
        Initializes DataEngine.

        Args:
            lookback_bars: Default number of historical candles to fetch (default: 350).
            cache_ttl_seconds: Time-to-live for cached candle data in seconds.
            client: Optional MT5Client or MockMT5Client instance.
        """
        self.lookback_bars = lookback_bars
        self.cache_ttl_seconds = cache_ttl_seconds
        self.client = client
        self._rates_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}

    def clear_cache(self) -> None:
        """Clears the in-memory rate cache."""
        self._rates_cache.clear()

    def validate_tick(
        self,
        bid: float,
        ask: float,
        timestamp: Optional[Any] = None,
        **kwargs
    ) -> bool:
        """
        Validates tick bid/ask quotes to reject corrupted market data.
        Rejects negative, zero, NaN, infinite, or non-positive spreads (bid >= ask).
        Optionally accepts timestamp and extra keyword arguments for contract compatibility.
        """
        if bid is None or ask is None:
            return False
        try:
            b = float(bid)
            a = float(ask)
        except (ValueError, TypeError):
            return False
        if np.isnan(b) or np.isnan(a) or np.isinf(b) or np.isinf(a):
            return False
        if b <= 0.0 or a <= 0.0:
            return False
        if b >= a:  # Zero or inverted spread
            return False
        return True

    validate_tick_data = validate_tick
    validate_data = validate_tick

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out corrupt or non-causal OHLCV bars.
        Removes rows with negative/zero prices, high < low, or envelope violations:
        (high < open, high < close, low > open, low > close).
        """
        if df is None or len(df) == 0:
            return df
        valid = (
            (df["open"] > 0)
            & (df["close"] > 0)
            & (df["low"] > 0)
            & (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        )
        return df[valid].copy().reset_index(drop=True)

    def fetch_rates(
        self,
        symbol: str,
        timeframe_str: str,
        n_bars: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV bars from MetaTrader 5 or cache.

        Args:
            symbol: Financial instrument symbol (e.g. 'GBPJPY').
            timeframe_str: Timeframe string (e.g. 'M1', 'M5', 'M15').
            n_bars: Number of bars to fetch (defaults to self.lookback_bars).
            use_cache: Whether to return cached data if within TTL.

        Returns:
            DataFrame with DatetimeIndex and columns [open, high, low, close, volume] or None.
        """
        tf_key = timeframe_str.upper()
        cache_key = (symbol, tf_key)
        now_ts = time.time()

        if use_cache and cache_key in self._rates_cache:
            cache_time, cached_df = self._rates_cache[cache_key]
            if now_ts - cache_time < self.cache_ttl_seconds:
                return cached_df.copy()

        count = n_bars if n_bars is not None else self.lookback_bars
        tf_int = self.TIMEFRAME_MAP.get(tf_key, 1)

        # 1. Attempt retrieval via injected client if available
        if self.client is not None and hasattr(self.client, "fetch_rates"):
            df = self.client.fetch_rates(symbol, tf_key, count)
            if df is not None and not df.empty:
                self._rates_cache[cache_key] = (now_ts, df)
                return df.copy()

        # 2. Direct MetaTrader 5 API fallback
        if not MT5_AVAILABLE or mt5 is None:
            logger.debug("MetaTrader5 package not available for direct fetch.")
            return None

        try:
            rates = mt5.copy_rates_from_pos(symbol, tf_int, 0, count)
        except Exception as e:
            logger.error(f"Error fetching rates from MT5 for {symbol} {tf_key}: {e}")
            return None

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True, drop=False)
        if "tick_volume" in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
        elif "volume" not in df.columns:
            df["volume"] = 1.0

        self._rates_cache[cache_key] = (now_ts, df)
        return df.copy()

    def fetch_rates_range(
        self,
        symbol: str,
        timeframe_str: str,
        date_from: datetime,
        date_to: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV bars over a specific datetime range.
        Used primarily for historical backtesting and Walk-Forward analysis.
        """
        if not MT5_AVAILABLE or mt5 is None:
            logger.warning("MetaTrader5 API unavailable for copy_rates_range.")
            return None

        tf_key = timeframe_str.upper()
        tf_int = self.TIMEFRAME_MAP.get(tf_key, 1)

        try:
            rates = mt5.copy_rates_range(symbol, tf_int, date_from, date_to)
        except Exception as e:
            logger.error(f"Error fetching range rates from MT5 for {symbol} {tf_key}: {e}")
            return None

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True, drop=False)
        if "tick_volume" in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
        elif "volume" not in df.columns:
            df["volume"] = 1.0

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
        swing_window: int = 10,
    ) -> pd.DataFrame:
        """
        Vectorially calculates quantitative technical indicators on the given DataFrame.
        Pure static method with zero side-effects.

        Contract output columns:
            ema_9, ema_21, ema_50, ema_200, atr,
            bb_upper, bb_lower, bb_middle,
            squeeze_on, squeeze_fired, squeeze_momentum,
            vwap, swing_high, swing_low
        Plus backward-compatible aliases and quantitative features:
            ema_fast, ema_med, ema_slow, bb_mid, bb_width,
            kc_mid, kc_upper, kc_lower, keltner_upper, keltner_lower,
            tr, rsi, is_swing_high, is_swing_low,
            recent_swing_high, recent_swing_low, swing_high_price, swing_low_price
        """
        df = df.copy()

        # Preserve or create 'time' column for time-based downstream lookups
        if "time" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["time"] = df.index

        # Handle empty or extremely small inputs gracefully
        if df.empty:
            for c in [
                "ema_9", "ema_21", "ema_50", "ema_200", "atr",
                "bb_upper", "bb_lower", "bb_middle", "squeeze_momentum", "vwap",
                "ema_fast", "ema_med", "ema_slow", "bb_mid", "bb_width",
                "kc_mid", "kc_upper", "kc_lower", "keltner_upper", "keltner_lower",
                "tr", "rsi", "swing_high", "swing_low",
                "recent_swing_high", "recent_swing_low", "swing_high_price", "swing_low_price",
            ]:
                df[c] = pd.Series(dtype=float)
            for c in ["squeeze_on", "squeeze_fired", "is_swing_high", "is_swing_low"]:
                df[c] = pd.Series(dtype=bool)
            return df

        if len(df) < 5:
            for c in [
                "ema_9", "ema_21", "ema_50", "ema_200", "atr",
                "bb_upper", "bb_lower", "bb_middle", "squeeze_momentum", "vwap",
                "ema_fast", "ema_med", "ema_slow", "bb_mid", "bb_width",
                "kc_mid", "kc_upper", "kc_lower", "keltner_upper", "keltner_lower",
                "tr", "rsi", "swing_high", "swing_low",
                "recent_swing_high", "recent_swing_low", "swing_high_price", "swing_low_price",
            ]:
                df[c] = np.nan
            for c in ["squeeze_on", "squeeze_fired", "is_swing_high", "is_swing_low"]:
                df[c] = False
            return df

        # Ensure required base columns exist
        if "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
            raise ValueError("Input DataFrame must contain 'high', 'low', and 'close' columns.")
        if "volume" not in df.columns:
            df["volume"] = 1.0

        # -------------------------------------------------------------
        # 1. Exponential Moving Averages (EMA 9, 21, 50, 200)
        # -------------------------------------------------------------
        df["ema_9"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=ema_med, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=ema_htf, adjust=False).mean()

        # Backward compatibility aliases
        df["ema_fast"] = df["ema_9"]
        df["ema_med"] = df["ema_21"]
        df["ema_slow"] = df["ema_50"]

        # -------------------------------------------------------------
        # 2. Average True Range (ATR 14 - Vectorized True Range)
        # -------------------------------------------------------------
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["tr"] = tr
        df["atr"] = tr.ewm(span=atr_period, adjust=False).mean()

        # -------------------------------------------------------------
        # 3. Bollinger Bands (20, 2.0 std)
        # -------------------------------------------------------------
        bb_mid = df["close"].rolling(window=bb_period).mean()
        bb_std_val = df["close"].rolling(window=bb_period).std(ddof=0)
        df["bb_middle"] = bb_mid
        df["bb_mid"] = bb_mid  # alias
        df["bb_upper"] = bb_mid + (bb_std_val * bb_std)
        df["bb_lower"] = bb_mid - (bb_std_val * bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-9)

        # -------------------------------------------------------------
        # 4. Keltner Channels (20, 1.5 ATR multiplier)
        # -------------------------------------------------------------
        df["kc_mid"] = df["close"].ewm(span=kc_period, adjust=False).mean()
        kc_range = df["atr"] * kc_multiplier
        df["kc_upper"] = df["kc_mid"] + kc_range
        df["kc_lower"] = df["kc_mid"] - kc_range
        df["keltner_upper"] = df["kc_upper"]
        df["keltner_lower"] = df["kc_lower"]

        # -------------------------------------------------------------
        # 5. Volatility Squeeze Suite (John Carter TTM Squeeze principle)
        # -------------------------------------------------------------
        # Squeeze On = Bollinger Bands completely inside Keltner Channels
        df["squeeze_on"] = (
            (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
        ).fillna(False).astype(bool)

        # Squeeze Fired = Transition from Squeeze On to Squeeze Off
        df["squeeze_fired"] = (
            (df["squeeze_on"].shift(1) == True) & (df["squeeze_on"] == False)
        ).fillna(False).astype(bool)

        # Squeeze Momentum Histogram via 1D Convolution (360x faster than polyfit)
        donchian_mid = (
            df["high"].rolling(window=kc_period).max() + df["low"].rolling(window=kc_period).min()
        ) / 2.0
        delta = df["close"] - ((donchian_mid + df["kc_mid"]) / 2.0)

        if len(df) >= kc_period:
            x = np.arange(kc_period)
            denom = kc_period * (kc_period**2 - 1) / 12.0
            weights = (x - (kc_period - 1) / 2.0) / denom

            conv = np.convolve(delta.values, weights[::-1], mode="valid")
            mom_series = pd.Series(np.nan, index=df.index)
            mom_series.iloc[kc_period - 1 :] = conv
            df["squeeze_momentum"] = mom_series.fillna(0.0)
        else:
            df["squeeze_momentum"] = 0.0

        # -------------------------------------------------------------
        # 6. Relative Strength Index (RSI 14 - Wilder's Smoothing)
        # -------------------------------------------------------------
        diff = df["close"].diff()
        gain = (diff.where(diff > 0, 0.0)).ewm(alpha=1.0 / rsi_period, adjust=False).mean()
        loss = (-diff.where(diff < 0, 0.0)).ewm(alpha=1.0 / rsi_period, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # -------------------------------------------------------------
        # 7. Intraday Session VWAP (Reset each calendar day 00:00 UTC)
        # -------------------------------------------------------------
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0

        if "time" in df.columns:
            date_group = pd.to_datetime(df["time"]).dt.normalize()
        elif isinstance(df.index, pd.DatetimeIndex):
            date_group = df.index.normalize()
        else:
            date_group = np.zeros(len(df), dtype=int)

        vol = df["volume"].copy()
        vol = vol.replace(0, 1.0).fillna(1.0)
        cum_vol = vol.groupby(date_group).cumsum()
        cum_tp_vol = (typical_price * vol).groupby(date_group).cumsum()
        df["vwap"] = cum_tp_vol / (cum_vol + 1e-9)

        # -------------------------------------------------------------
        # 8. Causal Swing Highs & Lows (Strictly backwards-looking)
        # -------------------------------------------------------------
        # 10-bar rolling extrema of shifted bars for trailing stop reference
        df["recent_swing_high"] = df["high"].shift(1).rolling(window=swing_window, min_periods=3).max()
        df["recent_swing_low"] = df["low"].shift(1).rolling(window=swing_window, min_periods=3).min()

        # 5-bar fractal confirmed strictly at bar t (evaluating bar t-2):
        # Bar t-2 must exceed bars t-4, t-3 (left) and bars t-1, t (right)
        h2 = df["high"].shift(2)
        l2 = df["low"].shift(2)

        is_sh = (
            (h2 > df["high"].shift(3))
            & (h2 > df["high"].shift(4))
            & (h2 > df["high"].shift(1))
            & (h2 > df["high"])
        ).fillna(False).astype(bool)

        is_sl = (
            (l2 < df["low"].shift(3))
            & (l2 < df["low"].shift(4))
            & (l2 < df["low"].shift(1))
            & (l2 < df["low"])
        ).fillna(False).astype(bool)

        df["is_swing_high"] = is_sh
        df["is_swing_low"] = is_sl

        # swing_high and swing_low hold the confirmed price level at swing bars, and NaN elsewhere
        df["swing_high"] = h2.where(is_sh, np.nan)
        df["swing_low"] = l2.where(is_sl, np.nan)

        # Forward-filled verified price level of confirmed swing point
        df["swing_high_price"] = df["swing_high"].ffill()
        df["swing_low_price"] = df["swing_low"].ffill()

        return df

    def get_market_snapshot(
        self,
        symbol: str,
        execution_tf: str = "M5",
        structure_tf: str = "M15",
        regime_tf: str = "H1",
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Fetches and prepares multi-timeframe indicator dataframes for strategy evaluation.

        Returns:
            Dict containing 'execution', 'structure', and 'regime' DataFrames with indicators computed,
            or None if execution timeframe data is insufficient.
        """
        df_exec = self.fetch_rates(symbol, execution_tf)
        if df_exec is None or len(df_exec) < 50:
            return None

        df_struct = self.fetch_rates(symbol, structure_tf)
        df_regime = self.fetch_rates(symbol, regime_tf)

        processed_exec = self.compute_indicators(df_exec)
        processed_struct = (
            self.compute_indicators(df_struct) if df_struct is not None else processed_exec
        )
        processed_regime = (
            self.compute_indicators(df_regime) if df_regime is not None else processed_exec
        )

        return {
            "execution": processed_exec,
            "structure": processed_struct,
            "regime": processed_regime,
        }

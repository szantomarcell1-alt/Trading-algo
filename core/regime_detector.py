"""
Market Microstructure & Regime Detection Engine
Implements Log-Price Hurst Exponent and Multi-Horizon EMA Regime Classification on Closed Bars.
"""

from enum import Enum
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    MEAN_REVERTING = "MEAN_REVERTING"
    RANDOM_WALK_CHOP = "RANDOM_WALK_CHOP"


class RegimeDetector:
    def __init__(
        self,
        hurst_period: int = 100,
        hurst_trend_threshold: float = 0.50,
        hurst_mr_threshold: float = 0.45,
    ):
        self.hurst_period = hurst_period
        self.hurst_trend_threshold = hurst_trend_threshold
        self.hurst_mr_threshold = hurst_mr_threshold

    @staticmethod
    def calculate_hurst_exponent(price_series: np.ndarray, max_lags: int = 25) -> float:
        series = np.asarray(price_series, dtype=np.float64)
        series = series[series > 0]
        n = len(series)
        if n < 30:
            return 0.50

        log_series = np.log(series)
        max_lag = min(max_lags, n // 3)
        if max_lag < 4:
            return 0.50

        lags = range(2, max_lag)
        rms_values = []
        valid_lags = []
        for lag in lags:
            diffs = log_series[lag:] - log_series[:-lag]
            rms = np.sqrt(np.mean(diffs ** 2))
            if rms > 1e-12:
                rms_values.append(rms)
                valid_lags.append(lag)

        if len(rms_values) < 3:
            return 0.50

        x = np.log(valid_lags)
        y = np.log(rms_values)

        try:
            poly = np.polyfit(x, y, 1)
            hurst = float(poly[0])
            return float(np.clip(hurst, 0.0, 1.0))
        except Exception:
            return 0.50

    @staticmethod
    def calculate_garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2
        gk_term = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        rolling_gk = gk_term.rolling(window=window).mean()
        return np.sqrt(rolling_gk)

    def analyze_regime(
        self,
        df: pd.DataFrame,
        fast_ema_col: str = "ema_fast",
        slow_ema_col: str = "ema_slow",
        htf_ema_col: str = "ema_200",
    ) -> Tuple[MarketRegime, Dict[str, Any]]:
        if len(df) < self.hurst_period + 1:
            return MarketRegime.RANDOM_WALK_CHOP, {"reason": "Insufficient history"}

        closed_df = df.iloc[:-1] if len(df) > self.hurst_period else df
        recent_closes = closed_df["close"].values[-self.hurst_period:]
        hurst = self.calculate_hurst_exponent(recent_closes)

        current_close = float(closed_df["close"].iloc[-1])
        fast_ema = float(closed_df[fast_ema_col].iloc[-1]) if fast_ema_col in closed_df.columns else current_close
        slow_ema = float(closed_df[slow_ema_col].iloc[-1]) if slow_ema_col in closed_df.columns else current_close
        htf_ema = float(closed_df[htf_ema_col].iloc[-1]) if htf_ema_col in closed_df.columns else current_close

        info = {
            "hurst": round(hurst, 3),
            "price": current_close,
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "htf_ema": htf_ema,
        }

        if self.hurst_mr_threshold <= hurst <= self.hurst_trend_threshold:
            info["reason"] = f"Random walk (Hurst: {hurst:.2f}) - Zajos piac"
            return MarketRegime.RANDOM_WALK_CHOP, info

        if hurst > self.hurst_trend_threshold:
            if current_close > htf_ema and fast_ema > slow_ema:
                info["reason"] = f"Erős emelkedő trend (Hurst: {hurst:.2f})"
                return MarketRegime.TRENDING_BULLISH, info
            elif current_close < htf_ema and fast_ema < slow_ema:
                info["reason"] = f"Erős csökkenő trend (Hurst: {hurst:.2f})"
                return MarketRegime.TRENDING_BEARISH, info
            else:
                info["reason"] = f"Hurst trend ({hurst:.2f}), de EMA ellentmondásos"
                return MarketRegime.RANDOM_WALK_CHOP, info

        if hurst < self.hurst_mr_threshold:
            info["reason"] = f"Sávos / Átlaghoz visszatérő piac (Hurst: {hurst:.2f})"
            return MarketRegime.MEAN_REVERTING, info

        return MarketRegime.RANDOM_WALK_CHOP, info
"""
Market Microstructure & Regime Detection Engine for GBPJPY
Calculates Hurst exponent via RMS lag differences and Garman-Klass volatility.
Classifies market into TRENDING_BULLISH, TRENDING_BEARISH, MEAN_REVERTING, RANDOM_WALK_CHOP.
"""

from enum import Enum
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    RANDOM_WALK_CHOP = "RANDOM_WALK_CHOP"


class RegimeResult(dict):
    """
    Regime result container supporting both attribute access and dict indexing.
    Supports unpacking as (regime, info).
    """
    def __init__(
        self,
        regime: MarketRegime,
        hurst: float,
        volatility: float,
        info: Optional[Dict[str, Any]] = None,
    ):
        raw_info = info or {}
        super().__init__(raw_info)
        self["regime"] = regime
        self["hurst"] = hurst
        self["volatility"] = volatility
        self.regime = regime
        self.hurst = hurst
        self.volatility = volatility
        self.info = raw_info

    def __iter__(self):
        yield self.regime
        yield self.info

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "regime": self.regime.value if isinstance(self.regime, MarketRegime) else str(self.regime),
            "hurst": self.hurst,
            "volatility": self.volatility,
        }
        d.update(self.info)
        return d


class RegimeDetector:
    """
    GBPJPY Regime Detector.
    Implements Hurst exponent estimation and Garman-Klass volatility.
    """
    def __init__(
        self,
        config: Optional[Any] = None,
        hurst_period: int = 100,
        hurst_trend_threshold: float = 0.55,
        hurst_mr_threshold: float = 0.45,
        max_lag: int = 20,
    ):
        if config is not None:
            strat_cfg = config.get("strategy_parameters", config.get("indicators", {})) if isinstance(config, dict) else getattr(config, "strategy", None)
            if strat_cfg is not None:
                if isinstance(strat_cfg, dict):
                    self.hurst_period = strat_cfg.get("hurst_period", hurst_period)
                    self.hurst_trend_threshold = strat_cfg.get("hurst_trend_threshold", hurst_trend_threshold)
                    self.hurst_mr_threshold = strat_cfg.get("hurst_mean_reversion_threshold", strat_cfg.get("hurst_mr_threshold", hurst_mr_threshold))
                else:
                    self.hurst_period = getattr(strat_cfg, "hurst_period", hurst_period)
                    self.hurst_trend_threshold = getattr(strat_cfg, "hurst_trend_threshold", hurst_trend_threshold)
                    self.hurst_mr_threshold = getattr(strat_cfg, "hurst_mean_reversion_threshold", hurst_mr_threshold)
            else:
                self.hurst_period = hurst_period
                self.hurst_trend_threshold = hurst_trend_threshold
                self.hurst_mr_threshold = hurst_mr_threshold
        else:
            self.hurst_period = hurst_period
            self.hurst_trend_threshold = hurst_trend_threshold
            self.hurst_mr_threshold = hurst_mr_threshold

        self.max_lag = max_lag

    def compute_hurst(
        self,
        series: Union[pd.Series, np.ndarray, list],
        max_lag: Optional[int] = None,
    ) -> float:
        """
        Calculates Hurst Exponent via RMS lag differences:
        tau(lag) = sqrt(mean((ts[lag:] - ts[:-lag])**2))
        Slope of log(tau) vs log(lag) via linear fit.
        """
        if max_lag is None:
            max_lag = self.max_lag

        ts = np.asarray(series, dtype=np.float64).ravel()
        ts = ts[np.isfinite(ts)]

        n = len(ts)
        if n < 10:
            return 0.50

        # Guard against zero-variance / flat price series
        if np.std(ts) < 1e-12 or np.all(ts == ts[0]):
            return 0.50

        effective_max_lag = min(max_lag, n // 2)
        if effective_max_lag < 4:
            return 0.50

        lags = list(range(2, effective_max_lag))
        rms_values = []
        valid_lags = []

        for lag in lags:
            diffs = ts[lag:] - ts[:-lag]
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
            if np.isnan(hurst) or np.isinf(hurst):
                return 0.50
            return float(np.clip(hurst, 0.0, 1.0))
        except Exception:
            return 0.50

    # Aliases
    calculate_hurst = compute_hurst
    calculate_hurst_exponent = compute_hurst

    def compute_garman_klass(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None,
    ) -> Union[float, pd.Series]:
        """
        Garman-Klass Volatility formula:
        sqrt(mean(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2))
        """
        if df is None or len(df) == 0:
            return 0.0 if window is None else pd.Series(dtype=np.float64)

        high = np.asarray(df["high"], dtype=np.float64)
        low = np.asarray(df["low"], dtype=np.float64)
        close = np.asarray(df["close"], dtype=np.float64)
        open_ = np.asarray(df["open"], dtype=np.float64)

        # Avoid log(0)
        low = np.where(low <= 0, 1e-6, low)
        open_ = np.where(open_ <= 0, 1e-6, open_)

        h_l = np.log(high / low)
        c_o = np.log(close / open_)

        term1 = 0.5 * (h_l ** 2)
        term2 = (2.0 * np.log(2.0) - 1.0) * (c_o ** 2)
        diff = term1 - term2

        if window is None:
            mean_diff = np.mean(diff)
            return float(np.sqrt(np.maximum(0.0, mean_diff)))
        else:
            diff_series = pd.Series(diff, index=df.index if isinstance(df, pd.DataFrame) else None)
            rolling_mean = diff_series.rolling(window=window).mean()
            return np.sqrt(np.maximum(0.0, rolling_mean))

    # Aliases
    calculate_garman_klass = compute_garman_klass
    calculate_garman_klass_volatility = compute_garman_klass

    def detect_regime(
        self,
        df_or_series: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> RegimeResult:
        """
        Classifies current market regime into:
        - TRENDING_BULLISH (Hurst > 0.55, upward trend)
        - TRENDING_BEARISH (Hurst > 0.55, downward trend)
        - MEAN_REVERTING (Hurst < 0.45)
        - RANDOM_WALK_CHOP (0.45 <= Hurst <= 0.55)
        """
        if isinstance(df_or_series, pd.DataFrame):
            df = df_or_series
            close_series = df["close"]
            volatility = self.compute_garman_klass(df)
            if not isinstance(volatility, (int, float)):
                volatility = float(volatility.iloc[-1]) if len(volatility) > 0 else 0.0
        else:
            df = None
            close_series = pd.Series(df_or_series)
            diffs = close_series.diff().dropna()
            volatility = float(diffs.std()) if len(diffs) > 0 else 0.0

        n = len(close_series)
        lookback = min(n, self.hurst_period)
        recent_closes = close_series.iloc[-lookback:] if lookback > 0 else close_series

        hurst = self.compute_hurst(recent_closes)

        # Determine trend direction
        if len(recent_closes) >= 2:
            price_change = float(recent_closes.iloc[-1]) - float(recent_closes.iloc[0])
        else:
            price_change = 0.0

        # Also check EMAs if present in DataFrame
        ema_bullish = None
        if df is not None and "ema_fast" in df.columns and "ema_slow" in df.columns:
            ema_bullish = float(df["ema_fast"].iloc[-1]) > float(df["ema_slow"].iloc[-1])
        elif df is not None and "ema_9" in df.columns and "ema_21" in df.columns:
            ema_bullish = float(df["ema_9"].iloc[-1]) > float(df["ema_21"].iloc[-1])

        info = {
            "hurst": round(hurst, 4),
            "volatility": round(volatility, 6),
            "price_change": round(price_change, 4),
        }

        if hurst > self.hurst_trend_threshold:
            is_bull = ema_bullish if ema_bullish is not None else (price_change >= 0)
            if is_bull:
                regime = MarketRegime.TRENDING_BULLISH
                info["reason"] = f"Bullish trend confirmed (Hurst: {hurst:.3f})"
            else:
                regime = MarketRegime.TRENDING_BEARISH
                info["reason"] = f"Bearish trend confirmed (Hurst: {hurst:.3f})"
        elif hurst < self.hurst_mr_threshold:
            regime = MarketRegime.MEAN_REVERTING
            info["reason"] = f"Mean reversion regime (Hurst: {hurst:.3f})"
        else:
            regime = MarketRegime.RANDOM_WALK_CHOP
            info["reason"] = f"Random walk chop (Hurst: {hurst:.3f})"

        return RegimeResult(regime=regime, hurst=hurst, volatility=volatility, info=info)

    def analyze_regime(
        self,
        df: pd.DataFrame,
        fast_ema_col: str = "ema_fast",
        slow_ema_col: str = "ema_slow",
        htf_ema_col: str = "ema_200",
    ) -> Tuple[MarketRegime, Dict[str, Any]]:
        """Legacy-compatible analyze_regime returning (regime, info_dict)."""
        res = self.detect_regime(df)
        info = res.to_dict()
        return res.regime, info

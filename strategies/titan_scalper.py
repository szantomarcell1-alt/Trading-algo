"""
Titan Quant Scalper - Apex Micro-Regime MT5 Scalping Strategy
Optimized specifically for $250-$300 Small Capital accounts on Pepperstone MT5.
Features separate microstructure calibration for XAUUSD (Gold) and Major Forex pairs.
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import logging

from core.mt5_client import SymbolSpecs
from core.regime_detector import RegimeDetector, MarketRegime
from config.settings import AppConfig
from .base_strategy import BaseStrategy
from .signals import TradeSignal, SignalType

logger = logging.getLogger("TitanScalper.Strategy")


class TitanScalperStrategy(BaseStrategy):
    def __init__(self, config: AppConfig):
        super().__init__(name="TitanQuantScalper")
        self.config = config
        self.regime_detector = RegimeDetector(
            hurst_period=config.strategy.hurst_period,
            hurst_trend_threshold=config.strategy.hurst_trend_threshold,
            hurst_mr_threshold=config.strategy.hurst_mean_reversion_threshold,
        )

    def evaluate(
        self,
        symbol: str,
        data_snapshot: Dict[str, pd.DataFrame],
        symbol_specs: SymbolSpecs,
        current_bid: float,
        current_ask: float,
        current_spread_pips: float,
    ) -> Optional[TradeSignal]:
        sym_cfg = self.config.get_symbol_config(symbol)
        max_spread = sym_cfg.max_spread_pips if sym_cfg else 2.0
        min_atr = sym_cfg.min_atr_pips if sym_cfg else 0.8

        # Step 1: Spread Filter Guard
        if current_spread_pips > max_spread:
            return None

        df_exec = data_snapshot.get("execution")    # M1 or M5
        df_struct = data_snapshot.get("structure")  # M5 or M15

        # Multi-timeframe Macro Trend Data Extraction (M15 and H1)
        # Supports direct keys 'm15' and 'h1', or hierarchical keys 'structure', 'macro', 'regime'
        df_m15 = data_snapshot.get("m15")
        df_h1 = data_snapshot.get("h1")
        if df_h1 is None:
            df_h1 = data_snapshot.get("macro")

        if df_m15 is None:
            if df_h1 is not None:
                df_m15 = data_snapshot.get("regime")
                if df_m15 is None:
                    df_m15 = data_snapshot.get("structure")
            else:
                df_m15 = data_snapshot.get("structure")
                df_h1 = data_snapshot.get("regime")

        if df_h1 is None and data_snapshot.get("regime") is not df_m15:
            df_h1 = data_snapshot.get("regime")

        if df_exec is None or len(df_exec) < 65:
            return None

        closed_candle = df_exec.iloc[-2]
        current_atr = float(closed_candle["atr"])
        atr_pips = current_atr / symbol_specs.pip_size

        if atr_pips < min_atr:
            return None

        # Extreme News Spike Filter
        median_atr = float(df_exec["atr"].iloc[-60:-2].median())
        last_bar_range = float(closed_candle["high"] - closed_candle["low"])
        if median_atr > 0:
            if current_atr > median_atr * 2.8 or last_bar_range > current_atr * 3.2:
                return None

        # Step 2: Session Regime Classification on M5
        regime_source = (
            df_struct
            if (df_struct is not None and len(df_struct) >= self.config.strategy.hurst_period + 2)
            else df_exec
        )
        regime, regime_info = self.regime_detector.analyze_regime(regime_source)

        # Macro Trend Filter: Require price above 200 EMA on BOTH M15 and H1 for BUY signals,
        # and below 200 EMA on BOTH M15 and H1 for SELL signals.
        m15_bull = True
        m15_bear = True
        if df_m15 is not None and len(df_m15) >= 2:
            if "ema_200" not in df_m15.columns and len(df_m15) >= 200:
                from core.data_engine import DataEngine
                df_m15 = DataEngine.compute_indicators(df_m15)
            if "ema_200" in df_m15.columns:
                m15_closed = df_m15.iloc[-2]
                m15_bull = float(m15_closed["close"]) > float(m15_closed["ema_200"])
                m15_bear = float(m15_closed["close"]) < float(m15_closed["ema_200"])

        h1_bull = True
        h1_bear = True
        if df_h1 is not None and len(df_h1) >= 2:
            if "ema_200" not in df_h1.columns and len(df_h1) >= 200:
                from core.data_engine import DataEngine
                df_h1 = DataEngine.compute_indicators(df_h1)
            if "ema_200" in df_h1.columns:
                h1_closed = df_h1.iloc[-2]
                h1_bull = float(h1_closed["close"]) > float(h1_closed["ema_200"])
                h1_bear = float(h1_closed["close"]) < float(h1_closed["ema_200"])

        if regime == MarketRegime.TRENDING_BULLISH and not (m15_bull and h1_bull):
            return None
        if regime == MarketRegime.TRENDING_BEARISH and not (m15_bear and h1_bear):
            return None

        if regime == MarketRegime.RANDOM_WALK_CHOP or regime == MarketRegime.MEAN_REVERTING:
            return None

        # Step 3: Strategy Mode Dispatch
        signal: Optional[TradeSignal] = None
        if regime in (MarketRegime.TRENDING_BULLISH, MarketRegime.TRENDING_BEARISH):
            signal = self._evaluate_squeeze_breakout(
                symbol=symbol,
                df_exec=df_exec,
                df_struct=df_struct,
                symbol_specs=symbol_specs,
                regime=regime,
                regime_info=regime_info,
                current_bid=current_bid,
                current_ask=current_ask,
                m15_bull=m15_bull,
                m15_bear=m15_bear,
                h1_bull=h1_bull,
                h1_bear=h1_bear,
            )
        elif regime == MarketRegime.MEAN_REVERTING:
            # Strictly deactivated: 100% pure Squeeze Breakout logic enforced
            signal = None

        min_conf = getattr(self.config.risk_management, "min_signal_confidence", 0.80)
        if signal and signal.confidence < min_conf:
            return None

        return signal

    def _evaluate_squeeze_breakout(
        self,
        symbol: str,
        df_exec: pd.DataFrame,
        df_struct: Optional[pd.DataFrame],
        symbol_specs: SymbolSpecs,
        regime: MarketRegime,
        regime_info: Dict[str, Any],
        current_bid: float,
        current_ask: float,
        m15_bull: bool = True,
        m15_bear: bool = True,
        h1_bull: bool = True,
        h1_bear: bool = True,
    ) -> Optional[TradeSignal]:
        c_last = df_exec.iloc[-2]
        c_prev = df_exec.iloc[-3]
        atr = float(c_last["atr"])

        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()

        # Dynamic timeframe detection (M1 vs M5+)
        bar_sec = (df_exec.index[-1] - df_exec.index[-2]).total_seconds() if len(df_exec) >= 2 else 60
        is_m5_plus = bar_sec >= 240

        # Small-account ($250-$300) Stop-Loss bounds in pips:
        if is_m5_plus:
            min_pips = 20.0 if is_gold else 8.0
            max_pips = 43.5 if is_gold else 18.0
        else:
            min_pips = 15.0 if is_gold else 6.0
            max_pips = 30.0 if is_gold else 14.0

        min_sl_dist = max(atr * 0.5, symbol_specs.pip_size * min_pips)
        max_sl_dist = symbol_specs.pip_size * max_pips

        m5_bullish = True
        m5_bearish = True
        if df_struct is not None and len(df_struct) >= 55:
            m5_closed = df_struct.iloc[-2]
            m5_bullish = (
                (m5_closed["close"] > m5_closed["ema_slow"])
                and (m5_closed["ema_fast"] >= m5_closed["ema_med"])
                and (m5_closed["ema_med"] >= m5_closed["ema_slow"])
            )
            m5_bearish = (
                (m5_closed["close"] < m5_closed["ema_slow"])
                and (m5_closed["ema_fast"] <= m5_closed["ema_med"])
                and (m5_closed["ema_med"] <= m5_closed["ema_slow"])
            )

        bar_range = max(1e-9, float(c_last["high"] - c_last["low"]))
        clv = (float(c_last["close"] - c_last["low"]) - float(c_last["high"] - c_last["close"])) / bar_range
        min_clv = 0.50

        recent_squeeze = bool(df_exec["squeeze_on"].iloc[-8:-1].any() or df_exec["squeeze_fired"].iloc[-5:-1].any())
        touched_bull_ema = (c_prev["low"] <= c_prev["ema_med"] * 1.0002) or (c_last["low"] <= c_last["ema_fast"] * 1.0001)
        touched_bear_ema = (c_prev["high"] >= c_prev["ema_med"] * 0.9998) or (c_last["high"] >= c_last["ema_fast"] * 0.9999)

        fresh_bull_impulse = touched_bull_ema and (c_last["close"] > c_prev["high"]) and (clv >= min_clv)
        fresh_bear_impulse = touched_bear_ema and (c_last["close"] < c_prev["low"]) and (clv <= -min_clv)

        # Squeeze Breakout requires BOTH recent volatility compression AND clean EMA bounce
        trigger_bull = recent_squeeze and fresh_bull_impulse
        trigger_bear = recent_squeeze and fresh_bear_impulse

        # Feature 8: 20-period volume SMA over prior 20 completed candles [-22:-2] (excluding breakout bar at -2)
        vol_20 = (
            float(df_exec["volume"].iloc[-22:-2].mean())
            if ("volume" in df_exec.columns and len(df_exec) >= 22)
            else (float(df_exec["volume"].iloc[:-2].mean()) if "volume" in df_exec.columns and len(df_exec) >= 3 else 1.0)
        )
        vol_surge = float(c_last["volume"]) > round(vol_20 * 1.15, 6) if vol_20 > 0 else True

        if regime == MarketRegime.TRENDING_BULLISH and m5_bullish and m15_bull and h1_bull:
            momentum_bullish = (
                c_last["squeeze_momentum"] > 0
                and c_last["squeeze_momentum"] > c_prev["squeeze_momentum"]
            )
            ema_bullish = (
                c_last["close"] > c_last["ema_fast"]
                and c_last["ema_fast"] > c_last["ema_med"]
                and c_last["ema_med"] >= c_last["ema_slow"]
                and c_last["ema_fast"] >= c_prev["ema_fast"]
            )
            vwap_bullish = c_last["close"] > c_last.get("vwap", c_last["close"])
            rsi_bullish = 50.0 <= c_last["rsi"] <= 66.0

            if momentum_bullish and ema_bullish and vwap_bullish and rsi_bullish and clv >= min_clv and trigger_bull and vol_surge:
                entry_price = current_ask
                recent_low = float(df_exec["low"].iloc[-4:-1].min())
                structural_sl_dist = entry_price - (recent_low - atr * 0.20)
                # Feature 9: Anchor tightly to recent swing level without clamping inside swing range
                sl_dist = max(min_sl_dist, structural_sl_dist)
                sl_price = entry_price - sl_dist

                # If structural stop is wider than small-account max_sl_dist, skip trade!
                if sl_dist > max_sl_dist:
                    return None

                risk_points = sl_dist / symbol_specs.point
                risk_pips = sl_dist / symbol_specs.pip_size
                if risk_pips <= 0:
                    return None

                tp_price = entry_price + sl_dist * self.config.execution.risk_reward_ratio

                return TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    entry_price=entry_price,
                    sl_price=round(sl_price, symbol_specs.digits),
                    tp_price=round(tp_price, symbol_specs.digits),
                    risk_distance_points=risk_points,
                    risk_distance_pips=risk_pips,
                    confidence=0.88,
                    strategy_mode="SQUEEZE_BREAKOUT_BULL",
                    regime=str(regime.value),
                    rationale=(
                        f"M15/M5 Trend kitörés: Hurst={regime_info.get('hurst'):.2f}, "
                        f"SL={risk_pips:.1f}p, CLV={clv:+.2f}, RSI={c_last['rsi']:.1f}"
                    ),
                )

        elif regime == MarketRegime.TRENDING_BEARISH and m5_bearish and m15_bear and h1_bear:
            momentum_bearish = (
                c_last["squeeze_momentum"] < 0
                and c_last["squeeze_momentum"] < c_prev["squeeze_momentum"]
            )
            ema_bearish = (
                c_last["close"] < c_last["ema_fast"]
                and c_last["ema_fast"] < c_last["ema_med"]
                and c_last["ema_med"] <= c_last["ema_slow"]
                and c_last["ema_fast"] <= c_prev["ema_fast"]
            )
            vwap_bearish = c_last["close"] < c_last.get("vwap", c_last["close"])
            rsi_bearish = 34.0 <= c_last["rsi"] <= 50.0

            if momentum_bearish and ema_bearish and vwap_bearish and rsi_bearish and clv <= -min_clv and trigger_bear and vol_surge:
                entry_price = current_bid
                recent_high = float(df_exec["high"].iloc[-4:-1].max())
                structural_sl_dist = (recent_high + atr * 0.20) - entry_price
                # Feature 9: Anchor tightly to recent swing level without clamping inside swing range
                sl_dist = max(min_sl_dist, structural_sl_dist)
                sl_price = entry_price + sl_dist

                if sl_dist > max_sl_dist:
                    return None

                risk_points = sl_dist / symbol_specs.point
                risk_pips = sl_dist / symbol_specs.pip_size
                if risk_pips <= 0:
                    return None

                tp_price = entry_price - sl_dist * self.config.execution.risk_reward_ratio

                return TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    entry_price=entry_price,
                    sl_price=round(sl_price, symbol_specs.digits),
                    tp_price=round(tp_price, symbol_specs.digits),
                    risk_distance_points=risk_points,
                    risk_distance_pips=risk_pips,
                    confidence=0.88,
                    strategy_mode="SQUEEZE_BREAKOUT_BEAR",
                    regime=str(regime.value),
                    rationale=(
                        f"M15/M5 Csökkenő kitörés: Hurst={regime_info.get('hurst'):.2f}, "
                        f"SL={risk_pips:.1f}p, CLV={clv:+.2f}, RSI={c_last['rsi']:.1f}"
                    ),
                )

        return None

    def _evaluate_mean_reversion(
        self,
        symbol: str,
        df_exec: pd.DataFrame,
        df_struct: Optional[pd.DataFrame],
        m15_bull: bool,
        m15_bear: bool,
        symbol_specs: SymbolSpecs,
        regime_info: Dict[str, Any],
        current_bid: float,
        current_ask: float,
    ) -> Optional[TradeSignal]:
        """
        Deactivated (Feature 11): Counter-trend mean reversion entries are strictly disabled
        in favor of 100% pure Squeeze Breakout logic for capital preservation on both EURUSD and XAUUSD.
        """
        return None

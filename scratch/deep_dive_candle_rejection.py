import sys
sys.path.insert(0, r"c:\Trading algo")

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from core.data_engine import DataEngine
from portfolio.portfolio_config import load_portfolio_config
from strategies.titan_scalper import TitanScalperStrategy
from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
from config.settings import load_config as load_root_config
from gbpjpy.config.settings import load_config as load_gbpjpy_config

def inspect_nov20():
    if not mt5.initialize():
        print("MT5 init failed")
        return
        
    root_cfg = load_root_config()
    titan = TitanScalperStrategy(root_cfg)
    
    gbp_cfg = load_gbpjpy_config()
    gbp_strat = GBPJPYDualRegimeScalper(config=gbp_cfg)
    
    symbols = ["EURUSD", "XAUUSD", "GBPJPY"]
    
    for sym in symbols:
        print("\n" + "#"*80)
        print(f"DEEP DIVE FOR {sym} on NOV 20")
        print("#"*80)
        
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 300)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = DataEngine.compute_indicators(df)
        
        # Today's cutoff
        cutoff = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
        today_df = df[df.index >= cutoff]
        print(f"Today bars count: {len(today_df)}")
        
        if sym in ["EURUSD", "XAUUSD"]:
            # Check Titan Scalper conditions on each bar
            reasons = {}
            for i in range(len(today_df)):
                idx_in_full = df.index.get_loc(today_df.index[i])
                if idx_in_full < 65:
                    continue
                slice_df = df.iloc[idx_in_full-65:idx_in_full+1]
                t_bar = today_df.index[i]
                
                # Check regime
                regime, info = titan.regime_detector.analyze_regime(slice_df)
                
                # Check closed candle
                c_last = slice_df.iloc[-2]
                c_prev = slice_df.iloc[-3]
                
                # Squeeze
                recent_squeeze = bool(slice_df["squeeze_on"].iloc[-8:-1].any() or slice_df["squeeze_fired"].iloc[-5:-1].any())
                
                # 200 EMA
                e200 = float(c_last.get("ema_200", 0))
                price = float(c_last["close"])
                
                # Volume surge
                vol_20 = float(slice_df["volume"].iloc[-22:-2].mean()) if len(slice_df) >= 22 else 1.0
                vol_surge = float(c_last["volume"]) > (vol_20 * 1.15)
                
                reason = f"Regime={regime.name}"
                if regime.name in ["TRENDING_BULLISH", "TRENDING_BEARISH"]:
                    reason += f" | Squeeze={recent_squeeze} | VolSurge={vol_surge}"
                
                reasons[reason] = reasons.get(reason, 0) + 1
                
            print("Distribution of bar states for", sym)
            for r, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {r}: {count} bars")
                
        elif sym == "GBPJPY":
            reasons = {}
            for i in range(len(today_df)):
                idx_in_full = df.index.get_loc(today_df.index[i])
                slice_df = df.iloc[max(0, idx_in_full-200):idx_in_full+1]
                t_bar = today_df.index[i]
                bar = slice_df.iloc[-1]
                
                hour = t_bar.hour
                is_london = (8 <= hour < 16) or (hour == 16 and t_bar.minute <= 30) or (12 <= hour < 16)
                is_asian = 1 <= hour < 8
                
                # Hurst
                ts = slice_df["close"].values
                hurst = gbp_strat.regime_detector.compute_hurst(ts) if len(ts) >= 50 else 0.5
                
                atr_pips = float(bar["atr"]) / 0.01 if "atr" in bar else 0
                sq_fired = bool(bar.get("squeeze_fired", False))
                
                session_str = "London" if is_london else ("Asian" if is_asian else "Other")
                reason = f"Session={session_str} | Hurst={hurst:.2f} (TrendThresh={gbp_strat.hurst_trend_threshold}) | ATR={atr_pips:.1f}p (Min={gbp_strat.min_atr_pips}) | SqFired={sq_fired}"
                reasons[reason] = reasons.get(reason, 0) + 1
                
            print("Sample of bar states for GBPJPY today:")
            for r, count in list(reasons.items())[:15]:
                print(f"  {r}: {count} bars")

    mt5.shutdown()

if __name__ == "__main__":
    inspect_nov20()

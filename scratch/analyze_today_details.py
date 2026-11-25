import sys
sys.path.insert(0, r"c:\Trading algo")

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

from core.data_engine import DataEngine
from strategies.titan_scalper import TitanScalperStrategy
from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
from config.settings import load_config as load_root_config
from gbpjpy.config.settings import load_config as load_gbpjpy_config

def analyze_today():
    if not mt5.initialize():
        print("MT5 init failed")
        return
        
    root_cfg = load_root_config()
    titan = TitanScalperStrategy(root_cfg)
    gbp_cfg = load_gbpjpy_config()
    gbp_strat = GBPJPYDualRegimeScalper(config=gbp_cfg)
    
    symbols = ["EURUSD", "XAUUSD", "GBPJPY"]
    today_start = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
    
    for sym in symbols:
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 350)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = DataEngine.compute_indicators(df)
        
        today_df = df[df.index >= today_start]
        if today_df.empty:
            continue
            
        high = today_df["high"].max()
        low = today_df["low"].min()
        open_p = today_df["open"].iloc[0]
        close_p = today_df["close"].iloc[-1]
        
        print(f"\n==================== {sym} TODAY (Nov 20) ====================")
        print(f"Open: {open_p:.5f} | High: {high:.5f} | Low: {low:.5f} | Close: {close_p:.5f}")
        range_points = (high - low)
        print(f"Total Daily Range: {range_points:.5f}")
        
        # Check sessions
        # Asian: 00:00 - 08:00
        # London: 08:00 - 13:00
        # NY Overlap: 13:00 - 17:00
        # Evening: 17:00 - 21:00
        
        regime_counts = {}
        sub_reasons = {}
        
        for i in range(len(today_df)):
            idx_in_full = df.index.get_loc(today_df.index[i])
            if idx_in_full < 65:
                continue
            slice_df = df.iloc[idx_in_full-65:idx_in_full+1]
            t = today_df.index[i]
            
            if sym in ["EURUSD", "XAUUSD"]:
                regime, info = titan.regime_detector.analyze_regime(slice_df)
                reg_name = regime.name
                regime_counts[reg_name] = regime_counts.get(reg_name, 0) + 1
                
                # Check why no signal if trending
                if reg_name in ["TRENDING_BULLISH", "TRENDING_BEARISH"]:
                    # Squeeze check
                    sq_on = slice_df["squeeze_on"].iloc[-8:-1].any()
                    sq_fired = slice_df["squeeze_fired"].iloc[-5:-1].any()
                    vol_20 = float(slice_df["volume"].iloc[-22:-2].mean()) if len(slice_df) >= 22 else 1.0
                    vol_surge = float(slice_df["volume"].iloc[-2]) > (vol_20 * 1.15)
                    sub = f"{reg_name}: sq={sq_on or sq_fired}, vol={vol_surge}"
                    sub_reasons[sub] = sub_reasons.get(sub, 0) + 1
            else:
                # GBPJPY
                ts = slice_df["close"].values
                hurst = gbp_strat.regime_detector.compute_hurst(ts) if len(ts) >= 50 else 0.5
                hour = t.hour
                is_london = (8 <= hour < 16) or (hour == 16 and t.minute <= 30)
                is_asian = (1 <= hour < 8)
                
                bar = slice_df.iloc[-1]
                sq_fired = bool(bar.get("squeeze_fired", False))
                atr_pips = float(bar.get("atr", 0)) / 0.01
                
                status = f"Hurst={hurst:.2f} (London={is_london}, Asian={is_asian}, sq_fired={sq_fired}, atr={atr_pips:.1f}p)"
                if is_london:
                    london_stat = f"London: Hurst={hurst:.2f}, sq={sq_fired}, atr={atr_pips:.1f}"
                    sub_reasons[london_stat] = sub_reasons.get(london_stat, 0) + 1
                elif is_asian:
                    asian_stat = f"Asian: Hurst={hurst:.2f}, atr={atr_pips:.1f}"
                    sub_reasons[asian_stat] = sub_reasons.get(asian_stat, 0) + 1
                    
        print(f"Regime breakdown ({sym}):")
        for k, v in regime_counts.items():
            print(f"  {k}: {v} bars")
        if sub_reasons:
            print("Key condition samples/details:")
            for k, v in list(sub_reasons.items())[:8]:
                print(f"  {k}: {v} bars")

if __name__ == "__main__":
    analyze_today()

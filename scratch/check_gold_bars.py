import sys
sys.path.insert(0, r"c:\Trading algo")

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

from core.data_engine import DataEngine
from strategies.titan_scalper import TitanScalperStrategy
from config.settings import load_config as load_root_config

def check_gold_bars():
    if not mt5.initialize():
        print("MT5 failed")
        return

    root_cfg = load_root_config()
    titan = TitanScalperStrategy(root_cfg)

    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 500)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df = DataEngine.compute_indicators(df)

    today_start = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
    today_indices = [i for i, t in enumerate(df.index) if t >= today_start]

    print(f"Total XAUUSD bars today: {len(today_indices)}")

    filter_tallies = {}

    for idx in today_indices:
        t = df.index[idx]
        slice_df = df.iloc[max(0, idx - 250) : idx + 1]
        if len(slice_df) < 100:
            continue

        # Let's inspect regime
        regime, regime_info = titan.regime_detector.analyze_regime(slice_df)
        c_last = slice_df.iloc[-2]
        c_prev = slice_df.iloc[-3]
        atr = float(c_last["atr"])
        
        # Check reasons
        if regime.name == "RANDOM_WALK_CHOP":
            reason = f"Regime Chop: {regime_info.get('reason')}"
            filter_tallies[reason] = filter_tallies.get(reason, 0) + 1
        elif regime.name == "MEAN_REVERTING":
            reason = "Regime Mean Reverting (Titan is Squeeze Breakout only)"
            filter_tallies[reason] = filter_tallies.get(reason, 0) + 1
        else:
            # TRENDING!
            # check squeeze
            recent_squeeze = bool(slice_df["squeeze_on"].iloc[-8:-1].any() or slice_df["squeeze_fired"].iloc[-5:-1].any())
            vol_20 = float(slice_df["volume"].iloc[-22:-2].mean()) if len(slice_df) >= 22 else 1.0
            vol_surge = float(c_last["volume"]) > round(vol_20 * 1.15, 6)
            
            # SL distance check
            recent_high = float(slice_df["high"].iloc[-4:-1].max())
            structural_sl_dist = (recent_high + atr * 0.20) - float(c_last["close"])
            max_sl_dist = 0.10 * 43.5 # $4.35
            
            reasons = []
            if not recent_squeeze:
                reasons.append("No Squeeze compression")
            if not vol_surge:
                reasons.append("No Volume Surge")
            if structural_sl_dist > max_sl_dist:
                reasons.append(f"SL too wide ({structural_sl_dist:.2f} > {max_sl_dist:.2f})")
                
            reason = f"{regime.name} but blocked by: " + (", ".join(reasons) if reasons else "Other indicators (RSI/CLV/EMA)")
            filter_tallies[reason] = filter_tallies.get(reason, 0) + 1

    print("\nXAUUSD rejection tally for today:")
    for k, v in sorted(filter_tallies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v} bars")

if __name__ == "__main__":
    check_gold_bars()

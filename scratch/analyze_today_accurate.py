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

def analyze_today_accurate():
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
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 600)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = DataEngine.compute_indicators(df)
        
        today_indices = [i for i, t in enumerate(df.index) if t >= today_start]
        print(f"\n==================== {sym} TODAY (Nov 20) ====================")
        print(f"Total bars today: {len(today_indices)}")
        
        regime_counts = {}
        titan_rejections = {}
        gbp_rejections = {}
        
        for idx in today_indices:
            t = df.index[idx]
            # Window of 250 bars
            slice_df = df.iloc[max(0, idx - 250) : idx + 1]
            if len(slice_df) < 120:
                continue
                
            if sym in ["EURUSD", "XAUUSD"]:
                # Check Titan Scalper
                snapshot = {
                    "execution": slice_df,
                    "structure": slice_df,
                    "regime": slice_df,
                    "macro": slice_df,
                }
                # Let's inspect regime
                regime, info = titan.regime_detector.analyze_regime(slice_df)
                reg_name = regime.name
                regime_counts[reg_name] = regime_counts.get(reg_name, 0) + 1
                
                # Check signals
                sigs = titan.evaluate_signals(snapshot, sym)
                if sigs:
                    print(f"[{sym}] SIGNAL AT {t}: {sigs}")
                else:
                    # Let's see what failed
                    c = slice_df.iloc[-2]
                    reason = info.get("reason", "")
                    if reg_name == "RANDOM_WALK_CHOP":
                        titan_rejections[reason] = titan_rejections.get(reason, 0) + 1
                    else:
                        # Regime was trending! Why no signal?
                        # check squeeze
                        sq = slice_df["squeeze_on"].iloc[-8:-1].any() or slice_df["squeeze_fired"].iloc[-5:-1].any()
                        vol_20 = float(slice_df["volume"].iloc[-22:-2].mean()) if len(slice_df) >= 22 else 1.0
                        vol_surge = float(c["volume"]) > (vol_20 * 1.15)
                        detail = f"{reg_name}: sq={sq}, vol={vol_surge}, rsi={float(c.get('rsi_14', 50)):.1f}"
                        titan_rejections[detail] = titan_rejections.get(detail, 0) + 1
            else:
                # GBPJPY
                snapshot = {
                    "execution": slice_df,
                    "structure": slice_df,
                    "regime": slice_df,
                    "m15": slice_df,
                    "h1": slice_df,
                }
                sigs = gbp_strat.generate_signals(snapshot)
                if sigs:
                    print(f"[GBPJPY] SIGNAL AT {t}: {sigs}")
                else:
                    bar = slice_df.iloc[-1]
                    hour = t.hour
                    is_london = (8 <= hour < 16) or (hour == 16 and t.minute <= 30)
                    is_asian = (1 <= hour < 8)
                    ts = slice_df["close"].values
                    hurst = gbp_strat.regime_detector.compute_hurst(ts) if len(ts) >= 50 else 0.5
                    sq = bool(bar.get("squeeze_fired", False))
                    sess = "London" if is_london else ("Asian" if is_asian else "OffSession")
                    detail = f"{sess} (Hurst={hurst:.2f}, sq_fired={sq})"
                    gbp_rejections[detail] = gbp_rejections.get(detail, 0) + 1
                    
        if sym in ["EURUSD", "XAUUSD"]:
            print(f"Regimes for {sym}:")
            for k, v in regime_counts.items():
                print(f"  {k}: {v}")
            print(f"Rejection reasons summary for {sym}:")
            for k, v in sorted(titan_rejections.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {k}: {v} bars")
        else:
            print(f"GBPJPY condition breakdown:")
            for k, v in sorted(gbp_rejections.items(), key=lambda x: x[1], reverse=True)[:8]:
                print(f"  {k}: {v} bars")

if __name__ == "__main__":
    analyze_today_accurate()

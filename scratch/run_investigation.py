import sys
sys.path.insert(0, r"c:\Trading algo")

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

from portfolio.portfolio_config import load_portfolio_config
from portfolio.portfolio_dispatcher import PortfolioDispatcher
from portfolio.portfolio_risk_manager import UnifiedPortfolioRiskManager

def run_investigation():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    cfg = load_portfolio_config()
    risk_mgr = UnifiedPortfolioRiskManager(cfg)
    dispatcher = PortfolioDispatcher(config=cfg, risk_manager=risk_mgr)

    symbols = ["EURUSD", "XAUUSD", "GBPJPY"]
    today_start = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)

    for sym in symbols:
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 600)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        
        from core.data_engine import DataEngine
        df = DataEngine.compute_indicators(df)

        today_indices = [i for i, t in enumerate(df.index) if t >= today_start]
        print(f"\n========================================================")
        print(f"INVESTIGATING {sym} TODAY (Nov 20, 2026)")
        print(f"========================================================")
        print(f"Total M5 candles today: {len(today_indices)}")
        if today_indices:
            print(f"From {df.index[today_indices[0]]} to {df.index[today_indices[-1]]}")
            print(f"Open: {df['open'].iloc[today_indices[0]]} | High: {df['high'].iloc[today_indices].max()} | Low: {df['low'].iloc[today_indices].min()} | Close: {df['close'].iloc[today_indices[-1]]}")

        signals_found = []
        for idx in today_indices:
            t = df.index[idx]
            slice_df = df.iloc[max(0, idx - 300) : idx + 1].copy()
            if len(slice_df) < 65:
                continue
            
            snapshot = {
                "execution": slice_df,
                "structure": slice_df,
                "regime": slice_df,
                "m15": slice_df,
                "h1": slice_df,
            }
            sigs = dispatcher.process_bar(sym, snapshot, t, current_spread_pips=1.0)
            if sigs:
                signals_found.append((t, sigs))

        print(f"Signals generated today for {sym}: {len(signals_found)}")
        for t, s in signals_found:
            print(f"  [{t}] {s}")

if __name__ == "__main__":
    run_investigation()

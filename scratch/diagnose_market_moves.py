import sys
sys.path.insert(0, r"c:\Trading algo")

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from core.data_engine import DataEngine
from portfolio.portfolio_config import load_portfolio_config
from portfolio.portfolio_dispatcher import PortfolioDispatcher
from portfolio.portfolio_risk_manager import UnifiedPortfolioRiskManager

def diagnose():
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return

    print("MT5 Initialized successfully.")
    cfg = load_portfolio_config()
    data_engine = DataEngine(lookback_bars=350)
    
    # We want to check candles from 2026-11-19 00:00 to now (2026-11-20 14:00)
    # Let's see what rates MT5 returns
    symbols = ["EURUSD", "XAUUSD", "GBPJPY"]
    
    for sym in symbols:
        print("\n" + "="*80)
        print(f"ANALYZING REAL MT5 BARS FOR {sym} (Nov 19 - Nov 20)")
        print("="*80)
        
        # Fetch M5 bars
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 600)
        if rates is None or len(rates) == 0:
            print(f"No rates for {sym}")
            continue
            
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        
        print(f"Fetched {len(df)} M5 bars. Earliest: {df.index[0]}, Latest: {df.index[-1]}")
        
        # Filter for Nov 19 00:00 UTC onwards
        cutoff = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
        test_indices = [i for i, t in enumerate(df.index) if t >= cutoff]
        print(f"Number of M5 bars since Nov 19 00:00 UTC: {len(test_indices)}")
        
        risk_mgr = UnifiedPortfolioRiskManager(cfg)
        dispatcher = PortfolioDispatcher(config=cfg, risk_manager=risk_mgr)
        
        signals_found = []
        filter_stats = {
            "evaluated": 0,
            "session_blocked": 0,
            "regime_chop": 0,
            "macro_ema_mismatch": 0,
            "no_squeeze": 0,
            "no_impulse_or_vol": 0,
            "signal_generated": 0,
        }
        
        # Test each candle
        for idx in test_indices:
            candle_t = df.index[idx]
            # Window slice up to this candle
            window_slice = df.iloc[max(0, idx - 300) : idx + 1].copy()
            if len(window_slice) < 65:
                continue
                
            filter_stats["evaluated"] += 1
            
            # Compute indicators
            window_slice = DataEngine.compute_indicators(window_slice)
            
            snapshot = {
                "execution": window_slice,
                "structure": window_slice,
                "regime": window_slice,
                "m15": window_slice,
                "h1": window_slice,
                "macro": window_slice,
            }
            
            sigs = dispatcher.process_bar(sym, snapshot, candle_t, current_spread_pips=1.0)
            if sigs:
                filter_stats["signal_generated"] += len(sigs)
                for s in sigs:
                    signals_found.append((candle_t, s))
                    
        print(f"Filter statistics for {sym}:")
        for k, v in filter_stats.items():
            print(f"  {k}: {v}")
            
        if signals_found:
            print(f"\nSIGNALS FOUND FOR {sym} ({len(signals_found)}):")
            for t, s in signals_found:
                print(f"  [{t}] {s}")
        else:
            print(f"\nNO SIGNALS FOUND for {sym} across all {len(test_indices)} bars!")

    mt5.shutdown()

if __name__ == "__main__":
    diagnose()

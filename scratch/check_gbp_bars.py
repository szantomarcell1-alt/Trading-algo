import sys
sys.path.insert(0, r"c:\Trading algo")

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

from core.data_engine import DataEngine
from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
from gbpjpy.config.settings import load_config as load_gbpjpy_config

def check_gbp_bars():
    if not mt5.initialize():
        print("MT5 failed")
        return

    gbp_cfg = load_gbpjpy_config()
    strat = GBPJPYDualRegimeScalper(config=gbp_cfg)

    rates = mt5.copy_rates_from_pos("GBPJPY", mt5.TIMEFRAME_M5, 0, 500)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df = DataEngine.compute_indicators(df)

    today_start = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
    today_indices = [i for i, t in enumerate(df.index) if t >= today_start]

    print(f"Total GBPJPY bars today: {len(today_indices)}")

    filter_tallies = {}

    for idx in today_indices:
        t = df.index[idx]
        slice_df = df.iloc[max(0, idx - 250) : idx + 1]
        if len(slice_df) < 100:
            continue

        hour = t.hour
        is_london = (8 <= hour < 16) or (hour == 16 and t.minute <= 30)
        is_asian = (1 <= hour < 8)

        ts = slice_df["close"].values
        hurst = strat.regime_detector.compute_hurst(ts)
        bar = slice_df.iloc[-1]
        sq_fired = bool(bar.get("squeeze_fired", False))
        sq_on = bool(bar.get("squeeze_on", False))
        atr_pips = float(bar.get("atr", 0)) / 0.01

        session = "London (Breakout)" if is_london else ("Asian (Mean Reversion)" if is_asian else "Off-Session")
        
        if is_london:
            # London Breakout requires hurst > 0.55 and squeeze_fired
            if hurst <= 0.55:
                reason = f"London: Hurst too low ({hurst:.2f} <= 0.55)"
            elif not sq_fired:
                reason = f"London: Hurst trending ({hurst:.2f} > 0.55) but squeeze_fired is False (sq_on={sq_on})"
            else:
                reason = "London: Conditions met for evaluation"
        elif is_asian:
            # Asian MR requires hurst < 0.45 and BB touch
            if hurst >= 0.45:
                reason = f"Asian: Hurst too high ({hurst:.2f} >= 0.45) - not mean reverting"
            else:
                # check BB touch
                bb_up = bar.get("bb_upper", 999)
                bb_low = bar.get("bb_lower", 0)
                c = bar["close"]
                if c < bb_up and c > bb_low:
                    reason = f"Asian: Hurst MR ({hurst:.2f} < 0.45) but price inside BB [{bb_low:.2f}, {bb_up:.2f}]"
                else:
                    reason = "Asian: BB touched, waiting for candle close / reversal confirmation"
        else:
            reason = "Off-Session (Rollover / Late NY / pre-Asian)"

        filter_tallies[reason] = filter_tallies.get(reason, 0) + 1

    print("\nGBPJPY rejection tally for today:")
    for k, v in sorted(filter_tallies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v} bars")

if __name__ == "__main__":
    check_gbp_bars()

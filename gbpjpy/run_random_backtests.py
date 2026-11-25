"""
GBPJPY Multi-Interval Random Backtest Runner on M5 Timeframe.
Benchmarks the Enhanced M5 Quantitative Scalper against the Baseline M5
across 7 random historical intervals (2020-2026) on real MetaTrader 5 broker data.
"""

from datetime import datetime, timezone, timedelta
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

# Add paths
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for p in [str(PARENT_DIR), str(CURRENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import MetaTrader5 as mt5

from gbpjpy.core.data_engine import DataEngine
from gbpjpy.core.regime_detector import RegimeDetector
from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
from gbpjpy.risk.risk_manager import RiskManager
from gbpjpy.backtest.engine import BacktestEngine, BacktestResult
from gbpjpy.backtest.metrics import calculate_metrics, PerformanceMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RandomBacktest")


def get_default_windows() -> List[Dict[str, Any]]:
    """Returns 7 representative random intervals spanning every year from 2020 to 2026."""
    return [
        {
            "id": 1,
            "year": 2020,
            "start": datetime(2020, 4, 20, tzinfo=timezone.utc),
            "end": datetime(2020, 5, 4, tzinfo=timezone.utc),
            "label": "2020 Post-Covid Consolidation",
            "desc": "Extreme volatility after March crash, central bank stimulus surge."
        },
        {
            "id": 2,
            "year": 2021,
            "start": datetime(2021, 8, 2, tzinfo=timezone.utc),
            "end": datetime(2021, 8, 16, tzinfo=timezone.utc),
            "label": "2021 Summer Range Consolidation",
            "desc": "Low-volatility summer range-bound choppy sideways market."
        },
        {
            "id": 3,
            "year": 2022,
            "start": datetime(2022, 10, 10, tzinfo=timezone.utc),
            "end": datetime(2022, 10, 24, tzinfo=timezone.utc),
            "label": "2022 UK Gilt Crisis & BOJ FX Interventions",
            "desc": "Historic GBP crash and BOJ massive direct yen intervention."
        },
        {
            "id": 4,
            "year": 2023,
            "start": datetime(2023, 2, 13, tzinfo=timezone.utc),
            "end": datetime(2023, 2, 27, tzinfo=timezone.utc),
            "label": "2023 Global Inflation & Rates Repricing",
            "desc": "Persistent inflation repricing, BoE rate hikes vs ultra-loose BOJ."
        },
        {
            "id": 5,
            "year": 2024,
            "start": datetime(2024, 7, 29, tzinfo=timezone.utc),
            "end": datetime(2024, 8, 12, tzinfo=timezone.utc),
            "label": "2024 Historic BOJ Hike & Carry Trade Unwind",
            "desc": "BOJ rate hike trigger, massive global yen carry trade liquidation."
        },
        {
            "id": 6,
            "year": 2025,
            "start": datetime(2025, 2, 10, tzinfo=timezone.utc),
            "end": datetime(2025, 2, 24, tzinfo=timezone.utc),
            "label": "2025 Modern Two-Way Scalping Regime",
            "desc": "Balanced two-way FX flow with mature microstructure patterns."
        },
        {
            "id": 7,
            "year": 2026,
            "start": datetime(2026, 5, 11, tzinfo=timezone.utc),
            "end": datetime(2026, 5, 25, tzinfo=timezone.utc),
            "label": "2026 Current Year Trading Environment",
            "desc": "Current year broker liquidity, tight spreads and session dynamics."
        },
    ]


def fetch_and_prepare_rates(
    symbol: str,
    timeframe: int,
    start: datetime,
    end: datetime,
    de: DataEngine,
    rd: RegimeDetector,
) -> Optional[pd.DataFrame]:
    """Fetches real historical rates from MT5 and computes technical indicators + Hurst."""
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) < 100:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["volume"] = df["tick_volume"]

    # Compute vectorized technical indicators
    df = de.compute_indicators(df)

    # Fast rolling Hurst exponent calculation
    prices = df["close"].values
    n = len(prices)
    hurst_arr = np.full(n, 0.50)
    for i in range(100, n, 1):
        h = rd.compute_hurst(prices[max(0, i - 100):i])
        hurst_arr[i] = h
    df["hurst"] = hurst_arr

    return df


def run_single_backtest(
    df: pd.DataFrame,
    mode_name: str,
    initial_balance: float = 250.0,
    leverage: float = 30.0,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
    commission_per_lot: float = 7.0,
    min_rr_ratio: float = 1.4,
    min_atr_pips: float = 8.0,
    use_macro_filter: bool = True,
) -> Dict[str, Any]:
    """Executes single backtest run and extracts key quantitative metrics."""
    rm = RiskManager(
        initial_balance=initial_balance,
        leverage=leverage,
        commission_per_lot=commission_per_lot,
        max_spread_pips=2.5,
        min_margin_level_percent=200.0,
    )
    strat = GBPJPYDualRegimeScalper(
        risk_manager=rm,
        min_rr_ratio=min_rr_ratio,
        min_atr_pips=min_atr_pips,
        use_macro_filter=use_macro_filter,
        macro_trend_buffer_pips=15.0,
        london_tp_multiple=2.0,
    )
    engine = BacktestEngine(
        initial_balance=initial_balance,
        leverage=leverage,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_per_lot=commission_per_lot,
        strategy=strat,
        risk_manager=rm,
        max_bars_held=48,
    )

    result = engine.run(df)
    m = result.metrics

    # Tally exit reasons
    exit_counts = {"TP": 0, "SL": 0, "TRAILING_STOP": 0, "TIME_DECAY": 0}
    for tr in result.trades:
        reason = tr.get("exit_reason", "")
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    avg_win = float(m.get("avg_win_usd", 0.0))
    avg_loss = float(m.get("avg_loss_usd", 0.0))

    return {
        "mode": mode_name,
        "bars_processed": len(df),
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "net_profit_usd": result.total_net_profit,
        "net_return_pct": round((result.total_net_profit / initial_balance) * 100.0, 2),
        "profit_factor": float(m.get("profit_factor", 0.0)),
        "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
        "sharpe_ratio": float(m.get("sharpe_ratio", 0.0)),
        "sortino_ratio": float(m.get("sortino_ratio", 0.0)),
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "exits": exit_counts,
        "final_balance": result.final_balance,
    }


def main():
    logger.info("Initializing MetaTrader 5 connection...")
    if not mt5.initialize():
        logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
        sys.exit(1)

    term = mt5.terminal_info()
    acc = mt5.account_info()
    logger.info(f"Connected to MT5 build {getattr(term, 'build', 'unknown')} | Broker: {getattr(acc, 'company', 'unknown')}")

    windows = get_default_windows()
    de = DataEngine()
    rd = RegimeDetector(hurst_period=100)

    all_results = []

    print("\n" + "=" * 115)
    print(" GBPJPY QUANTITATIVE M5 TIMEFRAME BENCHMARK: BASELINE vs ENHANCED FILTERS")
    print(" Capital: $250.00 | Leverage: 1:30 | Spread: 1.2 pips | Slippage: 0.5 pips | Commission: $7/lot")
    print(" Enhanced: Min R:R >= 1.4R | Min ATR >= 8.0p | Macro H1 EMA 200 Filter | London TP = 2.0R")
    print("=" * 115)

    for win in windows:
        w_id = win["id"]
        label = win["label"]
        s_date = win["start"].strftime("%Y-%m-%d")
        e_date = win["end"].strftime("%Y-%m-%d")

        print(f"\n>>> Running Interval #{w_id}: {label} ({s_date} to {e_date})")
        print(f"    Context: {win['desc']}")

        df_m5 = fetch_and_prepare_rates("GBPJPY", mt5.TIMEFRAME_M5, win["start"], win["end"], de, rd)

        if df_m5 is None:
            logger.warning(f"Insufficient data for interval #{w_id}. Skipping.")
            continue

        # 1. Baseline M5 Run (No strict R:R filter, low ATR threshold, no macro filter)
        res_baseline = run_single_backtest(
            df_m5,
            mode_name="Baseline M5",
            min_rr_ratio=1.0,
            min_atr_pips=3.0,
            use_macro_filter=False,
        )

        # 2. Enhanced M5 Run (1.4R filter, 8.0p ATR gate, Macro H1 200 EMA alignment)
        res_enhanced = run_single_backtest(
            df_m5,
            mode_name="Enhanced M5",
            min_rr_ratio=1.4,
            min_atr_pips=8.0,
            use_macro_filter=True,
        )

        all_results.append({
            "window": win,
            "baseline": res_baseline,
            "enhanced": res_enhanced
        })

        print(f"    [BASELINE] Trades: {res_baseline['total_trades']:2d} | WR: {res_baseline['win_rate']:5.1f}% | PnL: ${res_baseline['net_profit_usd']:+6.2f} ({res_baseline['net_return_pct']:+5.2f}%) | PF: {res_baseline['profit_factor']:4.2f} | MaxDD: {res_baseline['max_drawdown_pct']:4.2f}%")
        print(f"    [ENHANCED] Trades: {res_enhanced['total_trades']:2d} | WR: {res_enhanced['win_rate']:5.1f}% | PnL: ${res_enhanced['net_profit_usd']:+6.2f} ({res_enhanced['net_return_pct']:+5.2f}%) | PF: {res_enhanced['profit_factor']:4.2f} | MaxDD: {res_enhanced['max_drawdown_pct']:4.2f}%")
        print(f"               Exits: TP={res_enhanced['exits'].get('TP',0)} | SL={res_enhanced['exits'].get('SL',0)} | Trail={res_enhanced['exits'].get('TRAILING_STOP',0)} | Decay={res_enhanced['exits'].get('TIME_DECAY',0)}")

    mt5.shutdown()

    # Aggregate summaries
    print("\n" + "=" * 115)
    print(" AGGREGATE SUMMARY: BASELINE M5 vs ENHANCED M5 (7 RANDOM PERIODS 2020-2026)")
    print("=" * 115)

    header = f"{'Window':<28} | {'Base Trades':<11} {'Base WR':<8} {'Base PnL':<11} {'Base PF':<7} {'Base DD':<7} | {'Enh Trades':<10} {'Enh WR':<8} {'Enh PnL':<11} {'Enh PF':<7} {'Enh DD':<7}"
    print(header)
    print("-" * 115)

    tot_trades_base, tot_pnl_base = 0, 0.0
    tot_trades_enh, tot_pnl_enh = 0, 0.0

    for item in all_results:
        w_title = item["window"]["label"][:28]
        b = item["baseline"]
        e = item["enhanced"]

        tot_trades_base += b["total_trades"]
        tot_pnl_base += b["net_profit_usd"]
        tot_trades_enh += e["total_trades"]
        tot_pnl_enh += e["net_profit_usd"]

        line = (
            f"{w_title:<28} | "
            f"{b['total_trades']:<11} {b['win_rate']:<7.1f}% ${b['net_profit_usd']:<9.2f} {b['profit_factor']:<7.2f} {b['max_drawdown_pct']:<6.2f}% | "
            f"{e['total_trades']:<10} {e['win_rate']:<7.1f}% ${e['net_profit_usd']:<9.2f} {e['profit_factor']:<7.2f} {e['max_drawdown_pct']:<6.2f}%"
        )
        print(line)

    print("-" * 115)
    print(f"{'OVERALL TOTALS / COMPARISON':<28} | Baseline: {tot_trades_base} trades, Net: ${tot_pnl_base:+.2f} | Enhanced: {tot_trades_enh} trades, Net: ${tot_pnl_enh:+.2f}")
    print("=" * 115)

    # Save results to JSON
    output_path = CURRENT_DIR / "random_backtest_results.json"
    clean_results = []
    for item in all_results:
        clean_item = {
            "window": {
                "id": item["window"]["id"],
                "year": item["window"]["year"],
                "start": item["window"]["start"].isoformat(),
                "end": item["window"]["end"].isoformat(),
                "label": item["window"]["label"],
                "desc": item["window"]["desc"],
            },
            "baseline": item["baseline"],
            "enhanced": item["enhanced"]
        }
        clean_results.append(clean_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)

    logger.info(f"Results successfully saved to {output_path}")


if __name__ == "__main__":
    main()

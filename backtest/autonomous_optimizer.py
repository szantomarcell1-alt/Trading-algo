"""
Autonomous Quantitative Optimizer - Titan Quant Scalper
Executes iterative hypothesis testing with Walk-Forward In-Sample (2020-2023)
and Out-of-Sample (2024-2026) validation to achieve consistent weekly ~1% profit.
"""

import sys
import copy
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_config, AppConfig
from core.data_engine import DataEngine
from core.mt5_client import SymbolSpecs
from strategies.signals import TradeSignal, SignalType
from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics, calculate_metrics

console = Console()

MT5_TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 16385,
}

TF_HIERARCHY = {
    "M5": ("M5", "M15", "H1"),
    "M1": ("M1", "M5", "M15"),
}

DEFAULT_SPREADS = {"EURUSD": 0.6, "XAUUSD": 1.8, "GBPUSD": 1.0}


def fetch_historical_dataset(symbol: str, d_from: datetime, d_to: datetime, exec_tf: str = "M5"):
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None, None, None

        mt5.symbol_select(symbol, True)
        tf_exec_str, tf_struct_str, tf_regime_str = TF_HIERARCHY.get(exec_tf, ("M5", "M15", "H1"))

        mt5_exec = MT5_TIMEFRAMES[tf_exec_str]
        mt5_struct = MT5_TIMEFRAMES[tf_struct_str]
        mt5_regime = MT5_TIMEFRAMES[tf_regime_str]

        # Extra warmup for higher timeframes
        d_from_htf = d_from - timedelta(days=25)

        r_exec = mt5.copy_rates_range(symbol, mt5_exec, d_from, d_to)
        r_struct = mt5.copy_rates_range(symbol, mt5_struct, d_from_htf, d_to)
        r_regime = mt5.copy_rates_range(symbol, mt5_regime, d_from_htf, d_to)
        mt5.shutdown()

        def _to_df(r):
            if r is None or len(r) == 0:
                return None
            df = pd.DataFrame(r)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        return _to_df(r_exec), _to_df(r_struct), _to_df(r_regime)
    except Exception as e:
        console.print(f"[red]Adatlekérési hiba ({symbol}): {e}[/red]")
        return None, None, None


def evaluate_windows(
    symbol: str,
    df_exec: pd.DataFrame,
    df_struct: pd.DataFrame,
    df_regime: pd.DataFrame,
    config: AppConfig,
    n_windows: int = 25,
    window_bars: int = 4000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    max_start = len(df_exec) - window_bars - 50
    if max_start <= 0:
        return []

    rng = random.Random(seed)
    start_indices = sorted(rng.sample(range(0, max_start), min(n_windows, max_start)))
    spread = DEFAULT_SPREADS.get(symbol.upper(), 1.0)

    results = []
    for start_idx in start_indices:
        end_idx = start_idx + window_bars
        sub_exec = df_exec.iloc[start_idx : end_idx]
        w_start = sub_exec.index[0]
        w_end = sub_exec.index[-1]
        days = max(1, (w_end - w_start).days)

        sub_struct = df_struct.loc[:w_end].tail(window_bars // 3 + 200) if df_struct is not None else None
        sub_regime = df_regime.loc[:w_end].tail(window_bars // 9 + 200) if df_regime is not None else None

        engine = BacktestEngine(
            config=config,
            initial_balance=250.0,
            commission_per_lot_round_turn=config.risk_management.commission_per_lot_round_turn,
            fixed_spread_pips=spread,
            slippage_points=1.0,
        )

        m = engine.run(
            df_m1=sub_exec,
            df_m5=sub_exec,
            df_m15=sub_struct,
            symbol=symbol,
            df_h1=sub_regime,
        )
        weekly_ret = (m.total_return_pct / days) * 7.0

        results.append({
            "start": str(w_start)[:10],
            "end": str(w_end)[:10],
            "days": days,
            "trades": m.total_trades,
            "win_rate": m.win_rate_pct,
            "profit": m.total_net_profit,
            "return_pct": m.total_return_pct,
            "weekly_return": weekly_ret,
            "profit_factor": m.profit_factor,
            "max_dd": m.max_drawdown_pct,
        })

    return results


def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "n": 0, "win_windows_pct": 0.0, "avg_profit": 0.0, "avg_weekly": 0.0,
            "hit_1pct_rate": 0.0, "avg_win_rate": 0.0, "avg_pf": 0.0, "worst_dd": 0.0,
        }

    df = pd.DataFrame(results)
    n = len(df)
    win_windows = len(df[df["profit"] > 0])
    win_windows_pct = (win_windows / n) * 100.0

    hit_1pct = len(df[df["weekly_return"] >= 1.0])
    hit_1pct_rate = (hit_1pct / n) * 100.0

    return {
        "n": n,
        "win_windows_pct": win_windows_pct,
        "avg_profit": float(df["profit"].mean()),
        "median_profit": float(df["profit"].median()),
        "avg_weekly": float(df["weekly_return"].mean()),
        "hit_1pct_rate": hit_1pct_rate,
        "avg_win_rate": float(df["win_rate"].mean()),
        "avg_pf": float(df["profit_factor"].mean()),
        "avg_dd": float(df["max_dd"].mean()),
        "worst_dd": float(df["max_dd"].max()),
        "avg_trades": float(df["trades"].mean()),
    }


def main():
    console.print(
        Panel(
            "[bold cyan]AUTONOMOUS QUANT OPTIMIZATION LOOP - TITAN QUANT SCALPER[/bold cyan]\n"
            "[green]Fixed Capital: $250.00 | Symbols: EURUSD & XAUUSD | Timeframe: M5\n"
            "In-Sample: 2020 - 2023 | Out-of-Sample Validation: 2024 - 2026[/green]",
            border_style="cyan",
        )
    )

    # 1. Fetch In-Sample (2020-2023) and Out-of-Sample (2024-2026) data
    console.print("[cyan]In-Sample és Out-of-Sample adatok letöltése MT5-ből...[/cyan]")
    d_is_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    d_is_end = datetime(2023, 12, 31, tzinfo=timezone.utc)
    d_oos_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    d_oos_end = datetime(2026, 10, 30, tzinfo=timezone.utc)

    datasets = {}
    for sym in ["EURUSD", "XAUUSD"]:
        console.print(f"  Adatok betöltése: {sym}...")
        is_exec, is_struct, is_regime = fetch_historical_dataset(sym, d_is_start, d_is_end, "M5")
        oos_exec, oos_struct, oos_regime = fetch_historical_dataset(sym, d_oos_start, d_oos_end, "M5")
        datasets[sym] = {
            "is": (is_exec, is_struct, is_regime),
            "oos": (oos_exec, oos_struct, oos_regime),
        }
        console.print(
            f"    [green]{sym} In-Sample bars: {len(is_exec) if is_exec is not None else 0} | "
            f"Out-of-Sample bars: {len(oos_exec) if oos_exec is not None else 0}[/green]"
        )

    # Define Candidate Variations
    # Each variation explores quantitative hypotheses
    variations = [
        {
            "name": "Var 1 (Alapértelmezett M5, Risk 1.25x)",
            "config_tweaks": {"risk_multiplier": 1.25, "rr": 1.68, "be_r": 1.25},
        },
        {
            "name": "Var 2 (Kockázat 1.8x, Arany 0.01 lot feloldva)",
            "config_tweaks": {"risk_multiplier": 1.80, "rr": 1.68, "be_r": 1.25},
        },
        {
            "name": "Var 3 (BE Késleltetés 1.45R + R/R 1.68)",
            "config_tweaks": {"risk_multiplier": 1.80, "rr": 1.68, "be_r": 1.45},
        },
        {
            "name": "Var 4 (Aszimmetrikus R/R 1.75 + BE 1.45R)",
            "config_tweaks": {"risk_multiplier": 1.80, "rr": 1.75, "be_r": 1.45},
        },
        {
            "name": "Var 5 (Konzisztens R/R 1.60 + BE 1.35R)",
            "config_tweaks": {"risk_multiplier": 1.80, "rr": 1.60, "be_r": 1.35},
        },
    ]

    base_config = load_config()
    scoreboard = []

    for var in variations:
        v_name = var["name"]
        tweaks = var["config_tweaks"]
        console.print(f"\n[bold yellow]>>> Értékelés: {v_name} <<<[/bold yellow]")

        cfg = copy.deepcopy(base_config)
        cfg.risk_management.max_risk_multiplier_for_min_lot = tweaks["risk_multiplier"]
        cfg.execution.risk_reward_ratio = tweaks["rr"]
        cfg.execution.breakeven_trigger_r = tweaks["be_r"]
        cfg.risk_management.max_consecutive_losses = 3

        # Test on In-Sample (2020-2023)
        is_scores = {}
        for sym in ["EURUSD", "XAUUSD"]:
            is_exec, is_struct, is_regime = datasets[sym]["is"]
            res = evaluate_windows(
                symbol=sym,
                df_exec=is_exec,
                df_struct=is_struct,
                df_regime=is_regime,
                config=cfg,
                n_windows=20,
                window_bars=6000,
            )
            is_scores[sym] = calculate_summary(res)

        avg_is_win_pct = (is_scores["EURUSD"]["win_windows_pct"] + is_scores["XAUUSD"]["win_windows_pct"]) / 2.0
        avg_is_weekly = (is_scores["EURUSD"]["avg_weekly"] + is_scores["XAUUSD"]["avg_weekly"]) / 2.0
        avg_is_pf = (is_scores["EURUSD"]["avg_pf"] + is_scores["XAUUSD"]["avg_pf"]) / 2.0
        avg_is_dd = (is_scores["EURUSD"]["worst_dd"] + is_scores["XAUUSD"]["worst_dd"]) / 2.0

        console.print(
            f"  [cyan]In-Sample Eredmény:[/cyan] Nyertes ablakok: {avg_is_win_pct:.1f}% | "
            f"Átlagos Heti Profit: {avg_is_weekly:+.2f}% | PF: {avg_is_pf:.2f} | MaxDD: {avg_is_dd:.2f}%"
        )

        # Test on Out-of-Sample (2024-2026) to test real forward edge
        oos_scores = {}
        for sym in ["EURUSD", "XAUUSD"]:
            oos_exec, oos_struct, oos_regime = datasets[sym]["oos"]
            res = evaluate_windows(
                symbol=sym,
                df_exec=oos_exec,
                df_struct=oos_struct,
                df_regime=oos_regime,
                config=cfg,
                n_windows=20,
                window_bars=6000,
            )
            oos_scores[sym] = calculate_summary(res)

        avg_oos_win_pct = (oos_scores["EURUSD"]["win_windows_pct"] + oos_scores["XAUUSD"]["win_windows_pct"]) / 2.0
        avg_oos_weekly = (oos_scores["EURUSD"]["avg_weekly"] + oos_scores["XAUUSD"]["avg_weekly"]) / 2.0
        avg_oos_pf = (oos_scores["EURUSD"]["avg_pf"] + oos_scores["XAUUSD"]["avg_pf"]) / 2.0
        avg_oos_dd = (oos_scores["EURUSD"]["worst_dd"] + oos_scores["XAUUSD"]["worst_dd"]) / 2.0

        p_col = "bold green" if avg_oos_weekly > 0 else "bold red"
        console.print(
            f"  [magenta]Out-of-Sample Eredmény:[/magenta] Nyertes ablakok: {avg_oos_win_pct:.1f}% | "
            f"Heti Profit: [{p_col}]{avg_oos_weekly:+.2f}%[/{p_col}] | PF: {avg_oos_pf:.2f} | MaxDD: {avg_oos_dd:.2f}%"
        )

        scoreboard.append({
            "name": v_name,
            "tweaks": tweaks,
            "is_win_pct": avg_is_win_pct,
            "is_weekly": avg_is_weekly,
            "oos_win_pct": avg_oos_win_pct,
            "oos_weekly": avg_oos_weekly,
            "oos_pf": avg_oos_pf,
            "oos_max_dd": avg_oos_dd,
            "details": {"is": is_scores, "oos": oos_scores},
        })

        # Save progress immediately
        out_file = Path(__file__).parent / "autonomous_optimization_scoreboard.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(scoreboard, f, indent=2)

    # Display final scoreboard
    console.print()
    t = Table(title="AUTONÓM KVANTITATÍV SCOREBOARD (WALK-FORWARD)", show_header=True, header_style="bold magenta")
    t.add_column("Modell / Variáció", style="bold white", width=34)
    t.add_column("In-Sample Win%", justify="right", style="cyan", width=14)
    t.add_column("In-Sample Heti%", justify="right", style="cyan", width=14)
    t.add_column("Out-of-Sample Win%", justify="right", style="yellow", width=16)
    t.add_column("Out-of-Sample Heti%", justify="right", style="bold", width=16)
    t.add_column("OOS Profit Factor", justify="right", style="green", width=14)
    t.add_column("OOS Worst DD", justify="right", style="red", width=12)

    for item in scoreboard:
        w_style = "bold green" if item["oos_weekly"] > 0 else "red"
        t.add_row(
            item["name"][:32],
            f"{item['is_win_pct']:.1f}%",
            f"{item['is_weekly']:+.2f}%",
            f"{item['oos_win_pct']:.1f}%",
            f"[{w_style}]{item['oos_weekly']:+.2f}%[/{w_style}]",
            f"{item['oos_pf']:.2f}",
            f"{item['oos_max_dd']:.2f}%",
        )

    console.print(t)
    console.print()

    # Save scoreboard to file
    out_file = Path(__file__).parent / "autonomous_optimization_scoreboard.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(scoreboard, f, indent=2)
    console.print(f"[green]Teljes Scoreboard mentve: {out_file}[/green]")


if __name__ == "__main__":
    main()

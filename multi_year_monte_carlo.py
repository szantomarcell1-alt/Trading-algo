"""
Multi-Year Monte Carlo & Random Window Historical Benchmark (2020 - 2026)
Titan Quant Scalper - MT5 Historical Simulation
Supports flexible execution timeframes (M1, M5, M15) and dynamic Strike thresholds.
"""

import sys
import argparse
import random
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_config
from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics

console = Console()

MT5_TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}

TF_HIERARCHY = {
    "M1": ("M1", "M5", "M15"),
    "M5": ("M5", "M15", "H1"),
    "M15": ("M15", "H1", "H4"),
    "H1": ("H1", "H4", "D1"),
}


def generate_random_start_dates(
    start_date: str = "2020-01-01",
    end_date: str = "2026-10-30",
    n_windows: int = 40,
    seed: int = 42,
) -> List[datetime]:
    dt_from = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt_to = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    total_seconds = int((dt_to - dt_from).total_seconds())

    rng = random.Random(seed)
    random_offsets = sorted(rng.sample(range(0, total_seconds), n_windows))
    dates = [dt_from + timedelta(seconds=sec) for sec in random_offsets]
    return dates


def fetch_historical_window(
    symbol: str,
    start_dt: datetime,
    bars_needed: int = 15000,
    exec_tf: str = "M5",
) -> Optional[Dict[str, pd.DataFrame]]:
    try:
        import MetaTrader5 as mt5
        should_shutdown = False
        if not mt5.terminal_info():
            if not mt5.initialize():
                return None
            should_shutdown = True

        mt5.symbol_select(symbol, True)

        tf_exec_str, tf_struct_str, tf_regime_str = TF_HIERARCHY.get(
            exec_tf.upper(), ("M5", "M15", "H1")
        )
        mt5_exec = MT5_TIMEFRAMES.get(tf_exec_str, mt5.TIMEFRAME_M5)
        mt5_struct = MT5_TIMEFRAMES.get(tf_struct_str, mt5.TIMEFRAME_M15)
        mt5_regime = MT5_TIMEFRAMES.get(tf_regime_str, mt5.TIMEFRAME_H1)
        mt5_h1 = MT5_TIMEFRAMES.get("H1", mt5.TIMEFRAME_H1)

        # Calculate calendar days based on timeframe
        # In M5, 15,000 bars ≈ 75 trading days ≈ 115 calendar days
        if exec_tf.upper() == "M5":
            days_span = int(bars_needed / 200 * 1.5) + 30
            warmup_days = 20
        elif exec_tf.upper() == "M1":
            days_span = int(bars_needed / 1000 * 1.5) + 10
            warmup_days = 8
        else:
            days_span = 120
            warmup_days = 30

        end_dt_fetch = start_dt + timedelta(days=days_span)
        start_htf = start_dt - timedelta(days=warmup_days)

        r_exec = mt5.copy_rates_range(symbol, mt5_exec, start_dt, end_dt_fetch)
        r_struct = mt5.copy_rates_range(symbol, mt5_struct, start_htf, end_dt_fetch)
        r_regime = mt5.copy_rates_range(symbol, mt5_regime, start_htf, end_dt_fetch)
        r_h1 = mt5.copy_rates_range(symbol, mt5_h1, start_htf, end_dt_fetch)
        if should_shutdown:
            mt5.shutdown()

        if r_exec is None or len(r_exec) < min(1000, bars_needed):
            return None

        def _to_df(rates):
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        df_exec = _to_df(r_exec[:bars_needed])
        df_struct = _to_df(r_struct)
        df_regime = _to_df(r_regime)
        df_h1 = _to_df(r_h1)

        return {"exec": df_exec, "struct": df_struct, "regime": df_regime, "h1": df_h1}
    except Exception as e:
        console.print(f"[red]Adatletöltési hiba ({symbol} @ {start_dt}): {e}[/red]")
        return None


def run_multi_year_monte_carlo(
    symbols: List[str],
    exec_tf: str = "M5",
    n_windows: int = 40,
    bars_per_window: int = 15000,
    start_date: str = "2020-01-01",
    end_date: str = "2026-10-30",
    initial_balance: float = 250.0,
    consecutive_losses_limit: int = 3,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    config = load_config()
    config.risk_management.max_consecutive_losses = consecutive_losses_limit
    spread_map = {"EURUSD": 0.6, "XAUUSD": 1.8, "GBPUSD": 1.0}

    tf_exec_str, tf_struct_str, tf_regime_str = TF_HIERARCHY.get(
        exec_tf.upper(), ("M5", "M15", "H1")
    )

    all_results = {}
    random_dates = generate_random_start_dates(
        start_date=start_date, end_date=end_date, n_windows=n_windows, seed=seed
    )

    console.print(
        Panel(
            f"[bold cyan]TITAN QUANT SCALPER - {exec_tf} MONTE CARLO TESZT ({consecutive_losses_limit}-STRIKE, 2020 - 2026)[/bold cyan]\n"
            f"[green]Szimbólumok: {', '.join(symbols)} | Idősáv hierarchia: {tf_exec_str} / {tf_struct_str} / {tf_regime_str}\n"
            f"Mintaszám: {n_windows} ablak / szimbólum | Ablakméret: {bars_per_window} {exec_tf} gyertya[/green]",
            border_style="cyan",
        )
    )

    import MetaTrader5 as mt5
    mt5_initialized_here = False
    if not mt5.terminal_info():
        if mt5.initialize():
            mt5_initialized_here = True

    try:
        for sym in symbols:
            spread = spread_map.get(sym, 1.0)
            console.print(f"\n[bold yellow]>>> Szimuláció indítása: {sym} ({exec_tf}, {n_windows} ablak, Spread: {spread} pip, Strike: {consecutive_losses_limit}) <<<[/bold yellow]")
            sym_results = []

            for idx, start_dt in enumerate(random_dates):
                window_data = fetch_historical_window(
                    sym, start_dt, bars_needed=bars_per_window, exec_tf=exec_tf
                )
                if window_data is None:
                    continue

                df_exec = window_data["exec"]
                df_struct = window_data["struct"]
                df_regime = window_data["regime"]
                df_h1 = window_data.get("h1")

                w_start = df_exec.index[0]
                w_end = df_exec.index[-1]
                trading_days = max(1, (w_end - w_start).days)

                engine = BacktestEngine(
                    config=config,
                    initial_balance=initial_balance,
                    commission_per_lot_round_turn=config.risk_management.commission_per_lot_round_turn,
                    fixed_spread_pips=spread,
                    slippage_points=1.0,
                )

                if exec_tf.upper() == "M5":
                    m = engine.run(
                        df_m1=df_exec,
                        df_m5=df_exec,
                        df_m15=df_struct,
                        symbol=sym,
                        df_h1=df_h1 if df_h1 is not None else df_regime,
                    )
                else:
                    m = engine.run(
                        df_m1=df_exec,
                        df_m5=df_struct,
                        df_m15=df_regime,
                        symbol=sym,
                        df_h1=df_h1,
                    )

                est_weekly_return = (m.total_return_pct / trading_days) * 7.0
                p_style = "bold green" if m.total_net_profit >= 0 else "bold red"

                console.print(
                    f"  [{sym} {exec_tf}] Ablak {idx + 1:02d}/{n_windows:02d} "
                    f"({w_start.strftime('%Y-%m-%d')} -> {w_end.strftime('%Y-%m-%d')}, {len(df_exec)} bars): "
                    f"Trades: {m.total_trades:2d} | Win: {m.win_rate_pct:4.1f}% | "
                    f"Profit: [{p_style}]${m.total_net_profit:+5.2f}[/{p_style}] ({m.total_return_pct:+5.2f}%) | "
                    f"MaxDD: {m.max_drawdown_pct:4.2f}% | PF: {m.profit_factor:4.2f}"
                )

                sym_results.append({
                    "symbol": sym,
                    "tf": exec_tf,
                    "window_idx": idx + 1,
                    "start_date": str(w_start)[:16],
                    "end_date": str(w_end)[:16],
                    "duration_days": trading_days,
                    "bars": len(df_exec),
                    "trades": m.total_trades,
                    "winning_trades": m.winning_trades,
                    "losing_trades": m.losing_trades,
                    "win_rate": m.win_rate_pct,
                    "net_profit": m.total_net_profit,
                    "return_pct": m.total_return_pct,
                    "weekly_return_pct": est_weekly_return,
                    "profit_factor": m.profit_factor,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "max_drawdown_amount": m.max_drawdown_amount,
                    "sharpe": m.sharpe_ratio,
                })

            all_results[sym] = sym_results
    finally:
        if mt5_initialized_here:
            mt5.shutdown()

    # Save results to JSON (merge with existing file if present to preserve all symbols)
    output_filename = f"monte_carlo_{exec_tf.lower()}_{consecutive_losses_limit}strike_2020_2026.json"
    output_path = Path(__file__).parent / output_filename
    merged_results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                merged_results = json.load(f)
        except Exception:
            merged_results = {}
    merged_results.update(all_results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2)
    console.print(f"\n[green]Részletes ablakeredmények elmentve: {output_path}[/green]")

    return merged_results


def summarize_results(all_results: Dict[str, List[Dict[str, Any]]], exec_tf: str = "M5"):
    for sym, results in all_results.items():
        if not results:
            continue

        df = pd.DataFrame(results)
        n = len(df)
        win_runs = len(df[df["net_profit"] > 0])
        loss_runs = len(df[df["net_profit"] < 0])
        even_runs = len(df[df["net_profit"] == 0])

        avg_profit = df["net_profit"].mean()
        median_profit = df["net_profit"].median()
        avg_ret = df["return_pct"].mean()
        avg_weekly = df["weekly_return_pct"].mean()

        hit_1pct = len(df[df["weekly_return_pct"] >= 1.0])
        hit_1pct_rate = (hit_1pct / n) * 100.0

        avg_win_rate = df["win_rate"].mean()
        avg_pf = df["profit_factor"].mean()
        avg_dd = df["max_drawdown_pct"].mean()
        max_dd = df["max_drawdown_pct"].max()
        avg_trades = df["trades"].mean()

        console.print()
        t = Table(
            title=f"KVANTITATÍV MONTE CARLO ÖSSZESÍTŐ (2020-2026) - {sym} ({exec_tf}, 3-STRIKE)",
            show_header=True,
            header_style="bold magenta",
        )
        t.add_column("Kvantitatív Paraméter / Metrika", style="bold white", width=38)
        t.add_column("Érték", justify="right", style="cyan", width=22)
        t.add_column("Benchmark Referencia / Értékelés", justify="left", style="dim", width=30)

        t.add_row(f"Összes tesztelt {exec_tf} ablak", str(n), "Reprezentatív 6 éves minta")
        t.add_row("Nyereséges időablakok száma", f"[bold green]{win_runs} ({win_runs/n*100:.1f}%)[/bold green]", "Cél: > 65%")
        t.add_row("Veszteséges időablakok száma", f"[bold red]{loss_runs} ({loss_runs/n*100:.1f}%)[/bold red]", "Minimális kontrollált bukás")
        t.add_row("Semleges / Kötés nélküli ablak", f"{even_runs}", "Piac pangás / szűrők tiltottak")

        t.add_row("Átlagos Nettó Profit / ablak", f"${avg_profit:+.2f}", "Pozitív várható érték (Edge)")
        t.add_row("Medián Nettó Profit / ablak", f"${median_profit:+.2f}", "Outlierektől mentes stabil érték")
        t.add_row("Átlagos Hozam / ablak", f"{avg_ret:+.2f}%", "-")

        w_col = "bold green" if avg_weekly >= 1.0 else "yellow"
        t.add_row("Becsült Átlagos Heti Profit", f"[{w_col}]{avg_weekly:+.2f}% / hét[/{w_col}]", "Cél: ~1.00% / hét")
        t.add_row("Heti >= 1% Profit Elérési Arány", f"[{w_col}]{hit_1pct_rate:.1f}%[/{w_col}]", f"{hit_1pct}/{n} ablakban teljesült")

        t.add_row(f"Átlagos Kötésszám / ablak", f"{avg_trades:.1f} trade", "Bizonyított mintanagyság")
        t.add_row("Átlagos Nyerési Arány (Win Rate)", f"{avg_win_rate:.1f}%", "Cél: 55% - 75%")
        t.add_row("Átlagos Profit Factor (PF)", f"{avg_pf:.2f}", "Intézményi szint: > 1.50")
        t.add_row("Átlagos Visszaesés (Avg Drawdown)", f"{avg_dd:.2f}%", "Cél: < 3.00%")
        t.add_row("Legrosszabb Visszaesés (Worst Max DD)", f"[bold yellow]{max_dd:.2f}%[/bold yellow]", "Tőkevédelmi korlát: < 5.00%")

        console.print(t)
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Multi-Year Monte Carlo Backtester")
    parser.add_argument("-s", "--symbols", type=str, default="EURUSD,XAUUSD")
    parser.add_argument("--tf", type=str, default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("-w", "--windows", type=int, default=40)
    parser.add_argument("-n", "--bars", type=int, default=15000)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2026-10-30")
    parser.add_argument("--balance", type=float, default=250.0)
    parser.add_argument("--strike", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    sym_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    results = run_multi_year_monte_carlo(
        symbols=sym_list,
        exec_tf=args.tf,
        n_windows=args.windows,
        bars_per_window=args.bars,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        consecutive_losses_limit=args.strike,
        seed=args.seed,
    )
    summarize_results(results, exec_tf=args.tf)


if __name__ == "__main__":
    main()

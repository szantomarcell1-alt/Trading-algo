"""
Monte Carlo & Random Window Backtest Runner - Titan Quant Scalper
Executes N randomly sampled historical windows (e.g. 40 windows of 15,000 M1 bars)
and calculates statistical metrics: Weekly Profit Distribution, Win Rate Distribution,
Probability of achieving >= 1% weekly profit, and Max Drawdown bounds.
"""

import sys
import argparse
import random
from datetime import datetime, timezone
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
from core.data_engine import DataEngine
from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics

console = Console()


def load_full_symbol_data(symbol: str, total_bars: int = 100000):
    """
    Fetches available M1, M5, and M15 bars from MT5 into memory using chunks.
    """
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None, None, None

        mt5.symbol_select(symbol, True)
        console.print(f"[cyan]Adatok betöltése MT5-ből ({symbol}, cél: {total_bars} M1 gyertya)...[/cyan]")

        def _fetch_chunked(tf, count):
            chunk_size = 45000
            dfs = []
            fetched = 0
            while fetched < count:
                to_fetch = min(chunk_size, count - fetched)
                rates = mt5.copy_rates_from_pos(symbol, tf, fetched, to_fetch)
                if rates is None or len(rates) == 0:
                    break
                df_c = pd.DataFrame(rates)
                dfs.append(df_c)
                fetched += len(rates)
                if len(rates) < to_fetch:
                    break
            if not dfs:
                return None
            combined = pd.concat(dfs).drop_duplicates(subset=["time"]).sort_values("time")
            combined["time"] = pd.to_datetime(combined["time"], unit="s", utc=True)
            combined.set_index("time", inplace=True)
            combined.rename(columns={"tick_volume": "volume"}, inplace=True)
            return combined[["open", "high", "low", "close", "volume"]]

        df_m1 = _fetch_chunked(mt5.TIMEFRAME_M1, total_bars)
        df_m5 = _fetch_chunked(mt5.TIMEFRAME_M5, (total_bars // 5) + 500)
        df_m15 = _fetch_chunked(mt5.TIMEFRAME_M15, (total_bars // 15) + 500)
        mt5.shutdown()

        return df_m1, df_m5, df_m15
    except Exception as e:
        console.print(f"[red]MT5 betöltési hiba: {e}[/red]")
        return None, None, None


def run_monte_carlo(
    symbol: str,
    n_windows: int = 40,
    window_bars: int = 15000,
    fixed_spread: float = 0.6,
    initial_balance: float = 250.0,
) -> List[Dict[str, Any]]:
    config = load_config()
    df_m1, df_m5, df_m15 = load_full_symbol_data(symbol=symbol, total_bars=100000)
    if df_m1 is None or len(df_m1) < window_bars + 200:
        console.print(f"[red]Nincs elég adat a(z) {symbol} szimbólumhoz![/red]")
        return []

    console.print(f"[bold green]Összesen {len(df_m1)} M1 gyertya sikeresen betöltve ({df_m1.index[0].strftime('%Y-%m-%d')} -> {df_m1.index[-1].strftime('%Y-%m-%d')})![/bold green]")

    max_start = len(df_m1) - window_bars - 50
    if max_start <= 0:
        start_indices = [0]
    else:
        # Generate uniformly distributed random starting points across the dataset
        random.seed(42)
        start_indices = sorted(random.sample(range(0, max_start), min(n_windows, max_start)))

    results = []
    console.print(f"[bold yellow]Indul a {len(start_indices)} véletlenszerű {window_bars} gyertyás ablak szimulációja ({symbol})...[/bold yellow]")

    for idx, start_idx in enumerate(start_indices):
        end_idx = start_idx + window_bars
        sub_m1 = df_m1.iloc[start_idx : end_idx]
        start_dt = sub_m1.index[0]
        end_dt = sub_m1.index[-1]

        # Slice corresponding structure and regime with warmup lookback
        sub_m5 = df_m5.loc[:end_dt].tail(window_bars // 5 + 200) if df_m5 is not None else None
        sub_m15 = df_m15.loc[:end_dt].tail(window_bars // 15 + 200) if df_m15 is not None else None

        engine = BacktestEngine(
            config=config,
            initial_balance=initial_balance,
            commission_per_lot_round_turn=config.risk_management.commission_per_lot_round_turn,
            fixed_spread_pips=fixed_spread,
            slippage_points=1.0,
        )

        m = engine.run(
            df_m1=sub_m1,
            df_m5=sub_m5,
            df_m15=sub_m15,
            symbol=symbol,
        )

        trading_days = max(1, (end_dt - start_dt).days)
        est_weekly_return = (m.total_return_pct / trading_days) * 7.0

        p_color = "green" if m.total_net_profit >= 0 else "red"
        console.print(
            f"  [dim]Ablak {idx + 1:02d}/{len(start_indices):02d}:[/dim] "
            f"[{start_dt.strftime('%m-%d')} - {end_dt.strftime('%m-%d')}] | "
            f"Trades: {m.total_trades:2d} | Win: {m.win_rate_pct:4.1f}% | "
            f"Profit: [{p_color}]${m.total_net_profit:+5.2f}[/{p_color}] ({m.total_return_pct:+5.2f}%) | "
            f"MaxDD: {m.max_drawdown_pct:4.2f}%"
        )

        results.append({
            "window_id": idx + 1,
            "start_date": str(start_dt)[:16],
            "end_date": str(end_dt)[:16],
            "days": trading_days,
            "trades": m.total_trades,
            "win_rate": m.win_rate_pct,
            "net_profit": m.total_net_profit,
            "return_pct": m.total_return_pct,
            "weekly_return_pct": est_weekly_return,
            "profit_factor": m.profit_factor,
            "max_dd_pct": m.max_drawdown_pct,
            "sharpe": m.sharpe_ratio,
        })

    return results


def summarize_monte_carlo(results: List[Dict[str, Any]], symbol: str):
    if not results:
        return

    df_res = pd.DataFrame(results)

    n_runs = len(df_res)
    win_runs = len(df_res[df_res["net_profit"] > 0])
    loss_runs = len(df_res[df_res["net_profit"] < 0])
    breakeven_runs = len(df_res[df_res["net_profit"] == 0])

    avg_profit = df_res["net_profit"].mean()
    median_profit = df_res["net_profit"].median()
    avg_return_pct = df_res["return_pct"].mean()
    avg_weekly_return = df_res["weekly_return_pct"].mean()

    # Success rate of hitting >= 1% weekly profit
    hit_1pct_weekly = len(df_res[df_res["weekly_return_pct"] >= 1.0])
    hit_1pct_prob = (hit_1pct_weekly / n_runs) * 100.0

    avg_win_rate = df_res["win_rate"].mean()
    avg_pf = df_res["profit_factor"].mean()
    avg_max_dd = df_res["max_dd_pct"].mean()
    worst_dd = df_res["max_dd_pct"].max()
    avg_trades = df_res["trades"].mean()

    console.print()
    console.print(
        Panel(
            f"[bold cyan]MONTE CARLO STATISZTIKAI ÉRTÉKELÉS - {symbol}[/bold cyan]\n"
            f"[green]Mintaszám: {n_runs} véletlen ablak (egyenként 15 000 M1 gyertya, ~2 hét / ablak)[/green]",
            border_style="cyan",
        )
    )

    t = Table(title=f"Kvantitatív Összegzés ({symbol})", show_header=True, header_style="bold magenta")
    t.add_column("Kvantitatív Paraméter / Metrika", style="bold white", width=38)
    t.add_column("Érték", justify="right", style="cyan", width=22)
    t.add_column("Értékelés / Elvárás", justify="left", style="dim", width=28)

    t.add_row("Összes tesztelt ablak", str(n_runs), "Teljes reprezentatív minta")
    t.add_row("Nyereséges időszakok száma", f"[bold green]{win_runs} ({win_runs/n_runs*100:.1f}%)[/bold green]", "Cél: > 70%")
    t.add_row("Veszteséges időszakok száma", f"[bold red]{loss_runs} ({loss_runs/n_runs*100:.1f}%)[/bold red]", "Minimális veszteség")
    t.add_row("Nullás / Kötés nélküli időszak", f"{breakeven_runs}", "Piac pangás / szűrők aktívak")

    t.add_row("Átlagos Nettó Profit / ablak", f"[bold green]${avg_profit:+.2f}[/bold green]" if avg_profit >= 0 else f"[bold red]${avg_profit:+.2f}[/bold red]", "Pozitív várható érték")
    t.add_row("Medián Nettó Profit / ablak", f"${median_profit:+.2f}", "Outlierektől mentes középérték")
    t.add_row("Átlagos Hozam / ablak (~2 hét)", f"{avg_return_pct:+.2f}%", "-")
    
    w_style = "bold green" if avg_weekly_return >= 1.0 else "yellow"
    t.add_row("Becsült Átlagos Heti Hozam", f"[{w_style}]{avg_weekly_return:+.2f}% / hét[/{w_style}]", "Cél: ~1.00% / hét")
    t.add_row("Heti >= 1% Profit Elérési Esély", f"[{w_style}]{hit_1pct_prob:.1f}%[/{w_style}]", f"{hit_1pct_weekly}/{n_runs} ablakban elérve")

    t.add_row("Átlagos Kötésszám / ablak", f"{avg_trades:.1f} trade", "~4-8 trade / hét")
    t.add_row("Átlagos Nyerési Arány (Win Rate)", f"{avg_win_rate:.1f}%", "Cél: 60% - 75%")
    t.add_row("Átlagos Profit Factor (PF)", f"{avg_pf:.2f}", "> 1.50 jóváhagyva")
    t.add_row("Átlagos Maximális Visszaesés", f"{avg_max_dd:.2f}%", "Cél: < 3.00%")
    t.add_row("Legrosszabb Visszaesés (Worst DD)", f"[bold yellow]{worst_dd:.2f}%[/bold yellow]", "Tőkevédelmi limit: < 5.00%")

    console.print(t)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Random Window Backtester")
    parser.add_argument("-s", "--symbols", type=str, default="EURUSD,XAUUSD")
    parser.add_argument("-w", "--windows", type=int, default=40, help="Random ablakok száma szimbólumonként (alap: 40)")
    parser.add_argument("-n", "--bars", type=int, default=15000, help="Ablak méret gyertyában (alap: 15000)")
    parser.add_argument("--balance", type=float, default=250.0)

    args = parser.parse_args()
    sym_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    spread_map = {"EURUSD": 0.6, "XAUUSD": 1.8, "GBPUSD": 1.0}

    all_summaries = {}
    for sym in sym_list:
        spread = spread_map.get(sym, 1.0)
        results = run_monte_carlo(
            symbol=sym,
            n_windows=args.windows,
            window_bars=args.bars,
            fixed_spread=spread,
            initial_balance=args.balance,
        )
        if results:
            summarize_monte_carlo(results, symbol=sym)
            all_summaries[sym] = results


if __name__ == "__main__":
    main()

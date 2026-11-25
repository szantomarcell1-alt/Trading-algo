"""
Backtest Runner & Benchmark CLI - Titan Quant Scalper
Loads real Pepperstone MT5 candles or generates synthetic multi-regime data to run quantitative benchmarks.
Supports dynamic CLI parameters: symbols, execution timeframes (M1, M5, M15, H1), bar counts, and date ranges.
"""

import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, List
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

# Mapping of timeframe strings to MT5 constants
MT5_TIMEFRAMES = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}

# Multi-timeframe structure and regime hierarchy
TF_HIERARCHY = {
    "M1": ("M1", "M5", "M15"),
    "M5": ("M5", "M15", "H1"),
    "M15": ("M15", "H1", "H4"),
    "H1": ("H1", "H4", "D1"),
}

DEFAULT_SPREADS = {
    "EURUSD": 0.6,
    "GBPUSD": 1.0,
    "XAUUSD": 1.8,
    "US30": 3.0,
}


def generate_synthetic_forex_data(
    n_bars: int = 3000,
    start_price: float = 1.0850,
    dt_start: str = "2026-05-01 08:00:00",
) -> pd.DataFrame:
    """
    Generates realistic multi-regime M1 price series with realistic London/NY ATR (~3.5 - 5.5 pips).
    """
    np.random.seed(42)
    timestamps = pd.date_range(start=dt_start, periods=n_bars, freq="1min", tz="UTC")

    prices = [start_price]
    current_regime = 1
    regime_lengths = [350, 250, 400, 250, 450, 400]
    regime_idx = 0
    cur_len = 0

    for i in range(1, n_bars):
        if cur_len >= regime_lengths[regime_idx % len(regime_lengths)]:
            regime_idx += 1
            cur_len = 0
            current_regime = regime_idx % 4

        cur_len += 1

        if current_regime == 0:    # Mean-Reverting range
            drift = -0.18 * (prices[-1] - start_price)
            vol = 0.00028
        elif current_regime == 1:  # Bull Trend
            drift = 0.00014
            vol = 0.00032
        elif current_regime == 2:  # Bear Trend
            drift = -0.00015
            vol = 0.00034
        else:                      # Low-vol squeeze / chop
            drift = 0.0
            vol = 0.00018

        change = drift + np.random.normal(0, vol)
        prices.append(prices[-1] + change)

    prices = np.array(prices)
    noise_h = np.abs(np.random.normal(0, 0.00022, n_bars))
    noise_l = np.abs(np.random.normal(0, 0.00022, n_bars))

    opens = np.roll(prices, 1)
    opens[0] = start_price
    closes = prices
    highs = np.maximum(opens, closes) + noise_h
    lows = np.minimum(opens, closes) - noise_l
    volumes = np.random.randint(150, 1800, n_bars)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=timestamps,
    )


def fetch_real_mt5_data(
    symbol: str = "EURUSD",
    n_bars: int = 15000,
    days: Optional[int] = None,
    exec_tf: str = "M1",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Attempts to fetch real historical multi-timeframe candles from the open MT5 terminal (Read-Only).
    Resolves execution, structure, macro regime, and H1 macro timeframes dynamically.
    """
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None, None, None, None

        mt5.symbol_select(symbol, True)

        tf_exec_str, tf_struct_str, tf_regime_str = TF_HIERARCHY.get(
            exec_tf.upper(), ("M1", "M5", "M15")
        )
        mt5_exec = MT5_TIMEFRAMES.get(tf_exec_str, mt5.TIMEFRAME_M1)
        mt5_struct = MT5_TIMEFRAMES.get(tf_struct_str, mt5.TIMEFRAME_M5)
        mt5_regime = MT5_TIMEFRAMES.get(tf_regime_str, mt5.TIMEFRAME_M15)
        mt5_h1 = MT5_TIMEFRAMES.get("H1", mt5.TIMEFRAME_H1)

        def _to_df(rates):
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        if days is not None and days > 0:
            d_to = datetime.now(timezone.utc)
            d_from = d_to - timedelta(days=days)
            # Extra lookback for structure, regime, and macro warming
            d_from_htf = d_from - timedelta(days=max(10, days // 3))

            r_exec = mt5.copy_rates_range(symbol, mt5_exec, d_from, d_to)
            r_struct = mt5.copy_rates_range(symbol, mt5_struct, d_from_htf, d_to)
            r_regime = mt5.copy_rates_range(symbol, mt5_regime, d_from_htf, d_to)
            r_h1 = mt5.copy_rates_range(symbol, mt5_h1, d_from_htf, d_to)
        else:
            # Scaled bar counts based on timeframe ratios
            ratio_struct = 5 if tf_exec_str == "M1" else 3
            ratio_regime = 15 if tf_exec_str == "M1" else 9
            ratio_h1 = 60 if tf_exec_str == "M1" else 12

            r_exec = mt5.copy_rates_from_pos(symbol, mt5_exec, 0, n_bars)
            r_struct = mt5.copy_rates_from_pos(symbol, mt5_struct, 0, (n_bars // ratio_struct) + 400)
            r_regime = mt5.copy_rates_from_pos(symbol, mt5_regime, 0, (n_bars // ratio_regime) + 400)
            r_h1 = mt5.copy_rates_from_pos(symbol, mt5_h1, 0, (n_bars // ratio_h1) + 400)

        mt5.shutdown()

        df_exec = _to_df(r_exec)
        df_struct = _to_df(r_struct)
        df_regime = _to_df(r_regime)
        df_h1 = _to_df(r_h1)

        if df_exec is None or len(df_exec) < 200:
            return None, None, None, None

        return df_exec, df_struct, df_regime, df_h1
    except Exception as e:
        console.print(f"[dim red]MT5 adatlekérési hiba: {e}[/dim red]")
        return None, None, None, None


def display_benchmark_report(
    metrics: BacktestMetrics,
    symbol: str = "EURUSD",
    exec_tf: str = "M1",
    days_or_bars_str: str = "",
):
    tf_exec_str, tf_struct_str, tf_regime_str = TF_HIERARCHY.get(
        exec_tf.upper(), ("M1", "M5", "M15")
    )
    extra_info = f" ({days_or_bars_str})" if days_or_bars_str else ""

    console.print()
    console.print(
        Panel(
            f"[bold cyan]TITAN QUANT SCALPER - BENCHMARK & STRATÉGIA RIPORT[/bold cyan]\n"
            f"[green]Szimbólum: {symbol} | Idősáv hierarchia: {tf_exec_str} / {tf_struct_str} / {tf_regime_str}{extra_info}[/green]",
            border_style="cyan",
        )
    )

    table = Table(title="Kvantitatív Eredmények és Metrikák", show_header=True, header_style="bold magenta")
    table.add_column("Metrika Neve", style="bold white", width=32)
    table.add_column("Érték", justify="right", style="cyan", width=22)
    table.add_column("Benchmark Referencia", justify="left", style="dim", width=28)

    table.add_row("Kezdő tőkeállomány", f"${metrics.initial_balance:.2f}", f"Kis tőkés számla (${metrics.initial_balance:.0f})")
    table.add_row("Záró tőkeállomány", f"${metrics.final_balance:.2f}", "-")

    profit_color = "bold green" if metrics.total_net_profit >= 0 else "bold red"
    table.add_row("Nettó Profit (Levont költségekkel)", f"[{profit_color}]${metrics.total_net_profit:+.2f}[/{profit_color}]", "> $0")
    table.add_row("Teljes Hozam", f"[{profit_color}]{metrics.total_return_pct:+.2f}%[/{profit_color}]", "Pozitív elvárt hozam")

    table.add_row("Összes Kötésszám", str(metrics.total_trades), "Elégséges statisztikai minta")
    table.add_row("Nyertes kötések", f"{metrics.winning_trades} ({metrics.win_rate_pct:.1f}%)", "Cél: 55% - 75%")
    table.add_row("Vesztes kötések", f"{metrics.losing_trades}", "Kontrollált veszteségek")

    pf_style = "bold green" if metrics.profit_factor >= 1.5 else "yellow"
    table.add_row("Profit Factor (PF)", f"[{pf_style}]{metrics.profit_factor:.2f}[/{pf_style}]", "Intézményi szint: > 1.5")

    table.add_row("Átlagos Nyereség / Trade", f"${metrics.average_win:.2f}", "-")
    table.add_row("Átlagos Veszteség / Trade", f"${metrics.average_loss:.2f}", "Kisebb, mint az átl. nyereség")
    table.add_row("Payoff Ratio (Win/Loss)", f"{metrics.payoff_ratio:.2f}", "Aszimmetrikus hozam: > 1.2")

    dd_style = "bold green" if metrics.max_drawdown_pct < 5.0 else ("yellow" if metrics.max_drawdown_pct < 10.0 else "bold red")
    table.add_row("Maximális Visszaesés (Max DD)", f"[{dd_style}]{metrics.max_drawdown_pct:.2f}% (${metrics.max_drawdown_amount:.2f})[/{dd_style}]", "Cél: < 5.0% kis tőkén")

    daily_dd_val = getattr(metrics, "max_daily_drawdown_pct", 0.0)
    daily_dd_style = "bold green" if daily_dd_val <= 3.0 else "bold red"
    table.add_row("Napi Max Visszaesés (Daily DD)", f"[{daily_dd_style}]{daily_dd_val:.2f}%[/{daily_dd_style}]", "Szigorú plafon: <= 3.0%")

    sharpe_style = "bold green" if metrics.sharpe_ratio >= 2.0 else "yellow"
    table.add_row("Évesített Sharpe Ráta", f"[{sharpe_style}]{metrics.sharpe_ratio:.2f}[/{sharpe_style}]", "Top Quant: > 2.0")
    table.add_row("Sortino Ráta (Downside Vol)", f"{metrics.sortino_ratio:.2f}", "Top Quant: > 2.5")
    table.add_row("Calmar Ráta (Hozam / Max DD)", f"{metrics.calmar_ratio:.2f}", "> 2.0")

    table.add_row("Max Egymást Követő Nyereség", str(metrics.max_consecutive_wins), "-")
    table.add_row("Max Egymást Követő Veszteség", str(metrics.max_consecutive_losses), "Circuit breaker aktiválás előtt")

    console.print(table)
    console.print()


def get_mt5_or_config_balance(config) -> float:
    """
    Reads real account balance from active MT5 terminal; falls back to config.yaml initial_balance.
    """
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            acc = mt5.account_info()
            mt5.shutdown()
            if acc is not None and acc.balance > 0:
                console.print(
                    f"[bold green]MT5 Számlaegyenleg automatikusan kiolvasva: "
                    f"{acc.balance:.2f} {acc.currency} (Login: {acc.login})[/bold green]"
                )
                return float(acc.balance)
    except Exception:
        pass
    return float(getattr(config.risk_management, "initial_balance", 250.0))


def run_single_backtest(
    symbol: str,
    exec_tf: str = "M1",
    n_bars: int = 15000,
    days: Optional[int] = None,
    balance: Optional[float] = None,
    spread: Optional[float] = None,
) -> Optional[BacktestMetrics]:
    config = load_config()
    start_cap = balance if balance is not None else get_mt5_or_config_balance(config)
    spread_val = spread if spread is not None else DEFAULT_SPREADS.get(symbol.upper(), 1.0)

    df_exec, df_struct, df_regime, df_h1 = fetch_real_mt5_data(
        symbol=symbol, n_bars=n_bars, days=days, exec_tf=exec_tf
    )

    range_desc = f"{days} nap" if days else f"{n_bars} gyertya"
    if df_exec is not None:
        console.print(
            f"[bold green]VALÓS MT5 ADATSOR BETÖLTVE ({symbol}: {len(df_exec)} db {exec_tf} gyertya, {range_desc})![/bold green]"
        )
    else:
        if symbol.upper() != "EURUSD":
            console.print(f"[bold red]Nem sikerült MT5 adatot letölteni {symbol}-re ({exec_tf})![/bold red]")
            return None
        console.print(f"[yellow]MT5 nem érhető el -> Szintetikus adatsor generálása ({symbol} {exec_tf})...[/yellow]")
        df_exec = generate_synthetic_forex_data(n_bars=n_bars, start_price=1.0850)
        df_struct = df_exec.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        df_regime = df_exec.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        df_h1 = df_exec.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

    engine = BacktestEngine(
        config=config,
        initial_balance=start_cap,
        commission_per_lot_round_turn=config.risk_management.commission_per_lot_round_turn,
        fixed_spread_pips=spread_val,
        slippage_points=1.0,
    )

    console.print(
        f"[cyan]Eseményvezérelt backtest indítása ({symbol} {exec_tf} | Tőke: ${start_cap:.0f} | Spread: {spread_val} pip | Jutalék: $7/lot)...[/cyan]"
    )
    metrics = engine.run(
        df_m1=df_exec,
        df_m5=df_struct,
        df_m15=df_regime,
        symbol=symbol,
        df_h1=df_h1,
    )
    display_benchmark_report(metrics, symbol=symbol, exec_tf=exec_tf, days_or_bars_str=range_desc)
    return metrics


def display_tf_comparison_table(results: Dict[str, BacktestMetrics], symbol: str):
    console.print()
    table = Table(
        title=f"Multi-Timeframe Összehasonlító Benchmark - {symbol}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Idősáv (TF)", style="bold white", width=12)
    table.add_column("Kötések", justify="right", style="cyan", width=10)
    table.add_column("Win Rate %", justify="right", style="cyan", width=12)
    table.add_column("Profit Factor", justify="right", style="cyan", width=14)
    table.add_column("Nettó Profit", justify="right", style="bold", width=14)
    table.add_column("Hozam %", justify="right", style="bold", width=12)
    table.add_column("Max DD %", justify="right", style="yellow", width=12)
    table.add_column("Sharpe Ráta", justify="right", style="cyan", width=12)

    for tf, m in results.items():
        p_style = "green" if m.total_net_profit >= 0 else "red"
        table.add_row(
            tf,
            str(m.total_trades),
            f"{m.win_rate_pct:.1f}%",
            f"{m.profit_factor:.2f}",
            f"[{p_style}]${m.total_net_profit:+.2f}[/{p_style}]",
            f"[{p_style}]{m.total_return_pct:+.2f}%[/{p_style}]",
            f"{m.max_drawdown_pct:.2f}%",
            f"{m.sharpe_ratio:.2f}",
        )

    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Titan Quant Scalper - Fejlett Backtest & Benchmark Futtató"
    )
    parser.add_argument(
        "-s", "--symbols",
        type=str,
        default="EURUSD,XAUUSD",
        help="Tesztelendő szimbólumok vesszővel elválasztva (pl. EURUSD, XAUUSD)",
    )
    parser.add_argument(
        "--tf", "--timeframe",
        type=str,
        default="M1",
        choices=["M1", "M5", "M15", "H1"],
        help="Kereskedési (execution) idősík (alapértelmezett: M1)",
    )
    parser.add_argument(
        "-n", "--bars",
        type=int,
        default=15000,
        help="Betöltendő gyertyák száma (alapértelmezett: 15000)",
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=None,
        help="Visszamenőleges napok száma (ha meg van adva, a --bars felülbírálásra kerül)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=None,
        help="Kezdőtőke felülbírálása (alapértelmezett: MT5 számlaegyenleg vagy config)",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=None,
        help="Spread felülbírálása pipben (alapértelmezett: szimbólum szerinti)",
    )
    parser.add_argument(
        "--compare-tf",
        action="store_true",
        help="Több idősáv (M1, M5, M15) automatikus összehasonlítása az adott szimbólumon",
    )

    args = parser.parse_args()
    sym_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for sym in sym_list:
        if args.compare_tf:
            tf_results = {}
            for test_tf in ["M1", "M5", "M15"]:
                console.print(f"\n[bold yellow]>>> Idősáv teszt: {sym} @ {test_tf} <<<[/bold yellow]")
                m = run_single_backtest(
                    symbol=sym,
                    exec_tf=test_tf,
                    n_bars=args.bars,
                    days=args.days,
                    balance=args.balance,
                    spread=args.spread,
                )
                if m is not None:
                    tf_results[test_tf] = m
            if tf_results:
                display_tf_comparison_table(tf_results, symbol=sym)
        else:
            run_single_backtest(
                symbol=sym,
                exec_tf=args.tf,
                n_bars=args.bars,
                days=args.days,
                balance=args.balance,
                spread=args.spread,
            )


if __name__ == "__main__":
    main()

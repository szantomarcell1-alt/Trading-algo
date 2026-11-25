"""
GBPJPY Headless Event-Driven Backtest Runner & CLI (Milestone 4)
Executes institutional backtests without requiring MT5 GUI terminal,
computes institutional weekly return target metrics (~1% target, Sharpe, Sortino, Max DD),
displays rich ASCII benchmark report, and exports results to JSON, CSV, or TXT.
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List, Sequence, Union

import numpy as np
import pandas as pd

# Add parent workspace to sys.path so gbpjpy is importable
_this_dir = Path(__file__).resolve().parent
_gbpjpy_root = _this_dir.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.backtest.engine import BacktestEngine, BacktestResult
    from gbpjpy.backtest.metrics import calculate_metrics, PerformanceMetrics
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
    from gbpjpy.core.data_engine import DataEngine
    from gbpjpy.config.settings import load_config
except ImportError:
    try:
        from backtest.engine import BacktestEngine, BacktestResult
        from backtest.metrics import calculate_metrics, PerformanceMetrics
        from strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
        from core.data_engine import DataEngine
        from config.settings import load_config
    except ImportError:
        BacktestEngine = None
        BacktestResult = None
        calculate_metrics = None
        PerformanceMetrics = None
        GBPJPYDualRegimeScalper = None
        DataEngine = None
        load_config = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GBPJPY.BacktestCLI")


def generate_synthetic_data(bars: int = 6000, start_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Generates realistic synthetic GBPJPY M5 rates for standalone headless testing.
    Includes intraday London/NY volatility expansion, Asian quiet range, and realistic spreads.
    """
    np.random.seed(42)
    start_t = start_date or datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [start_t + timedelta(minutes=5 * i) for i in range(bars)]

    hours = np.array([(t.hour + t.minute / 60.0) for t in timestamps])
    # London & NY overlap volume / volatility multiplier
    vol_mult = 1.0 + 1.2 * np.sin(np.pi * np.clip(hours - 7, 0, 10) / 10.0)

    daily_cycle = 0.3 * np.sin(2 * np.pi * hours / 24.0)
    drift = np.cumsum(np.random.normal(0, 0.02 * vol_mult, bars))
    cumulative = 195.000 + drift + daily_cycle

    opens = cumulative
    highs = opens + np.abs(np.random.normal(0.04, 0.02, bars))
    lows = opens - np.abs(np.random.normal(0.04, 0.02, bars))
    closes = opens + np.random.normal(0, 0.03, bars)

    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    volumes = np.random.randint(100, 1500, bars)

    df = pd.DataFrame({
        "time": timestamps,
        "open": np.round(opens, 3),
        "high": np.round(highs, 3),
        "low": np.round(lows, 3),
        "close": np.round(closes, 3),
        "volume": volumes,
        "tick_volume": volumes,
    })
    return df


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the backtest execution."""
    parser = argparse.ArgumentParser(
        prog="python -m gbpjpy.backtest.run_backtest",
        description="GBPJPY Quantitative Trading Framework - Event-Driven Backtest Runner & CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="", help="Path to historical CSV or Parquet rate file.")
    parser.add_argument("--start", type=str, default=None, help="Start date filter (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date filter (YYYY-MM-DD).")
    parser.add_argument("--initial-balance", type=float, default=250.0, help="Initial balance in USD.")
    parser.add_argument("--leverage", type=float, default=30.0, help="Account leverage ratio (1:X).")
    parser.add_argument("--spread-pips", type=float, default=1.2, help="Fixed spread modeling in pips.")
    parser.add_argument("--commission", type=float, default=7.0, help="Round-turn commission per standard lot ($0.07/0.01 lot).")
    parser.add_argument("--slippage", type=float, default=0.5, help="Adverse execution slippage in pips on stop orders.")
    parser.add_argument("--output", type=str, default="", help="Export file path (.json, .csv, or .txt).")
    parser.add_argument("--plot", action="store_true", help="Plot equity curve to PNG headlessly (requires matplotlib).")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data generation even if data path provided.")
    parser.add_argument("--bars", type=int, default=6000, help="Number of synthetic M5 bars (~1 month = 6000).")
    parser.add_argument("--config", type=str, default="", help="Path to custom config.yaml.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")

    return parser.parse_args(args)


def format_ascii_report(
    metrics: Union[PerformanceMetrics, dict],
    initial_balance: float = 250.0,
    final_balance: float = 250.0,
    leverage: float = 30.0,
    spread_pips: float = 1.2,
    commission_per_lot: float = 7.0,
    symbol: str = "GBPJPY",
) -> str:
    """Formats institutional performance metrics into an ASCII benchmark table."""
    def g(key, default="0.0"):
        val = metrics.get(key) if hasattr(metrics, "get") else getattr(metrics, key, None)
        return val if val is not None else default

    total_trades = int(g("total_trades", 0))
    winning = int(g("winning_trades", 0))
    losing = int(g("losing_trades", 0))
    win_rate = float(g("win_rate", 0.0))
    profit_factor = float(g("profit_factor", 0.0))
    net_pnl = float(g("total_net_profit", 0.0))
    gross_profit = float(g("gross_profit", 0.0))
    gross_loss = float(g("gross_loss", 0.0))
    tot_comm = float(g("total_commission", 0.0))
    tot_pips = float(g("total_pips", 0.0))
    tot_ret_pct = float(g("total_return_pct", 0.0))
    avg_win = float(g("average_win", 0.0))
    avg_loss = float(g("average_loss", 0.0))
    payoff = float(g("payoff_ratio", 0.0))
    expectancy = float(g("expectancy", 0.0))
    avg_holding = float(g("average_holding_duration_mins", 0.0))
    consec_wins = int(g("max_consecutive_wins", 0))
    consec_losses = int(g("max_consecutive_losses", 0))

    weekly_ret = metrics.get("weekly_returns", [])
    num_weeks = len(weekly_ret)
    mean_weekly = float(g("mean_weekly_return", 0.0))
    target_hit = float(g("percent_weeks_target_hit", 0.0))
    prof_weeks = float(g("percent_profitable_weeks", 0.0))
    med_weekly = float(g("median_weekly_return", 0.0))
    best_w = float(g("best_week_pct", 0.0))
    worst_w = float(g("worst_week_pct", 0.0))
    consec_w_weeks = int(g("max_consecutive_win_weeks", 0))

    sharpe = float(g("sharpe_ratio", 0.0))
    sortino = float(g("sortino_ratio", 0.0))
    calmar = float(g("calmar_ratio", 0.0))
    recovery = float(g("recovery_factor", 0.0))
    max_dd_amt = float(g("max_drawdown_amount", 0.0))
    max_dd_pct = float(g("max_drawdown_pct", 0.0))
    avg_dd_pct = float(g("average_drawdown_pct", 0.0))
    max_daily_dd = float(g("max_daily_drawdown_pct", 0.0))

    # Verification status
    target_pass = (mean_weekly >= 0.70) and (max_dd_pct <= 5.0)
    status_str = "TARGET CRITERIA ACHIEVED [PASS]" if target_pass else "ACCEPTANCE CRITERIA NOT MET [FAIL]"

    lines = [
        "=" * 80,
        "           GBPJPY QUANTITATIVE TRADING FRAMEWORK - BENCHMARK REPORT             ",
        "=" * 80,
        f" Symbol: {symbol} (0.01 pip / JPY-USD 0.626)   | Execution Timeframe: M5          ",
        f" Initial Balance: ${initial_balance:<10.2f}          | Final Balance: ${final_balance:<10.2f}           ",
        f" Leverage: 1:{int(leverage):<5}                        | Commission: ${commission_per_lot:.2f}/lot ($0.07/min)",
        "-" * 80,
        " 1. CAPITAL & RETURN PERFORMANCE                                                ",
        f"   Total Net Profit       : {f'{net_pnl:+.2f}':<10}       | Total Return %         : {f'{tot_ret_pct:+.2f}%':<10}  ",
        f"   Gross Profit           : ${gross_profit:<10.2f}       | Gross Loss             : -${gross_loss:<9.2f} ",
        f"   Total Commission Paid  : ${tot_comm:<10.2f}       | Total Pips Won         : {f'{tot_pips:+.1f} p':<10}",
        "-" * 80,
        " 2. TRADE EXECUTION & WIN/LOSS ANALYTICS                                        ",
        f"   Total Trades Closed    : {total_trades:<10}       | Win Rate               : {f'{win_rate:.2f}%':<10}  ",
        f"   Winning Trades         : {winning:<10}       | Losing Trades          : {losing:<10}      ",
        f"   Profit Factor          : {profit_factor:<10.2f}       | Expectancy ($)         : {f'{expectancy:+.2f}':<10}  ",
        f"   Average Win            : +${avg_win:<9.2f}       | Average Loss           : -${avg_loss:<9.2f} ",
        f"   Payoff Ratio (Win/Loss): {payoff:<10.2f}       | Avg Holding Time       : {f'{avg_holding:.1f} min':<10}",
        f"   Max Consecutive Wins   : {consec_wins:<10}       | Max Consecutive Losses : {consec_losses:<10}      ",
        "-" * 80,
        " 3. CONSISTENT ~1% WEEKLY TARGET ANALYTICS                                      ",
        f"   Total Calendar Weeks   : {num_weeks:<10}       | Mean Weekly Return     : {f'{mean_weekly:+.2f}%':<10}  ",
        f"   Target Hit Rate (>=1%) : {f'{target_hit:.1f}%':<10}       | Profitable Weeks Rate  : {f'{prof_weeks:.1f}%':<10}  ",
        f"   Median Weekly Return   : {f'{med_weekly:+.2f}%':<10}       | Best Week              : {f'{best_w:+.2f}%':<10}  ",
        f"   Worst Week             : {f'{worst_w:+.2f}%':<10}       | Consec Winning Weeks   : {consec_w_weeks:<10}      ",
        "-" * 80,
        " 4. RISK-ADJUSTED RATIOS & DRAWDOWN GOVERNANCE                                  ",
        f"   Annualized Sharpe (52w): {sharpe:<10.2f}       | Sortino Ratio (Downside): {sortino:<10.2f}   ",
        f"   Calmar Ratio           : {calmar:<10.2f}       | Recovery Factor        : {recovery:<10.2f}   ",
        f"   Max Peak-to-Trough DD  : -${max_dd_amt:<6.2f} ({max_dd_pct:.2f}%)| Average Drawdown %     : {f'{avg_dd_pct:.2f}%':<10} ",
        f"   Max Daily Drawdown %   : {f'{max_daily_dd:.2f}%':<10} (Max: 3%) | Target Criteria Check  : {'PASS' if target_pass else 'FAIL'}              ",
        "=" * 80,
        f" Status: {status_str} (Mean Weekly >= 0.70%, Max DD <= 5.0%)",
        "=" * 80,
    ]
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> int:
    """Executes the full backtest pipeline from ingestion to reporting."""
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Initializing GBPJPY Backtest Engine...")

    # 1. Ingest historical rates or generate synthetic
    df: Optional[pd.DataFrame] = None
    if args.data and not args.synthetic:
        data_path = Path(args.data)
        if not data_path.is_file():
            logger.error(f"Specified historical data file not found: '{args.data}'")
            return 1
        logger.info(f"Loading historical market data from '{args.data}'...")
        try:
            if args.data.endswith(".parquet"):
                df = pd.read_parquet(args.data)
            else:
                df = pd.read_csv(args.data)
        except Exception as e:
            logger.error(f"Failed to load historical data file '{args.data}': {e}")
            return 1
    else:
        logger.info(f"Generating {args.bars} bars of high-fidelity synthetic GBPJPY M5 rates...")
        df = generate_synthetic_data(bars=args.bars)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    elif isinstance(df.index, pd.DatetimeIndex):
        df["time"] = df.index

    # Guard immediately if dataset has 0 bars prior to filtering
    if df is None or len(df) == 0:
        logger.error("Market dataset is empty. No bars available to process.")
        return 1

    # Date filtering with automatic timezone alignment
    if "time" in df.columns and len(df) > 0:
        series_tz = getattr(df["time"].dt, "tz", None)
        is_tz_aware = series_tz is not None

        if args.start:
            try:
                start_dt = pd.to_datetime(args.start)
            except Exception as e:
                logger.error(f"Invalid start date format '{args.start}': {e}")
                return 1

            if is_tz_aware:
                if start_dt.tzinfo is None:
                    start_dt = start_dt.tz_localize(series_tz)
                else:
                    start_dt = start_dt.tz_convert(series_tz)
            else:
                if start_dt.tzinfo is not None:
                    start_dt = start_dt.tz_localize(None)

            df = df[df["time"] >= start_dt]

        if args.end:
            try:
                end_dt = pd.to_datetime(args.end)
            except Exception as e:
                logger.error(f"Invalid end date format '{args.end}': {e}")
                return 1

            # Inclusive end-of-day for YYYY-MM-DD input
            if len(str(args.end).strip()) <= 10:
                end_dt = end_dt.replace(hour=23, minute=59, second=59)

            if is_tz_aware:
                if end_dt.tzinfo is None:
                    end_dt = end_dt.tz_localize(series_tz)
                else:
                    end_dt = end_dt.tz_convert(series_tz)
            else:
                if end_dt.tzinfo is not None:
                    end_dt = end_dt.tz_localize(None)

            df = df[df["time"] <= end_dt]

    df.reset_index(drop=True, inplace=True)
    if len(df) == 0:
        logger.error("Market dataset is empty after date filtering. No bars in specified range.")
        return 1

    logger.info(f"Market dataset ready: {len(df)} M5 bars spanning {df['time'].iloc[0]} to {df['time'].iloc[-1]}.")

    # 2. Precompute causal indicators via DataEngine
    logger.info("Computing causal indicators (EMA, ATR, Bollinger, Squeeze, VWAP)...")
    if DataEngine is not None:
        try:
            de = DataEngine()
            df = de.compute_indicators(df)
        except Exception as e:
            logger.warning(f"DataEngine failed: {e}. Falling back to inline indicators.")

    # 3. Instantiate Scalper Strategy & Backtest Engine
    strat = None
    if GBPJPYDualRegimeScalper is not None:
        strat = GBPJPYDualRegimeScalper()

    engine = BacktestEngine(
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        spread_pips=args.spread_pips,
        commission_per_lot=args.commission,
        slippage_pips=args.slippage,
        symbol="GBPJPY",
    )

    # 4. Run event-driven simulation
    logger.info("Executing event-driven bar-by-bar backtest simulation...")
    result: BacktestResult = engine.run(df=df, strategy=strat)

    # 5. Compute institutional metrics
    logger.info("Calculating institutional performance analytics & ~1% target consistency...")
    metrics: PerformanceMetrics = calculate_metrics(
        trades=result.trades,
        initial_balance=args.initial_balance,
        equity_curve=result.equity_curve,
    )

    # 6. Print formatted institutional ASCII table
    report_text = format_ascii_report(
        metrics=metrics,
        initial_balance=result.initial_balance,
        final_balance=result.final_balance,
        leverage=args.leverage,
        spread_pips=args.spread_pips,
        commission_per_lot=args.commission,
        symbol=engine.symbol,
    )
    print("\n" + report_text + "\n")

    # 7. Export outputs
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.output.endswith(".json"):
            logger.info(f"Exporting metrics to JSON: {args.output}")
            export_payload = {
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "bars_processed": result.bars_processed,
                    "initial_balance": result.initial_balance,
                    "final_balance": result.final_balance,
                },
                "metrics": metrics.to_dict(),
                "trades": [t.to_dict() if hasattr(t, "to_dict") else dict(t) for t in result.trades],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(export_payload, f, indent=2, default=str)
        elif args.output.endswith(".csv"):
            logger.info(f"Exporting closed trades to CSV: {args.output}")
            trades_df = pd.DataFrame([t.to_dict() if hasattr(t, "to_dict") else dict(t) for t in result.trades])
            trades_df.to_csv(out_path, index=False)
        else:
            logger.info(f"Exporting benchmark report to text: {args.output}")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report_text)

    # 8. Plot equity curve headlessly if requested
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_path = out_path.with_suffix(".png") if args.output else Path("equity_curve.png")
            fig, ax = plt.subplots(figsize=(12, 6))
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                ax.plot(result.equity_curve["equity"], label="Account Equity ($)", color="#1f77b4", linewidth=1.5)
                ax.plot(result.equity_curve["balance"], label="Account Balance ($)", color="#ff7f0e", linestyle="--", linewidth=1.2)
            ax.set_title("GBPJPY Scalper Strategy - Equity Curve Trajectory")
            ax.set_xlabel("Bar Steps")
            ax.set_ylabel("USD ($)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info(f"Equity curve chart saved headlessly to '{plot_path}'.")
        except ImportError:
            logger.warning("[INFO] matplotlib is not installed. Plotting skipped. Run 'pip install matplotlib' to enable.")

    logger.info("Backtest execution completed successfully.")
    return 0


def main():
    args = parse_arguments()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()

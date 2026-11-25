"""
Portfolio Command-Line Interface (CLI) Entrypoint
Supports headless multi-symbol backtest execution, performance verification,
paper trading (live MT5 market data, zero broker execution), and live trading
across EURUSD, XAUUSD, and GBPJPY on shared $250 capital.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from portfolio.portfolio_config import load_portfolio_config, PortfolioConfig
from portfolio.backtest_engine import PortfolioBacktestEngine, PortfolioBacktestResult
from portfolio.portfolio_metrics import evaluate_portfolio_performance
from portfolio.live_engine import PortfolioLiveEngine, TradingMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Portfolio.CLI")


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m portfolio.main",
        description="Unified Portfolio Trading System & Central Risk Engine CLI ($250 Capital, 1:30 Leverage)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "dry_run", "live", "backtest", "verify"],
        help="Operating mode: 'paper'/'dry_run' (virtual execution on live MT5 feed), 'live' (real broker trading), 'backtest', or 'verify'.",
    )
    parser.add_argument(
        "--live-confirm",
        type=str,
        default="",
        help="Mandatory safety confirmation token required to enable live broker trading with real money.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of polling cycles to execute (0 = run indefinitely until Ctrl+C).",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=1.0,
        help="Polling loop cycle sleep interval in seconds.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable interactive Rich console dashboard (recommended for headless background runs).",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=14,
        help="Number of simulated calendar weeks for multi-symbol backtesting.",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=250.0,
        help="Initial account balance in USD.",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=30.0,
        help="Account leverage ratio (1:30 EU ESMA retail leverage).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to export verification report (JSON or TXT).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logs.",
    )

    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = parse_arguments(args)

    if parsed.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_portfolio_config(
        initial_balance=parsed.initial_balance,
        leverage=parsed.leverage,
    )

    logger.info("=" * 70)
    logger.info("UNIFIED PORTFOLIO TRADING ENGINE - INITIALIZING")
    logger.info(f"Mode: {parsed.mode.upper()} | Capital: ${parsed.initial_balance:.2f} | Leverage: 1:{parsed.leverage:.0f}")
    logger.info("Symbols: EURUSD, XAUUSD ($4.50 Risk Cap), GBPJPY")
    logger.info("=" * 70)

    # 1. Backtest & Verification Modes
    if parsed.mode in ("backtest", "verify"):
        engine = PortfolioBacktestEngine(
            config=config,
            initial_balance=parsed.initial_balance,
        )

        logger.info(f"Running synchronized multi-symbol simulation over {parsed.weeks} weeks...")
        result: PortfolioBacktestResult = engine.run_portfolio_backtest(weeks=parsed.weeks)

        # Print ASCII Verification Summary
        print("\n" + result.report.format_summary_table() + "\n")

        # Export if path requested
        if parsed.output:
            out_path = Path(parsed.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.suffix.lower() == ".json":
                export_data = {
                    "initial_balance": result.initial_balance,
                    "final_balance": result.final_balance,
                    "final_equity": result.final_equity,
                    "total_trades": result.report.total_trades,
                    "win_rate_pct": result.report.win_rate_pct,
                    "profit_factor": result.report.profit_factor,
                    "average_weekly_return_pct": result.report.average_weekly_return_pct,
                    "winning_periods_pct": result.report.winning_periods_pct,
                    "max_portfolio_drawdown_pct": result.report.max_portfolio_drawdown_pct,
                    "max_daily_drawdown_pct": result.report.max_daily_drawdown_pct,
                    "all_criteria_passed": result.report.all_criteria_passed,
                    "weekly_returns": [w.to_dict() for w in result.report.weekly_returns],
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2)
                logger.info(f"Verification report exported to {out_path}")
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result.report.format_summary_table())
                logger.info(f"Verification summary text saved to {out_path}")

        return 0

    # 2. Live & Paper Trading Modes
    target_mode = TradingMode.LIVE if parsed.mode == "live" else TradingMode.PAPER

    live_engine = PortfolioLiveEngine(
        config=config,
        mode=target_mode,
        live_confirm_token=parsed.live_confirm,
        heartbeat_seconds=parsed.heartbeat,
        enable_rich_ui=not parsed.no_ui,
    )

    live_engine.start(max_cycles=parsed.cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())

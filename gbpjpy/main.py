"""
GBPJPY Quantitative Trading Framework - Unified Application Entry Point (Milestone 5)
Provides CLI execution across four operational modes:
  1. backtest     - Event-driven institutional backtesting with ASCII benchmark report.
  2. walk-forward - 24-fold Walk-Forward Analysis (WFE >= 60%) anti-overfitting gate.
  3. monte-carlo  - Shuffled trade permutation stress testing & drawdown modeling.
  4. stream       - Simulated or live real-time market stream with circuit breaker protection.

Usage:
  python main.py --mode backtest [--bars 6000] [--output results.json]
  python main.py --mode walk-forward [--folds 24]
  python main.py --mode monte-carlo [--iterations 1000]
  python main.py --mode stream [--cycles 10]
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Any, Union

import numpy as np
import pandas as pd

# Path setup to ensure local imports succeed
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for p in [str(PARENT_DIR), str(CURRENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from gbpjpy.config.settings import load_config, AppConfig
    from gbpjpy.core.data_engine import DataEngine
    from gbpjpy.core.mock_client import MockMT5Client
    from gbpjpy.core.regime_detector import RegimeDetector
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
    from gbpjpy.risk.risk_manager import RiskManager
    from gbpjpy.risk.execution_manager import ExecutionManager
    from gbpjpy.backtest.engine import BacktestEngine, BacktestResult
    from gbpjpy.backtest.metrics import calculate_metrics, PerformanceMetrics
    from gbpjpy.backtest.run_backtest import generate_synthetic_data, format_ascii_report
    from gbpjpy.backtest.walk_forward_validator import WalkForwardValidator, WalkForwardResult
    from gbpjpy.utils.logger import get_logger
except ImportError:
    try:
        from config.settings import load_config, AppConfig
        from core.data_engine import DataEngine
        from core.mock_client import MockMT5Client
        from core.regime_detector import RegimeDetector
        from strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
        from risk.risk_manager import RiskManager
        from risk.execution_manager import ExecutionManager
        from backtest.engine import BacktestEngine, BacktestResult
        from backtest.metrics import calculate_metrics, PerformanceMetrics
        from backtest.run_backtest import generate_synthetic_data, format_ascii_report
        from backtest.walk_forward_validator import WalkForwardValidator, WalkForwardResult
        from utils.logger import get_logger
    except ImportError:
        load_config = None
        AppConfig = None
        DataEngine = None
        MockMT5Client = None
        RegimeDetector = None
        GBPJPYDualRegimeScalper = None
        RiskManager = None
        ExecutionManager = None
        BacktestEngine = None
        BacktestResult = None
        calculate_metrics = None
        PerformanceMetrics = None
        generate_synthetic_data = None
        format_ascii_report = None
        WalkForwardValidator = None
        WalkForwardResult = None
        get_logger = None

if get_logger is not None:
    logger = get_logger("GBPJPY.Main")
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("GBPJPY.Main")


def format_walk_forward_ascii(result: Any, threshold: float = 0.60) -> str:
    """Generates institutional ASCII summary report for Walk-Forward Analysis."""
    folds_tot = getattr(result, "folds_total", result.get("folds_total", len(result.get("fold_details", [])) if isinstance(result, dict) else 0))
    folds_pass = getattr(result, "folds_passed", result.get("folds_passed", 0) if isinstance(result, dict) else 0)
    mean_is = getattr(result, "mean_is_return", result.get("mean_is_return", 0.0) if isinstance(result, dict) else 0.0)
    mean_oos = getattr(result, "mean_oos_return", result.get("mean_oos_return", 0.0) if isinstance(result, dict) else 0.0)
    wfe = getattr(result, "wfe", result.get("wfe", 0.0) if isinstance(result, dict) else 0.0)
    passed = getattr(result, "passed", result.get("passed", False) if isinstance(result, dict) else False)
    fold_details = getattr(result, "fold_details", result.get("fold_details", []) if isinstance(result, dict) else [])

    lines = [
        "=" * 80,
        "       GBPJPY QUANTITATIVE FRAMEWORK - WALK-FORWARD VALIDATION REPORT          ",
        "=" * 80,
        f" Total Folds Analyzed : {folds_tot:<6} | Passing Folds (WFE >= {threshold*100:.0f}%) : {folds_pass:<6}",
        f" Mean In-Sample Return: {mean_is*100:+.2f}%  | Mean Out-of-Sample Return: {mean_oos*100:+.2f}%",
        f" Composite WFE Ratio  : {wfe*100:.2f}%  | Standard Anti-Overfit Gate : >= {threshold*100:.0f}%",
        "-" * 80,
        " FOLD BREAKDOWN SUMMARY:                                                        ",
        "  Fold |   IS Return   |   OOS Return  |      WFE Ratio      | Status           ",
        "  " + "-" * 74,
    ]
    for f in fold_details[:10]:
        f_idx = f.get("fold", 1)
        is_ret = f.get("is_return", 0.0)
        oos_ret = f.get("oos_return", 0.0)
        f_wfe = f.get("wfe", 0.0)
        status_tag = "[PASS]" if f.get("passed", False) or f_wfe >= threshold else "[FAIL]"
        lines.append(
            f"  {f_idx:>4} | {is_ret*100:>11.2f}% | {oos_ret*100:>11.2f}% | {f_wfe*100:>17.2f}% | {status_tag:<16}"
        )
    if len(fold_details) > 10:
        lines.append(f"  ... and {len(fold_details) - 10} additional rolling folds.")
    lines.extend([
        "=" * 80,
        f" Status: {'ANTI-OVERFITTING GATE ACHIEVED [PASS]' if passed else 'WFE THRESHOLD FAILED [FAIL]'}",
        "=" * 80,
    ])
    return "\n".join(lines)


def format_monte_carlo_ascii(mc_res: Dict[str, Any], initial_balance: float = 250.0) -> str:
    """Generates institutional ASCII report for Monte Carlo permutation test."""
    mean_term = float(mc_res.get("mean_terminal_equity", initial_balance))
    worst_dd = float(mc_res.get("worst_drawdown", 0.0))
    median_dd = float(mc_res.get("median_drawdown", 0.0))
    p95_dd = float(mc_res.get("p95_drawdown", mc_res.get("drawdown_95th", 0.0)))
    p99_dd = float(mc_res.get("p99_drawdown", mc_res.get("drawdown_99th", 0.0)))
    ruin_prob = float(mc_res.get("ruin_probability", mc_res.get("probability_of_ruin", 0.0)))

    lines = [
        "=" * 80,
        "          GBPJPY QUANTITATIVE FRAMEWORK - MONTE CARLO STRESS REPORT            ",
        "=" * 80,
        f" Initial Capital      : ${initial_balance:<10.2f} | Mean Terminal Equity : ${mean_term:<10.2f}",
        f" Worst-Case Drawdown  : ${worst_dd:<10.2f} | Median Drawdown      : ${median_dd:<10.2f}",
        f" 95th Pctile Drawdown : ${p95_dd:<10.2f} | 99th Pctile Drawdown : ${p99_dd:<10.2f}",
        f" Probability of Ruin  : {ruin_prob*100:.2f}%      | Ruin Definition      : Balance <= $0.00",
        "-" * 80,
        " STRESS ASSESSMENT:                                                             ",
        f"   95% VaR Capital Retention: {(1.0 - p95_dd/initial_balance)*100:.1f}% capital safe under worst 5% sequence order.",
        "=" * 80,
        f" Status: {'STRESS TEST PASSED [PASS]' if ruin_prob == 0.0 else 'RISK EXCEEDED [FAIL]'}",
        "=" * 80,
    ]
    return "\n".join(lines)


def handle_backtest(args: argparse.Namespace) -> int:
    logger.info("Starting Backtest Mode...")
    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
    elif generate_synthetic_data is not None:
        df = generate_synthetic_data(bars=args.bars)
    else:
        from gbpjpy.tests.fixtures import generate_synthetic_m5_data
        df = generate_synthetic_m5_data(bars=args.bars)

    if DataEngine is not None:
        de = DataEngine()
        df = de.compute_indicators(df)

    strat = GBPJPYDualRegimeScalper() if GBPJPYDualRegimeScalper is not None else None
    engine = BacktestEngine(
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        spread_pips=args.spread_pips,
        commission_per_lot=args.commission,
        slippage_pips=args.slippage,
        symbol=getattr(args, "symbol", "GBPJPY"),
        strategy=strat,
    )
    result = engine.run(df=df, strategy=strat)

    metrics = calculate_metrics(
        result.trades,
        initial_balance=args.initial_balance,
        equity_curve=result.equity_curve
    ) if calculate_metrics is not None else {}

    if format_ascii_report is not None:
        report = format_ascii_report(
            metrics=metrics,
            initial_balance=result.initial_balance,
            final_balance=result.final_balance,
            leverage=args.leverage,
            spread_pips=args.spread_pips,
            commission_per_lot=args.commission,
            symbol=engine.symbol,
        )
        print("\n" + report + "\n")

    if args.output:
        metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else (dict(metrics) if isinstance(metrics, dict) else metrics.__dict__ if hasattr(metrics, "__dict__") else {})
        out_data = {
            "metrics": metrics_dict,
            "trades": len(result.trades),
            "initial_balance": result.initial_balance,
            "final_balance": result.final_balance,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, default=str)

    return 0


def handle_walk_forward(args: argparse.Namespace) -> int:
    logger.info(f"Starting Walk-Forward Analysis Mode ({args.folds} folds)...")
    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
    elif generate_synthetic_data is not None:
        df = generate_synthetic_data(bars=args.bars)
    else:
        from gbpjpy.tests.fixtures import generate_synthetic_m5_data
        df = generate_synthetic_m5_data(bars=args.bars)

    if DataEngine is not None and "atr" not in df.columns:
        de = DataEngine()
        df = de.compute_indicators(df)

    val = WalkForwardValidator(
        folds=args.folds,
        in_sample_bars=1000,
        out_of_sample_bars=250,
        initial_balance=args.initial_balance,
    )
    res = val.run(df=df, symbol=getattr(args, "symbol", "GBPJPY"))
    report = format_walk_forward_ascii(res)
    print("\n" + report + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "wfe": res.wfe,
                "passed": res.passed,
                "folds": res.folds_total,
                "mean_is_return": res.mean_is_return,
                "mean_oos_return": res.mean_oos_return,
            }, f, indent=2)

    return 0


def handle_monte_carlo(args: argparse.Namespace) -> int:
    logger.info(f"Starting Monte Carlo Simulation Mode ({args.iterations} iterations)...")
    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
    elif generate_synthetic_data is not None:
        df = generate_synthetic_data(bars=args.bars)
    else:
        from gbpjpy.tests.fixtures import generate_synthetic_m5_data
        df = generate_synthetic_m5_data(bars=args.bars)

    if DataEngine is not None and "atr" not in df.columns:
        de = DataEngine()
        df = de.compute_indicators(df)

    strat = GBPJPYDualRegimeScalper() if GBPJPYDualRegimeScalper is not None else None
    engine = BacktestEngine(initial_balance=args.initial_balance, symbol=getattr(args, "symbol", "GBPJPY"), strategy=strat)
    result = engine.run(df=df, strategy=strat)

    pnls = [t.get("net_pnl_usd", 0.0) if isinstance(t, dict) else float(t) for t in result.trades] if result.trades else []
    val = WalkForwardValidator(initial_balance=args.initial_balance)
    mc_res = val.monte_carlo_simulation(pnls, initial_balance=args.initial_balance, iterations=args.iterations)

    report = format_monte_carlo_ascii(mc_res, initial_balance=args.initial_balance)
    print("\n" + report + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dict(mc_res), f, indent=2)

    return 0


def handle_stream(args: argparse.Namespace) -> int:
    logger.info(f"Starting Stream Paper-Trading Mode (Cycles: {args.cycles or 'Continuous'})...")
    cfg = load_config() if load_config is not None else None
    client = MockMT5Client(initial_balance=args.initial_balance)
    client.connect()

    risk_mgr = RiskManager(cfg) if RiskManager is not None and cfg is not None else None
    exec_mgr = ExecutionManager(cfg, client, risk_mgr) if ExecutionManager is not None and cfg is not None and risk_mgr is not None else None

    cycles_run = 0
    while True:
        cycles_run += 1
        account = client.get_account_info()
        logger.info(
            f"Stream Cycle #{cycles_run}: Balance=${account.balance:.2f}, Margin Level={account.margin_level:.1f}%, Open Positions={len(client.get_open_positions())}"
        )
        if args.cycles > 0 and cycles_run >= args.cycles:
            break
        time.sleep(0.01)

    logger.info("Stream execution cycle limit reached. Exiting cleanly.")
    return 0


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gbpjpy",
        description="GBPJPY Quantitative Trading Framework - Unified Entry Point (Milestone 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="backtest",
                        choices=["backtest", "walk-forward", "monte-carlo", "stream"],
                        help="Execution mode.")
    parser.add_argument("--symbol", type=str, default="GBPJPY",
                        help="Currency pair symbol (default: GBPJPY).")
    parser.add_argument("--data", type=str, default="", help="Path to historical CSV data.")
    parser.add_argument("--bars", type=int, default=6000, help="Bars of synthetic data.")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--initial-balance", type=float, default=250.0, help="Initial balance.")
    parser.add_argument("--leverage", type=float, default=30.0, help="Leverage ratio.")
    parser.add_argument("--spread-pips", type=float, default=1.2, help="Spread in pips.")
    parser.add_argument("--commission", type=float, default=7.0, help="Commission/lot.")
    parser.add_argument("--slippage", type=float, default=0.2, help="Slippage in pips.")
    parser.add_argument("--folds", "--windows", dest="folds", type=int, default=24,
                        help="Walk-forward folds / evaluation windows.")
    parser.add_argument("--iterations", "--simulations", dest="iterations", type=int, default=1000,
                        help="Monte Carlo iterations / simulation runs.")
    parser.add_argument("--cycles", type=int, default=0, help="Stream mode max cycles.")
    parser.add_argument("--output", type=str, default=None, help="Output export JSON path.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output.")
    parsed = parser.parse_args(args)
    if not hasattr(parsed, "windows"):
        parsed.windows = parsed.folds
    if not hasattr(parsed, "simulations"):
        parsed.simulations = parsed.iterations
    return parsed


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = parse_arguments(args)
    if parsed.mode == "backtest":
        return handle_backtest(parsed)
    elif parsed.mode == "walk-forward":
        return handle_walk_forward(parsed)
    elif parsed.mode == "monte-carlo":
        return handle_monte_carlo(parsed)
    elif parsed.mode == "stream":
        return handle_stream(parsed)
    return 1


if __name__ == "__main__":
    sys.exit(main())

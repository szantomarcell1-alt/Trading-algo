"""
Quantitative Performance Metrics Module
Calculates Sharpe, Sortino, Calmar, Win Rate, Drawdowns, and Expectancy.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    initial_balance: float
    final_balance: float
    total_net_profit: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    average_trade_pnl: float
    average_win: float
    average_loss: float
    payoff_ratio: float
    max_drawdown_amount: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_daily_drawdown_pct: float = 0.0


def calculate_metrics(trades: List[Dict[str, Any]], initial_balance: float = 1000.0) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(
            initial_balance=initial_balance,
            final_balance=initial_balance,
            total_net_profit=0.0,
            total_return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            average_trade_pnl=0.0,
            average_win=0.0,
            average_loss=0.0,
            payoff_ratio=0.0,
            max_drawdown_amount=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            max_daily_drawdown_pct=0.0,
        )

    pnls = [t["profit"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_net_profit = sum(pnls)
    final_balance = initial_balance + total_net_profit
    total_return_pct = (total_net_profit / initial_balance) * 100.0

    total_trades = len(trades)
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate_pct = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else (99.0 if sum_wins > 0 else 0.0)

    average_trade_pnl = total_net_profit / total_trades if total_trades > 0 else 0.0
    average_win = (sum_wins / winning_trades) if winning_trades > 0 else 0.0
    average_loss = (sum_losses / losing_trades) if losing_trades > 0 else 0.0
    payoff_ratio = (average_win / average_loss) if average_loss > 0 else 0.0

    # Equity curve & Drawdowns
    balance_curve = [initial_balance]
    for p in pnls:
        balance_curve.append(balance_curve[-1] + p)

    balance_series = pd.Series(balance_curve)
    peak = balance_series.cummax()
    drawdown_series = balance_series - peak
    drawdown_pct_series = (drawdown_series / peak) * 100.0

    max_drawdown_amount = abs(float(drawdown_series.min()))
    max_drawdown_pct = abs(float(drawdown_pct_series.min()))

    # Returns & Sharpe / Sortino
    returns = pd.Series(pnls) / initial_balance
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1) if len(returns) > 1 else 0.0

    # Scalping frequency annualized scaling factor (assuming ~252 trading days, ~10 trades/day)
    annual_factor = np.sqrt(252 * 10)

    sharpe_ratio = (mean_ret / (std_ret + 1e-9)) * annual_factor if std_ret > 0 else 0.0

    # Downside deviation
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else 0.0
    sortino_ratio = (mean_ret / (downside_std + 1e-9)) * annual_factor if downside_std > 0 else 0.0

    # Calmar Ratio (Annualized Return / Max Drawdown %)
    calmar_ratio = (total_return_pct / (max_drawdown_pct + 1e-9)) if max_drawdown_pct > 0 else 0.0

    # Streaks
    max_consec_wins = 0
    cur_consec_wins = 0
    max_consec_losses = 0
    cur_consec_losses = 0

    for p in pnls:
        if p > 0:
            cur_consec_wins += 1
            cur_consec_losses = 0
            max_consec_wins = max(max_consec_wins, cur_consec_wins)
        elif p < 0:
            cur_consec_losses += 1
            cur_consec_wins = 0
            max_consec_losses = max(max_consec_losses, cur_consec_losses)

    # Max Daily Drawdown
    max_daily_drawdown_pct = 0.0
    daily_groups: Dict[Any, List[float]] = {}
    current_bal = initial_balance

    for t in trades:
        raw_time = t.get("exit_time") or t.get("entry_time")
        trade_date = None
        if raw_time is not None:
            if isinstance(raw_time, (datetime, pd.Timestamp)):
                trade_date = raw_time.date()
            elif isinstance(raw_time, str):
                try:
                    trade_date = pd.to_datetime(raw_time).date()
                except Exception:
                    trade_date = None
        if trade_date is None:
            trade_date = "default"

        if trade_date not in daily_groups:
            daily_groups[trade_date] = []
        daily_groups[trade_date].append(t.get("profit", 0.0))

    for d, day_pnls in daily_groups.items():
        day_start = current_bal
        day_peak = day_start
        day_running = day_start
        max_day_dd = 0.0
        for p in day_pnls:
            day_running += p
            if day_running > day_peak:
                day_peak = day_running
            dd_from_start = ((day_start - day_running) / day_start * 100.0) if day_start > 0 else 0.0
            dd_from_peak = ((day_peak - day_running) / day_peak * 100.0) if day_peak > 0 else 0.0
            dd = max(dd_from_start, dd_from_peak, 0.0)
            if dd > max_day_dd:
                max_day_dd = dd
        if max_day_dd > max_daily_drawdown_pct:
            max_daily_drawdown_pct = max_day_dd
        current_bal = day_running

    return BacktestMetrics(
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_net_profit=round(total_net_profit, 2),
        total_return_pct=round(total_return_pct, 2),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=round(win_rate_pct, 2),
        profit_factor=round(profit_factor, 2),
        average_trade_pnl=round(average_trade_pnl, 2),
        average_win=round(average_win, 2),
        average_loss=round(average_loss, 2),
        payoff_ratio=round(payoff_ratio, 2),
        max_drawdown_amount=round(max_drawdown_amount, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        sharpe_ratio=round(sharpe_ratio, 2),
        sortino_ratio=round(sortino_ratio, 2),
        calmar_ratio=round(calmar_ratio, 2),
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
        max_daily_drawdown_pct=round(max_daily_drawdown_pct, 2),
    )

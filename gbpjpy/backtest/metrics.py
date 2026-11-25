"""
GBPJPY Institutional Performance Metrics Module (Milestone 4)
Calculates institutional weekly return distribution, ~1% weekly consistency,
Sharpe, Sortino, Calmar, Recovery Factor, and peak-to-trough drawdowns.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Optional, Any, Union, Sequence
import math
import numpy as np
import pandas as pd


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts a value to float, handling None and non-convertible types."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class WeeklyReturn(float):
    """
    Float subclass representing weekly return percentage, with dict-like and attribute metadata.
    Enables seamless math operations (numpy, std, mean) and dictionary/attribute access.
    """
    def __new__(
        cls,
        value: float,
        year: int = 0,
        week: int = 0,
        net_pnl_usd: float = 0.0,
        trades_count: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
    ):
        obj = super().__new__(cls, float(value))
        obj.year = year
        obj.week = week
        obj.net_pnl_usd = round(float(net_pnl_usd), 4)
        obj.return_pct = round(float(value), 4)
        obj.trades_count = trades_count
        obj.winning_trades = winning_trades
        obj.losing_trades = losing_trades
        obj.week_key = f"{year}-W{week:02d}"
        obj.is_target_hit = bool(obj.return_pct >= 1.0)
        obj.is_profitable = bool(obj.return_pct > 0.0)
        obj._data = {
            "year": year,
            "week": week,
            "week_key": obj.week_key,
            "net_pnl_usd": obj.net_pnl_usd,
            "return_pct": obj.return_pct,
            "trades_count": trades_count,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "is_target_hit": obj.is_target_hit,
            "is_profitable": obj.is_profitable,
        }
        return obj

    def __getattr__(self, item: str) -> Any:
        if hasattr(self, "_data") and item in self._data:
            return self._data[item]
        raise AttributeError(f"'WeeklyReturn' object has no attribute '{item}'")

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self._data.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"WeeklyReturn({float(self):.2f}%, {self._data['week_key']}, pnl=${self.net_pnl_usd:.2f})"


@dataclass
class WeeklyPerformance:
    """Detailed summary of a single ISO calendar week."""
    week_key: str              # e.g. "2026-W02"
    net_pnl_usd: float
    return_pct: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    target_hit: bool           # return_pct >= 1.0


class PerformanceMetrics(dict):
    """
    Institutional Performance Metrics container.
    Inherits from dict and supports property/attribute access for all metrics and aliases.
    """
    _ALIASES = {
        "total_net_profit": "net_pnl",
        "net_pnl": "total_net_profit",
        "percent_weeks_target_hit": "target_hit_rate",
        "target_hit_rate": "percent_weeks_target_hit",
        "pct_weeks_above_1pct": "percent_weeks_target_hit",
        "mean_weekly_return": "avg_weekly_return",
        "avg_weekly_return": "mean_weekly_return",
        "max_drawdown": "max_drawdown_pct",
        "max_drawdown_pct": "max_drawdown",
        "pct_profitable_weeks": "percent_profitable_weeks",
        "percent_profitable_weeks": "pct_profitable_weeks",
        "avg_win_usd": "average_win",
        "average_win": "avg_win_usd",
        "avg_loss_usd": "average_loss",
        "average_loss": "avg_loss_usd",
        "win_rate": "win_rate_pct",
        "win_rate_pct": "win_rate",
    }

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()
        initial_data = dict(data or {})
        initial_data.update(kwargs)
        for k, v in initial_data.items():
            self[k] = v

    def __getattr__(self, name: str) -> Any:
        if super().__contains__(name):
            return super().__getitem__(name)
        alt = self._ALIASES.get(name)
        if alt and super().__contains__(alt):
            return super().__getitem__(alt)
        raise AttributeError(f"'PerformanceMetrics' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        alt = self._ALIASES.get(key)
        if alt and super().__contains__(alt):
            return super().__getitem__(alt)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        alt = self._ALIASES.get(key)
        if alt and super().__contains__(alt):
            return super().__getitem__(alt)
        return default

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        alt = self._ALIASES.get(str(key))
        if alt and super().__contains__(alt):
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


# Type alias for backward compatibility
BacktestMetrics = PerformanceMetrics


def _extract_trade_time(trade: Union[dict, Any]) -> Optional[datetime]:
    """Helper to extract datetime from exit_time or entry_time."""
    t = trade.get("exit_time") if isinstance(trade, dict) else getattr(trade, "exit_time", None)
    if t is None:
        t = trade.get("entry_time") if isinstance(trade, dict) else getattr(trade, "entry_time", None)
    if t is None:
        return None
    if isinstance(t, datetime):
        return t
    if isinstance(t, str):
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            try:
                res = pd.to_datetime(t)
                if pd.isna(res):
                    return None
                return res.to_pydatetime()
            except Exception:
                return None
    if hasattr(t, "to_pydatetime"):
        try:
            return t.to_pydatetime()
        except Exception:
            return None
    return None


def calculate_weekly_returns(
    trades: Sequence[Union[dict, Any]],
    initial_balance: float = 250.0,
) -> List[WeeklyReturn]:
    """
    Aggregates trade returns by ISO calendar week.
    Returns: List of WeeklyReturn instances in chronological order.
    Each WeeklyReturn behaves as a float (percentage) and provides dict/attribute metadata.
    """
    if not trades or initial_balance <= 0:
        return []

    # Group trades chronologically by (iso_year, iso_week)
    weeks: Dict[tuple, List[Union[dict, Any]]] = {}
    for tr in trades:
        t = _extract_trade_time(tr)
        if t is None:
            continue
        iso = t.isocalendar()
        key = (iso[0], iso[1])  # (year, week)
        if key not in weeks:
            weeks[key] = []
        weeks[key].append(tr)

    sorted_keys = sorted(weeks.keys())
    weekly_results: List[WeeklyReturn] = []

    for year, week in sorted_keys:
        w_trades = weeks[(year, week)]
        net_pnl = sum(
            _safe_float(tr.get("net_pnl_usd") if isinstance(tr, dict) else getattr(tr, "net_pnl_usd", 0.0), 0.0)
            for tr in w_trades
        )
        ret_pct = (net_pnl / initial_balance) * 100.0
        winning = sum(
            1 for tr in w_trades
            if _safe_float(tr.get("net_pnl_usd") if isinstance(tr, dict) else getattr(tr, "net_pnl_usd", 0.0), 0.0) > 0
        )
        losing = sum(
            1 for tr in w_trades
            if _safe_float(tr.get("net_pnl_usd") if isinstance(tr, dict) else getattr(tr, "net_pnl_usd", 0.0), 0.0) < 0
        )

        weekly_obj = WeeklyReturn(
            value=round(ret_pct, 4),
            year=year,
            week=week,
            net_pnl_usd=round(net_pnl, 4),
            trades_count=len(w_trades),
            winning_trades=winning,
            losing_trades=losing,
        )
        weekly_results.append(weekly_obj)

    return weekly_results


def aggregate_weekly_returns(
    trades: Sequence[Union[dict, Any]],
    initial_balance: float = 250.0,
) -> List[WeeklyReturn]:
    """Alias for calculate_weekly_returns."""
    return calculate_weekly_returns(trades, initial_balance)


def calculate_percent_weeks_target_hit(
    trades: Sequence[Union[dict, Any]],
    initial_balance: float = 250.0,
    target_pct: float = 1.0,
) -> float:
    """
    Calculates percentage of calendar weeks achieving >= target_pct return (e.g. 75.0 for 75.0%).
    """
    weekly = calculate_weekly_returns(trades, initial_balance)
    if not weekly:
        return 0.0
    hits = sum(1 for r in weekly if float(r) >= target_pct)
    return round((hits / len(weekly)) * 100.0, 2)


def calculate_sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """
    Calculates annualized Sharpe ratio from return series.
    Formula: (mean(excess) / std(excess)) * sqrt(periods_per_year).
    """
    if returns is None or len(returns) < 2:
        return 0.0

    r = np.array([float(x) for x in returns], dtype=float)
    # Check if returns are in percentage scale (> 0.20 on average) or decimal
    # Scale invariance: (c*mean)/(c*std) == mean/std, but risk_free_rate must match scale
    rf_per_period = (risk_free_rate / periods_per_year)
    excess = r - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std <= 1e-9:
        return 0.0

    mean = float(np.mean(excess))
    sharpe = (mean / std) * math.sqrt(periods_per_year)
    return round(float(sharpe), 4)


def calculate_sortino_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """
    Calculates annualized Sortino ratio isolating downside semi-deviation below risk_free_rate.
    Formula: (mean(excess) / downside_std) * sqrt(periods_per_year).
    """
    if returns is None or len(returns) == 0:
        return 0.0

    r = np.array([float(x) for x in returns], dtype=float)
    rf_per_period = (risk_free_rate / periods_per_year)
    excess = r - rf_per_period
    mean = float(np.mean(excess))

    downside = np.minimum(0.0, excess)
    downside_var = float(np.mean(downside ** 2))

    if downside_var <= 1e-12:
        return 999.0 if mean > 0 else 0.0

    downside_std = math.sqrt(downside_var)
    sortino = (mean / downside_std) * math.sqrt(periods_per_year)
    return round(float(sortino), 4)


def calculate_max_drawdown(
    equity_series: Union[Sequence[float], pd.Series, pd.DataFrame, np.ndarray],
) -> float:
    """
    Calculates maximum peak-to-trough drawdown percentage (e.g. 5.0 for 5.0%).
    """
    if equity_series is None:
        return 0.0

    if isinstance(equity_series, pd.DataFrame):
        if "equity" in equity_series.columns:
            arr = equity_series["equity"].to_numpy(dtype=float)
        elif "balance" in equity_series.columns:
            arr = equity_series["balance"].to_numpy(dtype=float)
        else:
            arr = equity_series.iloc[:, 0].to_numpy(dtype=float)
    elif isinstance(equity_series, pd.Series):
        arr = equity_series.to_numpy(dtype=float)
    else:
        arr = np.array([float(x) for x in equity_series], dtype=float)

    if len(arr) == 0:
        return 0.0

    peaks = np.maximum.accumulate(arr)
    # Avoid zero division
    safe_peaks = np.where(peaks > 0, peaks, 1.0)
    dd_pct = (peaks - arr) / safe_peaks * 100.0
    max_dd = float(np.max(dd_pct))
    return round(max(0.0, max_dd), 4)


def calculate_profit_factor(
    trades: Sequence[Union[dict, Any]],
) -> float:
    """
    Calculates profit factor (gross profits / gross losses).
    Returns 999.0 if gross loss is zero and gross profit > 0, or 0.0 if both are zero.
    """
    if not trades:
        return 0.0

    gross_profit = 0.0
    gross_loss = 0.0

    for tr in trades:
        val = tr.get("gross_pnl_usd") if isinstance(tr, dict) else getattr(tr, "gross_pnl_usd", None)
        if val is None:
            val = tr.get("net_pnl_usd", 0.0) if isinstance(tr, dict) else getattr(tr, "net_pnl_usd", 0.0)
        val = _safe_float(val, 0.0)
        if val > 0:
            gross_profit += val
        elif val < 0:
            gross_loss += abs(val)

    if gross_loss > 0:
        return round(gross_profit / gross_loss, 4)
    elif gross_profit > 0:
        return 999.0
    else:
        return 0.0


def calculate_metrics(
    trades: Sequence[Union[dict, Any]],
    initial_balance: float = 250.0,
    equity_curve: Optional[pd.DataFrame] = None,
) -> PerformanceMetrics:
    """
    Comprehensive institutional metrics calculation entry point.
    Returns PerformanceMetrics container with dual attribute/dictionary access and aliases.
    """
    total_trades = len(trades) if trades else 0

    if total_trades == 0:
        # Edge case: 0 trades
        m = PerformanceMetrics({
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate": 0.0,
            "win_rate_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "total_net_profit": 0.0,
            "net_pnl": 0.0,
            "initial_balance": float(initial_balance),
            "final_balance": float(initial_balance),
            "total_return_pct": 0.0,
            "average_win": 0.0,
            "avg_win_usd": 0.0,
            "average_loss": 0.0,
            "avg_loss_usd": 0.0,
            "payoff_ratio": 0.0,
            "expectancy": 0.0,
            "total_pips": 0.0,
            "total_commission": 0.0,
            "weekly_returns": [],
            "weekly_details": [],
            "percent_weeks_target_hit": 0.0,
            "target_hit_rate": 0.0,
            "pct_weeks_above_1pct": 0.0,
            "mean_weekly_return": 0.0,
            "avg_weekly_return": 0.0,
            "median_weekly_return": 0.0,
            "percent_profitable_weeks": 0.0,
            "pct_profitable_weeks": 0.0,
            "best_week_pct": 0.0,
            "worst_week_pct": 0.0,
            "max_consecutive_win_weeks": 0,
            "max_consecutive_loss_weeks": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "recovery_factor": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_amount": 0.0,
            "average_drawdown_pct": 0.0,
            "max_daily_drawdown_pct": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "average_holding_duration_mins": 0.0,
            "equity_curve": equity_curve if equity_curve is not None else pd.DataFrame([{"time": datetime.now(), "balance": initial_balance, "equity": initial_balance, "drawdown": 0.0, "drawdown_pct": 0.0}]),
        })
        return m

    # 1. PnL and trade counts
    gross_profit = 0.0
    gross_loss = 0.0
    net_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    breakeven_trades = 0
    total_commission = 0.0
    total_pips = 0.0
    holding_durations_min: List[float] = []

    consec_wins = 0
    max_consec_wins = 0
    consec_losses = 0
    max_consec_losses = 0

    daily_pnls: Dict[date, float] = {}

    for tr in trades:
        # PnL
        pnl = _safe_float(tr.get("net_pnl_usd") if isinstance(tr, dict) else getattr(tr, "net_pnl_usd", 0.0), 0.0)
        comm = _safe_float(tr.get("commission") if isinstance(tr, dict) else getattr(tr, "commission", 0.0), 0.0)
        pips = _safe_float(tr.get("pips") if isinstance(tr, dict) else getattr(tr, "pips", 0.0), 0.0)
        g_val = tr.get("gross_pnl_usd") if isinstance(tr, dict) else getattr(tr, "gross_pnl_usd", None)
        if g_val is None:
            g_val = pnl + comm
        else:
            g_val = _safe_float(g_val, pnl + comm)

        net_pnl += pnl
        total_commission += comm
        total_pips += pips

        if g_val > 0:
            gross_profit += g_val
        elif g_val < 0:
            gross_loss += abs(g_val)

        if pnl > 0:
            winning_trades += 1
            consec_wins += 1
            consec_losses = 0
            if consec_wins > max_consec_wins:
                max_consec_wins = consec_wins
        elif pnl < 0:
            losing_trades += 1
            consec_losses += 1
            consec_wins = 0
            if consec_losses > max_consec_losses:
                max_consec_losses = consec_losses
        else:
            breakeven_trades += 1
            consec_wins = 0
            consec_losses = 0

        # Holding duration
        t_entry = tr.get("entry_time") if isinstance(tr, dict) else getattr(tr, "entry_time", None)
        t_exit = tr.get("exit_time") if isinstance(tr, dict) else getattr(tr, "exit_time", None)
        if t_entry is not None and t_exit is not None:
            try:
                diff_sec = (pd.to_datetime(t_exit) - pd.to_datetime(t_entry)).total_seconds()
                holding_durations_min.append(max(0.0, diff_sec / 60.0))
            except Exception:
                pass

        # Daily PnL for daily drawdown guard
        t_for_day = _extract_trade_time(tr)
        if t_for_day is not None:
            d_key = t_for_day.date()
            daily_pnls[d_key] = daily_pnls.get(d_key, 0.0) + pnl

    final_balance = initial_balance + net_pnl
    total_return_pct = (net_pnl / initial_balance) * 100.0 if initial_balance > 0 else 0.0
    win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0

    avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
    avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    expectancy = (net_pnl / total_trades) if total_trades > 0 else 0.0
    profit_factor = calculate_profit_factor(trades)

    avg_holding_min = float(np.mean(holding_durations_min)) if holding_durations_min else 0.0

    # 2. Weekly Returns & Institutional Target (~1% weekly)
    weekly_returns = calculate_weekly_returns(trades, initial_balance)
    weekly_float_vals = [float(r) for r in weekly_returns]
    num_weeks = len(weekly_returns)

    if num_weeks > 0:
        pct_target_hit = calculate_percent_weeks_target_hit(trades, initial_balance, target_pct=1.0)
        mean_weekly = float(np.mean(weekly_float_vals))
        median_weekly = float(np.median(weekly_float_vals))
        profitable_weeks = sum(1 for r in weekly_float_vals if r > 0)
        pct_profitable_weeks = (profitable_weeks / num_weeks) * 100.0
        best_week = float(np.max(weekly_float_vals))
        worst_week = float(np.min(weekly_float_vals))

        # Consecutive winning / losing weeks
        cw = 0
        max_cw = 0
        cl = 0
        max_cl = 0
        for r in weekly_float_vals:
            if r > 0:
                cw += 1
                cl = 0
                max_cw = max(max_cw, cw)
            elif r < 0:
                cl += 1
                cw = 0
                max_cl = max(max_cl, cl)
            else:
                cw = 0
                cl = 0
        max_consec_win_weeks = max_cw
        max_consec_loss_weeks = max_cl

        # Sharpe & Sortino (annualized from weekly series, assuming 52 weeks)
        sharpe = calculate_sharpe_ratio([r / 100.0 for r in weekly_float_vals], risk_free_rate=0.0, periods_per_year=52)
        sortino = calculate_sortino_ratio([r / 100.0 for r in weekly_float_vals], risk_free_rate=0.0, periods_per_year=52)
    else:
        pct_target_hit = 0.0
        mean_weekly = 0.0
        median_weekly = 0.0
        pct_profitable_weeks = 0.0
        best_week = 0.0
        worst_week = 0.0
        max_consec_win_weeks = 0
        max_consec_loss_weeks = 0
        sharpe = 0.0
        sortino = 0.0

    # 3. Drawdown calculation
    if equity_curve is not None and len(equity_curve) > 0:
        if "equity" in equity_curve.columns:
            eq_series = equity_curve["equity"].to_numpy(dtype=float)
        elif "balance" in equity_curve.columns:
            eq_series = equity_curve["balance"].to_numpy(dtype=float)
        else:
            eq_series = equity_curve.iloc[:, 0].to_numpy(dtype=float)
    else:
        # Construct equity curve from trades
        eq_list = [float(initial_balance)]
        cur = float(initial_balance)
        for tr in trades:
            cur += _safe_float(tr.get("net_pnl_usd") if isinstance(tr, dict) else getattr(tr, "net_pnl_usd", 0.0), 0.0)
            eq_list.append(cur)
        eq_series = np.array(eq_list, dtype=float)

    peaks = np.maximum.accumulate(eq_series)
    safe_peaks = np.where(peaks > 0, peaks, 1.0)
    dd_pct_arr = (peaks - eq_series) / safe_peaks * 100.0
    dd_amt_arr = peaks - eq_series

    max_dd_pct = round(float(np.max(dd_pct_arr)), 4) if len(dd_pct_arr) > 0 else 0.0
    max_dd_amt = round(float(np.max(dd_amt_arr)), 4) if len(dd_amt_arr) > 0 else 0.0
    avg_dd_pct = round(float(np.mean(dd_pct_arr)), 4) if len(dd_pct_arr) > 0 else 0.0

    # Calmar & Recovery Factor
    annualized_return_pct = mean_weekly * 52.0
    calmar = round(annualized_return_pct / max_dd_pct, 4) if max_dd_pct > 0 else (999.0 if annualized_return_pct > 0 else 0.0)
    recovery_factor = round(net_pnl / max_dd_amt, 4) if max_dd_amt > 0 else (999.0 if net_pnl > 0 else 0.0)

    # Max daily drawdown percentage across all trading days
    max_daily_dd_pct = 0.0
    if initial_balance > 0:
        for d, d_pnl in daily_pnls.items():
            if d_pnl < 0:
                d_dd = (abs(d_pnl) / initial_balance) * 100.0
                if d_dd > max_daily_dd_pct:
                    max_daily_dd_pct = d_dd

    metrics_dict = {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "breakeven_trades": breakeven_trades,
        "win_rate": round(win_rate, 2),
        "win_rate_pct": round(win_rate, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": profit_factor,
        "total_net_profit": round(net_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "initial_balance": round(float(initial_balance), 2),
        "final_balance": round(float(final_balance), 2),
        "total_return_pct": round(total_return_pct, 2),
        "average_win": round(avg_win, 2),
        "avg_win_usd": round(avg_win, 2),
        "average_loss": round(avg_loss, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "expectancy": round(expectancy, 4),
        "total_pips": round(total_pips, 2),
        "total_commission": round(total_commission, 2),
        "weekly_returns": weekly_returns,
        "weekly_details": [r.to_dict() for r in weekly_returns],
        "percent_weeks_target_hit": round(pct_target_hit, 2),
        "target_hit_rate": round(pct_target_hit, 2),
        "pct_weeks_above_1pct": round(pct_target_hit, 2),
        "mean_weekly_return": round(mean_weekly, 2),
        "avg_weekly_return": round(mean_weekly, 2),
        "median_weekly_return": round(median_weekly, 2),
        "percent_profitable_weeks": round(pct_profitable_weeks, 2),
        "pct_profitable_weeks": round(pct_profitable_weeks, 2),
        "best_week_pct": round(best_week, 2),
        "worst_week_pct": round(worst_week, 2),
        "max_consecutive_win_weeks": max_consec_win_weeks,
        "max_consecutive_loss_weeks": max_consec_loss_weeks,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "recovery_factor": recovery_factor,
        "max_drawdown": max_dd_pct,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_amount": max_dd_amt,
        "average_drawdown_pct": avg_dd_pct,
        "max_daily_drawdown_pct": round(max_daily_dd_pct, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "average_holding_duration_mins": round(avg_holding_min, 1),
        "equity_curve": equity_curve,
    }

    return PerformanceMetrics(metrics_dict)

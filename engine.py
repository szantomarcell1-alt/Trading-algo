"""
Event-Driven Backtesting Engine
Models realistic CFD scalping friction: variable spreads, commissions, slippage,
Partial Take-Profit (50%), Commission-Adjusted Breakeven, and ATR Trailing Stops.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging

from utils.market_hours import MarketHoursGuard
from config.settings import AppConfig
from core.mt5_client import SymbolSpecs, AccountSnapshot
from core.data_engine import DataEngine
from strategies.titan_scalper import TitanScalperStrategy
from strategies.signals import SignalType
from risk.risk_manager import RiskManager
from .metrics import calculate_metrics, BacktestMetrics

logger = logging.getLogger("TitanScalper.Backtest")


class BacktestEngine:
    def __init__(
        self,
        config: AppConfig,
        initial_balance: float = 1000.0,
        commission_per_lot_round_turn: float = 7.0,
        fixed_spread_pips: float = 0.8,
        slippage_points: float = 1.0,
    ):
        self.config = config
        self.initial_balance = initial_balance
        self.commission_per_lot = commission_per_lot_round_turn
        self.fixed_spread_pips = fixed_spread_pips
        self.slippage_points = slippage_points

        self.strategy = TitanScalperStrategy(config)
        self.risk_manager = RiskManager(config)
        self.market_hours_guard = MarketHoursGuard(
    allowed_start_utc_hour=config.session_filter.allowed_start_utc_hour,
    allowed_end_utc_hour=config.session_filter.allowed_end_utc_hour,
    avoid_rollover_window=config.session_filter.avoid_rollover_window,
    avoid_friday_close=config.session_filter.avoid_friday_close,
)

    @staticmethod
    def _build_symbol_specs(symbol: str) -> SymbolSpecs:
        sym_u = symbol.upper()
        if "XAU" in sym_u or "GOLD" in sym_u:
            return SymbolSpecs(
                name=symbol, digits=2, point=0.01, tick_value=1.0, tick_size=0.01,
                contract_size=100.0, min_volume=0.01, max_volume=100.0, volume_step=0.01, pip_size=0.10
            )
        return SymbolSpecs(
            name=symbol, digits=5, point=0.00001, tick_value=1.0, tick_size=0.00001,
            contract_size=100000.0, min_volume=0.01, max_volume=100.0, volume_step=0.01, pip_size=0.0001
        )

    def run(
        self,
        df_m1: pd.DataFrame,
        df_m5: Optional[pd.DataFrame] = None,
        df_m15: Optional[pd.DataFrame] = None,
        symbol: str = "EURUSD",
        df_h1: Optional[pd.DataFrame] = None,
    ) -> BacktestMetrics:
        specs = self._build_symbol_specs(symbol)

        df_m1_proc = DataEngine.compute_indicators(df_m1)
        df_m5_proc = DataEngine.compute_indicators(df_m5) if df_m5 is not None else df_m1_proc
        df_m15_proc = DataEngine.compute_indicators(df_m15) if df_m15 is not None else df_m1_proc
        df_h1_proc = DataEngine.compute_indicators(df_h1) if df_h1 is not None else None

        balance = self.initial_balance
        equity = balance
        trades: List[Dict[str, Any]] = []

        open_position: Optional[Dict[str, Any]] = None
        spread_price_diff = self.fixed_spread_pips * specs.pip_size
        slippage_price_diff = self.slippage_points * specs.point

        min_warmup = 110
        total_bars = len(df_m1_proc)
        last_signal_bar_time = None

        last_m5_time = None
        cached_sub_m5 = None
        cached_sub_m15 = None
        cached_sub_h1 = None

        for i in range(min_warmup, total_bars):
            current_bar = df_m1_proc.iloc[i]
            bar_time = current_bar.name

            # 1. Manage existing open position
            if open_position is not None:
                is_buy = open_position["type"] == SignalType.BUY
                high = float(current_bar["high"])
                low = float(current_bar["low"])
                close_p = float(current_bar["close"])
                atr = float(current_bar["atr"])
                open_price = open_position["entry_price"]
                sl = open_position["sl"]
                tp = open_position["tp"]
                risk_dist = open_position["risk_dist"]

                # Check Stop Loss & Full Take Profit first on current bar extremes
                sl_hit = (low <= sl) if is_buy else (high >= sl)
                tp_hit = (high >= tp) if is_buy else (low <= tp)

                closed = False
                exit_price = 0.0
                exit_reason = ""

                if sl_hit:
                    exit_price = sl - (slippage_price_diff if is_buy else -slippage_price_diff)
                    exit_reason = "STOP_LOSS"
                    closed = True
                elif tp_hit:
                    exit_price = tp
                    exit_reason = "TAKE_PROFIT"
                    closed = True
                else:
                    # Intra-bar favorable excursion in R
                    curr_max_dist = (high - open_price) if is_buy else (open_price - low)
                    curr_max_r = (curr_max_dist / risk_dist) if risk_dist > 0 else 0.0

                    # A) Partial Take-Profit (50% at 1.5R)
                    if (
                        self.config.execution.use_partial_take_profit
                        and not open_position["partial_applied"]
                        and curr_max_r >= self.config.execution.partial_tp_trigger_r
                    ):
                        part_vol = round(int((open_position["volume"] * self.config.execution.partial_tp_ratio) / specs.volume_step) * specs.volume_step, 2)
                        rem_vol = round(open_position["volume"] - part_vol, 2)
                        if part_vol >= specs.min_volume and rem_vol >= specs.min_volume:
                            part_exit_p = (
                                open_price + risk_dist * self.config.execution.partial_tp_trigger_r
                                if is_buy
                                else open_price - risk_dist * self.config.execution.partial_tp_trigger_r
                            )
                            p_diff = (part_exit_p - open_price) if is_buy else (open_price - part_exit_p)
                            p_gross = p_diff * specs.contract_size * part_vol
                            p_comm = self.commission_per_lot * part_vol
                            open_position["realized_partial_pnl"] += (p_gross - p_comm)
                            open_position["volume"] = rem_vol
                        open_position["partial_applied"] = True

                    # B) Breakeven with commission buffer (min 0.8 pip)
                    if (
                        self.config.execution.use_breakeven
                        and not open_position["be_applied"]
                        and curr_max_r >= self.config.execution.breakeven_trigger_r
                    ):
                        eff_buf_pips = max(self.config.execution.breakeven_buffer_pips, 0.8)
                        be_sl = open_price + (eff_buf_pips * specs.pip_size * (1 if is_buy else -1))
                        if (is_buy and be_sl > open_position["sl"]) or (not is_buy and be_sl < open_position["sl"]):
                            open_position["sl"] = be_sl
                        open_position["be_applied"] = True

                    # C) Dynamic ATR Trailing Stop
                    if (
                        self.config.execution.use_trailing_stop
                        and curr_max_r >= self.config.execution.trailing_stop_activation_r
                    ):
                        trail_dist = atr * self.config.execution.trailing_stop_atr_multiplier
                        if is_buy:
                            cand_sl = close_p - trail_dist
                            if cand_sl > open_position["sl"] + specs.pip_size:
                                open_position["sl"] = cand_sl
                        else:
                            cand_sl = close_p + trail_dist
                            if cand_sl < open_position["sl"] - specs.pip_size:
                                open_position["sl"] = cand_sl

                    # D) Time-Decay Stop
                    open_position["bars_held"] += 1
                    if open_position["bars_held"] >= self.config.execution.max_holding_bars_m1:
                        close_dist = (close_p - open_price) if is_buy else (open_price - close_p)
                        close_r = (close_dist / risk_dist) if risk_dist > 0 else 0.0
                        if close_r < 0.5:
                            exit_price = close_p
                            exit_reason = "TIME_DECAY"
                            closed = True

                if closed:
                    price_diff = (exit_price - open_price) if is_buy else (open_price - exit_price)
                    gross_pnl = price_diff * specs.contract_size * open_position["volume"]
                    commission = self.commission_per_lot * open_position["volume"]
                    net_pnl = gross_pnl - commission + open_position["realized_partial_pnl"]

                    balance += net_pnl
                    equity = balance
                    self.risk_manager.register_trade_result(net_pnl, now=bar_time)

                    trades.append({
                        "entry_time": open_position["entry_time"],
                        "exit_time": bar_time,
                        "type": open_position["type"].value,
                        "volume": open_position["initial_volume"],
                        "entry_price": open_price,
                        "exit_price": exit_price,
                        "profit": net_pnl,
                        "reason": exit_reason,
                        "bars_held": open_position["bars_held"],
                        "strategy_mode": open_position.get("strategy_mode", "UNKNOWN"),
                        "rationale": open_position.get("rationale", ""),
                    })
                    open_position = None

            # 2. Check for new signal if no position is open
            if open_position is None:
                if self.config.session_filter.enabled:
                    bar_dt = bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time
                    session_ok, _ = self.market_hours_guard.is_trading_allowed(bar_dt)
                    if not session_ok:
                        continue

                sub_m1 = df_m1_proc.iloc[max(0, i - 300) : i + 1]
                closed_bar_time = sub_m1.index[-2]
                if last_signal_bar_time == closed_bar_time:
                    continue

                bid = float(current_bar["open"])
                ask = bid + spread_price_diff

                account_snap = AccountSnapshot(
                    login=12345,
                    balance=balance,
                    equity=equity,
                    margin=0.0,
                    margin_free=equity,
                    margin_level=9999.0,
                    profit=0.0,
                    currency=self.config.risk_management.account_currency,
                )

                can_trade, _ = self.risk_manager.can_open_new_trade(
                    account=account_snap,
                    open_positions_count=0,
                    symbol_positions_count=0,
                    now=bar_time,
                )

                if can_trade:
                    cur_m5_time = bar_time.floor("5min")
                    if last_m5_time != cur_m5_time or cached_sub_m5 is None:
                        last_m5_time = cur_m5_time
                        cached_sub_m5 = df_m5_proc.loc[:bar_time].tail(300) if df_m5_proc is not None else sub_m1
                        cached_sub_m15 = df_m15_proc.loc[:bar_time].tail(300) if df_m15_proc is not None else sub_m1
                        cached_sub_h1 = df_h1_proc.loc[:bar_time].tail(300) if df_h1_proc is not None else None

                    data_snapshot = {
                        "execution": sub_m1,
                        "structure": cached_sub_m5,
                        "regime": cached_sub_m15,
                        "macro": cached_sub_h1,
                        "m15": cached_sub_m15,
                        "h1": cached_sub_h1,
                    }

                    signal = self.strategy.evaluate(
                        symbol=symbol,
                        data_snapshot=data_snapshot,
                        symbol_specs=specs,
                        current_bid=bid,
                        current_ask=ask,
                        current_spread_pips=self.fixed_spread_pips,
                    )

                    if signal and signal.signal_type in (SignalType.BUY, SignalType.SELL):
                        last_signal_bar_time = closed_bar_time
                        vol = self.risk_manager.calculate_lot_size(
                            account=account_snap,
                            specs=specs,
                            risk_distance_points=signal.risk_distance_points,
                        )
                        if vol <= 0.0:
                            continue

                        entry_price = (
                            ask + slippage_price_diff
                            if signal.signal_type == SignalType.BUY
                            else bid - slippage_price_diff
                        )
                        open_position = {
                            "type": signal.signal_type,
                            "volume": vol,
                            "initial_volume": vol,
                            "entry_price": entry_price,
                            "sl": signal.sl_price,
                            "tp": signal.tp_price,
                            "risk_dist": abs(entry_price - signal.sl_price),
                            "entry_time": bar_time,
                            "bars_held": 0,
                            "be_applied": False,
                            "partial_applied": False,
                            "realized_partial_pnl": 0.0,
                            "strategy_mode": getattr(signal, "strategy_mode", "UNKNOWN"),
                            "rationale": getattr(signal, "rationale", ""),
                        }

        self.last_trades = trades
        return calculate_metrics(trades, initial_balance=self.initial_balance)
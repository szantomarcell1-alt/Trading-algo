"""
Trade Execution & Dynamic Position Management Module
Handles breakeven automation, partial profit taking, ATR trailing stops, and real-time decay exits.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import time
import logging

from config.settings import AppConfig
from core.mt5_client import MT5Client, OpenPosition, SymbolSpecs
from risk.risk_manager import RiskManager

logger = logging.getLogger("TitanScalper.ExecutionManager")


@dataclass
class PositionMetadata:
    ticket: int
    symbol: str
    is_buy: bool
    entry_price: float
    initial_sl: float
    initial_tp: float
    initial_risk_dist: float  # Absolute price distance to initial SL (1R)
    breakeven_applied: bool = False
    partial_tp_applied: bool = False
    entry_time: float = 0.0
    bars_held: int = 0


class ExecutionManager:
    def __init__(self, config: AppConfig, mt5_client: MT5Client, risk_manager: Optional[RiskManager] = None):
        self.config = config
        self.exec_cfg = config.execution
        self.mt5 = mt5_client
        self.risk_manager = risk_manager
        self.tracked_positions: Dict[int, PositionMetadata] = {}

    def register_new_position(
        self,
        ticket: int,
        symbol: str,
        is_buy: bool,
        entry_price: float,
        initial_sl: float,
        initial_tp: float,
    ):
        risk_dist = abs(entry_price - initial_sl)
        self.tracked_positions[ticket] = PositionMetadata(
            ticket=ticket,
            symbol=symbol,
            is_buy=is_buy,
            entry_price=entry_price,
            initial_sl=initial_sl,
            initial_tp=initial_tp,
            initial_risk_dist=risk_dist,
            breakeven_applied=False,
            partial_tp_applied=False,
            entry_time=time.time(),
            bars_held=0,
        )
        logger.info(
            f"Pozíció regisztrálva menedzselésre: #{ticket} | 1R távolság: {risk_dist:.5f}"
        )

    def synchronize_with_open_positions(self, live_positions: List[OpenPosition]):
        """
        Removes closed tickets, notifies RiskManager of closed PnL, and registers any untracked positions.
        """
        live_tickets = {p.ticket for p in live_positions}

        # Clean closed positions and report realized PnL to RiskManager
        closed_tickets = [t for t in list(self.tracked_positions.keys()) if t not in live_tickets]
        for t in closed_tickets:
            logger.info(f"Pozíció #{t} kikerült az aktív listából (lezárult).")
            if self.risk_manager is not None:
                realized_pnl = self.mt5.get_closed_position_net_profit(t)
                if realized_pnl is not None:
                    logger.info(f"Pozíció #{t} realizált nettó eredmény: {realized_pnl:+.2f}")
                    self.risk_manager.register_trade_result(realized_pnl)
            del self.tracked_positions[t]

        # Register existing open positions if bot was restarted
        for p in live_positions:
            if p.ticket not in self.tracked_positions:
                is_buy = p.type == 0
                specs = self.mt5.get_symbol_specs(p.symbol)
                min_default_risk = (specs.pip_size * 5.0) if specs else 0.0010
                raw_sl_dist = abs(p.price_open - p.sl) if p.sl > 0 else min_default_risk
                # Guard against already-breakeven positions having tiny 1R on restart
                already_be = (
                    (is_buy and p.sl >= p.price_open) or (not is_buy and 0 < p.sl <= p.price_open)
                )
                sl_dist = max(raw_sl_dist, min_default_risk) if already_be else raw_sl_dist

                self.tracked_positions[p.ticket] = PositionMetadata(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    is_buy=is_buy,
                    entry_price=p.price_open,
                    initial_sl=p.sl,
                    initial_tp=p.tp,
                    initial_risk_dist=sl_dist,
                    breakeven_applied=already_be,
                    partial_tp_applied=False,
                    entry_time=float(p.time_open) if p.time_open > 0 else time.time(),
                    bars_held=0,
                )

    def manage_positions(self, current_atr_by_symbol: Dict[str, float]):
        """
        Evaluates dynamic management rules across all open positions.
        """
        live_positions = self.mt5.get_open_positions()
        self.synchronize_with_open_positions(live_positions)

        now_ts = time.time()

        for pos in live_positions:
            meta = self.tracked_positions.get(pos.ticket)
            if not meta:
                continue

            specs = self.mt5.get_symbol_specs(pos.symbol)
            if not specs:
                continue

            quote = self.mt5.get_quote(pos.symbol)
            if not quote:
                continue

            bid, ask, _, _ = quote
            current_price = bid if meta.is_buy else ask
            current_atr = current_atr_by_symbol.get(pos.symbol, specs.pip_size * 4.0)

            # Calculate current floating R-multiple
            if meta.initial_risk_dist > 0:
                current_profit_dist = (current_price - meta.entry_price) if meta.is_buy else (meta.entry_price - current_price)
                current_r = current_profit_dist / meta.initial_risk_dist
            else:
                current_r = 0.0

            # 1. Breakeven Management
            if self.exec_cfg.use_breakeven and not meta.breakeven_applied:
                if current_r >= self.exec_cfg.breakeven_trigger_r:
                    self._apply_breakeven(pos, meta, specs)

            # 2. Partial Take-Profit Management
            if self.exec_cfg.use_partial_take_profit and not meta.partial_tp_applied:
                if current_r >= self.exec_cfg.partial_tp_trigger_r:
                    self._apply_partial_tp(pos, meta, specs)

            # 3. Dynamic Trailing Stop
            if self.exec_cfg.use_trailing_stop:
                if current_r >= self.exec_cfg.trailing_stop_activation_r:
                    self._apply_trailing_stop(pos, meta, specs, current_price, current_atr)

            # 4. Real-Time M1 Decay Exit (elapsed minutes instead of 2s heartbeat loops!)
            elapsed_minutes = int(max(0.0, now_ts - meta.entry_time) / 60.0)
            meta.bars_held = elapsed_minutes
            if meta.bars_held >= self.exec_cfg.max_holding_bars_m1:
                if current_r < 0.5:
                    logger.warning(
                        f"Pozíció #{pos.ticket} időtúllépés miatt lezárva: {meta.bars_held} perc eltelt, R={current_r:.2f}"
                    )
                    self.mt5.close_position(pos.ticket, pos.symbol, pos.volume, meta.is_buy)

    def _apply_breakeven(self, pos: OpenPosition, meta: PositionMetadata, specs: SymbolSpecs):
        # Ensure buffer covers Pepperstone Razor round-turn commission (~0.8 pip minimum)
        effective_buffer_pips = max(self.exec_cfg.breakeven_buffer_pips, 0.8)
        buffer_dist = effective_buffer_pips * specs.pip_size
        new_sl = (meta.entry_price + buffer_dist) if meta.is_buy else (meta.entry_price - buffer_dist)

        is_better = (new_sl > pos.sl) if meta.is_buy else (new_sl < pos.sl or pos.sl == 0)
        if is_better:
            logger.info(f"[bold green]Breakeven aktiválás #{pos.ticket}![/bold green] Új SL: {new_sl:.5f}")
            if self.mt5.modify_position(pos.ticket, pos.symbol, new_sl, pos.tp):
                meta.breakeven_applied = True
                pos.sl = new_sl

    def _apply_partial_tp(self, pos: OpenPosition, meta: PositionMetadata, specs: SymbolSpecs):
        step = specs.volume_step if specs.volume_step > 0 else 0.01
        raw_close = pos.volume * self.exec_cfg.partial_tp_ratio
        close_volume = round(int(raw_close / step) * step, 2)
        remaining = round(pos.volume - close_volume, 2)

        if remaining >= specs.min_volume and close_volume >= specs.min_volume:
            logger.info(
                f"[bold green]Részleges Take-Profit #{pos.ticket}![/bold green] "
                f"Zárás: {close_volume:.2f} lot / Marad: {remaining:.2f} lot"
            )
            if self.mt5.close_position(pos.ticket, pos.symbol, close_volume, meta.is_buy):
                meta.partial_tp_applied = True
                if not meta.breakeven_applied:
                    self._apply_breakeven(pos, meta, specs)
        else:
            # Cannot split 0.01 lot further -> mark as applied so we don't spam checks
            meta.partial_tp_applied = True

    def _apply_trailing_stop(
        self,
        pos: OpenPosition,
        meta: PositionMetadata,
        specs: SymbolSpecs,
        current_price: float,
        current_atr: float,
    ):
        trail_dist = current_atr * self.exec_cfg.trailing_stop_atr_multiplier
        if meta.is_buy:
            candidate_sl = current_price - trail_dist
            if candidate_sl > pos.sl + (specs.pip_size * 1.0):
                logger.info(f"Követő stop (BUY) frissítés #{pos.ticket}: {pos.sl:.5f} -> {candidate_sl:.5f}")
                if self.mt5.modify_position(pos.ticket, pos.symbol, candidate_sl, pos.tp):
                    pos.sl = candidate_sl
        else:
            candidate_sl = current_price + trail_dist
            if (candidate_sl < pos.sl or pos.sl == 0) and (candidate_sl < pos.sl - (specs.pip_size * 1.0) if pos.sl > 0 else True):
                logger.info(f"Követő stop (SELL) frissítés #{pos.ticket}: {pos.sl:.5f} -> {candidate_sl:.5f}")
                if self.mt5.modify_position(pos.ticket, pos.symbol, candidate_sl, pos.tp):
                    pos.sl = candidate_sl
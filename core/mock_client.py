"""
Mock MT5 Client for Dry-Run & Live Paper Trading
Reads live prices/candles from MT5 (Read-Only) while simulating orders, SL/TP hits, and ECN commissions in memory.
"""

from typing import Dict, List, Optional, Tuple
import logging
import time
from .mt5_client import MT5Client, SymbolSpecs, AccountSnapshot, OpenPosition

logger = logging.getLogger("TitanScalper.MockClient")


class MockMT5Client(MT5Client):
    """
    Live Paper-Trading client implementing identical interface to MT5Client.
    READ-ONLY from MetaTrader 5 (ticks & candles), ZERO live broker execution.
    """
    is_live_execution_allowed: bool = False
    is_paper_trading: bool = True

    def __init__(
        self,
        initial_balance: float = 275.0,
        currency: str = "USD",
        magic_number: int = 777001,
        slippage_points: int = 10,
        commission_per_lot: float = 7.0,
    ):
        super().__init__(
            terminal_path="",
            login=999999,
            password="",
            server="Pepperstone-PaperSim",
            timeout_ms=5000,
            magic_number=magic_number,
            slippage_points=slippage_points,
        )
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.currency = currency
        self.commission_per_lot = commission_per_lot
        self.mt5_live_feed = False
        self._positions: Dict[int, OpenPosition] = {}
        self._closed_pnl: Dict[int, float] = {}
        self._ticket_counter = 100000

        self._sim_quotes: Dict[str, Tuple[float, float]] = {
            "EURUSD": (1.08500, 1.08510),
            "GBPUSD": (1.27200, 1.27215),
            "XAUUSD": (2650.20, 2650.40),
            "GBPJPY": (210.150, 210.165),
            "US30": (43500.0, 43502.0),
        }

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self.mt5_live_feed = bool(mt5.initialize())
            if self.mt5_live_feed:
                acc = mt5.account_info()
                if acc is not None and acc.balance > 0:
                    self.initial_balance = float(acc.balance)
                    self.balance = float(acc.balance)
                    self.equity = float(acc.equity)
                    self.currency = str(acc.currency)
                    self.login = int(acc.login)
        except Exception:
            self.mt5_live_feed = False

        self.is_connected = True
        feed_status = "ÉLŐ MT5 ÁRFOLYAM & EGYENLEG (Read-Only)" if self.mt5_live_feed else "OFFLINE SZINTETIKUS ÁR"
        logger.info(
            f"[bold cyan][PAPER TRADING / DRY RUN][/bold cyan] Virtuális számla indítva ({feed_status}). "
            f"Kezdőtőke: {self.balance:.2f} {self.currency}"
        )
        return True

    def disconnect(self):
        if self.mt5_live_feed:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
            except Exception:
                pass
        self.is_connected = False
        logger.info(
            f"[DRY RUN] Virtuális kapcsolat bontva. Végső egyenleg: {self.balance:.2f} {self.currency} "
            f"(Nettó PnL: {self.balance - self.initial_balance:+.2f} {self.currency})"
        )

    def set_sim_price(self, symbol: str, bid: float, ask: float):
        self._sim_quotes[symbol] = (bid, ask)

    def ensure_symbol(self, symbol: str) -> bool:
        if self.mt5_live_feed:
            if super().ensure_symbol(symbol):
                return True

        if symbol not in self.symbol_specs_cache:
            sym_u = symbol.upper()
            is_gold = "XAU" in sym_u or "GOLD" in sym_u
            is_jpy = "JPY" in sym_u
            is_idx = any(idx in sym_u for idx in ("US30", "NAS", "SPX", "DE40", "DAX"))
            digits = 2 if is_gold else (3 if is_jpy else (1 if is_idx else 5))
            point = 10 ** (-digits)
            pip_size = 0.10 if is_gold else (0.01 if is_jpy else (1.0 if is_idx else 0.0001))
            contract_size = 100.0 if is_gold else (1.0 if is_idx else 100000.0)
            tick_value = 1.0 if not is_jpy else 0.65

            self.symbol_specs_cache[symbol] = SymbolSpecs(
                name=symbol,
                digits=digits,
                point=point,
                tick_value=tick_value,
                tick_size=point,
                contract_size=contract_size,
                min_volume=0.01,
                max_volume=100.0,
                volume_step=0.01,
                pip_size=pip_size,
            )
        return True

    def get_quote(self, symbol: str) -> Optional[Tuple[float, float, float, float]]:
        self.ensure_symbol(symbol)
        specs = self.get_symbol_specs(symbol)
        if specs is None:
            return None

        if self.mt5_live_feed and symbol not in self._sim_quotes_override_set():
            try:
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(symbol)
                # Auto-reconnect watchdog if tick is missing or connection lost
                if tick is None:
                    t_info = mt5.terminal_info()
                    if t_info is None or not getattr(t_info, "connected", False):
                        mt5.shutdown()
                        mt5.initialize()
                        tick = mt5.symbol_info_tick(symbol)

                if tick is not None and tick.bid > 0 and tick.ask > 0:
                    spread_points = (tick.ask - tick.bid) / specs.point
                    spread_pips = (tick.ask - tick.bid) / specs.pip_size
                    return tick.bid, tick.ask, spread_points, spread_pips
            except Exception:
                pass

        quote = self._sim_quotes.get(symbol, (1.08500, 1.08510))
        bid, ask = quote
        spread_points = (ask - bid) / specs.point
        spread_pips = (ask - bid) / specs.pip_size
        return bid, ask, spread_points, spread_pips

    def _sim_quotes_override_set(self):
        return getattr(self, "_manual_overrides", set())

    def _check_sl_tp_hits(self):
        """
        Automatically triggers virtual SL and TP closures when live price crosses levels.
        """
        for ticket, pos in list(self._positions.items()):
            quote = self.get_quote(pos.symbol)
            if not quote:
                continue
            bid, ask, _, _ = quote
            is_buy = pos.type == 0
            eval_price = bid if is_buy else ask

            if pos.sl > 0:
                sl_hit = (eval_price <= pos.sl) if is_buy else (eval_price >= pos.sl)
                if sl_hit:
                    logger.warning(f"[PAPER TRADING] #{ticket} {pos.symbol} STOP-LOSS KIÜTVE @ {eval_price:.5f}!")
                    self.close_position(ticket, pos.symbol, pos.volume, is_buy)
                    continue

            if pos.tp > 0:
                tp_hit = (eval_price >= pos.tp) if is_buy else (eval_price <= pos.tp)
                if tp_hit:
                    logger.info(f"[bold green][PAPER TRADING] #{ticket} {pos.symbol} TAKE-PROFIT TELJESÜLT @ {eval_price:.5f}![/bold green]")
                    self.close_position(ticket, pos.symbol, pos.volume, is_buy)

    def get_account_snapshot(self) -> Optional[AccountSnapshot]:
        self._check_sl_tp_hits()

        floating_profit = 0.0
        used_margin = 0.0

        for pos in self._positions.values():
            quote = self.get_quote(pos.symbol)
            if quote:
                bid, ask, _, _ = quote
                current_price = bid if pos.type == 0 else ask
                price_diff = (current_price - pos.price_open) if pos.type == 0 else (pos.price_open - current_price)
                specs = self.get_symbol_specs(pos.symbol)
                if specs and specs.point > 0 and getattr(specs, "tick_value", 0) > 0:
                    gross_pnl = (price_diff / specs.point) * specs.tick_value * pos.volume
                else:
                    gross_pnl = price_diff * (specs.contract_size if specs else 100000.0) * pos.volume
                pos_pnl = gross_pnl - (self.commission_per_lot * pos.volume)
                pos.profit = round(pos_pnl, 2)
                floating_profit += pos_pnl

                # Accurate institutional margin calculation (1:30 EU ESMA retail leverage default)
                contract_sz = specs.contract_size if specs else 100000.0
                leverage = getattr(self, "leverage", 30.0)
                if leverage <= 0:
                    leverage = 30.0
                sym_u = pos.symbol.upper()
                if "JPY" in sym_u:
                    # GBPJPY: notional in USD = volume * 100k * GBPUSD (approx 1.28)
                    notional_usd = pos.volume * contract_sz * 1.28
                elif "XAU" in sym_u or "GOLD" in sym_u:
                    notional_usd = pos.volume * contract_sz * pos.price_open
                else:
                    notional_usd = pos.volume * contract_sz * pos.price_open
                used_margin += notional_usd / leverage

        self.equity = round(self.balance + floating_profit, 2)
        margin_level = (self.equity / used_margin * 100.0) if used_margin > 0 else 9999.0

        return AccountSnapshot(
            login=self.login,
            balance=round(self.balance, 2),
            equity=self.equity,
            margin=round(used_margin, 2),
            margin_free=round(max(0.0, self.equity - used_margin), 2),
            margin_level=round(margin_level, 1),
            profit=round(floating_profit, 2),
            currency=self.currency,
        )

    def get_open_positions(self, symbol: Optional[str] = None) -> List[OpenPosition]:
        self.get_account_snapshot()
        positions = list(self._positions.values())
        if symbol:
            return [p for p in positions if p.symbol == symbol]
        return positions

    def send_market_order(
        self,
        symbol: str,
        order_type_str: str,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "SimOrder",
    ) -> Optional[int]:
        self.ensure_symbol(symbol)
        quote = self.get_quote(symbol)
        if quote is None:
            return None

        bid, ask, _, _ = quote
        is_buy = order_type_str.upper() == "BUY"
        price = ask if is_buy else bid

        self._ticket_counter += 1
        ticket = self._ticket_counter

        pos = OpenPosition(
            ticket=ticket,
            symbol=symbol,
            type=0 if is_buy else 1,
            volume=volume,
            price_open=price,
            sl=sl,
            tp=tp,
            profit=0.0,
            comment=f"[SIM] {comment}",
            magic=self.magic_number,
            time_open=int(time.time()),
        )
        self._positions[ticket] = pos

        logger.info(
            f"[bold cyan][PAPER ORDER OPENED][/bold cyan] #{ticket} {order_type_str} {volume:.2f} lot "
            f"{symbol} @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}"
        )
        return ticket

    def modify_position(self, ticket: int, symbol: str, sl: float, tp: float) -> bool:
        if ticket in self._positions:
            self._positions[ticket].sl = sl
            self._positions[ticket].tp = tp
            logger.info(f"[PAPER TRADING] Pozíció #{ticket} módosítva: SL={sl:.5f}, TP={tp:.5f}")
            return True
        return False

    def close_position(self, ticket: int, symbol: str, volume: float, is_buy: bool) -> bool:
        if ticket in self._positions:
            pos = self._positions[ticket]
            quote = self.get_quote(symbol)
            if quote:
                bid, ask, _, _ = quote
                exit_price = bid if is_buy else ask
                diff = (exit_price - pos.price_open) if is_buy else (pos.price_open - exit_price)
                specs = self.get_symbol_specs(symbol)
                if specs and specs.point > 0 and getattr(specs, "tick_value", 0) > 0:
                    gross_pnl = (diff / specs.point) * specs.tick_value * volume
                else:
                    gross_pnl = diff * (specs.contract_size if specs else 100000.0) * volume
                commission = self.commission_per_lot * volume
                realized_pnl = round(gross_pnl - commission, 2)

                self.balance = round(self.balance + realized_pnl, 2)
                self.equity = self.balance
                self._closed_pnl[ticket] = self._closed_pnl.get(ticket, 0.0) + realized_pnl

                if volume >= pos.volume - 1e-4:
                    del self._positions[ticket]
                else:
                    pos.volume = round(pos.volume - volume, 2)

                pnl_color = "bold green" if realized_pnl >= 0 else "bold red"
                logger.info(
                    f"[{pnl_color}][PAPER POSITION CLOSED][/{pnl_color}] #{ticket} {symbol} @ {exit_price:.5f} | "
                    f"Nettó PnL (jutalékkal): {realized_pnl:+.2f} {self.currency} | Új Egyenleg: {self.balance:.2f} {self.currency}"
                )
                return True
        return False

    def get_closed_position_net_profit(self, position_ticket: int) -> Optional[float]:
        return self._closed_pnl.get(position_ticket)
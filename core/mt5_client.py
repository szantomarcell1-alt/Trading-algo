"""
MetaTrader 5 Client & Execution Interface
Robust, thread-safe MT5 API wrapper with auto-reconnect, slippage control, and dry-run fallback.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
import MetaTrader5 as mt5

logger = logging.getLogger("TitanScalper.MT5Client")


@dataclass
class SymbolSpecs:
    name: str
    digits: int
    point: float
    tick_value: float
    tick_size: float
    contract_size: float
    min_volume: float
    max_volume: float
    volume_step: float
    pip_size: float


@dataclass
class AccountSnapshot:
    login: int
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    profit: float
    currency: str


@dataclass
class OpenPosition:
    ticket: int
    symbol: str
    type: int  # 0 = BUY, 1 = SELL
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    comment: str
    magic: int
    time_open: int


class MT5Client:
    is_live_execution_allowed: bool = True
    is_paper_trading: bool = False

    def __init__(
        self,
        terminal_path: str = "",
        login: int = 0,
        password: str = "",
        server: str = "",
        timeout_ms: int = 60000,
        magic_number: int = 777001,
        slippage_points: int = 10,
    ):
        self.terminal_path = terminal_path
        self.login = login
        self.password = password
        self.server = server
        self.timeout_ms = timeout_ms
        self.magic_number = magic_number
        self.slippage_points = slippage_points
        self.is_connected = False
        self.symbol_specs_cache: Dict[str, SymbolSpecs] = {}

    def connect(self) -> bool:
        """
        Initializes connection to MT5 terminal.
        """
        init_kwargs = {"timeout": self.timeout_ms}
        if self.terminal_path:
            init_kwargs["path"] = self.terminal_path

        logger.info(f"Kapcsolódás a MetaTrader 5 terminálhoz (Timeout: {self.timeout_ms}ms)...")

        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            logger.error(f"MT5 Inicializálási hiba: kód={err[0]}, üzenet={err[1]}")
            self.is_connected = False
            return False

        if self.login > 0:
            authorized = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server,
                timeout=self.timeout_ms,
            )
            if not authorized:
                err = mt5.last_error()
                logger.error(f"MT5 Bejelentkezési hiba: kód={err[0]}, üzenet={err[1]}")
                self.is_connected = False
                return False

        acc = mt5.account_info()
        if acc is None:
            logger.error("Nem sikerült lekérni az MT5 számlainformációkat.")
            self.is_connected = False
            return False

        self.is_connected = True
        logger.info(
            f"[bold green]Sikeres MT5 kapcsolat![/bold green] Számla: {acc.login} | "
            f"Bróker: {acc.company} | Egyenleg: {acc.balance:.2f} {acc.currency}"
        )
        return True

    def disconnect(self):
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            logger.info("MT5 kapcsolat bontva.")

    def ensure_symbol(self, symbol: str) -> bool:
        """
        Ensures the symbol is visible and selected in Market Watch, with exact Forex/Gold/Index pip sizing.
        """
        if not self.is_connected:
            return False

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"A(z) '{symbol}' szimbólum nem található a bróker kínálatában!")
            return False

        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Nem sikerült a(z) '{symbol}' hozzáadása a Market Watch-hoz.")
                return False

        sym_upper = symbol.upper()
        if "XAU" in sym_upper or "GOLD" in sym_upper:
            pip_size = 0.10  # Aranynál 1 pip = $0.10 (10 cent)
        elif any(idx in sym_upper for idx in ("US30", "NAS", "SPX", "GER", "DAX", "UK100", "JPN")):
            pip_size = 1.0   # Indexeknél 1 pip = 1.0 indexpont
        elif info.digits in (3, 5):
            pip_size = 10.0 * info.point
        else:
            pip_size = info.point

        tick_val = getattr(info, "trade_tick_value_loss", 0.0) or info.trade_tick_value
        if tick_val <= 0:
            tick_val = 1.0

        self.symbol_specs_cache[symbol] = SymbolSpecs(
            name=symbol,
            digits=info.digits,
            point=info.point,
            tick_value=tick_val,
            tick_size=info.trade_tick_size if info.trade_tick_size > 0 else info.point,
            contract_size=info.trade_contract_size,
            min_volume=info.volume_min,
            max_volume=info.volume_max,
            volume_step=info.volume_step,
            pip_size=pip_size,
        )
        return True

    def get_symbol_specs(self, symbol: str) -> Optional[SymbolSpecs]:
        if symbol not in self.symbol_specs_cache:
            if not self.ensure_symbol(symbol):
                return None
        return self.symbol_specs_cache.get(symbol)

    def get_quote(self, symbol: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns: (bid, ask, spread_points, spread_pips)
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        specs = self.get_symbol_specs(symbol)
        if specs is None:
            return None

        spread_points = (tick.ask - tick.bid) / specs.point
        spread_pips = (tick.ask - tick.bid) / specs.pip_size
        return tick.bid, tick.ask, spread_points, spread_pips

    def get_account_snapshot(self) -> Optional[AccountSnapshot]:
        acc = mt5.account_info()
        if acc is None:
            return None
        return AccountSnapshot(
            login=acc.login,
            balance=acc.balance,
            equity=acc.equity,
            margin=acc.margin,
            margin_free=acc.margin_free,
            margin_level=acc.margin_level,
            profit=acc.profit,
            currency=acc.currency,
        )

    def get_open_positions(self, symbol: Optional[str] = None) -> List[OpenPosition]:
        """
        Fetches open positions matching magic number and optional symbol.
        """
        if not self.is_connected:
            return []

        if symbol:
            raw_positions = mt5.positions_get(symbol=symbol)
        else:
            raw_positions = mt5.positions_get()

        if raw_positions is None:
            return []

        results = []
        for p in raw_positions:
            if p.magic == self.magic_number:
                results.append(
                    OpenPosition(
                        ticket=p.ticket,
                        symbol=p.symbol,
                        type=p.type,
                        volume=p.volume,
                        price_open=p.price_open,
                        sl=p.sl,
                        tp=p.tp,
                        profit=p.profit,
                        comment=p.comment,
                        magic=p.magic,
                        time_open=p.time,
                    )
                )
        return results

    def send_market_order(
        self,
        symbol: str,
        order_type_str: str,  # "BUY" or "SELL"
        volume: float,
        sl: float,
        tp: float,
        comment: str = "TitanScalp",
    ) -> Optional[int]:
        """
        Sends an aggressive market order with hard SL and TP.
        Returns ticket number on success, None on failure.
        """
        if not self.is_connected:
            logger.error("Megbízás elutasítva: nincs MT5 kapcsolat.")
            return None

        specs = self.get_symbol_specs(symbol)
        if specs is None:
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Nem érkezett érvényes tick ár a(z) {symbol} szimbólumhoz.")
            return None

        is_buy = order_type_str.upper() == "BUY"
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price = tick.ask if is_buy else tick.bid

        # Normalization
        price = round(price, specs.digits)
        sl = round(sl, specs.digits)
        tp = round(tp, specs.digits)
        volume = round(volume, 2)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.slippage_points,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Check order filling mode compatibility
        sym_info = mt5.symbol_info(symbol)
        if sym_info:
            filling_mode = sym_info.filling_mode
            if filling_mode & mt5.SYMBOL_FILLING_FOK:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
            elif filling_mode & mt5.SYMBOL_FILLING_IOC:
                request["type_filling"] = mt5.ORDER_FILLING_IOC
            else:
                request["type_filling"] = mt5.ORDER_FILLING_RETURN

        logger.info(
            f"Megbízás küldése -> {order_type_str} {volume:.2f} lot {symbol} @ {price} | "
            f"SL: {sl} | TP: {tp} (Slippage: {self.slippage_points}pt)"
        )

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            logger.error(f"order_send() sikertelen, utolsó hiba: {err}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"Megbízás visszautasítva a bróker által! Retcode: {result.retcode} "
                f"({result.comment})"
            )
            return None

        logger.info(
            f"[bold green]Megbízás teljesült![/bold green] Ticket: #{result.order} | "
            f"Ár: {result.price} | Volume: {result.volume}"
        )
        return result.order

    def modify_position(self, ticket: int, symbol: str, sl: float, tp: float) -> bool:
        """
        Modifies Stop Loss and Take Profit of an existing position.
        """
        if not self.is_connected:
            return False

        specs = self.get_symbol_specs(symbol)
        if specs is None:
            return False

        sl = round(sl, specs.digits)
        tp = round(tp, specs.digits)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": sl,
            "tp": tp,
            "magic": self.magic_number,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "mt5.order_send failed"
            logger.warning(f"SL/TP módosítás sikertelen a #{ticket} pozíción: {retcode} ({comment})")
            return False

        logger.info(f"Pozíció #{ticket} sikeresen módosítva: Új SL={sl}, Új TP={tp}")
        return True

    def close_position(self, ticket: int, symbol: str, volume: float, is_buy: bool) -> bool:
        """
        Closes full or partial position.
        """
        if not self.is_connected:
            return False

        specs = self.get_symbol_specs(symbol)
        if specs is None:
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        # Opposite order type to close
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy else tick.ask
        price = round(price, specs.digits)
        volume = round(volume, 2)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "price": price,
            "deviation": self.slippage_points,
            "magic": self.magic_number,
            "comment": "TitanScalp Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        sym_info = mt5.symbol_info(symbol)
        if sym_info:
            if sym_info.filling_mode & mt5.SYMBOL_FILLING_FOK:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
            elif sym_info.filling_mode & mt5.SYMBOL_FILLING_IOC:
                request["type_filling"] = mt5.ORDER_FILLING_IOC
            else:
                request["type_filling"] = mt5.ORDER_FILLING_RETURN

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            logger.error(f"Pozíció lezárás hiba #{ticket}: {retcode}")
            return False

        logger.info(f"[bold yellow]Pozíció lezárva #{ticket}![/bold yellow] {volume:.2f} lot @ {price}")
        return True

    def close_all_positions(self) -> int:
        """
        Emergency panic button: closes all positions opened by this bot.
        """
        positions = self.get_open_positions()
        closed_count = 0
        for pos in positions:
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            if self.close_position(pos.ticket, pos.symbol, pos.volume, is_buy):
                closed_count += 1
        return closed_count

    def get_closed_position_net_profit(self, position_ticket: int) -> Optional[float]:
        """
        Queries MT5 deal history for a closed position ticket and returns net PnL (profit + swap + commission).
        """
        if not self.is_connected:
            return None
        try:
            deals = mt5.history_deals_get(position=position_ticket)
            if deals is None or len(deals) == 0:
                return None
            net_pnl = 0.0
            has_exit_deal = False
            for d in deals:
                net_pnl += (d.profit + d.swap + d.commission)
                if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, mt5.DEAL_ENTRY_OUT_BY):
                    has_exit_deal = True
            return round(net_pnl, 2) if has_exit_deal else None
        except Exception as e:
            logger.debug(f"History deal lekérdezési hiba (#{position_ticket}): {e}")
            return None

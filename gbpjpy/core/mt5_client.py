"""
MetaTrader 5 Client & Execution Interface for GBPJPY
Thread-safe MT5 API wrapper with correct bitmask filling mode resolution,
JPY pip/tick sizing, and cross-currency execution controls.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
import math
import threading
import time

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

try:
    from gbpjpy.utils.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except (ImportError, ValueError):
        from utils.logger import get_logger

logger = get_logger("MT5Client")


@dataclass
class SymbolSpecs:
    name: str = "GBPJPY"
    digits: int = 3
    point: float = 0.001
    tick_value: float = 0.626076
    tick_size: float = 0.001
    contract_size: float = 100000.0
    min_volume: float = 0.01
    max_volume: float = 100.0
    volume_step: float = 0.01
    pip_size: float = 0.01

    @property
    def symbol(self) -> str:
        return self.name

    def calculate_tick_value(self, usdjpy_rate: float = 155.0) -> float:
        """Calculate USD tick value for 1 standard lot (100k GBP)."""
        return 100.0 / usdjpy_rate


@dataclass
class AccountSnapshot:
    login: int = 0
    balance: float = 250.0
    equity: float = 250.0
    margin: float = 0.0
    margin_free: float = 250.0
    margin_level: float = 0.0
    profit: float = 0.0
    currency: str = "USD"


@dataclass
class OpenPosition:
    ticket: int
    symbol: str
    type: int  # 0 = BUY, 1 = SELL
    volume: float
    price_open: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    comment: str = ""
    magic: int = 777101
    time_open: int = 0

    @property
    def entry_price(self) -> float:
        return self.price_open


class MT5Client:
    def __init__(
        self,
        terminal_path: str = "",
        login: int = 0,
        password: str = "",
        server: str = "",
        timeout_ms: int = 60000,
        magic_number: int = 777101,
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
        self._lock = threading.RLock()

    def connect(self) -> bool:
        """
        Initializes connection to MT5 terminal in a thread-safe manner.
        """
        with self._lock:
            if not MT5_AVAILABLE or mt5 is None:
                logger.error("MetaTrader5 package is not available in current Python environment.")
                self.is_connected = False
                return False

            init_kwargs = {"timeout": self.timeout_ms}
            if self.terminal_path:
                init_kwargs["path"] = self.terminal_path

            logger.info(f"Connecting to MT5 terminal (Timeout: {self.timeout_ms}ms)...")

            if not mt5.initialize(**init_kwargs):
                err = mt5.last_error()
                logger.error(f"MT5 Initialization error: code={err[0]}, message={err[1]}")
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
                    logger.error(f"MT5 Login failed: code={err[0]}, message={err[1]}")
                    self.is_connected = False
                    return False

            acc = mt5.account_info()
            if acc is None:
                logger.error("Failed to retrieve MT5 account info.")
                self.is_connected = False
                return False

            self.is_connected = True
            logger.info(
                f"MT5 Connected! Account: {acc.login} | Server: {acc.server} | "
                f"Balance: {acc.balance:.2f} {acc.currency}"
            )
            return True

    def disconnect(self):
        with self._lock:
            if self.is_connected and MT5_AVAILABLE and mt5 is not None:
                mt5.shutdown()
                self.is_connected = False
                logger.info("MT5 connection closed.")

    def ensure_symbol(self, symbol: str = "GBPJPY") -> bool:
        """
        Ensures symbol is visible in Market Watch and caches accurate SymbolSpecs.
        """
        with self._lock:
            if not self.is_connected or not MT5_AVAILABLE or mt5 is None:
                # Pre-populate default GBPJPY specs even if offline
                if symbol not in self.symbol_specs_cache:
                    self.symbol_specs_cache[symbol] = SymbolSpecs(name=symbol)
                return True

            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"Symbol '{symbol}' not found in broker offerings!")
                return False

            if not info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to add '{symbol}' to Market Watch.")
                    return False
                time.sleep(0.05)  # Allow tick subscription to settle

            # Pip size calculation: for GBPJPY (digits=3), point=0.001 -> pip=0.010
            digits = info.digits
            point = info.point
            pip_size = 10.0 * point if digits in (3, 5) else point

            # Tick value calculation: trade_tick_value_loss preferred, fallback to trade_tick_value or default 0.626076
            tick_val = getattr(info, "trade_tick_value_loss", 0.0) or getattr(info, "trade_tick_value", 0.0)
            if tick_val <= 0:
                tick_val = 0.626076  # Default GBPJPY tick value in USD

            self.symbol_specs_cache[symbol] = SymbolSpecs(
                name=symbol,
                digits=digits,
                point=point,
                tick_value=tick_val,
                tick_size=info.trade_tick_size if info.trade_tick_size > 0 else point,
                contract_size=info.trade_contract_size if info.trade_contract_size > 0 else 100000.0,
                min_volume=info.volume_min if info.volume_min > 0 else 0.01,
                max_volume=info.volume_max if info.volume_max > 0 else 100.0,
                volume_step=info.volume_step if info.volume_step > 0 else 0.01,
                pip_size=pip_size,
            )
            return True

    def get_symbol_specs(self, symbol: str = "GBPJPY") -> Optional[SymbolSpecs]:
        with self._lock:
            if symbol not in self.symbol_specs_cache:
                if not self.ensure_symbol(symbol):
                    return None
            return self.symbol_specs_cache.get(symbol)

    def get_quote(self, symbol: str = "GBPJPY") -> Optional[Tuple[float, float, float, float]]:
        """
        Returns: (bid, ask, spread_points, spread_pips)
        """
        with self._lock:
            if not self.is_connected or not MT5_AVAILABLE or mt5 is None:
                return None

            tick = mt5.symbol_info_tick(symbol)
            if tick is None or tick.bid <= 0 or tick.ask <= 0:
                return None

            specs = self.get_symbol_specs(symbol)
            if specs is None:
                return None

            spread_points = (tick.ask - tick.bid) / specs.point
            spread_pips = (tick.ask - tick.bid) / specs.pip_size
            return tick.bid, tick.ask, spread_points, spread_pips

    def get_account_snapshot(self) -> Optional[AccountSnapshot]:
        with self._lock:
            if not self.is_connected or not MT5_AVAILABLE or mt5 is None:
                return None
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

    def get_account_info(self) -> Optional[AccountSnapshot]:
        """Alias for get_account_snapshot for compatibility."""
        return self.get_account_snapshot()

    def get_open_positions(self, symbol: Optional[str] = None) -> List[OpenPosition]:
        with self._lock:
            if not self.is_connected or not MT5_AVAILABLE or mt5 is None:
                return []

            raw_positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
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

    def _resolve_filling_mode(self, symbol: str) -> int:
        """
        Safely determines order filling mode compatibility using bitmasks without non-existent constants.
        Bit 1 (0x02) = IOC, Bit 0 (0x01) = FOK.
        """
        if not MT5_AVAILABLE or mt5 is None:
            return 0

        sym_info = mt5.symbol_info(symbol)
        if sym_info is not None:
            filling_mode = getattr(sym_info, "filling_mode", 0)
            if filling_mode & 2:  # SYMBOL_FILLING_IOC bit
                return getattr(mt5, "ORDER_FILLING_IOC", 1)
            elif filling_mode & 1:  # SYMBOL_FILLING_FOK bit
                return getattr(mt5, "ORDER_FILLING_FOK", 0)
        return getattr(mt5, "ORDER_FILLING_RETURN", 2)

    def send_market_order(
        self,
        symbol: str,
        order_type_str: str,  # "BUY" or "SELL"
        volume: float,
        sl: float,
        tp: float,
        comment: str = "GBPJPY_Scalp",
    ) -> Optional[int]:
        """
        Sends market order with hard SL/TP and verified filling mode.
        """
        with self._lock:
            if not self.is_connected or not MT5_AVAILABLE or mt5 is None:
                logger.error("Order rejected: MT5 client not connected.")
                return None

            specs = self.get_symbol_specs(symbol)
            if specs is None:
                return None

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"No valid tick for {symbol}")
                return None

            is_buy = str(order_type_str).upper() in ("BUY", "0", "ORDER_TYPE_BUY")
            order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
            price = tick.ask if is_buy else tick.bid

            # Normalization
            price = round(price, specs.digits)
            sl = round(sl, specs.digits) if sl > 0 else 0.0
            tp = round(tp, specs.digits) if tp > 0 else 0.0

            # Volume step quantizing
            step = specs.volume_step if specs.volume_step > 0 else 0.01
            stepped_vol = math.floor(volume / step) * step
            volume = round(max(specs.min_volume, min(specs.max_volume, stepped_vol)), 2)

            type_filling = self._resolve_filling_mode(symbol)

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
                "type_filling": type_filling,
            }

            logger.info(
                f"Sending order -> {order_type_str} {volume:.2f} lot {symbol} @ {price:.3f} | "
                f"SL: {sl:.3f} | TP: {tp:.3f} | Filling: {type_filling}"
            )

            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                logger.error(f"order_send() returned None, last error: {err}")
                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order rejected by broker! Retcode: {result.retcode} ({result.comment})")
                return None

            logger.info(f"Order filled! Ticket: #{result.order} | Price: {result.price} | Volume: {result.volume}")
            return result.order

    def modify_position(
        self,
        ticket: int,
        symbol: Optional[str] = None,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> bool:
        with self._lock:
            if not self.is_connected or not MT5_AVAILABLE or mt5 is None:
                return False

            if symbol is None:
                for p in self.get_open_positions():
                    if p.ticket == ticket:
                        symbol = p.symbol
                        break
            if symbol is None:
                symbol = "GBPJPY"

            specs = self.get_symbol_specs(symbol)
            digits = specs.digits if specs else 3
            sl = round(sl, digits) if sl > 0 else 0.0
            tp = round(tp, digits) if tp > 0 else 0.0

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
                ret = result.retcode if result else "None"
                cmt = result.comment if result else "mt5.order_send failed"
                logger.warning(f"Position modify failed #{ticket}: {ret} ({cmt})")
                return False

            logger.info(f"Position #{ticket} modified: SL={sl:.3f}, TP={tp:.3f}")
            return True

    def close_position(
        self,
        ticket: int,
        symbol: Optional[str] = None,
        volume: Optional[float] = None,
        is_buy: Optional[bool] = None,
    ) -> bool:
        with self._lock:
            if not self.is_connected or not MT5_AVAILABLE or mt5 is None:
                return False

            if symbol is None or volume is None or is_buy is None:
                for p in self.get_open_positions():
                    if p.ticket == ticket:
                        symbol = symbol or p.symbol
                        volume = volume or p.volume
                        is_buy = is_buy if is_buy is not None else (p.type == 0)
                        break

            if symbol is None or volume is None or is_buy is None:
                logger.error(f"Cannot close position #{ticket}: details missing and not found in open positions.")
                return False

            specs = self.get_symbol_specs(symbol)
            if specs is None:
                return False

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False

            close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
            price = tick.bid if is_buy else tick.ask
            price = round(price, specs.digits)
            volume = round(volume, 2)

            type_filling = self._resolve_filling_mode(symbol)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "price": price,
                "deviation": self.slippage_points,
                "magic": self.magic_number,
                "comment": "GBPJPY Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                ret = result.retcode if result else "None"
                logger.error(f"Position close error #{ticket}: {ret}")
                return False

            logger.info(f"Position closed #{ticket}! {volume:.2f} lot @ {price:.3f}")
            return True

    def close_all_positions(self) -> int:
        with self._lock:
            positions = self.get_open_positions()
            closed_count = 0
            for pos in positions:
                is_buy = (pos.type == 0)
                if self.close_position(pos.ticket, pos.symbol, pos.volume, is_buy):
                    closed_count += 1
            return closed_count

    def get_closed_position_net_profit(self, position_ticket: int) -> Optional[float]:
        with self._lock:
            if not self.is_connected or not MT5_AVAILABLE or mt5 is None:
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
                logger.debug(f"History deals query failed (#{position_ticket}): {e}")
                return None

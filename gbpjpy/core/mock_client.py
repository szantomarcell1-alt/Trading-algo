"""
Mock MT5 Client for GBPJPY Dry-Run & Paper Trading
Handles accurate JPY-to-USD cross-currency tick valuation, margin accounting for $250 equity at 1:30 leverage,
and in-memory order/SL/TP simulation with Pepperstone Razor ECN commissions.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time

try:
    from gbpjpy.core.mt5_client import MT5Client, SymbolSpecs, AccountSnapshot, OpenPosition
    from gbpjpy.utils.logger import get_logger
except ImportError:
    try:
        from .mt5_client import MT5Client, SymbolSpecs, AccountSnapshot, OpenPosition
        from ..utils.logger import get_logger
    except (ImportError, ValueError):
        from core.mt5_client import MT5Client, SymbolSpecs, AccountSnapshot, OpenPosition
        from utils.logger import get_logger

logger = get_logger("MockClient")


class CloseResult:
    """Represents the outcome of a closed position, truthy in boolean expressions."""
    def __init__(self, ticket: int, pnl: float, exit_price: float, commission: float = 0.0):
        self.ticket = ticket
        self.pnl = round(pnl, 2)
        self.profit = self.pnl
        self.exit_price = exit_price
        self.commission = commission
        self.success = True

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"CloseResult(ticket={self.ticket}, pnl={self.pnl}, exit_price={self.exit_price})"


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    volume: float,
    order_type: str,
    specs: Optional[SymbolSpecs] = None,
    usdjpy_rate: Optional[float] = None,
    contract_size: float = 100000.0,
) -> float:
    """
    Universal Cross-Currency Gross PnL Calculation.
    Formula with usdjpy_rate: PnL (USD) = (price_diff * contract_size * volume) / usdjpy_rate
    Formula with specs: PnL (USD) = (price_diff / point) * tick_value * volume
    """
    is_buy = str(order_type).upper() in ("BUY", "0", "ORDER_TYPE_BUY")
    diff = (exit_price - entry_price) if is_buy else (entry_price - exit_price)

    if usdjpy_rate is not None and usdjpy_rate > 0:
        pnl_jpy = diff * contract_size * volume
        return round(pnl_jpy / usdjpy_rate, 4)

    if specs is not None:
        points = diff / specs.point
        return round(points * specs.tick_value * volume, 4)

    # Fallback to standard GBPJPY default tick value 0.626076
    points = diff / 0.001
    return round(points * 0.626076 * volume, 4)


class MockMT5Client(MT5Client):
    """
    Paper-Trading client implementing identical interface to MT5Client
    tailored specifically for GBPJPY with accurate USD cross-currency accounting.
    """

    def __init__(
        self,
        initial_balance: float = 250.0,
        currency: str = "USD",
        magic_number: int = 777002,
        slippage_points: int = 10,
        commission_per_lot: float = 7.0,
        leverage: float = 30.0,
        default_tick_value: float = 0.626076,
        use_live_feed: bool = False,
    ):
        super().__init__(
            terminal_path="",
            login=999999,
            password="",
            server="Pepperstone-GBPJPY-PaperSim",
            timeout_ms=5000,
            magic_number=magic_number,
            slippage_points=slippage_points,
        )
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.currency = currency
        self.commission_per_lot = commission_per_lot
        self.leverage = leverage
        self.default_tick_value = default_tick_value
        self.use_live_feed = use_live_feed
        self.mt5_live_feed = False

        self._positions: Dict[int, OpenPosition] = {}
        self._closed_pnl: Dict[int, float] = {}
        self._ticket_counter = 100000

        # Authentic default quotes for cross-currency paper trading
        self._sim_quotes: Dict[str, Tuple[float, float]] = {
            "GBPJPY": (195.500, 195.515),  # 1.5 pip spread
            "USDJPY": (159.720, 159.730),  # 100 JPY / 159.7266 = 0.626076
            "GBPUSD": (1.27200, 1.27215),
        }

    def connect(self, use_live_feed: Optional[bool] = None) -> bool:
        """
        Attempts connection to live MT5 terminal for read-only quotes and balance feed if requested.
        Defaults to offline synthetic simulation for isolated paper-trading and unit tests.
        """
        if use_live_feed is not None:
            self.use_live_feed = use_live_feed

        if self.use_live_feed:
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
                        if hasattr(acc, "leverage") and acc.leverage > 0:
                            self.leverage = float(acc.leverage)
            except Exception:
                self.mt5_live_feed = False
        else:
            self.mt5_live_feed = False

        self.is_connected = True
        logger.info(
            f"[GBPJPY PAPER TRADING] Virtual account initialized. "
            f"Capital: {self.balance:.2f} {self.currency} | Leverage: 1:{int(self.leverage)}"
        )
        return True

    def disconnect(self):
        """
        Disconnects paper simulation and reports net session performance.
        """
        if self.mt5_live_feed:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
            except Exception:
                pass
        self.is_connected = False
        net_pnl = self.balance - self.initial_balance
        pnl_pct = (net_pnl / self.initial_balance) * 100.0 if self.initial_balance > 0 else 0.0
        logger.info(
            f"[GBPJPY DRY RUN] Closed. Balance: {self.balance:.2f} {self.currency} "
            f"(Net PnL: {net_pnl:+.2f} {self.currency} / {pnl_pct:+.2f}%)"
        )

    def set_sim_price(self, symbol: str, bid: float, ask: float):
        """Updates synthetic quote for automated tests and offline simulation."""
        self._sim_quotes[symbol] = (bid, ask)

    def set_rates(self, symbol: str, bid: float, ask: float):
        """Alias for set_sim_price."""
        self.set_sim_price(symbol, bid, ask)

    def update_price(self, symbol: str, bid: float, ask: float):
        """Alias for set_sim_price."""
        self.set_sim_price(symbol, bid, ask)

    def ensure_symbol(self, symbol: str) -> bool:
        """
        Initializes symbol specifications with exact GBPJPY JPY-to-USD tick valuation.
        """
        if self.mt5_live_feed:
            if super().ensure_symbol(symbol):
                return True

        if symbol not in self.symbol_specs_cache:
            sym_u = symbol.upper()
            if "JPY" in sym_u:
                digits = 3
                point = 0.001
                pip_size = 0.01
                contract_size = 100000.0

                usdjpy_quote = self._sim_quotes.get("USDJPY")
                if usdjpy_quote and usdjpy_quote[0] > 0:
                    usdjpy_mid = (usdjpy_quote[0] + usdjpy_quote[1]) / 2.0
                    tick_value = (point * contract_size) / usdjpy_mid
                else:
                    tick_value = self.default_tick_value
            elif "XAU" in sym_u or "GOLD" in sym_u:
                digits = 2
                point = 0.01
                pip_size = 0.10
                contract_size = 100.0
                tick_value = 1.0
            else:
                digits = 5
                point = 0.00001
                pip_size = 0.0001
                contract_size = 100000.0
                tick_value = 1.0

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
        """Returns: (bid, ask, spread_points, spread_pips)"""
        self.ensure_symbol(symbol)
        specs = self.get_symbol_specs(symbol)
        if specs is None:
            return None

        # Prioritize explicit simulation quotes if present or if not using live feed
        if symbol in self._sim_quotes and not self.mt5_live_feed:
            bid, ask = self._sim_quotes[symbol]
            spread_points = (ask - bid) / specs.point
            spread_pips = (ask - bid) / specs.pip_size
            return bid, ask, spread_points, spread_pips

        if self.mt5_live_feed:
            try:
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None and tick.bid > 0 and tick.ask > 0:
                    spread_points = (tick.ask - tick.bid) / specs.point
                    spread_pips = (tick.ask - tick.bid) / specs.pip_size
                    return tick.bid, tick.ask, spread_points, spread_pips
            except Exception:
                pass

        quote = self._sim_quotes.get(symbol, (195.500, 195.515))
        bid, ask = quote
        spread_points = (ask - bid) / specs.point
        spread_pips = (ask - bid) / specs.pip_size
        return bid, ask, spread_points, spread_pips

    def calculate_pnl(
        self,
        symbol: str = "GBPJPY",
        order_type: str = "BUY",
        volume: float = 0.01,
        entry_price: float = 155.0,
        current_price: float = 155.2,
        usdjpy_rate: Optional[float] = None,
        specs: Optional[SymbolSpecs] = None,
    ) -> float:
        """Calculates PnL in USD using client specs and cross-currency conversion."""
        self.ensure_symbol(symbol)
        sym_specs = specs or self.get_symbol_specs(symbol)
        return calculate_pnl(
            entry_price=entry_price,
            exit_price=current_price,
            volume=volume,
            order_type=order_type,
            specs=sym_specs,
            usdjpy_rate=usdjpy_rate,
        )

    def get_spread_cost(
        self,
        symbol: str = "GBPJPY",
        volume: float = 0.01,
        spread_pips: float = 1.2,
        usdjpy_rate: float = 155.0,
    ) -> float:
        """Calculates spread cost in USD on position entry."""
        spread_price = spread_pips * 0.01
        return (spread_price * 100000.0 * volume) / usdjpy_rate

    def get_unrealized_pnl(self, ticket_or_pos: Union[int, OpenPosition]) -> float:
        """Returns floating PnL in USD for a given open position."""
        pos = ticket_or_pos if isinstance(ticket_or_pos, OpenPosition) else self._positions.get(ticket_or_pos)
        if pos is None:
            return 0.0

        quote = self.get_quote(pos.symbol)
        if not quote:
            return 0.0

        bid, ask, _, _ = quote
        current_price = bid if pos.type == 0 else ask
        specs = self.get_symbol_specs(pos.symbol)
        if not specs:
            return 0.0

        price_diff = (current_price - pos.price_open) if pos.type == 0 else (pos.price_open - current_price)
        points = price_diff / specs.point
        gross_pnl = points * specs.tick_value * pos.volume
        commission = self.commission_per_lot * pos.volume
        return round(gross_pnl - commission, 2)

    def _check_sl_tp_hits(self):
        """Monitors open positions and executes automatic exit if SL or TP is reached."""
        for ticket, pos in list(self._positions.items()):
            quote = self.get_quote(pos.symbol)
            if not quote:
                continue
            bid, ask, _, _ = quote
            is_buy = (pos.type == 0)
            eval_price = bid if is_buy else ask

            if pos.sl > 0:
                sl_hit = (eval_price <= pos.sl) if is_buy else (eval_price >= pos.sl)
                if sl_hit:
                    logger.warning(f"[GBPJPY PAPER] #{ticket} {pos.symbol} SL hit @ {eval_price:.3f}")
                    self.close_position(ticket, pos.symbol, pos.volume, is_buy)
                    continue

            if pos.tp > 0:
                tp_hit = (eval_price >= pos.tp) if is_buy else (eval_price <= pos.tp)
                if tp_hit:
                    logger.info(f"[GBPJPY PAPER] #{ticket} {pos.symbol} TP hit @ {eval_price:.3f}")
                    self.close_position(ticket, pos.symbol, pos.volume, is_buy)

    def get_account_snapshot(self) -> Optional[AccountSnapshot]:
        """Computes accurate USD equity, floating PnL, margin requirements, and margin level."""
        self._check_sl_tp_hits()

        floating_profit = 0.0
        used_margin = 0.0

        for pos in self._positions.values():
            quote = self.get_quote(pos.symbol)
            if quote:
                bid, ask, _, _ = quote
                current_price = bid if pos.type == 0 else ask
                specs = self.get_symbol_specs(pos.symbol)
                if specs:
                    price_diff = (current_price - pos.price_open) if pos.type == 0 else (pos.price_open - current_price)
                    points = price_diff / specs.point
                    gross_pnl = points * specs.tick_value * pos.volume
                    commission = self.commission_per_lot * pos.volume
                    pos_pnl = gross_pnl - commission
                    pos.profit = round(pos_pnl, 2)
                    floating_profit += pos_pnl

                    # Cross-currency margin in USD:
                    # GBPUSD approx = price_open * (tick_value / 100.0)
                    gbpusd_approx = pos.price_open * (specs.tick_value / 100.0)
                    notional_usd = pos.volume * specs.contract_size * gbpusd_approx
                    used_margin += notional_usd / self.leverage

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

    def get_account_info(self) -> Optional[AccountSnapshot]:
        """Alias for get_account_snapshot for compatibility."""
        return self.get_account_snapshot()

    def get_open_positions(self, symbol: Optional[str] = None) -> List[OpenPosition]:
        self.get_account_snapshot()
        positions = list(self._positions.values())
        if symbol:
            return [p for p in positions if p.symbol == symbol]
        return positions

    def open_position(
        self,
        symbol: str,
        order_type_str: str,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "SimOrder",
    ) -> OpenPosition:
        """
        Opens a simulated market order and returns the OpenPosition object.
        """
        self.ensure_symbol(symbol)
        quote = self.get_quote(symbol)
        specs = self.get_symbol_specs(symbol)
        digits = specs.digits if specs else 3

        if quote is not None:
            bid, ask, _, _ = quote
        else:
            bid, ask = 195.500, 195.515

        is_buy = str(order_type_str).upper() in ("BUY", "0", "ORDER_TYPE_BUY")
        price = ask if is_buy else bid

        price = round(price, digits)
        sl = round(sl, digits) if sl > 0 else 0.0
        tp = round(tp, digits) if tp > 0 else 0.0

        self._ticket_counter += 1
        ticket = self._ticket_counter

        pos = OpenPosition(
            ticket=ticket,
            symbol=symbol,
            type=0 if is_buy else 1,
            volume=round(volume, 2),
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
            f"[GBPJPY PAPER ORDER] #{ticket} {order_type_str} {volume:.2f} lot "
            f"{symbol} @ {price:.3f} | SL: {sl:.3f} | TP: {tp:.3f}"
        )
        return pos

    def send_market_order(
        self,
        symbol: str,
        order_type_str: str,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "SimOrder",
    ) -> Optional[int]:
        """MT5-compatible order execution returning ticket integer."""
        pos = self.open_position(symbol, order_type_str, volume, sl, tp, comment)
        return pos.ticket if pos else None

    def modify_position(
        self,
        ticket: int,
        symbol: Optional[str] = None,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> bool:
        """Modifies SL and TP of an open position."""
        if ticket in self._positions:
            pos = self._positions[ticket]
            sym = symbol or pos.symbol
            specs = self.get_symbol_specs(sym)
            digits = specs.digits if specs else 3

            pos.sl = round(sl, digits) if sl > 0 else 0.0
            pos.tp = round(tp, digits) if tp > 0 else 0.0
            logger.info(f"[GBPJPY PAPER] #{ticket} modified: SL={pos.sl:.3f}, TP={pos.tp:.3f}")
            return True
        return False

    def close_position(
        self,
        ticket_or_pos: Union[int, OpenPosition],
        symbol: Optional[str] = None,
        volume: Optional[float] = None,
        is_buy: Optional[bool] = None,
    ) -> Optional[CloseResult]:
        """
        Closes an open position and returns CloseResult (truthy object containing pnl/profit).
        """
        ticket = ticket_or_pos.ticket if isinstance(ticket_or_pos, OpenPosition) else ticket_or_pos

        if ticket not in self._positions:
            return None

        pos = self._positions[ticket]
        sym = symbol or pos.symbol
        vol = volume if volume is not None else pos.volume
        pos_is_buy = is_buy if is_buy is not None else (pos.type == 0)

        quote = self.get_quote(sym)
        if quote:
            bid, ask, _, _ = quote
            exit_price = bid if pos_is_buy else ask
        else:
            exit_price = pos.price_open

        specs = self.get_symbol_specs(sym)
        if specs:
            diff = (exit_price - pos.price_open) if pos_is_buy else (pos.price_open - exit_price)
            points = diff / specs.point
            gross_pnl = points * specs.tick_value * vol
            commission = self.commission_per_lot * vol
            realized_pnl = round(gross_pnl - commission, 2)
        else:
            diff = (exit_price - pos.price_open) if pos_is_buy else (pos.price_open - exit_price)
            points = diff / 0.001
            gross_pnl = points * self.default_tick_value * vol
            commission = self.commission_per_lot * vol
            realized_pnl = round(gross_pnl - commission, 2)

        self.balance = round(self.balance + realized_pnl, 2)
        self.equity = self.balance
        self._closed_pnl[ticket] = self._closed_pnl.get(ticket, 0.0) + realized_pnl

        if vol >= pos.volume - 1e-4:
            del self._positions[ticket]
        else:
            pos.volume = round(pos.volume - vol, 2)

        logger.info(
            f"[GBPJPY POSITION CLOSED] #{ticket} {sym} @ {exit_price:.3f} | "
            f"Net PnL: {realized_pnl:+.2f} {self.currency} | Balance: {self.balance:.2f} {self.currency}"
        )
        return CloseResult(ticket=ticket, pnl=realized_pnl, exit_price=exit_price, commission=commission)

    def close_all_positions(self) -> int:
        positions = list(self._positions.values())
        closed_count = 0
        for pos in positions:
            if self.close_position(pos.ticket):
                closed_count += 1
        return closed_count

    def get_closed_position_net_profit(self, position_ticket: int) -> Optional[float]:
        return self._closed_pnl.get(position_ticket)


# Alias MockClient to MockMT5Client for complete naming parity
MockClient = MockMT5Client

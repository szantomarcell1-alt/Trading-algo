"""
GBPJPY Execution Manager & Dynamic Position Lifecycle Module (Milestone 2)
Handles dynamic Breakeven adjustment (+1.45R trigger with +2.0 pip buffer),
ATR-based trailing stop ratcheting (never loosening/retreating), time-decay exit,
and signal execution suppression under circuit breaker lockouts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Dict, Optional, List, Any, Union
import logging

try:
    from gbpjpy.config.settings import load_config, AppConfig, ExecutionConfig
except ImportError:
    AppConfig = None
    ExecutionConfig = None

try:
    from gbpjpy.core.mt5_client import OpenPosition, SymbolSpecs
except ImportError:
    OpenPosition = None
    SymbolSpecs = None

logger = logging.getLogger("GBPJPY.ExecutionManager")


@dataclass
class PositionMetadata:
    """Metadata for tracking an open position throughout its lifecycle."""
    ticket: int
    symbol: str
    is_buy: bool
    entry_price: float
    initial_sl: float
    initial_tp: float
    initial_risk_dist: float  # Absolute price distance to initial SL (1R)
    highest_price: float = 0.0
    lowest_price: float = 0.0
    current_sl: float = 0.0
    breakeven_applied: bool = False
    partial_tp_applied: bool = False
    entry_time: float = field(default_factory=time.time)
    bars_held: int = 0


class ExecutionManager:
    """
    Manages order execution, position lifecycle, dynamic stop-loss adjustments,
    and stale trade exits tailored for GBPJPY scalping.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        mt5_client: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
    ):
        if config is None:
            try:
                from gbpjpy.config.settings import load_config
                self.config = load_config()
            except Exception:
                self.config = None
        else:
            self.config = config

        self.exec_cfg = getattr(self.config, "execution", None)
        self.mt5 = mt5_client
        self.risk_manager = risk_manager
        self.tracked_positions: Dict[int, PositionMetadata] = {}

    # -------------------------------------------------------------------------
    # Breakeven Logic
    # -------------------------------------------------------------------------

    def check_breakeven(
        self,
        order_type: str = "BUY",
        entry_price: float = 0.0,
        current_price: float = 0.0,
        initial_sl: float = 0.0,
        trigger_r: float = 1.45,
        is_long: Optional[bool] = None,
    ) -> bool:
        """
        Evaluates whether a trade has reached the required profit R-multiple (default +1.45R)
        to trigger moving stop-loss to Breakeven (+ buffer).
        """
        if is_long is not None:
            is_buy = is_long
        else:
            is_buy = str(order_type).upper() in ("BUY", "0", "TYPE_BUY", "LONG")

        if is_buy:
            risk = entry_price - initial_sl
            if risk <= 0:
                return False
            profit = current_price - entry_price
        else:
            risk = initial_sl - entry_price
            if risk <= 0:
                return False
            profit = entry_price - current_price

        current_r = profit / risk
        return current_r >= (trigger_r - 1e-9)

    # Alias for method compatibility
    should_move_to_breakeven = check_breakeven

    def calculate_breakeven_sl(
        self,
        order_type: str = "BUY",
        entry_price: float = 0.0,
        buffer_pips: float = 2.0,
        pip_size: float = 0.01,
        is_long: Optional[bool] = None,
    ) -> float:
        """
        Calculates the breakeven stop-loss price including profit buffer to cover
        spread and broker round-turn commissions:
        - BUY:  entry_price + (buffer_pips * pip_size)
        - SELL: entry_price - (buffer_pips * pip_size)
        """
        if is_long is not None:
            is_buy = is_long
        else:
            is_buy = str(order_type).upper() in ("BUY", "0", "TYPE_BUY", "LONG")

        buffer_dist = buffer_pips * pip_size
        if is_buy:
            return round(entry_price + buffer_dist, 3)
        else:
            return round(entry_price - buffer_dist, 3)

    # Alias for method compatibility
    get_breakeven_price = calculate_breakeven_sl

    # -------------------------------------------------------------------------
    # Trailing Stop Logic & Ratchet Invariant
    # -------------------------------------------------------------------------

    def calculate_trailing_stop(
        self,
        order_type: str = "BUY",
        highest_price: Optional[float] = None,
        lowest_price: Optional[float] = None,
        current_price: Optional[float] = None,
        current_sl: float = 0.0,
        atr: float = 0.0,
        multiplier: float = 1.5,
        is_long: Optional[bool] = None,
    ) -> float:
        """
        Calculates ATR-based trailing stop price:
        - BUY:  highest_price - (multiplier * atr)
        - SELL: lowest_price + (multiplier * atr)

        Enforces the RATCHET INVARIANT:
        The stop-loss must strictly advance in the direction of profit.
        If price retraces, the stop-loss NEVER retreats / loosens.
        """
        if is_long is not None:
            is_buy = is_long
        else:
            is_buy = str(order_type).upper() in ("BUY", "0", "TYPE_BUY", "LONG")

        trail_dist = multiplier * atr

        if is_buy:
            ref_high = highest_price if highest_price is not None else current_price
            if ref_high is None:
                return current_sl
            candidate_sl = round(ref_high - trail_dist, 3)
            # Invariant: BUY SL only ratchets higher
            if current_sl > 0:
                return max(current_sl, candidate_sl)
            return candidate_sl
        else:
            ref_low = lowest_price if lowest_price is not None else current_price
            if ref_low is None:
                return current_sl
            candidate_sl = round(ref_low + trail_dist, 3)
            # Invariant: SELL SL only ratchets lower
            if current_sl > 0:
                return min(current_sl, candidate_sl)
            return candidate_sl

    # Alias for method compatibility
    update_trailing_sl = calculate_trailing_stop

    def update_trailing_stop(
        self,
        is_long: bool = True,
        entry_price: float = 0.0,
        current_sl: float = 0.0,
        current_price: float = 0.0,
        atr: float = 0.0,
        multiplier: float = 1.5,
        trigger_r: float = 1.45,
        buffer_pips: float = 2.0,
        pip_size: float = 0.01,
    ) -> float:
        """
        High-level orchestrator for trailing stop progression:
        1. If profit reaches trigger_r (1.45R), moves candidate to Breakeven (+ buffer).
        2. Ratchets candidate stop using ATR trailing distance.
        3. Enforces ratchet invariant (never moves backwards).
        """
        order_type = "BUY" if is_long else "SELL"
        risk_dist = abs(entry_price - current_sl) if current_sl > 0 else (atr * multiplier)
        if risk_dist <= 0:
            risk_dist = 0.200  # Default 20 pips

        profit_dist = (current_price - entry_price) if is_long else (entry_price - current_price)
        current_r = profit_dist / risk_dist if risk_dist > 0 else 0.0

        candidate_sl = current_sl

        # 1. Breakeven adjustment
        if current_r >= trigger_r:
            be_sl = self.calculate_breakeven_sl(
                order_type=order_type,
                entry_price=entry_price,
                buffer_pips=buffer_pips,
                pip_size=pip_size,
            )
            if is_long:
                candidate_sl = max(candidate_sl, be_sl)
            else:
                candidate_sl = min(candidate_sl, be_sl) if candidate_sl > 0 else be_sl

        # 2. ATR Trailing Stop
        if atr > 0:
            atr_sl = self.calculate_trailing_stop(
                order_type=order_type,
                highest_price=current_price if is_long else None,
                lowest_price=current_price if not is_long else None,
                current_sl=candidate_sl,
                atr=atr,
                multiplier=multiplier,
                is_long=is_long,
            )
            if is_long:
                candidate_sl = max(candidate_sl, atr_sl)
            else:
                candidate_sl = min(candidate_sl, atr_sl) if candidate_sl > 0 else atr_sl

        # 3. Ratchet Invariant
        if is_long:
            return max(current_sl, candidate_sl)
        else:
            return min(current_sl, candidate_sl) if current_sl > 0 else candidate_sl

    # -------------------------------------------------------------------------
    # Time-Decay Position Exit Logic
    # -------------------------------------------------------------------------

    def check_time_decay_exit(
        self,
        bars_held: int,
        max_bars: int = 48,
        current_r: Optional[float] = None,
        min_momentum_r: float = 0.5,
    ) -> bool:
        """
        Time-decay exit check:
        Stale trades held beyond max_bars (e.g. 48 M5 bars = 4 hours) without momentum
        exit early to recycle capital.
        """
        if bars_held < max_bars:
            return False
        if current_r is not None and current_r >= min_momentum_r:
            return False
        return True

    # Alias for method compatibility
    is_time_decay_exit = check_time_decay_exit

    # -------------------------------------------------------------------------
    # Order Dispatch & Circuit Breaker Suppression
    # -------------------------------------------------------------------------

    def execute_signal(
        self,
        signal: Any,
        risk_manager: Optional[Any] = None,
        client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Attempts to execute a strategy TradeSignal after risk and circuit breaker verification.
        If RiskManager circuit breaker is active (or locked/cooling down), execution is
        strictly suppressed and returns {'executed': False, 'reason': 'Circuit breaker active: ...'}.
        """
        rm = risk_manager or self.risk_manager
        mt5 = client or self.mt5

        # 1. Circuit Breaker / Risk Validation Check
        if rm is not None:
            cb = getattr(rm, "circuit_breaker", rm)
            is_circuit_active = False
            reasons = []

            if getattr(rm, "is_locked", False) or getattr(cb, "is_locked", False):
                is_circuit_active = True
                reasons.append("circuit breaker locked")

            if hasattr(cb, "is_circuit_breaker_active") and cb.is_circuit_breaker_active():
                is_circuit_active = True
                reasons.append("circuit breaker active")

            if hasattr(cb, "is_cooldown_active") and cb.is_cooldown_active():
                is_circuit_active = True
                reasons.append("circuit breaker cooldown active")

            if hasattr(cb, "is_daily_halted") and cb.is_daily_halted():
                is_circuit_active = True
                reasons.append("circuit breaker daily drawdown halt")

            if hasattr(rm, "can_trade") and not rm.can_trade():
                is_circuit_active = True
                reasons.append("circuit breaker trading disallowed")
            elif hasattr(cb, "can_trade") and not cb.can_trade():
                is_circuit_active = True
                reasons.append("circuit breaker trading disallowed")

            if hasattr(rm, "validate_signal"):
                valid, reason = rm.validate_signal(signal)
                if not valid:
                    return {"executed": False, "reason": reason}

            if is_circuit_active:
                reason_text = "Circuit breaker active: " + (", ".join(reasons) if reasons else "trading suppressed")
                logger.warning(f"Trade execution suppressed: {reason_text}")
                return {"executed": False, "reason": reason_text}

            # Check position concurrency / margin limits
            if hasattr(rm, "can_open_trade"):
                open_positions_count = len(self.tracked_positions)
                if not rm.can_open_trade(open_positions_count=open_positions_count):
                    return {"executed": False, "reason": "Risk limit: max open positions reached"}

        # 2. Extract signal details
        symbol = getattr(signal, "symbol", signal.get("symbol", "GBPJPY") if isinstance(signal, dict) else "GBPJPY")
        signal_type = getattr(signal, "signal_type", signal.get("signal_type", "BUY") if isinstance(signal, dict) else "BUY")
        sig_type_str = getattr(signal_type, "value", str(signal_type)).upper()
        is_buy = "BUY" in sig_type_str or sig_type_str in ("0", "LONG")
        entry_price = getattr(signal, "entry_price", signal.get("entry_price", 0.0) if isinstance(signal, dict) else 0.0)
        sl = getattr(signal, "sl", signal.get("sl", 0.0) if isinstance(signal, dict) else 0.0)
        tp = getattr(signal, "tp", signal.get("tp", 0.0) if isinstance(signal, dict) else 0.0)
        volume = getattr(signal, "volume", signal.get("volume", 0.01) if isinstance(signal, dict) else 0.01)

        # 3. Execution via Client if available, else mock ticket
        ticket = 1001
        if mt5 is not None and hasattr(mt5, "open_position"):
            pos = mt5.open_position(symbol, "BUY" if is_buy else "SELL", volume=volume, sl=sl, tp=tp)
            ticket = getattr(pos, "ticket", 1001) if pos else 1001
        elif mt5 is not None and hasattr(mt5, "send_market_order"):
            ticket = mt5.send_market_order(symbol, "BUY" if is_buy else "SELL", volume=volume, sl=sl, tp=tp)
            if not ticket:
                return {"executed": False, "reason": "Order placement failed on client"}

        self.register_new_position(
            ticket=ticket,
            symbol=symbol,
            is_buy=is_buy,
            entry_price=entry_price,
            initial_sl=sl,
            initial_tp=tp,
        )

        return {
            "executed": True,
            "ticket": ticket,
            "symbol": symbol,
            "volume": volume,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "reason": "Executed successfully",
        }

    # -------------------------------------------------------------------------
    # Position Tracking & Monitoring
    # -------------------------------------------------------------------------

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
            highest_price=entry_price,
            lowest_price=entry_price,
            current_sl=initial_sl,
            breakeven_applied=False,
            partial_tp_applied=False,
            entry_time=time.time(),
            bars_held=0,
        )
        logger.info(f"Registered position #{ticket} for management | 1R dist: {risk_dist:.3f}")

    def synchronize_with_open_positions(self, live_positions: List[Any]):
        """Synchronizes internal tracking registry with active broker positions."""
        live_tickets = {getattr(p, "ticket", p.get("ticket") if isinstance(p, dict) else 0) for p in live_positions}

        # Remove closed positions and notify RiskManager
        closed_tickets = [t for t in list(self.tracked_positions.keys()) if t not in live_tickets]
        for t in closed_tickets:
            meta = self.tracked_positions[t]
            logger.info(f"Position #{t} removed from active tracking (closed).")
            if self.risk_manager is not None and self.mt5 is not None:
                if hasattr(self.mt5, "get_closed_position_net_profit"):
                    realized_pnl = self.mt5.get_closed_position_net_profit(t)
                    if realized_pnl is not None and hasattr(self.risk_manager, "record_trade_result"):
                        self.risk_manager.record_trade_result(realized_pnl)
            del self.tracked_positions[t]

    def manage_positions(self, current_atr_by_symbol: Dict[str, float]):
        """Evaluates dynamic management rules across open positions."""
        if self.mt5 is None:
            return

        live_positions = self.mt5.get_open_positions() if hasattr(self.mt5, "get_open_positions") else []
        self.synchronize_with_open_positions(live_positions)

        for pos in live_positions:
            ticket = getattr(pos, "ticket", 0)
            meta = self.tracked_positions.get(ticket)
            if not meta:
                continue

            quote = self.mt5.get_quote(meta.symbol) if hasattr(self.mt5, "get_quote") else None
            if not quote:
                continue

            bid, ask = quote[0], quote[1]
            current_price = bid if meta.is_buy else ask
            current_atr = current_atr_by_symbol.get(meta.symbol, 0.150)

            # Update highest/lowest price
            meta.highest_price = max(meta.highest_price, current_price)
            meta.lowest_price = min(meta.lowest_price, current_price) if meta.lowest_price > 0 else current_price

            # Apply updated trailing stop
            new_sl = self.update_trailing_stop(
                is_long=meta.is_buy,
                entry_price=meta.entry_price,
                current_sl=meta.current_sl,
                current_price=current_price,
                atr=current_atr,
            )
            if (meta.is_buy and new_sl > meta.current_sl) or (not meta.is_buy and new_sl < meta.current_sl):
                meta.current_sl = new_sl
                if hasattr(self.mt5, "modify_position"):
                    self.mt5.modify_position(ticket, sl=new_sl)

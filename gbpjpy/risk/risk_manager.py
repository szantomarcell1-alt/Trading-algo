"""
GBPJPY Risk Management Module (Milestone 2)
Provides small-capital institutional risk management ($250 equity, 1:30 leverage):
- Exact required margin calculation for GBPJPY with GBPUSD conversion
- Dynamic lot sizing with 1.0% risk parity and 0.01 min lot clamping
- Strict margin level guarding (>200% threshold)
- Single open position limit enforcement (max_open_positions: 1)
- 3-strike loss streak circuit breaker with cooldown
- 3.0% maximum daily drawdown circuit breaker
- Next-day calendar auto-reset mechanism
- Spread filter (max 2.5 - 3.0 pips)
- Wednesday 3x swap cost calculation
"""

from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Optional, Tuple, Union, Dict, Any
import math
import logging

try:
    from gbpjpy.core.mt5_client import SymbolSpecs, AccountSnapshot
    from gbpjpy.config.settings import load_config, AppConfig, InstrumentConfig
    from gbpjpy.utils.logger import get_logger
except ImportError:
    try:
        from core.mt5_client import SymbolSpecs, AccountSnapshot
        from config.settings import load_config, AppConfig, InstrumentConfig
        from utils.logger import get_logger
    except (ImportError, ValueError):
        SymbolSpecs = None
        AccountSnapshot = None
        AppConfig = None
        InstrumentConfig = None
        get_logger = logging.getLogger

logger = logging.getLogger("GBPJPY.RiskManager")


@dataclass
class AccountInfo:
    """Mock/Live trading account info calibrated for $250 equity and 1:30 leverage."""
    balance: float = 250.0
    equity: float = 250.0
    margin: float = 0.0
    free_margin: float = 250.0
    margin_level: float = 0.0
    leverage: float = 30.0
    currency: str = "USD"
    profit: float = 0.0
    login: int = 0

    @property
    def margin_free(self) -> float:
        return self.free_margin

    @margin_free.setter
    def margin_free(self, val: float):
        self.free_margin = val

    def update_margin(self, required_margin: float, unrealized_pnl: float = 0.0):
        self.margin = required_margin
        self.equity = self.balance + unrealized_pnl
        self.free_margin = self.equity - self.margin
        if self.margin > 0:
            self.margin_level = (self.equity / self.margin) * 100.0
        else:
            self.margin_level = 0.0


def calculate_required_margin(
    volume: float,
    leverage: float = 30.0,
    contract_size: float = 100000.0,
    gbpusd_rate: float = 1.28,
) -> float:
    """
    Margin required for GBP-base pair in USD account.
    Formula: (volume * contract_size * gbpusd_rate) / leverage
    """
    notional_usd = volume * contract_size * gbpusd_rate
    return round(notional_usd / leverage, 4)


def calculate_swap(
    volume: float,
    is_long: bool = True,
    rollover_dt: Optional[datetime] = None,
    base_swap_per_lot: float = -15.0,
) -> float:
    """
    Swap calculation with Wednesday 3x rollover modeling (T+2 weekend settlement).
    """
    if rollover_dt is None:
        rollover_dt = datetime.now(timezone.utc)
    multiplier = 3.0 if rollover_dt.weekday() == 2 else 1.0
    rate = base_swap_per_lot if is_long else (base_swap_per_lot * 0.5)
    return round(rate * volume * multiplier, 4)


def is_spread_acceptable(spread_pips: float, max_spread: float = 3.0) -> bool:
    """Spread filter: returns True if spread is <= max_spread pips."""
    return spread_pips <= max_spread


class CircuitBreaker:
    """
    Intraday Risk Circuit Breaker:
    - 3 consecutive losses: intraday cooldown / halt
    - 3.0% maximum daily drawdown: trading halted for rest of calendar day
    - Automatic next-day rollover reset
    """
    def __init__(
        self,
        max_consecutive_losses: int = 3,
        cooldown_minutes: int = 60,
        max_daily_drawdown_pct: float = 0.03,
        daily_starting_equity: float = 250.0,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.daily_starting_equity = daily_starting_equity

        self.consecutive_losses: int = 0
        self.daily_pnl: float = 0.0
        self.daily_halted: bool = False
        self.cooldown_active: bool = False
        self.cooldown_until: Optional[float] = None
        self.current_date: Optional[date] = None
        self.is_locked: bool = False

    def _ensure_date(self, current_dt: Optional[Union[date, datetime]] = None):
        if current_dt is None:
            return
        d = current_dt.date() if isinstance(current_dt, datetime) else current_dt
        if self.current_date is not None and d > self.current_date:
            self.reset_daily_limits(current_date=d)
        elif self.current_date is None:
            self.current_date = d

    def record_trade_result(
        self,
        pnl: float,
        is_win: Optional[bool] = None,
        trade_date: Optional[Union[date, datetime]] = None,
    ):
        if trade_date is not None:
            self._ensure_date(trade_date)
        elif self.current_date is None:
            self.current_date = datetime.now(timezone.utc).date()

        win = is_win if is_win is not None else (pnl > 0)
        self.daily_pnl += pnl

        if win:
            self.consecutive_losses = 0
            self.cooldown_active = False
            self.cooldown_until = None
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.cooldown_active = True
                if isinstance(trade_date, datetime):
                    now_ts = trade_date.timestamp()
                else:
                    now_ts = datetime.now(timezone.utc).timestamp()
                self.cooldown_until = now_ts + (self.cooldown_minutes * 60)

        max_allowed_loss = self.daily_starting_equity * self.max_daily_drawdown_pct
        if self.daily_pnl <= -max_allowed_loss:
            self.daily_halted = True

    def can_trade(self, current_date: Optional[Union[date, datetime]] = None) -> bool:
        if current_date is not None:
            self._ensure_date(current_date)
        if self.is_locked:
            return False
        if self.daily_halted:
            return False
        now_dt = current_date if isinstance(current_date, datetime) else None
        if self.is_cooldown_active(now=now_dt):
            return False
        return True

    def is_cooldown_active(self, now: Optional[datetime] = None) -> bool:
        if not self.cooldown_active:
            return False
        current_time = now if now is not None else datetime.now(timezone.utc)
        if self.cooldown_until is not None:
            now_ts = current_time.timestamp()
            if now_ts >= self.cooldown_until:
                self.cooldown_active = False
                self.cooldown_until = None
                self.consecutive_losses = 0
                return False
        return self.cooldown_active

    def is_daily_halted(self) -> bool:
        return self.daily_halted

    def is_active(self, now: Optional[datetime] = None) -> bool:
        return self.is_locked or self.daily_halted or self.is_cooldown_active(now=now)

    def is_circuit_breaker_active(self, now: Optional[datetime] = None) -> bool:
        return self.is_active(now=now)

    def in_cooldown(self, now: Optional[datetime] = None) -> bool:
        return self.is_cooldown_active(now=now)

    def trip_circuit_breaker(self, reason: str = ""):
        self.consecutive_losses = max(self.consecutive_losses, self.max_consecutive_losses)
        self.cooldown_active = True
        self.is_locked = True

    def reset_daily_limits(
        self,
        new_equity: Optional[float] = None,
        current_date: Optional[Union[date, datetime]] = None,
    ):
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.daily_halted = False
        self.cooldown_active = False
        self.cooldown_until = None
        self.is_locked = False
        if new_equity is not None:
            self.daily_starting_equity = new_equity
        if current_date is not None:
            self.current_date = current_date.date() if isinstance(current_date, datetime) else current_date

    def on_new_day(self, current_date: Optional[Union[date, datetime]] = None):
        self.reset_daily_limits(current_date=current_date)

    def set_daily_starting_equity(self, equity: float):
        self.daily_starting_equity = equity


class RiskManager:
    """
    Main Risk & Execution Manager for small capital GBPJPY quantitative trading.
    """
    def __init__(
        self,
        config: Optional[Union[AppConfig, Dict[str, Any]]] = None,
        max_risk_pct: Optional[float] = None,
        max_daily_dd_pct: Optional[float] = None,
        max_positions: Optional[int] = None,
        initial_balance: Optional[float] = None,
        leverage: Optional[float] = None,
        max_consecutive_losses: Optional[int] = None,
        consecutive_loss_cooldown_minutes: Optional[int] = None,
        min_margin_level_percent: Optional[float] = None,
        max_spread_pips: Optional[float] = None,
        commission_per_lot: Optional[float] = None,
        **kwargs
    ):
        # Default baseline configuration
        self.initial_balance: float = 250.0
        self.leverage: float = 30.0
        self.max_risk_pct: float = 0.01  # 1.0%
        self.max_daily_dd_pct: float = 0.03  # 3.0%
        self.max_positions: int = 1
        self.max_consecutive_losses: int = 3
        self.consecutive_loss_cooldown_minutes: int = 60
        self.min_margin_level_percent: float = 200.0
        self.max_spread_pips: float = 3.0
        self.commission_per_lot_round_turn: float = 7.0
        self.contract_size: float = 100000.0
        self.gbpusd_rate: float = 1.28

        # Ingest configuration object or dictionary if supplied
        cfg_obj = None
        if config is not None:
            cfg_obj = config
        else:
            try:
                cfg_obj = load_config()
            except Exception:
                cfg_obj = None

        if cfg_obj is not None:
            if isinstance(cfg_obj, dict):
                r_cfg = cfg_obj.get("risk", cfg_obj.get("risk_management", {}))
                s_cfg = cfg_obj.get("symbol", {})
                self.initial_balance = float(r_cfg.get("initial_balance", self.initial_balance))
                self.leverage = float(r_cfg.get("leverage", self.leverage))
                r_pct = float(r_cfg.get("risk_per_trade_percent", 1.0))
                self.max_risk_pct = r_pct / 100.0 if r_pct > 0.1 else r_pct
                dd_pct = float(r_cfg.get("max_daily_drawdown_percent", 3.0))
                self.max_daily_dd_pct = dd_pct / 100.0 if dd_pct > 0.1 else dd_pct
                self.max_positions = int(r_cfg.get("max_open_positions", self.max_positions))
                self.max_consecutive_losses = int(r_cfg.get("max_consecutive_losses", self.max_consecutive_losses))
                self.consecutive_loss_cooldown_minutes = int(r_cfg.get("consecutive_loss_cooldown_minutes", self.consecutive_loss_cooldown_minutes))
                self.min_margin_level_percent = float(r_cfg.get("min_margin_level_percent", self.min_margin_level_percent))
                self.commission_per_lot_round_turn = float(r_cfg.get("commission_per_lot_round_turn", self.commission_per_lot_round_turn))
                self.max_spread_pips = float(s_cfg.get("max_spread_pips", self.max_spread_pips))
            elif hasattr(cfg_obj, "risk_management"):
                rm_cfg = cfg_obj.risk_management
                self.initial_balance = float(getattr(rm_cfg, "initial_balance", self.initial_balance))
                self.leverage = float(getattr(rm_cfg, "leverage", self.leverage))
                r_pct = float(getattr(rm_cfg, "risk_per_trade_percent", 1.0))
                self.max_risk_pct = r_pct / 100.0 if r_pct > 0.1 else r_pct
                dd_pct = float(getattr(rm_cfg, "max_daily_drawdown_percent", 3.0))
                self.max_daily_dd_pct = dd_pct / 100.0 if dd_pct > 0.1 else dd_pct
                self.max_positions = int(getattr(rm_cfg, "max_open_positions", self.max_positions))
                self.max_consecutive_losses = int(getattr(rm_cfg, "max_consecutive_losses", self.max_consecutive_losses))
                self.consecutive_loss_cooldown_minutes = int(getattr(rm_cfg, "consecutive_loss_cooldown_minutes", self.consecutive_loss_cooldown_minutes))
                self.min_margin_level_percent = float(getattr(rm_cfg, "min_margin_level_percent", self.min_margin_level_percent))
                self.commission_per_lot_round_turn = float(getattr(rm_cfg, "commission_per_lot_round_turn", self.commission_per_lot_round_turn))

        # Explicit overrides from kwargs
        if max_risk_pct is not None:
            self.max_risk_pct = max_risk_pct / 100.0 if max_risk_pct > 0.1 else max_risk_pct
        if max_daily_dd_pct is not None:
            self.max_daily_dd_pct = max_daily_dd_pct / 100.0 if max_daily_dd_pct > 0.1 else max_daily_dd_pct
        if max_positions is not None:
            self.max_positions = max_positions
        if initial_balance is not None:
            self.initial_balance = initial_balance
        if leverage is not None:
            self.leverage = leverage
        if max_consecutive_losses is not None:
            self.max_consecutive_losses = max_consecutive_losses
        if consecutive_loss_cooldown_minutes is not None:
            self.consecutive_loss_cooldown_minutes = consecutive_loss_cooldown_minutes
        if min_margin_level_percent is not None:
            self.min_margin_level_percent = min_margin_level_percent
        if max_spread_pips is not None:
            self.max_spread_pips = max_spread_pips
        if commission_per_lot is not None:
            self.commission_per_lot_round_turn = commission_per_lot

        self.risk_per_trade_percent = self.max_risk_pct * 100.0

        # Subcomponent Circuit Breaker
        self.circuit_breaker = CircuitBreaker(
            max_consecutive_losses=self.max_consecutive_losses,
            cooldown_minutes=self.consecutive_loss_cooldown_minutes,
            max_daily_drawdown_pct=self.max_daily_dd_pct,
            daily_starting_equity=self.initial_balance,
        )

    # -------------------------------------------------------------------------
    # Circuit Breaker Delegation & State Aliases
    # -------------------------------------------------------------------------
    @property
    def is_locked(self) -> bool:
        return self.circuit_breaker.is_locked

    @is_locked.setter
    def is_locked(self, val: bool):
        self.circuit_breaker.is_locked = val

    @property
    def loss_streak(self) -> int:
        return self.circuit_breaker.consecutive_losses

    @property
    def daily_starting_equity(self) -> float:
        return self.circuit_breaker.daily_starting_equity

    @daily_starting_equity.setter
    def daily_starting_equity(self, val: float):
        self.circuit_breaker.daily_starting_equity = val

    def record_trade_result(
        self,
        pnl: float,
        is_win: Optional[bool] = None,
        trade_date: Optional[Union[date, datetime]] = None,
    ):
        self.circuit_breaker.record_trade_result(pnl=pnl, is_win=is_win, trade_date=trade_date)

    def record_loss(self, pnl: float):
        self.record_trade_result(pnl=pnl, is_win=False)

    def record_daily_pnl(self, pnl: float):
        self.record_trade_result(pnl=pnl)

    def can_trade(self, current_date: Optional[Union[date, datetime]] = None) -> bool:
        return self.circuit_breaker.can_trade(current_date=current_date)

    def is_cooldown_active(self, now: Optional[datetime] = None) -> bool:
        return self.circuit_breaker.is_cooldown_active(now=now)

    def in_cooldown(self, now: Optional[datetime] = None) -> bool:
        return self.circuit_breaker.in_cooldown(now=now)

    def is_circuit_breaker_active(self, now: Optional[datetime] = None) -> bool:
        return self.circuit_breaker.is_circuit_breaker_active(now=now)

    def is_daily_halted(self) -> bool:
        return self.circuit_breaker.is_daily_halted()

    def trip_circuit_breaker(self, reason: str = ""):
        self.circuit_breaker.trip_circuit_breaker(reason=reason)

    def reset_daily_limits(
        self,
        new_equity: Optional[float] = None,
        current_date: Optional[Union[date, datetime]] = None,
    ):
        self.circuit_breaker.reset_daily_limits(new_equity=new_equity, current_date=current_date)

    def on_new_day(self, current_date: Optional[Union[date, datetime]] = None):
        self.circuit_breaker.on_new_day(current_date=current_date)

    def reset_day(self, new_equity: Optional[float] = None):
        self.reset_daily_limits(new_equity=new_equity)

    def set_daily_starting_equity(self, equity: float):
        self.circuit_breaker.set_daily_starting_equity(equity=equity)

    def check_daily_drawdown(
        self,
        account: Union[AccountInfo, Any],
        starting_equity: Optional[float] = None,
    ) -> bool:
        start_eq = starting_equity if starting_equity is not None else self.circuit_breaker.daily_starting_equity
        loss = start_eq - account.equity
        max_loss = start_eq * self.max_daily_dd_pct
        return loss <= max_loss

    def check_daily_loss_limit(
        self,
        daily_pnl: float,
        starting_equity: Optional[float] = None,
    ) -> bool:
        start_eq = starting_equity if starting_equity is not None else self.circuit_breaker.daily_starting_equity
        max_loss = start_eq * self.max_daily_dd_pct
        return daily_pnl > -max_loss

    # -------------------------------------------------------------------------
    # Margin & Leverage Calculations
    # -------------------------------------------------------------------------
    def calculate_required_margin(
        self,
        volume: float,
        leverage: Optional[float] = None,
        contract_size: Optional[float] = None,
        gbpusd_rate: Optional[float] = None,
    ) -> float:
        lev = leverage if leverage is not None else self.leverage
        c_size = contract_size if contract_size is not None else self.contract_size
        rate = gbpusd_rate if gbpusd_rate is not None else self.gbpusd_rate
        return calculate_required_margin(volume=volume, leverage=lev, contract_size=c_size, gbpusd_rate=rate)

    def calculate_max_allowed_lots(
        self,
        equity: float,
        gbpusd_rate: Optional[float] = None,
        leverage: Optional[float] = None,
    ) -> float:
        lev = leverage if leverage is not None else self.leverage
        rate = gbpusd_rate if gbpusd_rate is not None else self.gbpusd_rate
        raw_max = (equity * lev) / (self.contract_size * rate)
        stepped = math.floor(raw_max / 0.01) * 0.01
        return round(max(0.01, stepped), 2)

    def check_margin_level(
        self,
        account: Optional[Union[AccountInfo, Any]] = None,
        equity: Optional[float] = None,
        total_margin: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> bool:
        if account is not None:
            if getattr(account, "margin", 0.0) <= 0:
                return True
            m_level = getattr(account, "margin_level", 0.0)
            if m_level <= 0 and account.margin > 0:
                m_level = (account.equity / account.margin) * 100.0
            return m_level >= self.min_margin_level_percent

        eq = equity if equity is not None else self.initial_balance
        m = total_margin if total_margin is not None else margin
        if m is None or m <= 0:
            return True
        level = (eq / m) * 100.0
        return level >= self.min_margin_level_percent

    def is_margin_sufficient(self, *args, **kwargs) -> bool:
        return self.check_margin_level(*args, **kwargs)

    def can_open_position(
        self,
        equity: float,
        required_margin: float,
        current_margin: float = 0.0,
        now: Optional[Union[date, datetime]] = None,
        current_date: Optional[Union[date, datetime]] = None,
    ) -> bool:
        dt = current_date or now
        if not self.can_trade(current_date=dt):
            return False
        if equity < required_margin:
            return False
        total_m = current_margin + required_margin
        if total_m > 0:
            level = (equity / total_m) * 100.0
            if level < self.min_margin_level_percent:
                return False
        return True

    def validate_leverage(
        self,
        volume: float,
        equity: float,
        gbpusd_rate: Optional[float] = None,
        max_leverage: Optional[float] = None,
    ) -> bool:
        rate = gbpusd_rate if gbpusd_rate is not None else self.gbpusd_rate
        max_lev = max_leverage if max_leverage is not None else self.leverage
        if equity <= 0:
            return False
        effective_leverage = (volume * self.contract_size * rate) / equity
        return effective_leverage <= max_lev

    # -------------------------------------------------------------------------
    # Position Sizing & Clamping
    # -------------------------------------------------------------------------
    def clamp_lot_size(
        self,
        volume: float,
        min_lot: float = 0.01,
        max_lot: float = 100.0,
        step: float = 0.01,
    ) -> float:
        if volume <= 0:
            return 0.0
        clamped = max(min_lot, min(max_lot, volume))
        stepped = math.floor(clamped / step) * step
        return round(max(min_lot, stepped), 2)

    def calculate_lot_size(
        self,
        account: Optional[Union[AccountInfo, Any]] = None,
        equity: Optional[float] = None,
        sl_pips: Optional[float] = None,
        risk_distance_points: Optional[float] = None,
        specs: Optional[Any] = None,
        risk_pct: Optional[float] = None,
        gbpusd_rate: Optional[float] = None,
        **kwargs
    ) -> float:
        # Determine effective equity
        eff_equity = None
        if account is not None and hasattr(account, "equity"):
            eff_equity = float(account.equity)
        elif equity is not None:
            eff_equity = float(equity)
        else:
            eff_equity = self.initial_balance

        rate = gbpusd_rate if gbpusd_rate is not None else self.gbpusd_rate

        # Minimum required margin for 0.01 lot
        min_margin = self.calculate_required_margin(volume=0.01, gbpusd_rate=rate)
        if eff_equity < min_margin:
            return 0.0

        # Points distance
        pts = 0.0
        if risk_distance_points is not None:
            pts = float(risk_distance_points)
        elif sl_pips is not None:
            pts = float(sl_pips) * 10.0

        if pts <= 0:
            return 0.0

        # Risk percentage
        r_fraction = self.max_risk_pct
        if risk_pct is not None:
            r_fraction = risk_pct / 100.0 if risk_pct > 0.1 else risk_pct

        risk_amount = eff_equity * r_fraction

        # Tick value
        tick_val = 0.626076
        if specs is not None and hasattr(specs, "tick_value"):
            tick_val = float(specs.tick_value)

        # Loss per standard lot
        loss_per_lot = (pts * tick_val) + self.commission_per_lot_round_turn
        if loss_per_lot <= 0:
            return 0.0

        raw_lot = risk_amount / loss_per_lot
        stepped_lot = math.floor(raw_lot / 0.01) * 0.01

        # Enforce 0.01 lot minimum parity
        bounded_lot = max(0.01, stepped_lot)

        # Enforce leverage ceiling
        max_lot = self.calculate_max_allowed_lots(equity=eff_equity, gbpusd_rate=rate)
        final_lot = min(bounded_lot, max_lot)

        # Enforce minimum margin level requirement (e.g. 200% margin level safety gate)
        if self.min_margin_level_percent > 0:
            max_margin = eff_equity / (self.min_margin_level_percent / 100.0)
            max_lot_margin = math.floor(((max_margin * self.leverage) / (self.contract_size * rate)) / 0.01) * 0.01
            final_lot = min(final_lot, max(0.01, round(max_lot_margin, 2)))

        return round(final_lot, 2)

    # -------------------------------------------------------------------------
    # Trade Approval & Order Validation
    # -------------------------------------------------------------------------
    def is_spread_acceptable(self, spread_pips: float, max_spread: Optional[float] = None) -> bool:
        lim = max_spread if max_spread is not None else self.max_spread_pips
        return is_spread_acceptable(spread_pips=spread_pips, max_spread=lim)

    def can_trade_spread(self, spread_pips: float) -> bool:
        return self.is_spread_acceptable(spread_pips=spread_pips)

    def calculate_swap(
        self,
        volume: float,
        is_long: bool = True,
        rollover_dt: Optional[datetime] = None,
    ) -> float:
        return calculate_swap(volume=volume, is_long=is_long, rollover_dt=rollover_dt)

    def can_open_trade(
        self,
        account: Optional[Union[AccountInfo, Any]] = None,
        open_positions_count: int = 0,
        now: Optional[Union[date, datetime]] = None,
        current_date: Optional[Union[date, datetime]] = None,
        **kwargs
    ) -> bool:
        if open_positions_count >= self.max_positions:
            return False
        dt = current_date or now
        if not self.can_trade(current_date=dt):
            return False
        if account is not None:
            if not self.check_margin_level(account=account):
                return False
            if not self.check_daily_drawdown(account=account):
                return False
        return True

    def can_open_new_trade(
        self,
        account: Union[AccountInfo, Any],
        open_positions_count: int,
        symbol_positions_count: int = 0,
        now: Optional[Union[date, datetime]] = None,
        current_date: Optional[Union[date, datetime]] = None,
    ) -> Tuple[bool, str]:
        dt = current_date or now
        if not self.can_trade(current_date=dt):
            return False, "Circuit breaker active or daily loss limit reached"
        if open_positions_count >= self.max_positions:
            return False, f"Maximum open positions reached ({open_positions_count}/{self.max_positions})"
        if symbol_positions_count >= self.max_positions:
            return False, f"Maximum symbol positions reached ({symbol_positions_count}/{self.max_positions})"
        if not self.check_margin_level(account=account):
            return False, f"Margin level below required {self.min_margin_level_percent}%"
        if not self.check_daily_drawdown(account=account):
            return False, f"Daily drawdown limit of {self.max_daily_dd_pct*100}% exceeded"
        return True, "Risk criteria satisfied"

    def validate_order(
        self,
        equity: Optional[float] = None,
        volume: float = 0.01,
        gbpusd_rate: Optional[float] = None,
        spread_pips: Optional[float] = None,
        now: Optional[Union[date, datetime]] = None,
        current_date: Optional[Union[date, datetime]] = None,
        **kwargs
    ) -> Tuple[bool, str]:
        dt = current_date or now
        if not self.can_trade(current_date=dt):
            return False, "Order rejected: circuit breaker is active"

        if spread_pips is not None and not self.is_spread_acceptable(spread_pips):
            return False, f"Order rejected: spread {spread_pips:.1f} pips exceeds limit {self.max_spread_pips:.1f} pips"

        rate = gbpusd_rate if gbpusd_rate is not None else self.gbpusd_rate

        if equity is not None:
            req_margin = self.calculate_required_margin(volume=volume, gbpusd_rate=rate)
            if equity < req_margin:
                return False, f"Order rejected: equity {equity:.2f} insufficient for margin {req_margin:.2f}"

            effective_leverage = (volume * self.contract_size * rate) / equity
            if effective_leverage > self.leverage:
                return False, f"Order rejected: excessive leverage {effective_leverage:.1f}:1 exceeds maximum 1:{int(self.leverage)}"

            margin_level = (equity / req_margin) * 100.0
            if margin_level < self.min_margin_level_percent:
                return False, f"Order rejected: margin level {margin_level:.1f}% below minimum {self.min_margin_level_percent}%"

        return True, "Order valid"

    def validate_signal(
        self,
        signal: Any,
        now: Optional[Union[date, datetime]] = None,
        current_date: Optional[Union[date, datetime]] = None,
    ) -> Tuple[bool, str]:
        dt = current_date or now
        if dt is None and signal is not None:
            dt = getattr(signal, "timestamp", signal.get("timestamp", None) if isinstance(signal, dict) else None)
        if not self.can_trade(current_date=dt):
            return False, "Order blocked: circuit breaker is active"
        return True, "Signal approved"

    def validate_order_entry(
        self,
        account: Optional[Union[AccountInfo, Any]] = None,
        volume: float = 0.01,
        entry_price: float = 0.0,
        now: Optional[Union[date, datetime]] = None,
        current_date: Optional[Union[date, datetime]] = None,
        **kwargs
    ) -> Tuple[bool, str]:
        """Validate order entry against account margin, leverage, and circuit breaker gates."""
        dt = current_date or now
        if not self.can_trade(current_date=dt):
            return False, "Order rejected: circuit breaker is active or daily loss limit reached"

        eq = getattr(account, "equity", None) if account is not None else None
        if eq is None and account is not None:
            eq = getattr(account, "balance", None)
        if eq is None:
            eq = self.initial_balance

        return self.validate_order(
            equity=eq,
            volume=volume,
            now=dt,
            current_date=dt,
            **kwargs
        )


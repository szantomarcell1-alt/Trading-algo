"""
Central Portfolio Risk Manager Module
Implements institutional cross-asset risk management for a shared $250 USD capital account
at 1:30 leverage across EURUSD, XAUUSD ($4.50 Gold Cap), and GBPJPY.

Guarantees:
1. Dynamic margin level >= 200.0% (Zero-Margin Deadlock Guard).
2. Maximum 2 open positions portfolio-wide; maximum 1 open position per symbol.
3. Pre-trade risk budgeting enforcing combined daily drawdown <= 3.0% ($7.50 loss ceiling).
4. Portfolio institutional maximum drawdown strictly in [3.0%, 6.0%) with 5.5% Master Kill-Switch.
5. Shared 3-strike circuit breaker with 60-minute candle-timestamped cooldown across all pairs.
6. Gold micro-lot $4.50 risk ceiling for 0.01 lot XAUUSD trades.
7. Multi-signal conflict and priority resolution.
"""

from datetime import datetime, date, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import math

from portfolio.portfolio_config import PortfolioConfig, SymbolRiskProfile, load_portfolio_config

logger = logging.getLogger("Portfolio.RiskManager")


class UnifiedPortfolioRiskManager:
    """
    Central Risk Guardrail for Multi-Asset Trading ($250 Capital, 1:30 Leverage).
    Coordinates risk across EURUSD, XAUUSD, and GBPJPY.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config: PortfolioConfig = config if config is not None else load_portfolio_config()

        # Balance and Equity Tracking
        self.initial_balance: float = float(self.config.initial_balance)
        self.balance: float = self.initial_balance
        self.equity: float = self.initial_balance
        self.all_time_peak_equity: float = self.initial_balance

        # Daily Session Tracking
        self.current_day: Optional[date] = None
        self.daily_start_equity: float = self.initial_balance
        self.daily_peak_equity: float = self.initial_balance
        self.daily_realized_pnl: float = 0.0
        self.daily_circuit_breaker_tripped: bool = False

        # Loss Streak & Cooldown Tracking
        self.consecutive_losses: int = 0
        self.cooldown_until: Optional[datetime] = None

        # Master Kill-Switch
        self.master_kill_tripped: bool = False

        # Active Position Registry
        # ticket -> position_dict: {"symbol": str, "volume": float, "margin": float, ...}
        self.open_positions: Dict[int, Dict[str, Any]] = {}
        self._next_ticket: int = 1000

        # Market prices cache for margin estimation
        self._latest_prices: Dict[str, float] = {
            "EURUSD": 1.0850,
            "XAUUSD": 2650.0,
            "GBPJPY": 195.00,
        }
        self._latest_gbpusd: float = 1.2800

    # -------------------------------------------------------------------------
    # Margin Calculations
    # -------------------------------------------------------------------------

    @property
    def circuit_breaker_active(self) -> bool:
        """Returns True if either daily drawdown circuit breaker or loss streak cooldown is active."""
        if self.daily_circuit_breaker_tripped:
            return True
        if self.cooldown_until is not None:
            now_utc = datetime.now(timezone.utc)
            cd = self.cooldown_until
            if cd.tzinfo is None:
                cd = cd.replace(tzinfo=timezone.utc)
            return now_utc < cd
        return False

    def update_market_price(self, symbol: str, price: float, gbpusd_rate: Optional[float] = None) -> None:
        """Caches latest asset price for dynamic margin and valuation."""
        sym = symbol.upper().strip()
        self._latest_prices[sym] = float(price)
        if gbpusd_rate is not None and gbpusd_rate > 0:
            self._latest_gbpusd = float(gbpusd_rate)

    def calculate_required_margin(
        self,
        symbol: str,
        volume: float,
        price: Optional[float] = None,
        leverage: Optional[float] = None,
    ) -> float:
        """
        Calculates exact required margin in account currency (USD).
        - EURUSD: (volume * 100,000 * EURUSD_price) / leverage
        - GBPJPY: (volume * 100,000 * GBPUSD_price) / leverage
        - XAUUSD: (volume * 100 * Gold_price) / leverage
        """
        sym = symbol.upper().strip()
        lev = float(leverage if leverage is not None else self.config.leverage)
        if lev <= 0:
            lev = 30.0

        profile = self.config.get_symbol_profile(sym)
        contract_size = profile.contract_size if profile else 100000.0

        if price is None or price <= 0:
            price = self._latest_prices.get(sym, 1.0)

        if sym == "EURUSD":
            # EUR base currency -> notional in USD = volume * 100k * price
            notional_usd = volume * contract_size * price
            return round(notional_usd / lev, 4)

        elif sym == "GBPJPY":
            # GBP base currency -> notional in USD = volume * 100k * GBPUSD_rate
            gbpusd = self._latest_gbpusd
            notional_usd = volume * contract_size * gbpusd
            return round(notional_usd / lev, 4)

        elif sym == "XAUUSD":
            # Gold CFD -> 100 oz per lot. 0.01 lot = 1.0 oz.
            notional_usd = volume * contract_size * price
            return round(notional_usd / lev, 4)

        else:
            notional_usd = volume * contract_size * price
            return round(notional_usd / lev, 4)

    def get_used_margin(self) -> float:
        """Sums committed margin across all currently open positions."""
        return sum(pos.get("margin", 0.0) for pos in self.open_positions.values())

    def get_free_margin(self, current_equity: Optional[float] = None) -> float:
        """Calculates free margin available in USD."""
        eq = current_equity if current_equity is not None else self.equity
        return max(0.0, eq - self.get_used_margin())

    def calculate_projected_margin_level(
        self,
        new_margin: float,
        current_equity: Optional[float] = None,
    ) -> float:
        """
        Calculates projected margin level if new position is opened:
        Margin Level = (Equity / (Current Used Margin + New Margin)) * 100%
        """
        eq = current_equity if current_equity is not None else self.equity
        total_margin = self.get_used_margin() + new_margin
        if total_margin <= 0.0:
            return 999999.0
        return (eq / total_margin) * 100.0

    # -------------------------------------------------------------------------
    # Calendar & Daily Tracking
    # -------------------------------------------------------------------------

    def _ensure_daily_reset(self, candle_timestamp: datetime, current_equity: Optional[float] = None) -> None:
        """Resets daily drawdown metrics upon advancing to a new UTC calendar day."""
        dt = candle_timestamp.date() if isinstance(candle_timestamp, datetime) else candle_timestamp
        eq = current_equity if current_equity is not None else self.equity

        if self.current_day is None or dt != self.current_day:
            self.current_day = dt
            self.daily_start_equity = eq
            self.daily_peak_equity = eq
            self.daily_realized_pnl = 0.0
            self.daily_circuit_breaker_tripped = False
            logger.info(
                f"[Portfolio Daily Reset] New UTC Day {dt} initialized. "
                f"Starting Equity: ${eq:.2f} USD"
            )
        else:
            self.daily_peak_equity = max(self.daily_peak_equity, eq)

    # -------------------------------------------------------------------------
    # Trade Approval & Pre-Trade Risk Budgeting
    # -------------------------------------------------------------------------

    def can_open_trade(
        self,
        symbol: str,
        signal: Any,
        current_price: float,
        candle_timestamp: datetime,
        current_equity: Optional[float] = None,
        spread_pips: Optional[float] = None,
    ) -> Tuple[bool, str, float]:
        """
        Master Risk Validation Gate.
        Evaluates incoming trade signal against all institutional invariants.

        Returns:
            (approved: bool, reason: str, approved_volume: float)
        """
        sym = symbol.upper().strip()
        profile = self.config.get_symbol_profile(sym)
        if profile is None or not profile.enabled:
            return False, f"Symbol {sym} is not enabled in portfolio configuration", 0.0

        # Synchronize candle time and latest price
        if candle_timestamp.tzinfo is None:
            candle_timestamp = candle_timestamp.replace(tzinfo=timezone.utc)
        self.update_market_price(sym, current_price)

        eff_equity = current_equity if current_equity is not None else self.equity
        self._ensure_daily_reset(candle_timestamp, eff_equity)

        # 1. Master Portfolio Kill-Switch (< 6.0% Total Drawdown, trigger at 5.5%)
        self.all_time_peak_equity = max(self.all_time_peak_equity, eff_equity)
        if self.all_time_peak_equity > 0:
            total_dd_pct = ((self.all_time_peak_equity - eff_equity) / self.all_time_peak_equity) * 100.0
            if total_dd_pct >= self.config.max_total_drawdown_percent or self.master_kill_tripped:
                self.master_kill_tripped = True
                return (
                    False,
                    f"Master Portfolio Kill-Switch Tripped: Drawdown {total_dd_pct:.2f}% >= "
                    f"{self.config.max_total_drawdown_percent}%. All trading permanently locked.",
                    0.0,
                )

        # 2. Daily Max Drawdown Circuit Breaker (<= 3.0%)
        if self.daily_start_equity > 0:
            dd_start = ((self.daily_start_equity - eff_equity) / self.daily_start_equity) * 100.0
            dd_peak = (
                ((self.daily_peak_equity - eff_equity) / self.daily_peak_equity) * 100.0
                if self.daily_peak_equity > 0
                else 0.0
            )
            current_daily_dd = max(0.0, max(dd_start, dd_peak))
            if current_daily_dd >= self.config.max_daily_drawdown_percent or self.daily_circuit_breaker_tripped:
                self.daily_circuit_breaker_tripped = True
                return (
                    False,
                    f"Daily Drawdown Limit Reached: {current_daily_dd:.2f}% >= "
                    f"{self.config.max_daily_drawdown_percent}%. Trading halted for today.",
                    0.0,
                )

        # 3. Shared 3-Strike Loss Streak Circuit Breaker (60-minute candle timestamp cooldown)
        if self.cooldown_until is not None:
            if candle_timestamp < self.cooldown_until:
                remaining_secs = (self.cooldown_until - candle_timestamp).total_seconds()
                remaining_mins = max(1, int(remaining_secs / 60))
                return (
                    False,
                    f"Shared 3-strike circuit breaker active: {self.consecutive_losses} consecutive losses. "
                    f"Cooldown active for {remaining_mins} more minutes (until {self.cooldown_until.isoformat()}).",
                    0.0,
                )
            else:
                logger.info(
                    f"Shared circuit breaker cooldown elapsed at {candle_timestamp.isoformat()}. "
                    f"Resetting consecutive losses."
                )
                self.cooldown_until = None
                self.consecutive_losses = 0

        # 4. Concurrency Guard: Portfolio-Wide Maximum 2 Open Positions
        if len(self.open_positions) >= self.config.max_open_positions:
            return (
                False,
                f"Portfolio concurrency cap reached: {len(self.open_positions)}/"
                f"{self.config.max_open_positions} positions open.",
                0.0,
            )

        # 5. Concurrency Guard: Symbol-Specific Maximum 1 Open Position
        symbol_positions = sum(1 for p in self.open_positions.values() if p["symbol"] == sym)
        if symbol_positions >= self.config.max_open_positions_per_symbol:
            return (
                False,
                f"Symbol {sym} already has an active position ({symbol_positions}/"
                f"{self.config.max_open_positions_per_symbol}).",
                0.0,
            )

        # 6. Spread Filter Guard
        if spread_pips is not None and spread_pips > profile.max_spread_pips:
            return (
                False,
                f"Spread {spread_pips:.2f} pips exceeds maximum threshold {profile.max_spread_pips:.2f} pips.",
                0.0,
            )

        # 7. Signal Quality & SL Distance Extraction
        confidence = float(getattr(signal, "confidence", 1.0))
        if confidence < self.config.min_signal_confidence:
            return (
                False,
                f"Signal confidence {confidence:.2f} below threshold {self.config.min_signal_confidence:.2f}.",
                0.0,
            )

        # Extract stop-loss distance
        sl_price = getattr(signal, "sl_price", getattr(signal, "sl", 0.0))
        entry_price = getattr(signal, "entry_price", current_price)
        if entry_price <= 0:
            entry_price = current_price

        if sl_price <= 0:
            return False, "Invalid Stop-Loss price (must be positive).", 0.0

        sl_dist_price = abs(entry_price - sl_price)
        if sl_dist_price <= 0:
            return False, "Zero stop-loss distance.", 0.0

        risk_distance_points = getattr(signal, "risk_distance_points", None)
        if risk_distance_points is None or risk_distance_points <= 0:
            risk_distance_points = sl_dist_price / profile.point

        # 8. Sizing & Risk Cap Calculation
        # Determine base loss for 1 standard lot
        ticks = risk_distance_points * (profile.point / profile.point)
        commission_per_lot = float(self.config.commission_per_lot_round_turn)
        loss_per_lot = (ticks * profile.tick_value) + commission_per_lot

        if loss_per_lot <= 0:
            return False, "Calculated loss per lot is non-positive.", 0.0

        # Risk budget ($2.50 baseline for 1.0% on $250)
        base_eq = max(eff_equity, self.initial_balance)
        risk_budget = base_eq * (self.config.risk_per_trade_percent / 100.0)

        # Micro-Lot Risk Cap Guard ($4.50 max cap on 0.01 lot for Gold)
        min_lot_loss = profile.min_volume * loss_per_lot
        if sym == "XAUUSD":
            max_tolerable_risk = risk_budget * self.config.max_risk_multiplier_for_min_lot
        else:
            max_tolerable_risk = risk_budget * self.config.max_risk_multiplier_for_min_lot

        if min_lot_loss > max_tolerable_risk:
            return (
                False,
                f"[{sym}] KOCKÁZATI VÉSZFÉK: A minimális {profile.min_volume:.2f} lot kockázata "
                f"({min_lot_loss:.2f} USD) meghaladja a megengedett plafont "
                f"({max_tolerable_risk:.2f} USD). Kötés elutasítva!",
                0.0,
            )

        # 9. Pre-Trade Daily Risk Budgeting
        # Ensure existing daily DD + pending trade potential loss <= 3.20% (3.0% + 0.20% buffer)
        if self.daily_start_equity > 0:
            dd_start = ((self.daily_start_equity - eff_equity) / self.daily_start_equity) * 100.0
            dd_peak = (
                ((self.daily_peak_equity - eff_equity) / self.daily_peak_equity) * 100.0
                if self.daily_peak_equity > 0
                else 0.0
            )
            current_daily_dd = max(0.0, max(dd_start, dd_peak))
            potential_trade_dd_pct = (min_lot_loss / self.daily_start_equity) * 100.0
            max_allowed_daily_dd = self.config.max_daily_drawdown_percent + self.config.daily_drawdown_buffer_percent

            if (current_daily_dd + potential_trade_dd_pct) > max_allowed_daily_dd:
                return (
                    False,
                    f"[{sym}] Pre-trade daily risk budget exceeded: Current Daily DD ({current_daily_dd:.2f}%) + "
                    f"Trade Risk ({potential_trade_dd_pct:.2f}%) > Allowed ({max_allowed_daily_dd:.2f}%).",
                    0.0,
                )

        # 10. Margin Requirement & Projected Margin Level (>= 200.0%)
        # Size volume: for small accounts (<= $500), enforce strictly micro-lot (0.01)
        if eff_equity <= 500.0:
            approved_volume = profile.min_volume
        else:
            raw_lot = risk_budget / loss_per_lot
            step = profile.volume_step if profile.volume_step > 0 else 0.01
            stepped_lot = math.floor(raw_lot / step) * step
            approved_volume = max(profile.min_volume, min(profile.max_volume, stepped_lot))
        approved_volume = round(approved_volume, 2)

        # Check required margin
        req_margin = self.calculate_required_margin(sym, approved_volume, entry_price)
        proj_margin_level = self.calculate_projected_margin_level(req_margin, eff_equity)

        if proj_margin_level < self.config.min_margin_level_percent:
            return (
                False,
                f"[{sym}] Low Margin Level Guard: Projected margin level {proj_margin_level:.1f}% < "
                f"{self.config.min_margin_level_percent:.1f}%. (Required Margin: ${req_margin:.2f})",
                0.0,
            )

        # Free margin solvency check
        if req_margin > self.get_free_margin(eff_equity):
            return (
                False,
                f"[{sym}] Insufficient free margin: required ${req_margin:.2f} > free ${self.get_free_margin(eff_equity):.2f}.",
                0.0,
            )

        return True, "Trade approved by Unified Portfolio Risk Manager", approved_volume

    # -------------------------------------------------------------------------
    # Position Lifecycle & State Updates
    # -------------------------------------------------------------------------

    def register_position_opened(
        self,
        symbol: str,
        volume: float,
        entry_price: float,
        entry_time: datetime,
        sl: float,
        tp: float,
        ticket: Optional[int] = None,
    ) -> int:
        """Records a newly opened trade and reserves required margin."""
        sym = symbol.upper().strip()
        t = ticket if ticket is not None else self._next_ticket
        self._next_ticket = max(self._next_ticket + 1, t + 1)

        req_margin = self.calculate_required_margin(sym, volume, entry_price)
        self.open_positions[t] = {
            "ticket": t,
            "symbol": sym,
            "volume": volume,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "sl": sl,
            "tp": tp,
            "margin": req_margin,
        }
        return t

    def register_trade_close(
        self,
        symbol: str,
        realized_pnl: float,
        exit_time: datetime,
        ticket: Optional[int] = None,
    ) -> None:
        """
        Updates portfolio capital, daily realized PnL, all-time peak equity,
        and shared 3-strike circuit breaker status upon trade close.
        """
        sym = symbol.upper().strip()
        if exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)

        self._ensure_daily_reset(exit_time)

        # Remove from open positions if ticket is known
        if ticket is not None and ticket in self.open_positions:
            del self.open_positions[ticket]
        else:
            # Fallback: remove first matching symbol position
            for t, pos in list(self.open_positions.items()):
                if pos["symbol"] == sym:
                    del self.open_positions[t]
                    break

        # Balance and PnL updates
        self.balance += realized_pnl
        self.equity = self.balance
        self.daily_realized_pnl += realized_pnl
        self.all_time_peak_equity = max(self.all_time_peak_equity, self.balance)
        self.daily_peak_equity = max(self.daily_peak_equity, self.balance)

        # Consecutive Loss Tracking & Circuit Breaker
        if realized_pnl < 0:
            self.consecutive_losses += 1
            logger.warning(
                f"[{sym}] Loss registered: ${realized_pnl:.2f}. "
                f"Consecutive portfolio losses: {self.consecutive_losses}"
            )
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                cooldown_dur = timedelta(minutes=self.config.consecutive_loss_cooldown_minutes)
                self.cooldown_until = exit_time + cooldown_dur
                logger.error(
                    f"[SHARED CIRCUIT BREAKER TRIPPED] {self.consecutive_losses} consecutive losses. "
                    f"Portfolio-wide trading halted for {self.config.consecutive_loss_cooldown_minutes} minutes "
                    f"until {self.cooldown_until.isoformat()}."
                )
        else:
            if self.consecutive_losses > 0:
                logger.info(
                    f"[{sym}] Win registered: +${realized_pnl:.2f}. "
                    f"Consecutive loss streak ({self.consecutive_losses}) broken."
                )
            self.consecutive_losses = 0

    # -------------------------------------------------------------------------
    # Multi-Signal Conflict Resolution
    # -------------------------------------------------------------------------

    def resolve_conflicts(
        self,
        candidate_signals: List[Dict[str, Any]],
        candle_timestamp: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Resolves collisions when multiple symbols trigger simultaneous trade signals on the same bar.
        Ranks signals by:
        1. Confidence score (descending)
        2. Expected Risk:Reward ratio (descending)
        3. Lowest required margin (ascending)

        Greedily approves signals until position capacity or margin levels are reached.
        """
        if not candidate_signals:
            return []

        def sort_key(item: Dict[str, Any]) -> Tuple[float, float, float]:
            sig = item.get("signal")
            sym = item.get("symbol", "")
            conf = float(getattr(sig, "confidence", 1.0))
            entry = float(getattr(sig, "entry_price", item.get("price", 1.0)))
            sl = float(getattr(sig, "sl_price", getattr(sig, "sl", 1.0)))
            tp = float(getattr(sig, "tp_price", getattr(sig, "tp", 1.0)))
            sl_dist = abs(entry - sl)
            tp_dist = abs(tp - entry)
            rr = (tp_dist / sl_dist) if sl_dist > 0 else 1.0
            margin = self.calculate_required_margin(sym, 0.01, entry)
            # higher conf, higher RR, lower margin
            return (-conf, -rr, margin)

        sorted_candidates = sorted(candidate_signals, key=sort_key)
        approved_list: List[Dict[str, Any]] = []

        # Available slots
        available_slots = self.config.max_open_positions - len(self.open_positions)
        if available_slots <= 0:
            return []

        # Temporary simulation of newly accepted positions for margin stacking
        sim_used_margin = self.get_used_margin()
        sim_symbols = {p["symbol"] for p in self.open_positions.values()}

        for cand in sorted_candidates:
            if len(approved_list) >= available_slots:
                break

            sym = cand.get("symbol", "").upper().strip()
            if sym in sim_symbols:
                continue

            sig = cand.get("signal")
            price = float(cand.get("price", getattr(sig, "entry_price", 0.0)))
            spread = cand.get("spread_pips")

            ok, reason, vol = self.can_open_trade(
                symbol=sym,
                signal=sig,
                current_price=price,
                candle_timestamp=candle_timestamp,
                spread_pips=spread,
            )

            if ok and vol > 0:
                cand_margin = self.calculate_required_margin(sym, vol, price)
                # Verify combined projected margin level
                if (sim_used_margin + cand_margin) > 0:
                    proj_ml = (self.equity / (sim_used_margin + cand_margin)) * 100.0
                    if proj_ml < self.config.min_margin_level_percent:
                        continue

                sim_used_margin += cand_margin
                sim_symbols.add(sym)
                cand_copy = dict(cand)
                cand_copy["approved_volume"] = vol
                cand_copy["required_margin"] = cand_margin
                approved_list.append(cand_copy)

        return approved_list

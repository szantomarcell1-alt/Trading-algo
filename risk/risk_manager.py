"""
Risk & Money Management Module
Handles precise fractional Kelly / ATR position sizing for small capital,
daily drawdown circuit breakers, margin guards, and consecutive loss cooldowns.
"""

from datetime import datetime, timezone, date
from typing import Optional, Tuple
import math
import logging

from config.settings import AppConfig
from core.mt5_client import SymbolSpecs, AccountSnapshot

logger = logging.getLogger("TitanScalper.RiskManager")


class RiskManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.risk_cfg = config.risk_management

        # Daily tracking
        self.current_day: Optional[date] = None
        self.daily_start_equity: float = 0.0
        self.daily_peak_equity: float = 0.0
        self.daily_circuit_breaker_tripped: bool = False

        # Loss streak tracking
        self.consecutive_losses: int = 0
        self.cooldown_until: Optional[datetime] = None

        # Lifetime portfolio tracking (Institutional Hard DD Ceiling)
        self.all_time_peak_equity: float = float(getattr(self.risk_cfg, "initial_balance", 250.0))

    def reset_daily_metrics_if_new_day(self, current_equity: float, now: datetime = None):
        if current_equity is None or not math.isfinite(current_equity) or current_equity <= 0.0:
            return

        if now is None:
            now = datetime.now(timezone.utc)

        today = now.date()
        if self.current_day is None or self.current_day != today:
            self.current_day = today
            self.daily_start_equity = current_equity
            self.daily_peak_equity = current_equity
            self.daily_circuit_breaker_tripped = False
            logger.info(
                f"[bold magenta]Új kereskedési nap indul ({today})[/bold magenta] | "
                f"Kezdő tőke: {current_equity:.2f} {self.risk_cfg.account_currency}"
            )
        else:
            self.daily_peak_equity = max(self.daily_peak_equity, current_equity)

    def register_trade_result(self, profit: float, now: Optional[datetime] = None):
        """
        Updates streak stats when a trade closes.
        """
        if profit < 0:
            self.consecutive_losses += 1
            logger.warning(f"Veszteséges trade regisztrálva! Sorozatban: {self.consecutive_losses} veszteség.")
            if self.consecutive_losses >= self.risk_cfg.max_consecutive_losses:
                cooldown_secs = self.risk_cfg.consecutive_loss_cooldown_minutes * 60
                ref_time = now if now is not None else datetime.now(timezone.utc)
                ref_ts = ref_time.timestamp() if hasattr(ref_time, "timestamp") else float(ref_time)
                self.cooldown_until = ref_ts + cooldown_secs
                logger.error(
                    f"[bold red]CIRCUIT BREAKER AKTIVÁLVA![/bold red] {self.consecutive_losses} egymást követő veszteség. "
                    f"Pihenő idő {self.risk_cfg.consecutive_loss_cooldown_minutes} percig."
                )
        else:
            if self.consecutive_losses > 0:
                logger.info(f"Nyertes trade! Veszteség-sorozat megszakadt (előző: {self.consecutive_losses}).")
            self.consecutive_losses = 0

    def can_open_new_trade(
        self,
        account: AccountSnapshot,
        open_positions_count: int,
        symbol_positions_count: int,
        now: Optional[datetime] = None,
        pending_risk_amount: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Evaluates strict risk criteria before approving any new order.
        """
        # 0. Sanity check: equity validity
        if account.equity is None or not math.isfinite(account.equity) or account.equity <= 0.0:
            return (
                False,
                f"Érvénytelen vagy nem pozitív számla tőke (Equity: {account.equity})! Kereskedés tiltva a tőke védelmében.",
            )

        if now is None:
            now = datetime.now(timezone.utc)

        # 1. Update daily baseline
        self.reset_daily_metrics_if_new_day(account.equity, now)

        # 2. Check Lifetime Portfolio Drawdown Ceiling (Institutional Hard Cap)
        self.all_time_peak_equity = max(self.all_time_peak_equity, account.equity)
        if self.all_time_peak_equity > 0:
            total_dd_pct = ((self.all_time_peak_equity - account.equity) / self.all_time_peak_equity) * 100.0
            max_total_dd = getattr(self.risk_cfg, "max_total_drawdown_percent", 8.5)
            if total_dd_pct >= max_total_dd:
                return (
                    False,
                    f"Intézményi Max Drawdown elérve ({total_dd_pct:.2f}% >= {max_total_dd}%). "
                    f"Kereskedés leállítva a tőke védelmében!",
                )

        # 3. Check Daily Max Drawdown Circuit Breaker & Pre-Trade Risk Budgeting
        if self.daily_start_equity > 0:
            self.daily_peak_equity = max(self.daily_peak_equity, account.equity)
            dd_start = ((self.daily_start_equity - account.equity) / self.daily_start_equity) * 100.0
            dd_peak = (
                ((self.daily_peak_equity - account.equity) / self.daily_peak_equity) * 100.0
                if self.daily_peak_equity > 0
                else 0.0
            )
            current_drawdown_pct = max(dd_start, dd_peak)

            if current_drawdown_pct >= self.risk_cfg.max_daily_drawdown_percent or self.daily_circuit_breaker_tripped:
                self.daily_circuit_breaker_tripped = True
                return (
                    False,
                    f"Napi Max Drawdown elérve ({current_drawdown_pct:.2f}% >= {self.risk_cfg.max_daily_drawdown_percent}%). "
                    f"Kereskedés mára leállítva a tőke védelmében!",
                )

            # Pre-Trade Risk Budgeting check
            if pending_risk_amount > 0.0:
                potential_risk_pct = (pending_risk_amount / self.daily_start_equity) * 100.0
                max_allowed_daily_dd = self.risk_cfg.max_daily_drawdown_percent + 0.20
                if (current_drawdown_pct + potential_risk_pct) > max_allowed_daily_dd:
                    return (
                        False,
                        f"Kockázati költségvetés túllépve: Jelenlegi napi DD ({current_drawdown_pct:.2f}%) + "
                        f"függő kockázat ({potential_risk_pct:.2f}%) > {max_allowed_daily_dd:.2f}%. "
                        f"Új kötés elutasítva!",
                    )

        # 4. Check Consecutive Loss Cooldown
        if self.cooldown_until:
            current_ts = now.timestamp() if hasattr(now, "timestamp") else float(now)
            if current_ts < self.cooldown_until:
                remaining_mins = int((self.cooldown_until - current_ts) / 60)
                return False, f"Veszteség-sorozat miatti kényszerpihenő aktív: még {remaining_mins} perc."
            else:
                self.cooldown_until = None
                self.consecutive_losses = 0
                logger.info("Kényszerpihenő lejárt. Kereskedés újra engedélyezve.")

        # 5. Check Margin Level (Zero-Margin Deadlock Guard)
        if account.margin > 0.0 and account.margin_level < self.risk_cfg.min_margin_level_percent:
            return (
                False,
                f"Alacsony Margin Level ({account.margin_level:.1f}% < {self.risk_cfg.min_margin_level_percent}%). "
                f"Új pozíció nem nyitható!",
            )

        # 6. Check Max Open Positions Limit
        if open_positions_count >= self.risk_cfg.max_open_positions:
            return (
                False,
                f"Elérte a maximális egyidejű pozíciószámot ({open_positions_count}/{self.risk_cfg.max_open_positions}).",
            )

        # 7. Check Max Open Positions Per Symbol
        if symbol_positions_count >= self.risk_cfg.max_open_positions_per_symbol:
            return (
                False,
                f"A szimbólumon már fut egy nyitott pozíció ({symbol_positions_count}/{self.risk_cfg.max_open_positions_per_symbol}).",
            )

        return True, "Kockázati feltételek teljesültek (Jóváhagyva)"

    def calculate_lot_size(
        self,
        account: AccountSnapshot,
        specs: SymbolSpecs,
        risk_distance_points: float,
    ) -> float:
        """
        Calculates exact lot size including ECN commission and enforces a Small-Capital Hard Risk Cap.
        Returns 0.0 if even the minimum lot (0.01) exceeds the maximum allowed risk threshold.
        """
        if account.equity is None or not math.isfinite(account.equity) or account.equity <= 0.0:
            return 0.0

        if risk_distance_points <= 0:
            return 0.0

        # Amount to risk in account currency
        base_equity = max(account.equity, getattr(self.risk_cfg, "initial_balance", 250.0))
        risk_amount = base_equity * (self.risk_cfg.risk_per_trade_percent / 100.0)

        # Monetary risk of 1.0 standard lot including round-turn ECN commission
        ticks = risk_distance_points * (specs.point / specs.tick_size)
        commission_per_lot = getattr(self.risk_cfg, "commission_per_lot_round_turn", 7.0)
        loss_per_lot = (ticks * specs.tick_value) + commission_per_lot

        if loss_per_lot <= 0:
            return 0.0

        # Small-Capital Hard Risk Guard: check if min_volume (0.01 lot) violates risk ceiling
        min_lot_loss = specs.min_volume * loss_per_lot
        multiplier = getattr(self.risk_cfg, "max_risk_multiplier_for_min_lot", 1.80)
        max_tolerable_risk = risk_amount * multiplier

        if min_lot_loss > max_tolerable_risk:
            logger.warning(
                f"[{specs.name}] KOCKÁZATI VÉSZFÉK: A minimális {specs.min_volume:.2f} lot kockázata "
                f"({min_lot_loss:.2f} {account.currency}) meghaladja a megengedett plafont "
                f"({max_tolerable_risk:.2f} {account.currency}). Kötés elutasítva!"
            )
            return 0.0

        # Pre-Trade Daily Risk Budgeting: check if existing daily drawdown + pending trade risk exceeds daily limit + 0.20%
        if self.daily_start_equity > 0:
            dd_start = ((self.daily_start_equity - account.equity) / self.daily_start_equity) * 100.0
            dd_peak = (
                ((self.daily_peak_equity - account.equity) / self.daily_peak_equity) * 100.0
                if self.daily_peak_equity > 0
                else 0.0
            )
            current_drawdown_pct = max(0.0, max(dd_start, dd_peak))
            potential_trade_dd_pct = (min_lot_loss / self.daily_start_equity) * 100.0
            max_allowed_daily_dd = self.risk_cfg.max_daily_drawdown_percent + 0.20
            if (current_drawdown_pct + potential_trade_dd_pct) > max_allowed_daily_dd:
                logger.warning(
                    f"[{specs.name}] KOCKÁZATI PLAFON TÚLLÉPÉS: Jelenlegi napi DD ({current_drawdown_pct:.2f}%) + "
                    f"kötési kockázat ({potential_trade_dd_pct:.2f}%) > "
                    f"engedélyezett ({max_allowed_daily_dd:.2f}%). Kötés elutasítva!"
                )
                return 0.0

        raw_lot = risk_amount / loss_per_lot

        # Step quantization
        step = specs.volume_step if specs.volume_step > 0 else 0.01
        stepped_lot = math.floor(raw_lot / step) * step

        # Bounds clamp
        bounded_lot = max(specs.min_volume, min(specs.max_volume, stepped_lot))
        return round(bounded_lot, 2)

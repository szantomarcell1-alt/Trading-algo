"""
Unified Portfolio Live & Paper Trading Engine
Orchestrates real-time multi-symbol execution across EURUSD, XAUUSD, and GBPJPY on a shared $250 capital base.

Guarantees 100% isolation between Paper Trading (read-only MT5 market data, zero broker orders)
and Live Trading (real broker execution, requiring explicit confirmation).
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from portfolio.portfolio_config import PortfolioConfig, load_portfolio_config, SymbolRiskProfile
from portfolio.portfolio_risk_manager import UnifiedPortfolioRiskManager
from portfolio.portfolio_dispatcher import PortfolioDispatcher
from core.mt5_client import MT5Client, SymbolSpecs, AccountSnapshot, OpenPosition
from core.mock_client import MockMT5Client
from core.data_engine import DataEngine
from utils.market_hours import MarketHoursGuard

logger = logging.getLogger("Portfolio.LiveEngine")
console = Console()


class TradingMode(str, Enum):
    PAPER = "paper"
    DRY_RUN = "dry_run"
    LIVE = "live"


class PaperTradeSafetyGuard:
    """
    Impenetrable safety barrier ensuring that paper-trading cannot execute live orders
    on MetaTrader 5 under any circumstances.
    """
    @staticmethod
    def verify_safety(client: Any, mode: TradingMode) -> None:
        if mode in (TradingMode.PAPER, TradingMode.DRY_RUN):
            if not isinstance(client, MockMT5Client):
                raise RuntimeError(
                    "BIZTONSÁGI VÉSZLEÁLLÍTÁS: Paper trading módban szigorúan tilos éles MT5Client használata! "
                    f"Észlelt kliens típus: {type(client).__name__}"
                )
            if getattr(client, "is_live_execution_allowed", False):
                raise RuntimeError(
                    "BIZTONSÁGI VÉSZLEÁLLÍTÁS: A kliens live_execution_allowed flagje aktív paper módban!"
                )
            if not getattr(client, "is_paper_trading", False):
                raise RuntimeError(
                    "BIZTONSÁGI VÉSZLEÁLLÍTÁS: A kliens is_paper_trading flagje nem aktív paper módban!"
                )


@dataclass
class ManagedPositionMetadata:
    ticket: int
    symbol: str
    is_buy: bool
    volume: float
    entry_price: float
    initial_sl: float
    initial_tp: float
    risk_dist_1r: float
    breakeven_applied: bool = False
    trailing_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = 0.0
    entry_time: float = 0.0


class PortfolioLiveEngine:
    """
    Unified Multi-Symbol Live & Paper-Trading Engine.
    Executes real-time market data polling, strategy dispatch, centralized risk checks,
    and dynamic position management across EURUSD, XAUUSD, and GBPJPY.
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        mode: TradingMode = TradingMode.PAPER,
        live_confirm_token: str = "",
        heartbeat_seconds: float = 1.0,
        enable_rich_ui: bool = True,
    ):
        self.config: PortfolioConfig = config if config is not None else load_portfolio_config()
        self.mode: TradingMode = mode
        self.heartbeat_seconds = heartbeat_seconds
        self.enable_rich_ui = enable_rich_ui
        self.is_running: bool = False

        # Activity log buffer for UI
        self.event_log: List[Tuple[str, str]] = []

        # 1. Initialize Client with Strict Safety Enforcer
        if self.mode == TradingMode.LIVE:
            required_token = "YES_I_CONFIRM_LIVE_TRADING_WITH_REAL_MONEY"
            if live_confirm_token != required_token and live_confirm_token != "YES":
                self._log_event(
                    "HIBA",
                    f"Éles kereskedés megtagadva: Hiányzó vagy hibás megerősítő token! Visszaállás PAPER módra.",
                )
                self.mode = TradingMode.PAPER

        if self.mode == TradingMode.LIVE:
            self._log_event(
                "FIGYELEM",
                "[bold red]ÉLES KERESKEDÉSI MÓD AKTÍV! Valós megbízások kerülnek kiküldésre az MT5-be![/bold red]",
            )
            self.client = MT5Client(
                terminal_path="",
                login=0,
                password="",
                server="",
                timeout_ms=5000,
                magic_number=777001,
                slippage_points=10,
            )
        else:
            self._log_event(
                "BIZTONSÁG",
                "[bold cyan]PAPER TRADING AKTÍV! MT5 adatfolyam olvasás, NULLA megbízás a brókerhez.[/bold cyan]",
            )
            self.client = MockMT5Client(
                initial_balance=self.config.initial_balance,
                currency=self.config.account_currency,
                magic_number=777001,
                slippage_points=10,
                commission_per_lot=self.config.commission_per_lot_round_turn,
            )

        # Enforce safety check
        PaperTradeSafetyGuard.verify_safety(self.client, self.mode)

        # 2. Risk Manager & Strategy Dispatcher
        self.risk_manager = UnifiedPortfolioRiskManager(self.config)
        self.dispatcher = PortfolioDispatcher(config=self.config, risk_manager=self.risk_manager)
        self.data_engine = DataEngine(lookback_bars=self.config.lookback_bars)

        # 3. Market Hours Guard
        self.market_hours_guard = MarketHoursGuard(
            allowed_start_utc_hour=0,
            allowed_end_utc_hour=24,
            avoid_rollover_window=True,
            avoid_friday_close=True,
        )

        # 4. Position Tracking & Metadata
        self.tracked_positions: Dict[int, ManagedPositionMetadata] = {}
        self.active_symbols: List[str] = [
            sym for sym, prof in self.config.symbols.items() if prof.enabled
        ]
        self.last_candle_times: Dict[str, Optional[datetime]] = {sym: None for sym in self.active_symbols}
        self.current_quotes: Dict[str, Tuple[float, float, float, float]] = {}

    def _log_event(self, category: str, message: str) -> None:
        timestamp_str = datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")
        self.event_log.append((timestamp_str, f"[{category}] {message}"))
        if len(self.event_log) > 100:
            self.event_log.pop(0)
        logger.info(f"[{category}] {message}")

    def start(self, max_cycles: int = 0) -> None:
        """
        Connects to MetaTrader 5 and starts the live portfolio execution loop.
        """
        PaperTradeSafetyGuard.verify_safety(self.client, self.mode)

        self._log_event("START", f"Kapcsolódás a piachoz (Mód: {self.mode.value.upper()})...")
        if not self.client.connect():
            self._log_event("HIBA", "Nem sikerült a kapcsolat a MetaTrader 5 terminállal!")
            return

        for sym in self.active_symbols:
            self.client.ensure_symbol(sym)

        self.is_running = True
        self._log_event("START", f"Portfólió motor elindult. Aktív szimbólumok: {', '.join(self.active_symbols)}")

        try:
            self._run_loop(max_cycles=max_cycles)
        except KeyboardInterrupt:
            self._log_event("STOP", "Felhasználói megszakítás (Ctrl+C). Leállás...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully disconnects client and releases resources."""
        self.is_running = False
        try:
            self.client.disconnect()
        except Exception:
            pass
        self._log_event("STOP", "Portfólió motor leállt.")

    def _run_loop(self, max_cycles: int = 0) -> None:
        cycles = 0
        while self.is_running:
            cycles += 1
            loop_start = time.time()
            now_utc = datetime.now(timezone.utc)

            try:
                self.step(now_utc)
            except Exception as e:
                self._log_event("EXCEPT", f"Ciklus hiba: {e}")

            if max_cycles > 0 and cycles >= max_cycles:
                self._log_event("STOP", f"Elérte a maximális ciklusszámot ({max_cycles}).")
                break

            elapsed = time.time() - loop_start
            sleep_time = max(0.05, self.heartbeat_seconds - elapsed)
            time.sleep(sleep_time)

    def step(self, now_utc: datetime) -> None:
        """
        Single atomic execution step:
        1. Fetch quotes & check market session
        2. Synchronize and manage active positions (SL/TP, Breakeven, Trailing)
        3. Evaluate new candle signals
        4. Central risk assessment & execution
        5. Render status dashboard
        """
        PaperTradeSafetyGuard.verify_safety(self.client, self.mode)

        # 1. Update live quotes
        gbpusd_quote = self.client.get_quote("GBPUSD")
        gbpusd_rate = gbpusd_quote[0] if gbpusd_quote else 1.28

        for sym in self.active_symbols:
            q = self.client.get_quote(sym)
            if q:
                self.current_quotes[sym] = q
                self.risk_manager.update_market_price(sym, q[0], gbpusd_rate=gbpusd_rate)

        # 2. Account Snapshot
        account = self.client.get_account_snapshot()
        if not account:
            return

        # 3. Position Management & Lifecycle
        self._manage_open_positions(now_utc, account)

        # 4. Check Market Session (Rollover / Friday Close)
        trading_allowed, session_reason = self.market_hours_guard.is_trading_allowed(now_utc)
        curr_session_name = self.market_hours_guard.get_current_session_name(now_utc)

        # Track session state transitions to prevent flooding the event log with identical messages
        if not hasattr(self, "_last_session_state"):
            self._last_session_state = None

        curr_state_key = (curr_session_name, trading_allowed)
        if curr_state_key != self._last_session_state:
            self._last_session_state = curr_state_key
            if trading_allowed:
                self._log_event("SESSION", f"{curr_session_name} aktív | Kereskedés ENGEDÉLYEZVE, jelek folyamatos pásztázása...")
            else:
                self._log_event("SESSION", f"Új kötések tiltva: {session_reason}")

        # 5. Evaluate Candidate Signals
        if trading_allowed:
            self._evaluate_signals(now_utc, account)

        # 6. Render Dashboard
        if self.enable_rich_ui:
            self._render_dashboard(now_utc, account, trading_allowed, session_reason)

    def _manage_open_positions(self, now_utc: datetime, account: AccountSnapshot) -> None:
        """
        Monitors open positions, executes dynamic Breakeven and Trailing Stop,
        and detects closed positions to update the central risk manager.
        """
        open_positions = self.client.get_open_positions()
        current_open_tickets = {p.ticket for p in open_positions}

        # Check for closed positions
        for ticket, meta in list(self.tracked_positions.items()):
            if ticket not in current_open_tickets:
                # Position closed!
                net_profit = self.client.get_closed_position_net_profit(ticket)
                if net_profit is None:
                    # Estimate from last known quote
                    specs = self.client.get_symbol_specs(meta.symbol)
                    quote = self.current_quotes.get(meta.symbol)
                    if quote and specs:
                        exit_price = quote[0] if meta.is_buy else quote[1]
                        diff = (exit_price - meta.entry_price) if meta.is_buy else (meta.entry_price - exit_price)
                        points = diff / specs.point
                        gross = points * specs.tick_value * meta.volume
                        net_profit = gross - (self.config.commission_per_lot_round_turn * meta.volume)
                    else:
                        net_profit = 0.0

                self.risk_manager.register_trade_close(
                    symbol=meta.symbol,
                    realized_pnl=net_profit,
                    exit_time=now_utc,
                    ticket=ticket,
                )
                self._log_event(
                    "CLOSED",
                    f"#{ticket} {meta.symbol} lezárva | Nettó PnL: {net_profit:+.2f} USD | Új Egyenleg: {account.balance:.2f} USD",
                )
                del self.tracked_positions[ticket]

        # Manage existing active positions
        for pos in open_positions:
            ticket = pos.ticket
            sym = pos.symbol
            quote = self.current_quotes.get(sym)
            specs = self.client.get_symbol_specs(sym)
            if not quote or not specs:
                continue

            bid, ask, _, _ = quote
            is_buy = pos.type == 0
            curr_price = bid if is_buy else ask

            if ticket not in self.tracked_positions:
                # Register untracked position
                risk_dist = abs(pos.price_open - pos.sl) if pos.sl > 0 else (specs.pip_size * 20.0)
                self.tracked_positions[ticket] = ManagedPositionMetadata(
                    ticket=ticket,
                    symbol=sym,
                    is_buy=is_buy,
                    volume=pos.volume,
                    entry_price=pos.price_open,
                    initial_sl=pos.sl,
                    initial_tp=pos.tp,
                    risk_dist_1r=risk_dist,
                    highest_price=curr_price,
                    lowest_price=curr_price,
                    entry_time=time.time(),
                )

            meta = self.tracked_positions[ticket]
            meta.highest_price = max(meta.highest_price, curr_price)
            meta.lowest_price = min(meta.lowest_price, curr_price)

            # Calculate profit distance in R
            gain = (curr_price - meta.entry_price) if is_buy else (meta.entry_price - curr_price)
            r_multiple = gain / meta.risk_dist_1r if meta.risk_dist_1r > 0 else 0.0

            # 1. Dynamic Breakeven Rule
            # For GBPJPY: target +1.45R -> Move SL to entry_price + 0.1R
            # For EURUSD / XAUUSD: target +1.00R -> Move SL to entry_price
            be_threshold = 1.45 if sym == "GBPJPY" else 1.00
            if not meta.breakeven_applied and r_multiple >= be_threshold:
                new_sl = round(
                    (meta.entry_price + (0.1 * meta.risk_dist_1r)) if is_buy else (meta.entry_price - (0.1 * meta.risk_dist_1r)),
                    specs.digits,
                )
                # Verify new_sl is better than current SL
                is_sl_improved = (new_sl > pos.sl) if is_buy else (new_sl < pos.sl or pos.sl == 0)
                if is_sl_improved:
                    if self.client.modify_position(ticket, sym, new_sl, pos.tp):
                        meta.breakeven_applied = True
                        pos.sl = new_sl
                        self._log_event(
                            "BREAKEVEN",
                            f"#{ticket} {sym} Dinamikus Breakeven aktiválva (+{r_multiple:.2f}R)! Új SL: {new_sl:.5f}",
                        )

            # 2. Trailing Stop Rule (when profit >= 1.8R)
            if r_multiple >= 1.80:
                trail_dist = meta.risk_dist_1r * 1.2
                proposed_sl = round(
                    (curr_price - trail_dist) if is_buy else (curr_price + trail_dist),
                    specs.digits,
                )
                is_trail_better = (proposed_sl > pos.sl) if is_buy else (proposed_sl < pos.sl or pos.sl == 0)
                if is_trail_better:
                    if self.client.modify_position(ticket, sym, proposed_sl, pos.tp):
                        pos.sl = proposed_sl
                        meta.trailing_active = True
                        self._log_event(
                            "TRAILING",
                            f"#{ticket} {sym} Trailing Stop frissítve (+{r_multiple:.2f}R)! Új SL: {proposed_sl:.5f}",
                        )

    def _evaluate_signals(self, now_utc: datetime, account: AccountSnapshot) -> None:
        """
        Polls multi-timeframe candles, coordinates candidate trades across active symbols,
        and executes validated trades through Central Risk Management.
        """
        open_positions = self.client.get_open_positions()
        if len(open_positions) >= self.config.max_open_positions:
            return

        data_by_symbol: Dict[str, Any] = {}
        prices: Dict[str, float] = {}
        spreads: Dict[str, float] = {}

        for sym in self.active_symbols:
            existing_sym_positions = [p for p in open_positions if p.symbol == sym]
            if len(existing_sym_positions) >= self.config.max_open_positions_per_symbol:
                continue

            snapshot = self.data_engine.get_market_snapshot(
                symbol=sym,
                execution_tf=self.config.execution_tf,
                structure_tf=self.config.structure_tf,
                regime_tf=self.config.regime_tf,
                macro_tf="H1",
            )
            if snapshot and "execution" in snapshot and len(snapshot["execution"]) >= 50:
                data_by_symbol[sym] = snapshot
                q = self.current_quotes.get(sym)
                if q:
                    prices[sym] = q[1]  # ask
                    spreads[sym] = q[3]  # spread pips

        if not data_by_symbol:
            return

        approved_trades = self.dispatcher.dispatch_market_step(
            data_by_symbol=data_by_symbol,
            candle_timestamp=now_utc,
            prices=prices,
            spreads=spreads,
        )

        for trade in approved_trades:
            current_open = self.client.get_open_positions()
            if len(current_open) >= self.config.max_open_positions:
                break

            sym = trade["symbol"]
            if any(p.symbol == sym for p in current_open):
                continue

            sig = trade["signal"]
            vol = trade["approved_volume"]
            entry_p = trade["price"]

            is_buy = getattr(sig, "is_buy", False)
            is_sell = getattr(sig, "is_sell", False)
            if not (is_buy or is_sell):
                s_str = str(getattr(sig, "signal_type", getattr(sig, "type", ""))).upper()
                is_buy = "BUY" in s_str

            order_type_str = "BUY" if is_buy else "SELL"
            sl_price = float(getattr(sig, "sl_price", getattr(sig, "sl", 0.0)))
            tp_price = float(getattr(sig, "tp_price", getattr(sig, "tp", 0.0)))

            ticket = self.client.send_market_order(
                symbol=sym,
                order_type_str=order_type_str,
                volume=vol,
                sl=sl_price,
                tp=tp_price,
                comment="UnifiedPort",
            )

            if ticket is not None:
                self.risk_manager.register_position_opened(
                    symbol=sym,
                    volume=vol,
                    entry_price=entry_p,
                    entry_time=now_utc,
                    sl=sl_price,
                    tp=tp_price,
                    ticket=ticket,
                )
                risk_dist = abs(entry_p - sl_price)
                self.tracked_positions[ticket] = ManagedPositionMetadata(
                    ticket=ticket,
                    symbol=sym,
                    is_buy=is_buy,
                    volume=vol,
                    entry_price=entry_p,
                    initial_sl=sl_price,
                    initial_tp=tp_price,
                    risk_dist_1r=risk_dist,
                    highest_price=entry_p,
                    lowest_price=entry_p,
                    entry_time=time.time(),
                )
                mode_tag = "PAPER ORDER" if self.mode != TradingMode.LIVE else "LIVE ORDER"
                self._log_event(
                    "EXECUTED",
                    f"[{mode_tag}] #{ticket} {sym} {order_type_str} {vol:.2f} lot @ {entry_p:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f}",
                )

    def _render_dashboard(
        self,
        now_utc: datetime,
        account: AccountSnapshot,
        trading_allowed: bool = True,
        session_reason: str = "",
    ) -> None:
        """
        Prints a rich terminal status overview periodically.
        """
        # Periodic refresh (every 3 seconds)
        if int(now_utc.timestamp()) % 3 != 0:
            return

        session_name = self.market_hours_guard.get_current_session_name(now_utc)
        cb_status = self.risk_manager.circuit_breaker_active
        cb_text = "[bold red]LEÁLLÍTVA (COOLDOWN)[/bold red]" if cb_status else "[bold green]NORMÁL[/bold green]"

        mode_badge = (
            "[bold white on red] LIVE (ÉLES PÉNZ) [/bold white on red]"
            if self.mode == TradingMode.LIVE
            else "[bold white on cyan] PAPER (VIRTUÁLIS) [/bold white on cyan]"
        )

        table_acc = Table(title="📊 Számla Áttekintés ($250 Tőkekeret)", show_header=True)
        table_acc.add_column("Egyenleg", justify="right")
        table_acc.add_column("Tőkeérték (Equity)", justify="right")
        table_acc.add_column("Használt Fedezet", justify="right")
        table_acc.add_column("Fedezeti Szint", justify="right")
        table_acc.add_column("Lebegő PnL", justify="right")
        table_acc.add_column("Circuit Breaker", justify="center")

        pnl_style = "green" if account.profit >= 0 else "red"
        ml_style = "green" if account.margin_level >= 200.0 else "bold red"

        table_acc.add_row(
            f"${account.balance:.2f}",
            f"${account.equity:.2f}",
            f"${account.margin:.2f}",
            f"[{ml_style}]{account.margin_level:.1f}%[/{ml_style}]",
            f"[{pnl_style}]${account.profit:+.2f}[/{pnl_style}]",
            cb_text,
        )

        table_sym = Table(title="📈 Élő Piaci Árak (MT5 Feed)", show_header=True)
        table_sym.add_column("Szimbólum", justify="left")
        table_sym.add_column("Bid", justify="right")
        table_sym.add_column("Ask", justify="right")
        table_sym.add_column("Spread (pip)", justify="right")
        table_sym.add_column("Pozíciók", justify="center")

        open_positions = self.client.get_open_positions()
        for sym in self.active_symbols:
            q = self.current_quotes.get(sym)
            bid_str = f"{q[0]:.5f}" if q else "-"
            ask_str = f"{q[1]:.5f}" if q else "-"
            spread_str = f"{q[3]:.1f}" if q else "-"
            pos_count = sum(1 for p in open_positions if p.symbol == sym)
            table_sym.add_row(sym, bid_str, ask_str, spread_str, str(pos_count))

        console.clear()
        console.print(
            Panel(
                f"{mode_badge} | [bold]Idő UTC:[/bold] {now_utc.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"[bold]Szeánsz:[/bold] {session_name} | [bold]Nyitott pozíciók:[/bold] {len(open_positions)}/2",
                title="⚡ UNIFIED PORTFOLIO LIVE ENGINE",
            )
        )
        console.print(table_acc)
        console.print(table_sym)

        # Live Scanning Status Indicator
        if trading_allowed:
            scan_text = "[bold green]● AKTÍV PÁSZTÁZÁS[/bold green] (Minden beérkező M5/M15 gyertya valós időben kiértékelve, várakozás belépési jelre)"
            border_style = "green"
        else:
            scan_text = f"[bold yellow]⏸ PIHENŐ SZEÁNSZ[/bold yellow] ({session_reason})"
            border_style = "yellow"
        console.print(Panel(scan_text, title="🔍 Piacfigyelő Állapot", border_style=border_style))

        if self.event_log:
            console.print("\n[bold]Eseménynapló (utolsó 8 bejegyzés):[/bold]")
            for t_str, msg in self.event_log[-8:]:
                console.print(f"  [dim]{t_str}[/dim] {msg}")

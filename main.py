"""
Main Trading Bot Entry Point - Titan Quant Scalper
Executes real-time multi-symbol scalping loop with Rich Live terminal monitoring.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

from config.settings import load_config, AppConfig
from utils.logger import setup_logger
from utils.market_hours import MarketHoursGuard
from core.mt5_client import MT5Client, SymbolSpecs, AccountSnapshot
from core.mock_client import MockMT5Client
from core.data_engine import DataEngine
from strategies.titan_scalper import TitanScalperStrategy
from strategies.signals import TradeSignal, SignalType
from risk.risk_manager import RiskManager
from risk.execution_manager import ExecutionManager

console = Console()


class TitanTradingEngine:
    def __init__(self, config: AppConfig, override_mode: str = None):
        self.config = config
        self.mode = override_mode if override_mode else config.general.mode

        self.logger = setup_logger(
            name="TitanScalper",
            log_level=config.general.log_level,
        )

        self.market_hours_guard = MarketHoursGuard(
            allowed_start_utc_hour=config.session_filter.allowed_start_utc_hour,
            allowed_end_utc_hour=config.session_filter.allowed_end_utc_hour,
            avoid_rollover_window=config.session_filter.avoid_rollover_window,
            avoid_friday_close=config.session_filter.avoid_friday_close,
        )

        # Initialize MT5 Client (Live or Mock/Dry-Run)
        if self.mode == "live":
            self.client = MT5Client(
                terminal_path=config.mt5.terminal_path,
                login=config.mt5.login,
                password=config.mt5.password,
                server=config.mt5.server,
                timeout_ms=config.mt5.timeout_ms,
                magic_number=config.mt5.magic_number,
                slippage_points=config.mt5.slippage_points,
            )
        else:
            self.logger.info("[DRY RUN MÓD] Virtuális számlás szimuláció aktiválva.")
            self.client = MockMT5Client(
                initial_balance=getattr(config.risk_management, "initial_balance", 250.0),
                currency=config.risk_management.account_currency,
                magic_number=config.mt5.magic_number,
                slippage_points=config.mt5.slippage_points,
            )

        self.data_engine = DataEngine(lookback_bars=config.timeframes.lookback_bars)
        self.strategy = TitanScalperStrategy(config)
        self.risk_manager = RiskManager(config)
        self.execution_manager = ExecutionManager(config, self.client, self.risk_manager)

        self.enabled_instruments = config.get_enabled_symbols()
        self.last_signal_bar_time: Dict[str, Any] = {}
        self.is_running = False

    def start(self, max_cycles: int = 0):
        self.logger.info(
            f"[bold green]Titan Quant Scalper v{self.config.general.version} indítása...[/bold green] "
            f"Mód: {self.mode.upper()}"
        )

        connected = self.client.connect()
        if not connected:
            if self.mode == "live":
                self.logger.error(
                    "Nem sikerült kapcsolódni az MT5 terminálhoz! "
                    "Győződj meg róla, hogy a MetaTrader 5 fut és az Algo Trading engedélyezve van az Eszközök -> Beállítások -> Szakértők fülön!"
                )
                return
            else:
                self.client.connect()

        # Warm up symbol specifications
        for inst in self.enabled_instruments:
            if not self.client.ensure_symbol(inst.name):
                self.logger.warning(f"Figyelem: A(z) '{inst.name}' szimbólum nem elérhető a brókernél!")

        self.is_running = True
        self.logger.info("Fő kereskedési ciklus indul. Leállítás: Ctrl + C.")

        try:
            self._main_loop(max_cycles=max_cycles)
        except KeyboardInterrupt:
            self.logger.info("Felhasználói leállítás érzékelve (Ctrl+C). Kilépés...")
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        self.client.disconnect()
        self.logger.info("Titan Quant Scalper biztonságosan leállt.")

    def _main_loop(self, max_cycles: int = 0):
        cycle_count = 0
        while self.is_running:
            try:
                cycle_count += 1
                if max_cycles > 0 and cycle_count > max_cycles:
                    self.logger.info(f"Elérte a maximális teszt ciklusszámot ({max_cycles}). Leállás.")
                    break
                loop_start = time.time()
                now_utc = datetime.now(timezone.utc)

                # 1. Market session check
                session_allowed, session_reason = self.market_hours_guard.is_trading_allowed(now_utc)
                session_name = self.market_hours_guard.get_current_session_name(now_utc)

                # 2. Account snapshot
                account = self.client.get_account_snapshot()
                if not account:
                    time.sleep(self.config.general.heartbeat_interval_seconds)
                    continue

                # 3. Position management across all symbols (BE, Trailing SL, Partial TP)
                current_atrs: Dict[str, float] = {}
                open_positions = self.client.get_open_positions()

                # Process each instrument
                for inst in self.enabled_instruments:
                    sym = inst.name
                    quote = self.client.get_quote(sym)
                    if not quote:
                        continue

                    bid, ask, spread_pts, spread_pips = quote
                    specs = self.client.get_symbol_specs(sym)
                    if not specs:
                        continue

                    # Fetch multi-timeframe candle snapshot
                    snapshot = self.data_engine.get_market_snapshot(
                        symbol=sym,
                        execution_tf=self.config.timeframes.execution_tf,
                        structure_tf=self.config.timeframes.structure_tf,
                        regime_tf=self.config.timeframes.regime_tf,
                    )

                    if snapshot and "execution" in snapshot:
                        current_atr = snapshot["execution"]["atr"].iloc[-1]
                        current_atrs[sym] = current_atr
                    else:
                        continue

                    # Only look for new signals if session allows trading
                    if not session_allowed:
                        continue

                    # Filter by risk manager
                    sym_positions = [p for p in open_positions if p.symbol == sym]
                    can_trade, risk_reason = self.risk_manager.can_open_new_trade(
                        account=account,
                        open_positions_count=len(open_positions),
                        symbol_positions_count=len(sym_positions),
                        now=now_utc,
                    )

                    if not can_trade:
                        continue

                    # Prevent multiple entries on the same M1 candle
                    last_closed_bar_time = snapshot["execution"].index[-2]
                    if self.last_signal_bar_time.get(sym) == last_closed_bar_time:
                        continue

                    # Evaluate Strategy
                    signal: Optional[TradeSignal] = self.strategy.evaluate(
                        symbol=sym,
                        data_snapshot=snapshot,
                        symbol_specs=specs,
                        current_bid=bid,
                        current_ask=ask,
                        current_spread_pips=spread_pips,
                    )

                    if signal and signal.signal_type in (SignalType.BUY, SignalType.SELL):
                        # Calculate exact lot size based on small-capital fractional risk
                        lot_size = self.risk_manager.calculate_lot_size(
                            account=account,
                            specs=specs,
                            risk_distance_points=signal.risk_distance_points,
                        )

                        if lot_size <= 0.0:
                            # Mark bar as processed so we don't spam warnings every 2 seconds
                            self.last_signal_bar_time[sym] = last_closed_bar_time
                            continue

                        self.logger.info(
                            f"[bold yellow]ÚJ JELZÉS DETEKTÁLVA![/bold yellow] {signal.signal_type.value} {sym} | "
                            f"Lot: {lot_size:.2f} | Belépő: {signal.entry_price:.5f} | "
                            f"SL: {signal.sl_price:.5f} | TP: {signal.tp_price:.5f} | "
                            f"Konfidencia: {signal.confidence:.0%} | Indok: {signal.rationale}"
                        )

                        ticket = self.client.send_market_order(
                            symbol=sym,
                            order_type_str=signal.signal_type.value,
                            volume=lot_size,
                            sl=signal.sl_price,
                            tp=signal.tp_price,
                            comment=f"Titan_{signal.strategy_mode[:8]}",
                        )

                        self.last_signal_bar_time[sym] = last_closed_bar_time

                        if ticket:
                            is_buy = signal.signal_type == SignalType.BUY
                            self.execution_manager.register_new_position(
                                ticket=ticket,
                                symbol=sym,
                                is_buy=is_buy,
                                entry_price=signal.entry_price,
                                initial_sl=signal.sl_price,
                                initial_tp=signal.tp_price,
                            )
                            # Refresh positions immediately so max_open_positions is respected
                            open_positions = self.client.get_open_positions()

                # Dynamic position management
                self.execution_manager.manage_positions(current_atrs)

                # Sleep until next heartbeat
                elapsed = time.time() - loop_start
                sleep_time = max(0.1, self.config.general.heartbeat_interval_seconds - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.exception(f"Hiba a kereskedési ciklusban: {e}")
                time.sleep(2)


def main():
    # By default, route directly to the approved Unified Portfolio Engine (EURUSD, XAUUSD, GBPJPY)
    if "--engine" not in sys.argv or "--engine" in sys.argv and "legacy" not in sys.argv:
        from portfolio.main import main as portfolio_main
        cleaned_args = [a for a in sys.argv[1:] if not a.startswith("--engine")]
        sys.exit(portfolio_main(cleaned_args))

    parser = argparse.ArgumentParser(description="Titan Quant Scalper - MT5 Algorithmic Engine (Legacy Mode)")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["unified", "legacy"],
        default="legacy",
        help="Trading engine selection: 'unified' (default) or 'legacy'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "dry_run", "backtest"],
        default=None,
        help="Futási mód: 'live' (valódi MT5 bekötés), 'dry_run' (virtuális szimuláció) vagy 'backtest'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Egyedi config.yaml fájl elérési útja",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Maximális ciklusszám teszteléshez (0 = végtelen futás)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "backtest":
        from backtest.run_backtest import main as run_bt
        run_bt()
        return

    engine = TitanTradingEngine(config, override_mode=args.mode)
    engine.start(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()

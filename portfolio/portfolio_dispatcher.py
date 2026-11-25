"""
Portfolio Strategy Dispatcher Module
Routes market data to symbol-specific trading strategies:
- EURUSD & XAUUSD -> TitanScalperStrategy (Dual EMA, Hurst, Volume Surge)
- GBPJPY -> GBPJPYDualRegimeScalper (London Breakout + Asian Mean Reversion)

Provides multi-signal conflict resolution and integration with UnifiedPortfolioRiskManager.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import pandas as pd

from portfolio.portfolio_config import PortfolioConfig, load_portfolio_config, SymbolRiskProfile
from portfolio.portfolio_risk_manager import UnifiedPortfolioRiskManager
from core.mt5_client import SymbolSpecs

try:
    from strategies.titan_scalper import TitanScalperStrategy
    from strategies.signals import TradeSignal as RootTradeSignal, SignalType as RootSignalType
except ImportError:
    TitanScalperStrategy = None
    RootTradeSignal = None
    RootSignalType = None

try:
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
    from gbpjpy.strategies.signals import TradeSignal as GBPJPYTradeSignal, SignalType as GBPJPYSignalType
except ImportError:
    GBPJPYDualRegimeScalper = None
    GBPJPYTradeSignal = None
    GBPJPYSignalType = None

try:
    from config.settings import load_config as load_root_config
except ImportError:
    load_root_config = None

try:
    from gbpjpy.config.settings import load_config as load_gbpjpy_config
except ImportError:
    load_gbpjpy_config = None

logger = logging.getLogger("Portfolio.Dispatcher")


class PortfolioDispatcher:
    """
    Multi-Symbol Signal Router and Orchestrator.
    Directs incoming market bar streams to their designated quantitative strategies,
    aggregates candidate signals, and coordinates with UnifiedPortfolioRiskManager.
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        risk_manager: Optional[UnifiedPortfolioRiskManager] = None,
    ):
        self.config: PortfolioConfig = config if config is not None else load_portfolio_config()
        self.risk_manager: UnifiedPortfolioRiskManager = (
            risk_manager if risk_manager is not None else UnifiedPortfolioRiskManager(self.config)
        )

        # Strategy registry: symbol -> strategy instance
        self.strategies: Dict[str, Any] = {}
        self._init_default_strategies()

    def _init_default_strategies(self) -> None:
        """Initializes default production strategies for EURUSD, XAUUSD, and GBPJPY."""
        # Root engine TitanScalperStrategy for EURUSD and XAUUSD
        if TitanScalperStrategy is not None and load_root_config is not None:
            try:
                root_cfg = load_root_config()
                titan_strat = TitanScalperStrategy(root_cfg)
                self.strategies["EURUSD"] = titan_strat
                self.strategies["XAUUSD"] = titan_strat
            except Exception as e:
                logger.warning(f"Could not initialize TitanScalperStrategy: {e}")

        # GBPJPY Dual Regime Scalper
        if GBPJPYDualRegimeScalper is not None and load_gbpjpy_config is not None:
            try:
                gbp_cfg = load_gbpjpy_config()
                gbp_strat = GBPJPYDualRegimeScalper(config=gbp_cfg)
                self.strategies["GBPJPY"] = gbp_strat
            except Exception as e:
                logger.warning(f"Could not initialize GBPJPYDualRegimeScalper: {e}")

    def register_strategy(self, symbol: str, strategy: Any) -> None:
        """Registers a custom strategy instance for a specific symbol."""
        self.strategies[symbol.upper().strip()] = strategy

    def get_strategy(self, symbol: str) -> Optional[Any]:
        """Retrieves active strategy for a given symbol."""
        return self.strategies.get(symbol.upper().strip())

    def _build_symbol_specs(self, profile: SymbolRiskProfile) -> SymbolSpecs:
        """Constructs SymbolSpecs dataclass from SymbolRiskProfile."""
        return SymbolSpecs(
            name=profile.symbol,
            digits=profile.digits,
            point=profile.point,
            tick_value=profile.tick_value,
            tick_size=profile.point,
            contract_size=profile.contract_size,
            min_volume=profile.min_volume,
            max_volume=profile.max_volume,
            volume_step=profile.volume_step,
            pip_size=profile.pip_size,
        )

    def process_bar(
        self,
        symbol: str,
        bar: Union[Dict[str, Any], pd.DataFrame, pd.Series],
        candle_timestamp: datetime,
        current_spread_pips: Optional[float] = None,
    ) -> List[Any]:
        """
        Dispatches a single symbol's candle data to its strategy.
        Returns a list of candidate TradeSignal objects.
        """
        sym = symbol.upper().strip()
        strat = self.get_strategy(sym)
        if strat is None:
            return []

        profile = self.config.get_symbol_profile(sym)
        if profile is None or not profile.enabled:
            return []

        if candle_timestamp.tzinfo is None:
            candle_timestamp = candle_timestamp.replace(tzinfo=timezone.utc)

        # Handle GBPJPYDualRegimeScalper interface
        if isinstance(strat, GBPJPYDualRegimeScalper) or hasattr(strat, "generate_signals"):
            # Wrap bar into DataFrame if dict / Series
            if isinstance(bar, pd.DataFrame):
                df_input = bar
            elif isinstance(bar, pd.Series):
                df_input = pd.DataFrame([bar])
            elif isinstance(bar, dict):
                if "execution" in bar and isinstance(bar["execution"], pd.DataFrame):
                    df_input = bar["execution"]
                elif "GBPJPY" in bar and isinstance(bar["GBPJPY"], pd.DataFrame):
                    df_input = bar["GBPJPY"]
                elif any(isinstance(v, pd.DataFrame) for v in bar.values()):
                    for v in bar.values():
                        if isinstance(v, pd.DataFrame):
                            df_input = v
                            break
                else:
                    row = dict(bar)
                    if "time" not in row:
                        row["time"] = candle_timestamp
                    df_input = pd.DataFrame([row])
            else:
                return []

            try:
                # Pass full bar dictionary or extracted DataFrame
                input_payload = bar if (isinstance(bar, dict) and "execution" in bar) else df_input
                signals = strat.generate_signals(input_payload)
                return signals if signals else []
            except Exception as e:
                logger.error(f"[{sym}] Error generating signals with GBPJPYDualRegimeScalper: {e}")
                return []

        # Handle TitanScalperStrategy interface
        elif isinstance(strat, TitanScalperStrategy) or hasattr(strat, "evaluate"):
            specs = self._build_symbol_specs(profile)

            # Ensure DatetimeIndex for TitanScalperStrategy
            df_bar = bar
            if isinstance(df_bar, pd.DataFrame) and not isinstance(df_bar.index, pd.DatetimeIndex) and "time" in df_bar.columns:
                df_bar = df_bar.set_index("time", drop=False)

            # Build data snapshot
            if isinstance(df_bar, dict) and "execution" in df_bar:
                data_snapshot = df_bar
                close_price = float(data_snapshot["execution"].iloc[-1]["close"])
            elif isinstance(df_bar, pd.DataFrame):
                data_snapshot = {
                    "execution": df_bar,
                    "structure": df_bar,
                    "regime": df_bar,
                    "m15": df_bar,
                    "h1": df_bar,
                }
                close_price = float(df_bar.iloc[-1]["close"])
            elif isinstance(df_bar, dict):
                df_single = pd.DataFrame([df_bar])
                if "time" in df_single.columns:
                    df_single = df_single.set_index("time", drop=False)
                data_snapshot = {
                    "execution": df_single,
                    "structure": df_single,
                    "regime": df_single,
                    "m15": df_single,
                    "h1": df_single,
                }
                close_price = float(df_bar.get("close", 1.0))
            else:
                return []

            spread = current_spread_pips if current_spread_pips is not None else 1.0
            bid = close_price
            ask = close_price + (spread * profile.pip_size)

            try:
                sig = strat.evaluate(
                    symbol=sym,
                    data_snapshot=data_snapshot,
                    symbol_specs=specs,
                    current_bid=bid,
                    current_ask=ask,
                    current_spread_pips=spread,
                )
                if sig is not None:
                    # Filter out HOLD signals
                    sig_type = getattr(sig, "signal_type", getattr(sig, "type", "HOLD"))
                    sig_type_str = str(getattr(sig_type, "value", sig_type)).upper()
                    if "BUY" in sig_type_str or "SELL" in sig_type_str:
                        return [sig]
                return []
            except Exception as e:
                logger.error(f"[{sym}] Error evaluating TitanScalperStrategy: {e}")
                return []

        return []

    def dispatch_market_step(
        self,
        data_by_symbol: Dict[str, Any],
        candle_timestamp: datetime,
        prices: Optional[Dict[str, float]] = None,
        spreads: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Coordinates a complete multi-symbol M5 simulation step.
        Gathers signals from EURUSD, XAUUSD, and GBPJPY, resolves any conflicts,
        and returns validated trades with approved lot sizes and required margins.
        """
        if candle_timestamp.tzinfo is None:
            candle_timestamp = candle_timestamp.replace(tzinfo=timezone.utc)

        raw_candidates: List[Dict[str, Any]] = []

        for sym, bar_data in data_by_symbol.items():
            sym_clean = sym.upper().strip()
            profile = self.config.get_symbol_profile(sym_clean)
            if profile is None or not profile.enabled:
                continue

            spread = spreads.get(sym_clean, 1.0) if spreads else 1.0
            signals = self.process_bar(sym_clean, bar_data, candle_timestamp, spread)

            for sig in signals:
                # Check for BUY or SELL
                is_buy = getattr(sig, "is_buy", False)
                is_sell = getattr(sig, "is_sell", False)
                if not (is_buy or is_sell):
                    sig_type = str(getattr(sig, "signal_type", getattr(sig, "type", ""))).upper()
                    is_buy = "BUY" in sig_type
                    is_sell = "SELL" in sig_type

                if not (is_buy or is_sell):
                    continue

                entry_price = float(getattr(sig, "entry_price", 0.0))
                if entry_price <= 0 and prices and sym_clean in prices:
                    entry_price = prices[sym_clean]

                raw_candidates.append({
                    "symbol": sym_clean,
                    "signal": sig,
                    "price": entry_price,
                    "spread_pips": spread,
                    "timestamp": candle_timestamp,
                })

        # Resolve conflicts and approve through UnifiedPortfolioRiskManager
        approved_trades = self.risk_manager.resolve_conflicts(raw_candidates, candle_timestamp)
        return approved_trades

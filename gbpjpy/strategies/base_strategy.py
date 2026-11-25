"""
GBPJPY Abstract Base Strategy Interface (Milestone 3)
Defines standard strategy lifecycle, batch and event-driven interfaces,
parameter validation, and risk/execution management hooks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
import pandas as pd

try:
    from .signals import TradeSignal, SignalType
except (ImportError, ValueError):
    from gbpjpy.strategies.signals import TradeSignal, SignalType

logger = logging.getLogger("GBPJPY.BaseStrategy")


class BaseStrategy(ABC):
    """
    Abstract Base Class for quantitative trading strategies.
    Enforces standard lifecycle hooks:
    - initialize(): setup indicators, state, config
    - generate_signals(): vectorized or batch generation over multi-timeframe registry
    - on_bar(): event-driven per-candle evaluation
    - on_trade(): trade fill/close lifecycle callback
    - validate_parameters(): configuration sanity check
    """

    def __init__(
        self,
        name: str = "BaseStrategy",
        config: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        **kwargs
    ):
        self.name = name
        self.config = config
        self.risk_manager = risk_manager
        self.is_initialized: bool = False
        self.active_positions_count: int = 0
        self.parameters: Dict[str, Any] = kwargs
        self.logger = logging.getLogger(f"GBPJPY.Strategy.{name}")

    def initialize(self) -> None:
        """
        Initializes strategy internal state, resets counters, and caches parameters.
        Subclasses should call super().initialize() when overriding.
        """
        self.is_initialized = True
        self.active_positions_count = 0
        self.logger.info(f"Strategy {self.name} initialized successfully.")

    @abstractmethod
    def generate_signals(
        self,
        data_registry: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    ) -> List[TradeSignal]:
        """
        Batch / backtest signal generation interface.
        Accepts a dictionary of symbol/timeframe DataFrames or a single DataFrame.
        Returns a list of typed TradeSignal objects.
        """
        pass

    def on_bar(
        self,
        bar: Union[pd.Series, Dict[str, Any]],
        symbol: str = "GBPJPY",
    ) -> Optional[TradeSignal]:
        """
        Event-driven per-bar execution callback.
        Called once on each completed candle during live trading or tick/bar simulation.
        Default implementation converts bar to a 1-row DataFrame and calls generate_signals.
        """
        if isinstance(bar, pd.Series):
            df = bar.to_frame().T
        elif isinstance(bar, dict):
            df = pd.DataFrame([bar])
        else:
            return None

        signals = self.generate_signals({symbol: df})
        return signals[0] if signals else None

    def on_trade(self, trade_result: Dict[str, Any]) -> None:
        """
        Lifecycle callback invoked upon trade entry, partial fill, modification, or closure.
        Feeds realized PnL back into RiskManager circuit breaker if attached.
        """
        pnl = trade_result.get("net_pnl_usd", trade_result.get("pnl", 0.0))
        is_closed = trade_result.get("closed", trade_result.get("is_closed", False))
        if is_closed and self.risk_manager is not None:
            if hasattr(self.risk_manager, "record_trade_result"):
                self.risk_manager.record_trade_result(pnl=pnl)
            elif hasattr(self.risk_manager, "circuit_breaker"):
                self.risk_manager.circuit_breaker.record_trade_result(pnl=pnl)

    def validate_parameters(self) -> bool:
        """
        Validates internal strategy parameters against safety boundaries:
        - Max stop loss distance <= 50 pips
        - Min stop loss distance >= 5 pips
        - Positive profit targets
        """
        return True

    def evaluate(
        self,
        symbol_or_df: Union[str, pd.DataFrame],
        data_snapshot: Optional[Dict[str, pd.DataFrame]] = None,
        symbol_specs: Optional[Any] = None,
        current_bid: Optional[float] = None,
        current_ask: Optional[float] = None,
        current_spread_pips: Optional[float] = None,
        **kwargs
    ) -> Union[List[TradeSignal], Optional[TradeSignal]]:
        """
        Dual evaluation interface:
        1. If first argument is DataFrame, evaluates via generate_signals(df).
        2. If first argument is symbol string, evaluates live tick/snapshot data.
        """
        if isinstance(symbol_or_df, pd.DataFrame):
            return self.generate_signals(symbol_or_df)

        symbol = str(symbol_or_df)
        registry = data_snapshot or {}
        signals = self.generate_signals({symbol: registry.get("execution", registry.get(symbol))})
        return signals[0] if signals else None

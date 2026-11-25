"""
Abstract Base Strategy Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
from core.mt5_client import SymbolSpecs
from .signals import TradeSignal


class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        data_snapshot: Dict[str, pd.DataFrame],
        symbol_specs: SymbolSpecs,
        current_bid: float,
        current_ask: float,
        current_spread_pips: float,
    ) -> Optional[TradeSignal]:
        """
        Evaluates incoming market data and produces a TradeSignal if conditions are satisfied.
        """
        pass

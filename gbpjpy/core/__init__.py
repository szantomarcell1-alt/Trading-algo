"""
GBPJPY Core Execution, Data Engine, and Client Interfaces
"""

from .mt5_client import (
    MT5Client,
    SymbolSpecs,
    AccountSnapshot,
    OpenPosition,
)
from .mock_client import (
    MockMT5Client,
    MockClient,
    CloseResult,
    calculate_pnl,
)
from .data_engine import DataEngine
from .regime_detector import (
    RegimeDetector,
    MarketRegime,
    RegimeResult,
)

__all__ = [
    "MT5Client",
    "SymbolSpecs",
    "AccountSnapshot",
    "OpenPosition",
    "MockMT5Client",
    "MockClient",
    "CloseResult",
    "calculate_pnl",
    "DataEngine",
    "RegimeDetector",
    "MarketRegime",
    "RegimeResult",
]

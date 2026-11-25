"""
Trade Signals Data Model
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    symbol: str
    signal_type: SignalType
    entry_price: float
    sl_price: float
    tp_price: float
    risk_distance_points: float
    risk_distance_pips: float
    confidence: float  # 0.0 to 1.0 (statistical quality score)
    strategy_mode: str  # "SQUEEZE_BREAKOUT" or "LIQUIDITY_SWEEP_REVERSAL"
    regime: str
    rationale: str

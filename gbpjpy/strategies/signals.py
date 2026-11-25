"""
GBPJPY Quantitative Strategy Signals Module (Milestone 3)
Defines SignalType enum and TradeSignal dataclass with dual attribute/dictionary access,
exact cross-currency pip valuation, and backtest/execution lifecycle compatibility.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Union


class SignalType(str, Enum):
    """
    Standard trading signal directives.
    Inherits from str for string comparison and backward compatibility with test suites.
    """
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    def __str__(self) -> str:
        return self.value


@dataclass
class TradeSignal:
    """
    Typed institutional trade signal container.
    Guarantees strict attribute contracts while providing backward-compatible dictionary
    access (e.g. signal['sl']) and legacy property aliases (sl_price, tp_price).
    """
    symbol: str = "GBPJPY"
    signal_type: Union[SignalType, str] = SignalType.HOLD
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    volume: float = 0.01
    reason: str = ""
    timestamp: Optional[datetime] = None
    timeframe: str = "M5"
    regime: str = "TRENDING"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize signal_type to SignalType enum if valid string
        if isinstance(self.signal_type, str):
            val = self.signal_type.upper().strip()
            if "BUY" in val or val in ("0", "LONG"):
                self.signal_type = SignalType.BUY
            elif "SELL" in val or val in ("1", "SHORT"):
                self.signal_type = SignalType.SELL
            elif "HOLD" in val or val in ("NONE", "FLAT"):
                self.signal_type = SignalType.HOLD

        # Default timestamp to current UTC if omitted
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        elif self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    # -------------------------------------------------------------------------
    # Legacy & Ergonomic Property Aliases
    # -------------------------------------------------------------------------

    @property
    def sl_price(self) -> float:
        """Alias for sl."""
        return self.sl

    @sl_price.setter
    def sl_price(self, val: float):
        self.sl = val

    @property
    def tp_price(self) -> float:
        """Alias for tp."""
        return self.tp

    @tp_price.setter
    def tp_price(self, val: float):
        self.tp = val

    @property
    def type(self) -> Union[SignalType, str]:
        """Alias for signal_type."""
        return self.signal_type

    @type.setter
    def type(self, val: Union[SignalType, str]):
        self.signal_type = val

    @property
    def is_buy(self) -> bool:
        """True if signal represents a long entry."""
        s = str(getattr(self.signal_type, "value", self.signal_type)).upper()
        return "BUY" in s or s in ("0", "LONG")

    @property
    def is_sell(self) -> bool:
        """True if signal represents a short entry."""
        s = str(getattr(self.signal_type, "value", self.signal_type)).upper()
        return "SELL" in s or s in ("1", "SHORT")

    @property
    def is_hold(self) -> bool:
        """True if signal is neutral / hold."""
        s = str(getattr(self.signal_type, "value", self.signal_type)).upper()
        return "HOLD" in s or s in ("NONE", "FLAT")

    @property
    def risk_distance_points(self) -> float:
        """Distance between entry and SL in points (0.001 per point on GBPJPY)."""
        return round(abs(self.entry_price - self.sl) / 0.001, 1)

    @property
    def risk_distance_pips(self) -> float:
        """Distance between entry and SL in pips (0.01 per pip on GBPJPY)."""
        return round(abs(self.entry_price - self.sl) / 0.01, 2)

    @property
    def profit_target_pips(self) -> float:
        """Distance between entry and TP in pips."""
        return round(abs(self.tp - self.entry_price) / 0.01, 2)

    @property
    def risk_reward_ratio(self) -> float:
        """Reward-to-Risk ratio (R-multiple)."""
        risk = abs(self.entry_price - self.sl)
        if risk <= 1e-9:
            return 0.0
        return round(abs(self.tp - self.entry_price) / risk, 2)

    @property
    def confidence(self) -> float:
        """Quality score or confidence rating (0.0 to 1.0)."""
        return float(self.metadata.get("confidence", 1.0))

    # -------------------------------------------------------------------------
    # Dictionary Emulation for Test Compatibility
    # -------------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(f"TradeSignal has no attribute or metadata key '{key}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.metadata

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes TradeSignal to a flat dictionary."""
        sig_str = self.signal_type.value if isinstance(self.signal_type, SignalType) else str(self.signal_type)
        d = {
            "symbol": self.symbol,
            "signal_type": sig_str,
            "entry_price": self.entry_price,
            "sl": self.sl,
            "tp": self.tp,
            "sl_price": self.sl,
            "tp_price": self.tp,
            "volume": self.volume,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timeframe": self.timeframe,
            "regime": self.regime,
            "risk_distance_pips": self.risk_distance_pips,
            "risk_distance_points": self.risk_distance_points,
            "risk_reward_ratio": self.risk_reward_ratio,
            "metadata": dict(self.metadata),
        }
        return d

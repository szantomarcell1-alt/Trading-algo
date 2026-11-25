"""
Unified Portfolio Configuration Module
Defines shared account settings ($250 USD, 1:30 leverage), institutional risk limits,
and multi-symbol specifications for EURUSD, XAUUSD, and GBPJPY.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import copy


@dataclass
class SymbolRiskProfile:
    """Institutional specifications and risk thresholds for a portfolio symbol."""
    symbol: str
    digits: int
    point: float
    pip_size: float
    contract_size: float
    tick_value: float                      # Tick value in USD for 1 standard lot
    max_spread_pips: float
    min_atr_pips: float
    min_volume: float = 0.01
    max_volume: float = 0.05
    volume_step: float = 0.01
    default_leverage: float = 30.0
    margin_currency: str = "USD"
    enabled: bool = True

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)


@dataclass
class PortfolioConfig:
    """
    Central Portfolio Configuration Container.
    Governs shared account capital ($250 USD), leverage (1:30), institutional drawdown limits,
    shared circuit breaker parameters, and symbol specifications.
    """
    # Shared Capital Base & Broker Settings
    account_currency: str = "USD"
    initial_balance: float = 250.0
    leverage: float = 30.0

    # Risk Budgeting
    risk_per_trade_percent: float = 1.0           # 1.0% ($2.50) baseline risk per trade
    max_risk_multiplier_for_min_lot: float = 1.80 # Allows $4.50 max risk cap for 0.01 lot XAUUSD
    commission_per_lot_round_turn: float = 7.0    # Pepperstone ECN Razor: $7.00/lot ($0.07/0.01 lot)
    min_signal_confidence: float = 0.80

    # Concurrency and Margin Guards
    max_open_positions: int = 2                   # Max 2 concurrent positions across the portfolio
    max_open_positions_per_symbol: int = 1        # Max 1 open position per symbol
    min_margin_level_percent: float = 200.0       # Minimum 200% margin level gate

    # Drawdown Ceilings
    max_daily_drawdown_percent: float = 3.0       # Combined daily DD cap <= 3.0% ($7.50 on $250)
    daily_drawdown_buffer_percent: float = 0.20   # 0.20% buffer for pre-trade risk budgeting (3.20% total)
    max_total_drawdown_percent: float = 5.5       # Master kill-switch strictly < 6.0% (target [3.0%, 6.0%))

    # Shared Circuit Breaker
    max_consecutive_losses: int = 3               # 3 consecutive losses across portfolio
    consecutive_loss_cooldown_minutes: int = 60   # 60-minute candle timestamp cooldown

    # Timeframe Settings
    execution_tf: str = "M5"
    structure_tf: str = "M15"
    regime_tf: str = "H1"
    lookback_bars: int = 350

    # Target Verification Gates
    target_weekly_return_percent: float = 1.0     # ~1.0% average weekly return
    target_min_weekly_return_percent: float = 0.75
    target_max_weekly_return_percent: float = 1.25
    target_winning_periods_percent: float = 65.0  # >= 65% winning evaluation periods

    # Symbol Risk Profiles
    symbols: Dict[str, SymbolRiskProfile] = field(default_factory=dict)

    def __post_init__(self):
        if not self.symbols:
            self.symbols = self._default_symbols()

    @staticmethod
    def _default_symbols() -> Dict[str, SymbolRiskProfile]:
        return {
            "EURUSD": SymbolRiskProfile(
                symbol="EURUSD",
                digits=5,
                point=0.00001,
                pip_size=0.0001,
                contract_size=100000.0,
                tick_value=1.00,
                max_spread_pips=1.0,
                min_atr_pips=1.2,
                min_volume=0.01,
                max_volume=0.05,
                volume_step=0.01,
                default_leverage=30.0,
                margin_currency="EUR",
                enabled=True,
            ),
            "XAUUSD": SymbolRiskProfile(
                symbol="XAUUSD",
                digits=2,
                point=0.01,
                pip_size=0.10,
                contract_size=100.0,
                tick_value=1.00,
                max_spread_pips=2.5,
                min_atr_pips=4.5,
                min_volume=0.01,
                max_volume=0.05,
                volume_step=0.01,
                default_leverage=30.0,
                margin_currency="USD",
                enabled=True,
            ),
            "GBPJPY": SymbolRiskProfile(
                symbol="GBPJPY",
                digits=3,
                point=0.001,
                pip_size=0.01,
                contract_size=100000.0,
                tick_value=0.626076,
                max_spread_pips=2.5,
                min_atr_pips=4.0,
                min_volume=0.01,
                max_volume=0.05,
                volume_step=0.01,
                default_leverage=30.0,
                margin_currency="GBP",
                enabled=True,
            ),
        }

    def get_symbol_profile(self, symbol: str) -> Optional[SymbolRiskProfile]:
        """Fetches symbol risk profile case-insensitively."""
        sym_norm = symbol.upper().strip()
        for k, v in self.symbols.items():
            if k.upper().strip() == sym_norm:
                return v
        return None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        data = copy.deepcopy(self.__dict__)
        data["symbols"] = {k: v.to_dict() for k, v in self.symbols.items()}
        return data


def load_portfolio_config(**overrides) -> PortfolioConfig:
    """Factory helper creating PortfolioConfig with optional parameter overrides."""
    cfg = PortfolioConfig()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

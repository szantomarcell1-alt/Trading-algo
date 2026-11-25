"""
Config Settings Module for GBPJPY Quant Scalper
Loads and validates settings from config.yaml and provides typed dataclasses.
Fully isolated within GBPJPY workspace.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Workspace paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = WORKSPACE_ROOT
CONFIG_PATH = str(WORKSPACE_ROOT / "config" / "config.yaml")


@dataclass
class InstrumentConfig:
    name: str = "GBPJPY"
    symbol: str = "GBPJPY"
    enabled: bool = True
    digits: int = 3
    point: float = 0.001
    pip_size: float = 0.01
    contract_size: float = 100000.0
    tick_value: float = 0.626076
    max_spread_pips: float = 2.5
    min_atr_pips: float = 4.0

    def calculate_tick_value(self, usdjpy_rate: float = 155.0) -> float:
        """Calculate USD tick value for 1 standard lot (100k GBP)."""
        return 100.0 / usdjpy_rate


# Alias for compatibility with SymbolSpecs references
SymbolSpecs = InstrumentConfig

# Module-level default GBPJPY specifications
GBPJPY_SPECS = InstrumentConfig()


def calculate_tick_value(usdjpy_rate: float = 155.0) -> float:
    """Standalone tick value calculation for GBPJPY given USDJPY exchange rate."""
    return 100.0 / usdjpy_rate


@dataclass
class GeneralConfig:
    bot_name: str = "GBPJPY Quant Scalper"
    version: str = "1.0.0"
    mode: str = "dry_run"  # "live", "dry_run", "backtest"
    log_level: str = "INFO"
    heartbeat_interval_seconds: float = 2.0


@dataclass
class MT5Config:
    terminal_path: str = ""
    login: int = 0
    password: str = ""
    server: str = ""
    timeout_ms: int = 60000
    magic_number: int = 777101
    slippage_points: int = 10


@dataclass
class TimeframesConfig:
    execution_tf: str = "M5"
    structure_tf: str = "M15"
    regime_tf: str = "H1"
    lookback_bars: int = 350


@dataclass
class RiskManagementConfig:
    account_currency: str = "USD"
    initial_balance: float = 250.0
    leverage: float = 30.0
    risk_per_trade_percent: float = 1.0
    max_risk_multiplier_for_min_lot: float = 1.80
    commission_per_lot_round_turn: float = 7.0
    min_signal_confidence: float = 0.80
    max_open_positions: int = 1
    max_open_positions_per_symbol: int = 1
    max_daily_drawdown_percent: float = 3.0
    max_consecutive_losses: int = 3
    consecutive_loss_cooldown_minutes: int = 60
    min_margin_level_percent: float = 200.0


@dataclass
class ExecutionConfig:
    risk_reward_ratio: float = 1.68
    use_breakeven: bool = True
    breakeven_trigger_r: float = 1.45
    breakeven_buffer_pips: float = 2.0
    use_partial_take_profit: bool = False
    partial_tp_trigger_r: float = 1.5
    partial_tp_ratio: float = 0.5
    use_trailing_stop: bool = True
    trailing_stop_activation_r: float = 1.2
    trailing_stop_atr_multiplier: float = 1.5
    max_holding_bars_m1: int = 55


@dataclass
class SessionFilterConfig:
    enabled: bool = True
    allowed_start_utc_hour: int = 1
    allowed_end_utc_hour: int = 20
    avoid_rollover_window: bool = True
    rollover_buffer_minutes: int = 30
    avoid_friday_close: bool = True


@dataclass
class StrategyParametersConfig:
    hurst_period: int = 100
    hurst_trend_threshold: float = 0.55
    hurst_mean_reversion_threshold: float = 0.45
    bb_length: int = 20
    bb_std: float = 2.0
    kc_length: int = 20
    kc_multiplier: float = 1.5
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50
    ema_trend_htf: int = 200
    rsi_length: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    min_rr_ratio: float = 1.4
    min_atr_pips: float = 8.0
    use_macro_filter: bool = True
    macro_trend_buffer_pips: float = 15.0
    london_tp_multiple: float = 2.0


class AppConfig(dict):
    """
    Combined configuration class that functions as a dictionary for testing suites
    while providing typed attribute access for application code.
    """

    def __init__(
        self,
        general: Optional[GeneralConfig] = None,
        mt5: Optional[MT5Config] = None,
        instruments: Optional[List[InstrumentConfig]] = None,
        timeframes: Optional[TimeframesConfig] = None,
        risk_management: Optional[RiskManagementConfig] = None,
        execution: Optional[ExecutionConfig] = None,
        session_filter: Optional[SessionFilterConfig] = None,
        strategy: Optional[StrategyParametersConfig] = None,
        raw_dict: Optional[Dict[str, Any]] = None,
    ):
        raw = raw_dict or {}
        super().__init__(raw)

        self.general = general or GeneralConfig()
        self.mt5 = mt5 or MT5Config()
        self.instruments = instruments or [InstrumentConfig()]
        self.timeframes = timeframes or TimeframesConfig()
        self.risk_management = risk_management or RiskManagementConfig()
        self.execution = execution or ExecutionConfig()
        self.session_filter = session_filter or SessionFilterConfig()
        self.strategy = strategy or StrategyParametersConfig()

        # Ensure required dictionary sections exist for dict-based tests
        if "symbol" not in self:
            inst = self.instruments[0] if self.instruments else InstrumentConfig()
            self["symbol"] = {
                "name": inst.name,
                "symbol": inst.name,
                "digits": inst.digits,
                "point": inst.point,
                "pip_size": inst.pip_size,
                "contract_size": inst.contract_size,
                "tick_value": inst.tick_value,
                "max_spread_pips": inst.max_spread_pips,
                "min_atr_pips": inst.min_atr_pips,
            }
        if "risk" not in self:
            self["risk"] = {
                "account_currency": self.risk_management.account_currency,
                "initial_balance": self.risk_management.initial_balance,
                "leverage": self.risk_management.leverage,
                "risk_per_trade_percent": self.risk_management.risk_per_trade_percent,
                "max_risk_multiplier_for_min_lot": self.risk_management.max_risk_multiplier_for_min_lot,
                "commission_per_lot_round_turn": self.risk_management.commission_per_lot_round_turn,
                "min_signal_confidence": self.risk_management.min_signal_confidence,
                "max_open_positions": self.risk_management.max_open_positions,
                "max_daily_drawdown_percent": self.risk_management.max_daily_drawdown_percent,
                "max_consecutive_losses": self.risk_management.max_consecutive_losses,
                "min_margin_level_percent": self.risk_management.min_margin_level_percent,
            }
        if "indicators" not in self:
            self["indicators"] = {
                "hurst_period": self.strategy.hurst_period,
                "bb_length": self.strategy.bb_length,
                "bb_std": self.strategy.bb_std,
                "kc_length": self.strategy.kc_length,
                "kc_multiplier": self.strategy.kc_multiplier,
                "ema_fast": self.strategy.ema_fast,
                "ema_medium": self.strategy.ema_medium,
                "ema_slow": self.strategy.ema_slow,
                "ema_trend_htf": self.strategy.ema_trend_htf,
                "rsi_length": self.strategy.rsi_length,
                "atr_period": self.strategy.atr_period,
            }
        if "sessions" not in self:
            self["sessions"] = {
                "enabled": self.session_filter.enabled,
                "allowed_start_utc_hour": self.session_filter.allowed_start_utc_hour,
                "allowed_end_utc_hour": self.session_filter.allowed_end_utc_hour,
                "avoid_rollover_window": self.session_filter.avoid_rollover_window,
                "rollover_buffer_minutes": self.session_filter.rollover_buffer_minutes,
                "avoid_friday_close": self.session_filter.avoid_friday_close,
            }

    def get_symbol_config(self, symbol_name: str = "GBPJPY") -> Optional[InstrumentConfig]:
        for inst in self.instruments:
            if inst.name.upper() == symbol_name.upper():
                return inst
        return None

    def get_enabled_symbols(self) -> List[InstrumentConfig]:
        return [inst for inst in self.instruments if inst.enabled]


def _filter_dict(dataclass_cls, d: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to safely filter dictionary keys matching dataclass fields."""
    valid_fields = {f for f in dataclass_cls.__dataclass_fields__}
    return {k: v for k, v in d.items() if k in valid_fields}


def load_config(config_path: Optional[str] = None) -> AppConfig:
    if config_path is None:
        config_path = CONFIG_PATH

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    general_raw = raw.get("general", {})
    mt5_raw = raw.get("mt5", {})
    inst_raw = raw.get("instruments", {}).get("symbols", [])
    if not inst_raw and "symbol" in raw:
        inst_raw = [raw["symbol"]]

    tf_raw = raw.get("timeframes", {})
    risk_raw = raw.get("risk_management", raw.get("risk", {}))
    exec_raw = raw.get("execution_and_trade_management", raw.get("execution", {}))
    sess_raw = raw.get("session_filter", raw.get("sessions", {}))
    strat_raw = raw.get("strategy_parameters", raw.get("indicators", {}))

    instruments = [
        InstrumentConfig(**_filter_dict(InstrumentConfig, s))
        for s in inst_raw
    ]
    if not instruments:
        instruments = [InstrumentConfig()]

    return AppConfig(
        general=GeneralConfig(**_filter_dict(GeneralConfig, general_raw)),
        mt5=MT5Config(**_filter_dict(MT5Config, mt5_raw)),
        instruments=instruments,
        timeframes=TimeframesConfig(**_filter_dict(TimeframesConfig, tf_raw)),
        risk_management=RiskManagementConfig(**_filter_dict(RiskManagementConfig, risk_raw)),
        execution=ExecutionConfig(**_filter_dict(ExecutionConfig, exec_raw)),
        session_filter=SessionFilterConfig(**_filter_dict(SessionFilterConfig, sess_raw)),
        strategy=StrategyParametersConfig(**_filter_dict(StrategyParametersConfig, strat_raw)),
        raw_dict=raw,
    )

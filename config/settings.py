"""
Config Settings Module
Loads and validates settings from config.yaml and environment variables.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import os


@dataclass
class InstrumentConfig:
    name: str
    enabled: bool = True
    max_spread_pips: float = 2.0
    min_atr_pips: float = 3.0


@dataclass
class GeneralConfig:
    bot_name: str = "Titan Quant Scalper"
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
    magic_number: int = 777001
    slippage_points: int = 10


@dataclass
class TimeframesConfig:
    execution_tf: str = "M1"
    structure_tf: str = "M5"
    regime_tf: str = "M15"
    lookback_bars: int = 300


@dataclass
class RiskManagementConfig:
    account_currency: str = "USD"
    initial_balance: float = 250.0
    risk_per_trade_percent: float = 1.0
    max_risk_multiplier_for_min_lot: float = 1.80
    commission_per_lot_round_turn: float = 7.0
    min_signal_confidence: float = 0.80
    max_open_positions: int = 2
    max_open_positions_per_symbol: int = 1
    max_daily_drawdown_percent: float = 3.0
    max_total_drawdown_percent: float = 8.5
    max_consecutive_losses: int = 3
    consecutive_loss_cooldown_minutes: int = 60
    min_margin_level_percent: float = 160.0


@dataclass
class ExecutionConfig:
    risk_reward_ratio: float = 2.0
    use_breakeven: bool = True
    breakeven_trigger_r: float = 1.45
    breakeven_buffer_pips: float = 2.0
    use_partial_take_profit: bool = True
    partial_tp_trigger_r: float = 1.5
    partial_tp_ratio: float = 0.5
    use_trailing_stop: bool = True
    trailing_stop_activation_r: float = 1.2
    trailing_stop_atr_multiplier: float = 1.5
    max_holding_bars_m1: int = 45


@dataclass
class SessionFilterConfig:
    enabled: bool = True
    allowed_start_utc_hour: int = 7
    allowed_end_utc_hour: int = 20
    avoid_rollover_window: bool = True
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


@dataclass
class AppConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    instruments: List[InstrumentConfig] = field(default_factory=list)
    timeframes: TimeframesConfig = field(default_factory=TimeframesConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    session_filter: SessionFilterConfig = field(default_factory=SessionFilterConfig)
    strategy: StrategyParametersConfig = field(default_factory=StrategyParametersConfig)

    def get_symbol_config(self, symbol_name: str) -> Optional[InstrumentConfig]:
        for inst in self.instruments:
            if inst.name.upper() == symbol_name.upper():
                return inst
        return None

    def get_enabled_symbols(self) -> List[InstrumentConfig]:
        return [inst for inst in self.instruments if inst.enabled]


def load_config(config_path: Optional[str] = None) -> AppConfig:
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = str(base_dir / "config" / "config.yaml")

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    general_raw = raw.get("general", {})
    mt5_raw = raw.get("mt5", {})
    inst_raw = raw.get("instruments", {}).get("symbols", [])
    tf_raw = raw.get("timeframes", {})
    risk_raw = raw.get("risk_management", {})
    exec_raw = raw.get("execution_and_trade_management", {})
    sess_raw = raw.get("session_filter", {})
    strat_raw = raw.get("strategy_parameters", {})

    instruments = [
        InstrumentConfig(
            name=s.get("name", ""),
            enabled=s.get("enabled", True),
            max_spread_pips=float(s.get("max_spread_pips", 2.0)),
            min_atr_pips=float(s.get("min_atr_pips", 3.0)),
        )
        for s in inst_raw
    ]

    return AppConfig(
        general=GeneralConfig(**general_raw),
        mt5=MT5Config(**mt5_raw),
        instruments=instruments,
        timeframes=TimeframesConfig(**tf_raw),
        risk_management=RiskManagementConfig(**risk_raw),
        execution=ExecutionConfig(**exec_raw),
        session_filter=SessionFilterConfig(**sess_raw),
        strategy=StrategyParametersConfig(**strat_raw),
    )

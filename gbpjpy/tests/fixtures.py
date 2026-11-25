"""
Test Fixtures and Synthetic Generators for GBPJPY Testing Suite.

Provides:
- Synthetic M5 OHLCV bar generators (trend, mean-reversion, squeeze, volatility spike).
- Mock AccountInfo generator ($250 equity, 1:30 leverage).
- Mock SymbolSpecs generator (GBPJPY specifications).
- Mock trade and trade signal generators.
- Mathematical reference calculation utilities (Oracles).
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union, Sequence
import numpy as np
import pandas as pd


@dataclass
class MockSymbolSpecs:
    """GBPJPY Symbol specifications according to institutional forex standards."""
    name: str = "GBPJPY"
    symbol: str = "GBPJPY"
    digits: int = 3
    point: float = 0.001
    pip_size: float = 0.01
    contract_size: float = 100000.0
    tick_value: float = 0.626   # Typical USD tick value at USDJPY ~159.7 (100 JPY / 159.7)
    tick_size: float = 0.001
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01
    spread: float = 1.2        # Typical spread in pips
    currency_base: str = "GBP"
    currency_profit: str = "JPY"

    def calculate_tick_value(self, usdjpy_rate: float = 155.0) -> float:
        """
        Calculate USD tick value for 1 standard lot (100,000 GBP).
        1 tick = 0.001 JPY * 100,000 = 100 JPY.
        In USD: 100 JPY / usdjpy_rate.
        """
        return 100.0 / usdjpy_rate


@dataclass
class MockAccountInfo:
    """Mock trading account info calibrated for $250 equity and 1:30 leverage."""
    balance: float = 250.0
    equity: float = 250.0
    margin: float = 0.0
    free_margin: float = 250.0
    margin_level: float = 0.0
    leverage: float = 30.0
    currency: str = "USD"
    profit: float = 0.0

    def update_margin(self, required_margin: float, unrealized_pnl: float = 0.0):
        """Update margin and margin level."""
        self.margin = required_margin
        self.equity = self.balance + unrealized_pnl
        self.free_margin = self.equity - self.margin
        if self.margin > 0:
            self.margin_level = (self.equity / self.margin) * 100.0
        else:
            self.margin_level = 0.0


@dataclass
class MockTradeSignal:
    """Mock trade signal emitted by strategy."""
    symbol: str = "GBPJPY"
    signal_type: str = "BUY"  # "BUY" or "SELL"
    entry_price: float = 155.000
    sl: float = 154.800
    tp: float = 155.400
    volume: float = 0.01
    reason: str = "LONDON_BREAKOUT"
    timestamp: datetime = field(default_factory=lambda: datetime(2026, 1, 15, 8, 30))


@dataclass
class MockTrade:
    """Mock executed trade record."""
    ticket: int = 1001
    symbol: str = "GBPJPY"
    type: str = "BUY"
    volume: float = 0.01
    entry_price: float = 155.000
    exit_price: float = 155.200
    entry_time: datetime = field(default_factory=lambda: datetime(2026, 1, 15, 8, 30))
    exit_time: datetime = field(default_factory=lambda: datetime(2026, 1, 15, 10, 00))
    sl: float = 154.800
    tp: float = 155.400
    gross_pnl_usd: float = 1.29
    commission: float = 0.07
    net_pnl_usd: float = 1.22
    pips: float = 20.0


# -------------------------------------------------------------------------
# Synthetic Bar Generators
# -------------------------------------------------------------------------

def generate_synthetic_m5_data(
    bars: int = 500,
    start_price: float = 155.000,
    regime: str = "trend_up",
    start_time: Optional[datetime] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic GBPJPY M5 OHLCV bar series for testing.

    Supported regimes:
    - 'trend_up': Steady upward drift with realistic noise
    - 'trend_down': Steady downward drift with realistic noise
    - 'mean_reversion': Mean-reverting Ornstein-Uhlenbeck oscillating series
    - 'squeeze': Low-volatility squeeze followed by explosive expansion
    - 'volatility_spike': Calm market with sudden 100+ pip shock candle
    - 'random_walk': Geometric Brownian motion
    """
    rng = np.random.default_rng(seed)

    if start_time is None:
        start_time = datetime(2026, 1, 15, 0, 0, 0)

    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(bars)]

    close_prices = np.zeros(bars)
    close_prices[0] = start_price

    if regime == "trend_up":
        drift = 0.015  # positive drift per 5m bar (~1.5 pips/bar)
        noise = rng.normal(0, 0.03, size=bars)
        for i in range(1, bars):
            close_prices[i] = close_prices[i - 1] + drift + noise[i]

    elif regime == "trend_down":
        drift = -0.015
        noise = rng.normal(0, 0.03, size=bars)
        for i in range(1, bars):
            close_prices[i] = close_prices[i - 1] + drift + noise[i]

    elif regime == "mean_reversion":
        # Ornstein-Uhlenbeck process around start_price
        theta = 0.15   # Mean reversion speed
        mu = start_price
        sigma = 0.05   # Volatility
        for i in range(1, bars):
            dx = theta * (mu - close_prices[i - 1]) + rng.normal(0, sigma)
            close_prices[i] = close_prices[i - 1] + dx

    elif regime == "squeeze":
        # Bars 0 to 60% of data: ultra-low volatility (BB compresses inside Keltner)
        # Remaining bars: strong breakout trend
        squeeze_bars = int(bars * 0.6)
        low_noise = rng.normal(0, 0.005, size=squeeze_bars)
        for i in range(1, squeeze_bars):
            close_prices[i] = close_prices[i - 1] + low_noise[i]

        breakout_noise = rng.normal(0.04, 0.02, size=bars - squeeze_bars)
        for i in range(squeeze_bars, bars):
            close_prices[i] = close_prices[i - 1] + breakout_noise[i - squeeze_bars]

    elif regime == "volatility_spike":
        # Normal series with sudden 150-pip jump at bar 50%
        spike_idx = int(bars * 0.5)
        noise = rng.normal(0, 0.02, size=bars)
        for i in range(1, bars):
            delta = noise[i]
            if i == spike_idx:
                delta += 1.500  # 150 pip sudden shock
            close_prices[i] = close_prices[i - 1] + delta

    else:  # random_walk
        noise = rng.normal(0, 0.03, size=bars)
        for i in range(1, bars):
            close_prices[i] = close_prices[i - 1] + noise[i]

    # Generate plausible Open, High, Low from Close prices
    open_prices = np.zeros(bars)
    open_prices[0] = start_price
    for i in range(1, bars):
        open_prices[i] = close_prices[i - 1]

    high_prices = np.zeros(bars)
    low_prices = np.zeros(bars)
    volumes = rng.integers(100, 1500, size=bars)

    for i in range(bars):
        body_max = max(open_prices[i], close_prices[i])
        body_min = min(open_prices[i], close_prices[i])
        upper_wick = abs(rng.normal(0.015, 0.01))
        lower_wick = abs(rng.normal(0.015, 0.01))
        high_prices[i] = round(body_max + upper_wick, 3)
        low_prices[i] = round(body_min - lower_wick, 3)
        open_prices[i] = round(open_prices[i], 3)
        close_prices[i] = round(close_prices[i], 3)

    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps),
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
        "tick_volume": volumes
    })

    df.set_index("time", inplace=True, drop=False)
    return df


# -------------------------------------------------------------------------
# Mathematical Reference Functions (Oracles)
# -------------------------------------------------------------------------

def reference_pnl_usd(
    order_type: str,
    volume_lots: float,
    entry_price: float,
    exit_price: float,
    contract_size: float = 100000.0,
    usdjpy_rate: float = 155.0
) -> float:
    """
    Mathematical Oracle for cross-currency JPY-to-USD profit calculation.
    PnL (JPY) = (exit_price - entry_price) * contract_size * volume (for BUY)
    PnL (USD) = PnL (JPY) / usdjpy_rate
    """
    if str(order_type).upper() == "BUY":
        pnl_jpy = (exit_price - entry_price) * contract_size * volume_lots
    else:
        pnl_jpy = (entry_price - exit_price) * contract_size * volume_lots

    return round(pnl_jpy / usdjpy_rate, 4)


def reference_required_margin_usd(
    volume_lots: float,
    leverage: float = 30.0,
    contract_size: float = 100000.0,
    gbpusd_rate: float = 1.28
) -> float:
    """
    Mathematical Oracle for required margin for GBP-base pair in USD account.
    Base currency is GBP -> Notional = volume * contract_size (in GBP).
    In USD = Notional * gbpusd_rate.
    Required Margin = (Notional in USD) / leverage.
    """
    notional_gbp = volume_lots * contract_size
    notional_usd = notional_gbp * gbpusd_rate
    return round(notional_usd / leverage, 4)


def reference_garman_klass_vol(df: pd.DataFrame) -> float:
    """
    Standard Garman-Klass Volatility formula:
    0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    """
    h_l = np.log(df["high"] / df["low"])
    c_o = np.log(df["close"] / df["open"])
    term1 = 0.5 * (h_l ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (c_o ** 2)
    gk = np.sqrt(np.mean(term1 - term2))
    return float(gk)


def reference_hurst_exponent(series: Union[pd.Series, np.ndarray], max_lag: int = 20) -> float:
    """
    Calculates the Hurst Exponent via Rescaled Range (R/S) analysis.
    """
    ts = np.asarray(series)
    lags = range(2, max_lag)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    # Polyfit on log-log plot
    reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(reg[0])


class MockBenchmarkStrategy:
    """
    Deterministic benchmark strategy for Walk-Forward Analysis and Anti-Overfitting testing.
    Emits periodic breakout orders on trending price series to generate authentic positive WFE.
    """
    def __init__(self, step_bars: int = 50, tp_pips: float = 8.0, sl_pips: float = 40.0):
        self.step_bars = int(step_bars)
        self.tp_pips = float(tp_pips)
        self.sl_pips = float(sl_pips)
        self.risk_manager = None

    def generate_signals(self, window_df: pd.DataFrame) -> List[Any]:
        if len(window_df) < 5:
            return []
        bar = window_df.iloc[-1]
        idx = len(window_df)
        c = float(bar["close"])
        if idx % self.step_bars == 0:
            from gbpjpy.strategies.signals import TradeSignal, SignalType
            t = bar["time"] if "time" in bar else datetime.now(timezone.utc)
            return [TradeSignal(
                symbol="GBPJPY",
                signal_type=SignalType.BUY,
                entry_price=c,
                sl=round(c - (self.sl_pips * 0.01), 3),
                tp=round(c + (self.tp_pips * 0.01), 3),
                volume=0.01,
                reason="BenchmarkBreakoutSignal",
                timestamp=t,
            )]
        return []

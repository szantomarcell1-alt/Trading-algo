"""
Milestone 1 Core Test Suite for GBPJPY Framework
Thorough verification of Workspace Scaffolding, Configuration, MT5 Client,
Data Engine Indicator Suite, Mock Execution Client, and Structured Logger.

Run via:
    python "c:\\Trading algo\\gbpjpy\\test_m1_core.py"
"""

import math
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Guarantee gbpjpy package is available on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config.settings import (
    load_config,
    GBPJPY_SPECS,
    calculate_tick_value,
    InstrumentConfig,
    RiskManagementConfig,
    WORKSPACE_ROOT,
)
from core.mt5_client import MT5Client, SymbolSpecs, AccountSnapshot, OpenPosition
from core.data_engine import DataEngine
from core.mock_client import MockMT5Client, MockClient, calculate_pnl, CloseResult
from utils.logger import setup_logger, get_logger


def create_synthetic_bars(
    bars: int = 400,
    start_price: float = 195.500,
    regime: str = "trend_up",
    start_time: datetime = datetime(2026, 1, 15, 0, 0, 0),
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic GBPJPY bars for indicator verification."""
    rng = np.random.default_rng(seed)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(bars)]
    closes = np.zeros(bars)
    closes[0] = start_price

    if regime == "trend_up":
        drift = 0.015
        noise = rng.normal(0, 0.03, size=bars)
        for i in range(1, bars):
            closes[i] = closes[i - 1] + drift + noise[i]
    elif regime == "squeeze":
        squeeze_bars = int(bars * 0.6)
        low_noise = rng.normal(0, 0.003, size=squeeze_bars)
        for i in range(1, squeeze_bars):
            closes[i] = closes[i - 1] + low_noise[i]
        breakout = rng.normal(0.04, 0.02, size=bars - squeeze_bars)
        for i in range(squeeze_bars, bars):
            closes[i] = closes[i - 1] + breakout[i - squeeze_bars]
    else:
        noise = rng.normal(0, 0.03, size=bars)
        for i in range(1, bars):
            closes[i] = closes[i - 1] + noise[i]

    opens = np.zeros(bars)
    opens[0] = start_price
    for i in range(1, bars):
        opens[i] = closes[i - 1]

    highs = np.zeros(bars)
    lows = np.zeros(bars)
    volumes = rng.integers(100, 1500, size=bars)

    for i in range(bars):
        body_max = max(opens[i], closes[i])
        body_min = min(opens[i], closes[i])
        highs[i] = round(body_max + abs(rng.normal(0.015, 0.01)), 3)
        lows[i] = round(body_min - abs(rng.normal(0.015, 0.01)), 3)
        opens[i] = round(opens[i], 3)
        closes[i] = round(closes[i], 3)

    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    df.set_index("time", inplace=True, drop=False)
    return df


class TestM1ConfigAndSpecs(unittest.TestCase):
    """Verifies configuration loading, schema contracts, and GBPJPY specs."""

    def test_load_config_and_schema(self):
        cfg = load_config()
        self.assertIsNotNone(cfg)
        self.assertIsInstance(cfg, dict, "AppConfig must support dict interface")
        self.assertEqual(cfg.general.bot_name, "GBPJPY Quant Scalper")
        self.assertEqual(cfg.timeframes.execution_tf, "M5")
        self.assertEqual(cfg.risk_management.initial_balance, 250.0)
        self.assertEqual(cfg.risk_management.leverage, 30.0)
        self.assertEqual(cfg.risk_management.max_open_positions, 1)
        self.assertEqual(cfg.risk_management.min_margin_level_percent, 200.0)
        self.assertEqual(cfg.session_filter.allowed_start_utc_hour, 1)
        self.assertEqual(cfg.session_filter.allowed_end_utc_hour, 20)
        self.assertTrue(cfg.session_filter.avoid_rollover_window)

    def test_gbpjpy_symbol_specs(self):
        cfg = load_config()
        sym = cfg.get_symbol_config("GBPJPY")
        self.assertIsNotNone(sym)
        self.assertEqual(sym.digits, 3, "GBPJPY digits must be 3")
        self.assertAlmostEqual(sym.point, 0.001, places=4)
        self.assertAlmostEqual(sym.pip_size, 0.01, places=4)
        self.assertEqual(sym.contract_size, 100000.0)
        self.assertAlmostEqual(sym.tick_value, 0.626076, places=4)

    def test_dynamic_tick_value_calculation(self):
        # 1 tick = 0.001 * 100k = 100 JPY. In USD = 100 / usdjpy_rate
        val_155 = calculate_tick_value(155.0)
        self.assertAlmostEqual(val_155, 100.0 / 155.0, places=4)
        val_160 = calculate_tick_value(160.0)
        self.assertAlmostEqual(val_160, 100.0 / 160.0, places=4)
        # Check method on GBPJPY_SPECS
        self.assertAlmostEqual(GBPJPY_SPECS.calculate_tick_value(155.0), 100.0 / 155.0, places=4)

    def test_workspace_isolation_paths(self):
        cfg = load_config()
        cfg_str = str(cfg).lower()
        self.assertNotIn("xauusd", cfg_str, "Config must not contain XAUUSD references")
        self.assertTrue(str(WORKSPACE_ROOT).lower().endswith("gbpjpy"))


class TestM1Logger(unittest.TestCase):
    """Verifies logger creates isolated logs in gbpjpy/logs."""

    def test_logger_initialization_and_path(self):
        logger = setup_logger("TestLogger")
        self.assertIsNotNone(logger)
        logger.info("Milestone 1 test log entry.")
        log_dir = WORKSPACE_ROOT / "logs"
        self.assertTrue(log_dir.exists(), "Logs directory must exist inside gbpjpy")
        log_file = log_dir / "gbpjpy_scalper.log"
        self.assertTrue(log_file.exists(), "gbpjpy_scalper.log must exist inside gbpjpy/logs")


class TestM1MT5Client(unittest.TestCase):
    """Verifies MT5 client thread lock, filling mode bitmask, and spec defaults."""

    def test_client_initialization_and_thread_lock(self):
        client = MT5Client(magic_number=777101, timeout_ms=5000)
        self.assertTrue(hasattr(client, "_lock"), "MT5Client must maintain a thread lock")
        self.assertEqual(client.magic_number, 777101)
        self.assertFalse(client.is_connected)

    def test_filling_mode_bitmask_no_attribute_error(self):
        client = MT5Client()
        # Must resolve safely without throwing AttributeError on SYMBOL_FILLING_FOK
        mode = client._resolve_filling_mode("GBPJPY")
        self.assertIsInstance(mode, int)

    def test_symbol_specs_offline_fallback(self):
        client = MT5Client()
        client.ensure_symbol("GBPJPY")
        specs = client.get_symbol_specs("GBPJPY")
        self.assertIsNotNone(specs)
        self.assertEqual(specs.digits, 3)
        self.assertEqual(specs.point, 0.001)
        self.assertEqual(specs.pip_size, 0.01)
        self.assertEqual(specs.contract_size, 100000.0)


class TestM1DataEngine(unittest.TestCase):
    """Verifies DataEngine indicator vectorization, 14 contract columns, and convolution."""

    def test_contract_columns_presence(self):
        df = create_synthetic_bars(bars=350, regime="trend_up")
        res = DataEngine.compute_indicators(df)

        required_columns = [
            "ema_9", "ema_21", "ema_50", "ema_200", "atr",
            "bb_upper", "bb_lower", "bb_middle",
            "squeeze_on", "squeeze_fired", "squeeze_momentum",
            "vwap", "swing_high", "swing_low"
        ]
        for col in required_columns:
            self.assertIn(col, res.columns, f"Mandatory column '{col}' missing from indicators")

    def test_indicator_mathematical_properties(self):
        df = create_synthetic_bars(bars=350, regime="trend_up")
        res = DataEngine.compute_indicators(df)

        # 1. Bollinger Bands Invariants (Upper >= Middle >= Lower)
        valid_res = res.iloc[50:]
        self.assertTrue((valid_res["bb_upper"] >= valid_res["bb_middle"]).all())
        self.assertTrue((valid_res["bb_middle"] >= valid_res["bb_lower"]).all())

        # 2. Keltner Channels Invariant (Upper > Lower)
        self.assertTrue((valid_res["kc_upper"] > valid_res["kc_lower"]).all())
        self.assertTrue((valid_res["keltner_upper"] > valid_res["keltner_lower"]).all())

        # 3. ATR must be strictly positive
        self.assertTrue((valid_res["atr"] > 0).all())

        # 4. Squeeze Momentum must be finite float series
        self.assertFalse(np.isnan(res["squeeze_momentum"].iloc[-1]))

    def test_ttm_squeeze_behavior(self):
        df = create_synthetic_bars(bars=300, regime="squeeze")
        res = DataEngine.compute_indicators(df)
        self.assertTrue(res["squeeze_on"].any(), "Squeeze On must be detected during compression")
        self.assertTrue(isinstance(res["squeeze_on"].iloc[0], (bool, np.bool_)))
        self.assertTrue(isinstance(res["squeeze_fired"].iloc[0], (bool, np.bool_)))

    def test_vwap_daily_reset(self):
        day1 = datetime(2026, 1, 15, 0, 0, 0)
        day2 = datetime(2026, 1, 16, 0, 0, 0)
        df1 = create_synthetic_bars(bars=288, start_price=195.0, start_time=day1)
        df2 = create_synthetic_bars(bars=288, start_price=196.5, start_time=day2)
        combined = pd.concat([df1, df2])

        res = DataEngine.compute_indicators(combined)
        day2_first = res.loc[res["time"] == day2].iloc[0]
        typical_p = (day2_first["high"] + day2_first["low"] + day2_first["close"]) / 3.0
        self.assertAlmostEqual(day2_first["vwap"], typical_p, places=2,
                               msg="VWAP must reset to typical price at 00:00 UTC day boundary")

    def test_causal_swing_points_no_lookahead(self):
        df = create_synthetic_bars(bars=200, regime="trend_up")
        res = DataEngine.compute_indicators(df)

        swing_highs = res["swing_high"].dropna()
        self.assertTrue(len(swing_highs) > 0, "Should identify at least one swing high")

        for idx in swing_highs.index[:10]:
            curr_val = res.loc[idx, "swing_high"]
            loc = res.index.get_loc(idx)
            past_highs = res["high"].iloc[max(0, loc - 10):loc + 1]
            self.assertIn(round(curr_val, 3), [round(x, 3) for x in past_highs],
                          "Confirmed swing high must be a past bar's high (causal)")

    def test_rates_caching(self):
        engine = DataEngine(cache_ttl_seconds=2.0)
        # Mock cached entry
        dummy_df = pd.DataFrame({"close": [195.0, 195.5]})
        engine._rates_cache[("GBPJPY", "M5")] = (time.time(), dummy_df)
        cached = engine.fetch_rates("GBPJPY", "M5", use_cache=True)
        self.assertEqual(len(cached), 2)
        engine.clear_cache()
        self.assertEqual(len(engine._rates_cache), 0)


class TestM1MockClient(unittest.TestCase):
    """Verifies cross-currency PnL math, realistic margin for $250 equity, and order lifecycle."""

    def test_cross_currency_pnl_formula(self):
        specs = SymbolSpecs(
            name="GBPJPY",
            digits=3,
            point=0.001,
            tick_value=0.626076,
            contract_size=100000.0,
            pip_size=0.01,
        )
        # Test BUY: 20 pips gain on 0.01 lot
        # (0.200 / 0.001) * 0.626076 * 0.01 = 200 * 0.00626076 = $1.252 USD
        pnl_buy = calculate_pnl(195.000, 195.200, 0.01, "BUY", specs)
        self.assertAlmostEqual(pnl_buy, 1.2522, places=3)

        # Test SELL: 15 pips gain on 0.01 lot
        # (0.150 / 0.001) * 0.626076 * 0.01 = 150 * 0.00626076 = $0.939 USD
        pnl_sell = calculate_pnl(195.300, 195.150, 0.01, "SELL", specs)
        self.assertAlmostEqual(pnl_sell, 0.9391, places=3)

        # Test with dynamic usdjpy_rate: 155.0
        # JPY PnL = 0.200 * 100k * 0.01 = 200 JPY -> 200 / 155.0 = 1.2903 USD
        pnl_dynamic = calculate_pnl(195.000, 195.200, 0.01, "BUY", usdjpy_rate=155.0)
        self.assertAlmostEqual(pnl_dynamic, 1.2903, places=3)

    def test_margin_and_equity_accounting(self):
        client = MockMT5Client(initial_balance=250.0, leverage=30.0)
        client.connect()
        client.set_sim_price("GBPJPY", 195.500, 195.515)

        # Open 0.01 lot BUY
        pos = client.open_position("GBPJPY", "BUY", 0.01, sl=195.000, tp=196.000)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.volume, 0.01)
        self.assertEqual(pos.price_open, 195.515, "BUY must fill at ASK price")

        snap = client.get_account_snapshot()
        self.assertIsNotNone(snap)
        # Margin for 0.01 lot GBPJPY at 1:30 should be around $40 - $45 USD
        self.assertTrue(38.0 <= snap.margin <= 48.0, f"Expected margin ~$42, got {snap.margin}")
        # Margin level on $250 equity should be ~500% to 650%, safely above 200% guard
        self.assertTrue(snap.margin_level > 500.0, f"Expected margin level > 500%, got {snap.margin_level}%")
        self.assertTrue(snap.margin_free > 200.0)

    def test_mock_order_lifecycle_and_commission(self):
        client = MockMT5Client(initial_balance=250.0, commission_per_lot=7.0)
        client.connect()
        client.set_sim_price("GBPJPY", 195.000, 195.015)

        pos = client.open_position("GBPJPY", "BUY", 0.01, sl=194.500, tp=196.000)
        ticket = pos.ticket
        self.assertEqual(len(client.get_open_positions()), 1)

        # Modify SL/TP
        mod_ok = client.modify_position(ticket, sl=195.020, tp=196.200)
        self.assertTrue(mod_ok)
        self.assertEqual(client._positions[ticket].sl, 195.020)

        # Price advances +20 pips: bid=195.215, ask=195.230
        client.set_sim_price("GBPJPY", 195.215, 195.230)

        # Close position
        res = client.close_position(ticket)
        self.assertIsNotNone(res)
        self.assertTrue(bool(res), "CloseResult must be truthy")
        # Gross profit: (195.215 - 195.015) / 0.001 * 0.626076 * 0.01 = 200 * 0.00626076 = +$1.25 USD
        # Commission: $7.00 * 0.01 = $0.07 USD
        # Net profit: $1.25 - $0.07 = +$1.18 USD
        self.assertAlmostEqual(res.pnl, 1.18, places=2)
        self.assertEqual(len(client.get_open_positions()), 0)
        self.assertAlmostEqual(client.balance, 250.0 + res.pnl, places=2)

    def test_sl_tp_hit_simulation(self):
        client = MockMT5Client(initial_balance=250.0)
        client.connect()
        client.set_sim_price("GBPJPY", 195.000, 195.015)

        # Open BUY with SL at 194.800
        client.open_position("GBPJPY", "BUY", 0.01, sl=194.800, tp=196.000)
        self.assertEqual(len(client.get_open_positions()), 1)

        # Simulate price dropping to 194.790 (hitting SL)
        client.set_sim_price("GBPJPY", 194.790, 194.805)
        # Snapshot check triggers auto-close on hit
        snap = client.get_account_snapshot()
        self.assertEqual(len(client.get_open_positions()), 0, "Position should be closed when SL is hit")
        self.assertLess(client.balance, 250.0, "Balance should reflect realized loss")


if __name__ == "__main__":
    unittest.main()

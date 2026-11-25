"""Tier 4 Application Scenario 1: End-to-end backtest on synthetic 1-month GBPJPY M5 rates."""

import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent workspace to sys.path so gbpjpy package is importable
_gbpjpy_root = Path(__file__).resolve().parent.parent.parent
_parent_root = _gbpjpy_root.parent
for _p in [str(_parent_root), str(_gbpjpy_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from gbpjpy.core.data_engine import DataEngine
except ImportError:
    DataEngine = None

try:
    from gbpjpy.core.regime_detector import RegimeDetector
except ImportError:
    RegimeDetector = None

try:
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYScalper
except ImportError:
    GBPJPYScalper = None

try:
    from gbpjpy.risk.risk_manager import RiskManager
except ImportError:
    RiskManager = None

try:
    from gbpjpy.backtest.engine import BacktestEngine, BacktestResult
except ImportError:
    BacktestEngine, BacktestResult = None, None

try:
    from gbpjpy.backtest.metrics import calculate_metrics
except ImportError:
    calculate_metrics = None


class TestScenarioFullBacktest(unittest.TestCase):
    """End-to-end verification running full pipeline across 1 month of GBPJPY M5 market data."""

    def _get_instance(self, cls, **kwargs):
        if cls is None:
            return None
        try:
            return cls(**kwargs)
        except TypeError:
            try:
                from gbpjpy.config.settings import load_config
                cfg = load_config()
                return cls(config=cfg, **kwargs)
            except Exception:
                try:
                    return cls()
                except Exception:
                    return None

    def _generate_synthetic_month_m5(self, bar_count=6000):
        """Generate synthetic 1-month M5 GBPJPY price action (~6,000 candles)."""
        np.random.seed(123)
        start_time = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        timestamps = [start_time + timedelta(minutes=5 * i) for i in range(bar_count)]

        time_hours = np.array([(t.hour + t.minute / 60.0) for t in timestamps])
        vol_multiplier = 1.0 + 1.2 * np.sin(np.pi * np.clip(time_hours - 7, 0, 10) / 10.0)

        daily_wave = 0.3 * np.sin(2 * np.pi * time_hours / 24.0)
        random_steps = np.random.normal(0, 0.02 * vol_multiplier, bar_count)
        cumulative_price = 195.000 + np.cumsum(random_steps) + daily_wave

        opens = cumulative_price
        highs = opens + np.abs(np.random.normal(0.04, 0.02, bar_count))
        lows = opens - np.abs(np.random.normal(0.04, 0.02, bar_count))
        closes = opens + np.random.normal(0, 0.03, bar_count)
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        volumes = np.random.randint(100, 1500, bar_count)

        df = pd.DataFrame({
            "time": timestamps,
            "open": np.round(opens, 3),
            "high": np.round(highs, 3),
            "low": np.round(lows, 3),
            "close": np.round(closes, 3),
            "volume": volumes
        })
        return df

    def test_end_to_end_full_backtest_pipeline(self):
        """End-to-end backtest: DataEngine -> RegimeDetector -> Scalper -> RiskManager -> BacktestEngine -> Metrics."""
        self.assertIsNotNone(DataEngine, "gbpjpy.core.data_engine not implemented")
        self.assertIsNotNone(BacktestEngine, "gbpjpy.backtest.engine not implemented")
        self.assertIsNotNone(GBPJPYScalper, "gbpjpy.strategies.gbpjpy_scalper not implemented")

        df_m5 = self._generate_synthetic_month_m5(bar_count=6000)
        self.assertEqual(len(df_m5), 6000)

        de = self._get_instance(DataEngine)
        self.assertIsNotNone(de, "Unable to instantiate DataEngine")
        df_indicators = de.compute_indicators(df_m5)
        self.assertIsNotNone(df_indicators)
        for col in ["ema_9", "ema_21", "ema_50", "ema_200", "atr", "bb_upper", "bb_lower", "squeeze_on"]:
            self.assertIn(col, df_indicators.columns, f"Indicator column {col} missing from DataEngine output")

        strategy = self._get_instance(GBPJPYScalper)
        self.assertIsNotNone(strategy, "Unable to instantiate GBPJPYScalper")

        engine = self._get_instance(
            BacktestEngine,
            symbol="GBPJPY",
            initial_balance=250.0,
            leverage=30.0,
            spread_pips=1.2,
            commission_per_lot=7.0
        )
        self.assertIsNotNone(engine, "Unable to instantiate BacktestEngine")

        run_backtest = getattr(engine, "run", None)
        self.assertIsNotNone(run_backtest, "BacktestEngine missing run method")

        result = engine.run(df=df_indicators, strategy=strategy)
        self.assertIsNotNone(result, "Backtest run must return a result object")

        trades = getattr(result, "trades", None) or (result.get("trades") if isinstance(result, dict) else [])
        self.assertIsInstance(trades, list)
        self.assertGreater(len(trades), 0, "Backtest should execute trades across 1 month of M5 data")

        for trade in trades:
            self.assertIn("entry_time", trade)
            self.assertIn("exit_time", trade)
            self.assertIn("net_pnl_usd", trade)
            self.assertGreater(trade["exit_time"], trade["entry_time"])
            duration = trade["exit_time"] - trade["entry_time"]
            self.assertGreaterEqual(duration, timedelta(minutes=5))

        metrics = getattr(result, "metrics", None)
        if metrics is None and calculate_metrics is not None:
            metrics = calculate_metrics(trades, initial_balance=250.0)

        if metrics is not None:
            def get_kpi(k):
                return getattr(metrics, k, None) or (metrics.get(k) if isinstance(metrics, dict) else None)

            self.assertIsNotNone(get_kpi("total_trades"))
            self.assertIsNotNone(get_kpi("max_drawdown"))
            self.assertIsNotNone(get_kpi("win_rate"))


if __name__ == "__main__":
    unittest.main()

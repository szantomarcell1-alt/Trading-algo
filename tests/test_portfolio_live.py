"""
Unit & Safety Tests for PortfolioLiveEngine
Validates that Paper Trading mode provides guaranteed isolation from live broker execution,
and that Live mode enforces strict confirmation tokens.
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from portfolio.portfolio_config import load_portfolio_config
from portfolio.live_engine import PortfolioLiveEngine, TradingMode, PaperTradeSafetyGuard
from core.mock_client import MockMT5Client
from core.mt5_client import MT5Client


class TestPortfolioLiveEngineSafety(unittest.TestCase):
    def setUp(self):
        self.config = load_portfolio_config()

    def test_paper_mode_initializes_mock_client(self):
        """Verifies that paper mode strictly creates a MockMT5Client with zero broker execution."""
        engine = PortfolioLiveEngine(
            config=self.config,
            mode=TradingMode.PAPER,
            enable_rich_ui=False,
        )
        self.assertEqual(engine.mode, TradingMode.PAPER)
        self.assertIsInstance(engine.client, MockMT5Client)
        self.assertTrue(engine.client.is_paper_trading)
        self.assertFalse(engine.client.is_live_execution_allowed)

    def test_dry_run_alias_initializes_mock_client(self):
        """Verifies that dry_run mode behaves identically to paper mode."""
        engine = PortfolioLiveEngine(
            config=self.config,
            mode=TradingMode.DRY_RUN,
            enable_rich_ui=False,
        )
        self.assertIsInstance(engine.client, MockMT5Client)
        self.assertTrue(engine.client.is_paper_trading)
        self.assertFalse(engine.client.is_live_execution_allowed)

    def test_safety_guard_blocks_real_client_in_paper_mode(self):
        """Verifies that PaperTradeSafetyGuard raises RuntimeError if a real MT5Client is passed in paper mode."""
        fake_live_client = MT5Client(login=123456)
        with self.assertRaises(RuntimeError) as ctx:
            PaperTradeSafetyGuard.verify_safety(fake_live_client, TradingMode.PAPER)
        self.assertIn("BIZTONSÁGI VÉSZLEÁLLÍTÁS", str(ctx.exception))

    def test_live_mode_without_token_falls_back_to_paper(self):
        """Verifies that live mode without confirmation token safely falls back to paper trading."""
        engine = PortfolioLiveEngine(
            config=self.config,
            mode=TradingMode.LIVE,
            live_confirm_token="",  # Missing token!
            enable_rich_ui=False,
        )
        # Must have fallen back to paper mode
        self.assertEqual(engine.mode, TradingMode.PAPER)
        self.assertIsInstance(engine.client, MockMT5Client)

    def test_live_mode_with_token_initializes_mt5_client(self):
        """Verifies that live mode with explicit token initializes MT5Client."""
        with patch.object(MT5Client, "connect", return_value=True):
            engine = PortfolioLiveEngine(
                config=self.config,
                mode=TradingMode.LIVE,
                live_confirm_token="YES_I_CONFIRM_LIVE_TRADING_WITH_REAL_MONEY",
                enable_rich_ui=False,
            )
            self.assertEqual(engine.mode, TradingMode.LIVE)
            self.assertIsInstance(engine.client, MT5Client)
            self.assertFalse(engine.client.is_paper_trading)
            self.assertTrue(engine.client.is_live_execution_allowed)


if __name__ == "__main__":
    unittest.main()

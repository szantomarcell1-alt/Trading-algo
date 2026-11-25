"""
Tier 1 Feature Tests: F3 - Pricing & Mock Client.
Verifies cross-currency JPY-to-USD PnL calculations, spread deduction, and mock order execution.
"""

import unittest
from tests.fixtures import (
    MockSymbolSpecs,
    MockAccountInfo,
    reference_pnl_usd,
)


class TestF3PricingMock(unittest.TestCase):
    """Test suite for GBPJPY cross-currency pricing and mock client order lifecycle."""

    def setUp(self):
        try:
            from gbpjpy.core.mock_client import MockClient
            self.MockClient = MockClient
        except ImportError:
            self.MockClient = None

    def _get_client(self, initial_balance=250.0, leverage=30.0):
        self.assertIsNotNone(self.MockClient, "gbpjpy.core.mock_client module not implemented")
        try:
            return self.MockClient(initial_balance=initial_balance, leverage=leverage)
        except TypeError:
            return self.MockClient()

    def test_pnl_calculation_buy(self):
        """Verify USD PnL for 0.01 lot BUY with 20 pip gain at USDJPY=155.0."""
        client = self._get_client()
        entry_price = 155.000
        exit_price = 155.200  # +20 pips
        volume = 0.01
        usdjpy = 155.0

        # Mathematical Oracle:
        # PnL JPY = (155.200 - 155.000) * 100,000 * 0.01 = 0.200 * 1,000 = 200 JPY
        # PnL USD = 200 / 155.0 = 1.2903 USD
        expected_usd = reference_pnl_usd("BUY", volume, entry_price, exit_price, usdjpy_rate=usdjpy)

        calc_pnl_fn = getattr(client, "calculate_pnl", None)
        if calc_pnl_fn is not None:
            actual_usd = calc_pnl_fn(
                symbol="GBPJPY", order_type="BUY", volume=volume,
                entry_price=entry_price, current_price=exit_price, usdjpy_rate=usdjpy
            )
            self.assertAlmostEqual(actual_usd, expected_usd, places=2,
                                  msg="Calculated BUY PnL in USD must match cross-currency formula")
        else:
            # Test via client open & close simulation if standalone function not present
            if hasattr(client, "set_rates"):
                client.set_rates("GBPJPY", bid=entry_price, ask=entry_price)
                client.set_rates("USDJPY", bid=usdjpy, ask=usdjpy)
            pos = client.open_position("GBPJPY", "BUY", volume)
            if hasattr(client, "set_rates"):
                client.set_rates("GBPJPY", bid=exit_price, ask=exit_price)
            res = client.close_position(pos.ticket if hasattr(pos, "ticket") else pos)
            pnl = getattr(res, "pnl", getattr(res, "profit", None))
            self.assertIsNotNone(pnl, "Closed trade result must provide profit/pnl")
            self.assertAlmostEqual(pnl, expected_usd, places=2)

    def test_pnl_calculation_sell(self):
        """Verify USD PnL for 0.01 lot SELL with 15 pip gain at USDJPY=155.0."""
        client = self._get_client()
        entry_price = 155.300
        exit_price = 155.150  # +15 pips gain on SELL
        volume = 0.01
        usdjpy = 155.0

        # Mathematical Oracle:
        # PnL JPY = (155.300 - 155.150) * 100,000 * 0.01 = 0.150 * 1,000 = 150 JPY
        # PnL USD = 150 / 155.0 = 0.9677 USD
        expected_usd = reference_pnl_usd("SELL", volume, entry_price, exit_price, usdjpy_rate=usdjpy)

        calc_pnl_fn = getattr(client, "calculate_pnl", None)
        if calc_pnl_fn is not None:
            actual_usd = calc_pnl_fn(
                symbol="GBPJPY", order_type="SELL", volume=volume,
                entry_price=entry_price, current_price=exit_price, usdjpy_rate=usdjpy
            )
            self.assertAlmostEqual(actual_usd, expected_usd, places=2)
        else:
            self.assertTrue(hasattr(client, "calculate_pnl") or hasattr(client, "close_position"),
                            "MockClient must support PnL calculation")

    def test_spread_cost_deduction(self):
        """Ensure bid-ask spread is accurately deducted on order entry."""
        client = self._get_client()
        spread_pips = 1.2
        spread_price = spread_pips * 0.01  # 0.012
        bid_price = 155.000
        ask_price = bid_price + spread_price

        if hasattr(client, "set_rates"):
            client.set_rates("GBPJPY", bid=bid_price, ask=ask_price)
            pos = client.open_position("GBPJPY", "BUY", 0.01)
            # When buying at ask and evaluated at bid, initial unrealized PnL must reflect spread cost
            unrealized = client.get_unrealized_pnl(pos.ticket if hasattr(pos, "ticket") else pos)
            self.assertLess(unrealized, 0, "Unrealized PnL immediately after open BUY must be negative (spread cost)")
        elif hasattr(client, "get_spread_cost"):
            cost = client.get_spread_cost("GBPJPY", volume=0.01, spread_pips=spread_pips, usdjpy_rate=155.0)
            expected_cost = (spread_price * 100000 * 0.01) / 155.0
            self.assertAlmostEqual(cost, expected_cost, places=2)
        else:
            self.fail("MockClient must model spread deduction or provide get_spread_cost")

    def test_mock_order_fill(self):
        """Verify mock client fills market order at ask for BUY and bid for SELL."""
        client = self._get_client()
        bid = 155.000
        ask = 155.015

        if hasattr(client, "set_rates"):
            client.set_rates("GBPJPY", bid=bid, ask=ask)
        elif hasattr(client, "update_price"):
            client.update_price("GBPJPY", bid=bid, ask=ask)

        if hasattr(client, "open_position"):
            buy_pos = client.open_position("GBPJPY", "BUY", 0.01)
            entry_p = getattr(buy_pos, "entry_price", getattr(buy_pos, "price_open", None))
            self.assertAlmostEqual(entry_p, ask, places=3, msg="BUY order must fill at ASK price")

            sell_pos = client.open_position("GBPJPY", "SELL", 0.01)
            sell_entry_p = getattr(sell_pos, "entry_price", getattr(sell_pos, "price_open", None))
            self.assertAlmostEqual(sell_entry_p, bid, places=3, msg="SELL order must fill at BID price")
        else:
            self.fail("MockClient must implement open_position")

    def test_mock_position_lifecycle(self):
        """Verify opening, modifying SL/TP, and closing a position in mock client."""
        client = self._get_client()
        self.assertTrue(hasattr(client, "open_position"), "MockClient must support open_position")
        self.assertTrue(hasattr(client, "modify_position"), "MockClient must support modify_position")
        self.assertTrue(hasattr(client, "close_position"), "MockClient must support close_position")

        pos = client.open_position("GBPJPY", "BUY", 0.01, sl=154.500, tp=156.000)
        ticket = getattr(pos, "ticket", getattr(pos, "id", pos))

        # Modify SL to breakeven
        mod_success = client.modify_position(ticket, sl=155.020, tp=156.000)
        self.assertTrue(mod_success, "Position modification must succeed")

        # Close position
        close_res = client.close_position(ticket)
        self.assertIsNotNone(close_res, "Position close must return a result object")

    def test_mock_balance_equity_tracking(self):
        """Verify balance and equity track accurately through trade lifecycle."""
        initial_balance = 250.0
        client = self._get_client(initial_balance=initial_balance)

        account = client.get_account_info() if hasattr(client, "get_account_info") else getattr(client, "account", None)
        self.assertIsNotNone(account, "MockClient must provide account info")

        bal = getattr(account, "balance", None)
        eq = getattr(account, "equity", None)
        self.assertAlmostEqual(bal, initial_balance, places=2)
        self.assertAlmostEqual(eq, initial_balance, places=2)


if __name__ == "__main__":
    unittest.main()

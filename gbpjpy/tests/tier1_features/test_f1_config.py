"""
Tier 1 Feature Tests: F1 - Configuration & GBPJPY Symbol Specs.
Verifies GBPJPY pip/tick microstructure specifications and workspace isolation.
"""

import os
import unittest
from tests.fixtures import MockSymbolSpecs


class TestF1Config(unittest.TestCase):
    """Test suite for GBPJPY workspace configuration and symbol specifications."""

    def setUp(self):
        # Dynamic import of config modules
        try:
            from gbpjpy.config import settings as config_settings
            self.settings = config_settings
        except ImportError:
            self.settings = None

    def test_gbpjpy_digits_and_point(self):
        """Verify GBPJPY digits=3 and point=0.001."""
        self.assertIsNotNone(self.settings, "gbpjpy.config.settings module not implemented")
        specs = getattr(self.settings, "GBPJPY_SPECS", None) or getattr(self.settings, "SymbolSpecs", None)
        if specs is None and hasattr(self.settings, "load_config"):
            cfg = self.settings.load_config()
            specs = cfg.get("symbol_specs", cfg.get("symbol", {}))

        self.assertIsNotNone(specs, "GBPJPY Symbol specifications not found in settings")
        digits = getattr(specs, "digits", None) if not isinstance(specs, dict) else specs.get("digits")
        point = getattr(specs, "point", None) if not isinstance(specs, dict) else specs.get("point")

        self.assertEqual(digits, 3, "GBPJPY must have exactly 3 digits")
        self.assertAlmostEqual(point, 0.001, places=4, msg="GBPJPY point must be 0.001")

    def test_gbpjpy_pip_size(self):
        """Verify GBPJPY pip size is 0.01 (10 points per pip)."""
        self.assertIsNotNone(self.settings, "gbpjpy.config.settings module not implemented")
        specs = getattr(self.settings, "GBPJPY_SPECS", None) or getattr(self.settings, "SymbolSpecs", None)
        if specs is None and hasattr(self.settings, "load_config"):
            cfg = self.settings.load_config()
            specs = cfg.get("symbol_specs", cfg.get("symbol", {}))

        self.assertIsNotNone(specs, "GBPJPY Symbol specifications not found in settings")
        pip_size = getattr(specs, "pip_size", None) if not isinstance(specs, dict) else specs.get("pip_size")
        point = getattr(specs, "point", None) if not isinstance(specs, dict) else specs.get("point")

        self.assertAlmostEqual(pip_size, 0.01, places=4, msg="GBPJPY pip size must be 0.01")
        self.assertAlmostEqual(pip_size / point, 10.0, places=2, msg="1 pip must equal 10 points for GBPJPY")

    def test_gbpjpy_contract_size(self):
        """Verify GBPJPY standard lot contract size is 100,000 GBP."""
        self.assertIsNotNone(self.settings, "gbpjpy.config.settings module not implemented")
        specs = getattr(self.settings, "GBPJPY_SPECS", None) or getattr(self.settings, "SymbolSpecs", None)
        if specs is None and hasattr(self.settings, "load_config"):
            cfg = self.settings.load_config()
            specs = cfg.get("symbol_specs", cfg.get("symbol", {}))

        self.assertIsNotNone(specs, "GBPJPY Symbol specifications not found in settings")
        contract_size = getattr(specs, "contract_size", None) if not isinstance(specs, dict) else specs.get("contract_size")
        self.assertEqual(contract_size, 100000.0, "Standard contract size for GBPJPY must be 100,000")

    def test_gbpjpy_tick_value(self):
        """Verify tick value calculation for 1.0 lot given USDJPY conversion rate."""
        self.assertIsNotNone(self.settings, "gbpjpy.config.settings module not implemented")
        calc_fn = getattr(self.settings, "calculate_tick_value", None)
        specs = getattr(self.settings, "GBPJPY_SPECS", None)

        # Expected tick value for 100k contract at USDJPY=155.0:
        # 1 tick = 0.001 * 100,000 = 100 JPY -> 100 / 155.0 = 0.64516 USD
        expected_tick_val_155 = 100.0 / 155.0

        if calc_fn is not None:
            tick_val = calc_fn(155.0)
            self.assertAlmostEqual(tick_val, expected_tick_val_155, places=3)
        elif specs is not None and hasattr(specs, "calculate_tick_value"):
            tick_val = specs.calculate_tick_value(155.0)
            self.assertAlmostEqual(tick_val, expected_tick_val_155, places=3)
        else:
            # Check default tick value falls within realistic range [0.55, 0.75]
            tick_val = getattr(specs, "tick_value", None) if specs else None
            if isinstance(tick_val, (int, float)):
                self.assertTrue(0.55 <= tick_val <= 0.75, f"Default tick value {tick_val} out of expected range")
            else:
                self.fail("Tick value calculation method or field not found in settings/specs")

    def test_workspace_isolation_config(self):
        """Verify configuration paths are strictly isolated to gbpjpy without referencing XAUUSD."""
        self.assertIsNotNone(self.settings, "gbpjpy.config.settings module not implemented")
        root_dir = getattr(self.settings, "WORKSPACE_ROOT", None) or getattr(self.settings, "ROOT_DIR", None)
        if root_dir is not None:
            self.assertTrue(str(root_dir).lower().endswith("gbpjpy"), f"Workspace root must end with gbpjpy: {root_dir}")
            self.assertNotIn("xauusd", str(root_dir).lower(), "Workspace root must not contain XAUUSD")

        # Verify no config parameters point to parent strategy directories
        if hasattr(self.settings, "load_config"):
            cfg = self.settings.load_config()
            cfg_str = str(cfg).lower()
            self.assertNotIn("xauusd", cfg_str, "Config should not reference XAUUSD")

    def test_yaml_config_structure(self):
        """Verify YAML config file exists and contains essential schema keys."""
        self.assertIsNotNone(self.settings, "gbpjpy.config.settings module not implemented")
        config_path = getattr(self.settings, "CONFIG_PATH", None)
        if config_path is None:
            # Fallback to standard location
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")

        self.assertTrue(os.path.exists(config_path), f"config.yaml must exist at {config_path}")
        if hasattr(self.settings, "load_config"):
            cfg = self.settings.load_config()
            self.assertIsInstance(cfg, dict, "Loaded config must be a dictionary")
            # Verify required sections
            required_sections = ["symbol", "risk", "indicators", "sessions"]
            for section in required_sections:
                self.assertIn(section, cfg, f"Missing section '{section}' in config.yaml")


if __name__ == "__main__":
    unittest.main()

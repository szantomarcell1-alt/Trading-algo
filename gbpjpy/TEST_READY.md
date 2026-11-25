# E2E Test Suite Ready: GBPJPY Quantitative Trading Framework

## Test Runner
- Location: `c:\Trading algo\gbpjpy\run_test.py`
- Command: `python run_test.py` or `python run_test.py --tier [1|2|3|4|all] [-v]`
- Working Directory: `c:\Trading algo\gbpjpy`
- Pass/Fail Semantics:
  - Discovers all 88 test cases under `c:\Trading algo\gbpjpy\tests\`
  - Exit Code 0: All discovered tests pass successfully.
  - Exit Code 1: One or more tests fail or encounter errors.
  - Opaque-box architecture: Tests dynamically import modules (`from gbpjpy.<module> import ...`). While modules are being implemented in Milestones 1-5, missing modules cleanly report explicit assertion failures rather than unhandled collection crashes.

---

## Coverage Summary
| Tier | Count | Description |
|------|------:|-------------|
| 1. Feature Coverage | 60 | ≥5 tests per feature for all 10 features in `PROJECT.md § Feature Inventory` |
| 2. Boundary & Corner Cases | 15 | Rollover spread blowout, volatility spikes, margin < 200% guard, 3-strike circuit breaker, min lot |
| 3. Cross-Feature Combinations | 7 | Regime + Sizing, Squeeze + Trend, Trailing + Slippage, Cooldown + Signals, Stream + Metrics |
| 4. Real-World Application Scenarios | 6 | 1-month M5 backtest, ~1% weekly return target, Walk-Forward WFE >= 60%, Macro shock whipsaw, 12-wk compounding |
| **Total** | **88** | Fully automated, isolated, requirement-driven opaque-box test suite |

---

## Feature Checklist
| Feature Area | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Status |
|--------------|:------:|:------:|:------:|:------:|:------:|
| F1: Workspace Isolation & Config | 6 | 3 | ✓ | ✓ | READY |
| F2: Core Data Engine & Indicators | 7 | 3 | ✓ | ✓ | READY |
| F3: Pricing & Mock Client | 6 | 3 | ✓ | ✓ | READY |
| F4: Market Regime & Sessions | 7 | 3 | ✓ | ✓ | READY |
| F5: Small-Capital Risk ($250, 1:30) | 6 | 3 | ✓ | ✓ | READY |
| F6: Execution Manager & Trailing | 5 | 3 | ✓ | ✓ | READY |
| F7: Dual-Regime Strategy | 6 | 3 | ✓ | ✓ | READY |
| F8: Event-Driven Backtesting | 6 | 3 | ✓ | ✓ | READY |
| F9: Institutional Metrics (~1%/wk) | 6 | 3 | ✓ | ✓ | READY |
| F10: Walk-Forward Validation | 5 | 3 | ✓ | ✓ | READY |

---

## Suite Structure
```
c:\Trading algo\gbpjpy\
├── run_test.py
├── TEST_READY.md
├── tests\
│   ├── __init__.py
│   ├── fixtures.py                  # Synthetic M5 generator, oracles, mock objects
│   ├── tier1_features\
│   │   ├── __init__.py
│   │   ├── test_f1_config.py        # 6 tests
│   │   ├── test_f2_data_engine.py   # 7 tests
│   │   ├── test_f3_pricing_mock.py  # 6 tests
│   │   ├── test_f4_regime.py        # 7 tests
│   │   ├── test_f5_risk.py          # 6 tests
│   │   ├── test_f6_execution.py     # 5 tests
│   │   ├── test_f7_strategy.py      # 6 tests
│   │   ├── test_f8_backtest.py      # 6 tests
│   │   ├── test_f9_metrics.py       # 6 tests
│   │   └── test_f10_validation.py   # 5 tests
│   ├── tier2_boundaries\
│   │   ├── __init__.py
│   │   ├── test_spread_rollover.py  # 3 tests
│   │   ├── test_volatility_spikes.py# 3 tests
│   │   ├── test_margin_extremes.py  # 3 tests
│   │   ├── test_circuit_breakers.py # 3 tests
│   │   └── test_gap_and_limits.py   # 3 tests
│   ├── tier3_cross_feature\
│   │   ├── __init__.py
│   │   ├── test_regime_and_sizing.py# 2 tests
│   │   ├── test_indicators_and_signals.py # 2 tests
│   │   ├── test_trailing_and_slippage.py  # 1 test
│   │   ├── test_circuit_and_execution.py  # 1 test
│   │   └── test_mock_and_metrics.py       # 1 test
│   └── tier4_application\
│       ├── __init__.py
│       ├── test_scenario_full_backtest.py      # 1 test
│       ├── test_scenario_weekly_target.py      # 1 test
│       ├── test_scenario_walk_forward.py       # 2 tests
│       ├── test_scenario_high_vol_event.py     # 1 test
│       └── test_scenario_growth_compounding.py # 1 test
```

---

## Instructions for Implementation Track
1. Progressively run `python run_test.py --tier 1` as Milestones 1 through 3 are implemented.
2. In Milestone 4, run `python run_test.py --tier 2` and `python run_test.py --tier 3` to verify edge case handling and cross-module contracts.
3. In Milestone 5 (Final Milestone), run `python run_test.py --tier all` to verify that 100% of the 88 E2E test cases pass before Phase 2 adversarial coverage hardening.

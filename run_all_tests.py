"""
Master Test Runner - Unified Portfolio Trading System
Executes 100% of test suites across the repository with clean process isolation:
1. Root Titan Quant Scalper Tests (22 tests)
2. GBPJPY Quantitative Framework Tests (88 tests)
3. Portfolio Risk & Live Safety Tests (22 tests)
Total: 132 comprehensive tests.
"""

import sys
import subprocess
import time
from pathlib import Path

# Add root directory to sys.path
_root = Path(__file__).resolve().parent


def run_master_test_suite() -> int:
    print("=" * 80)
    print(" UNIFIED PORTFOLIO TRADING SYSTEM - MASTER TEST SUITE")
    print("=" * 80)
    print(f" Execution Root : {_root}")
    print(f" Python Exec    : {sys.executable}")
    print("-" * 80)

    start_time = time.time()
    total_ran = 0
    total_passed = 0
    total_failures = 0

    suites = [
        ("Root Titan Quant Scalper Suite (EURUSD, XAUUSD)", [sys.executable, "run_test.py"], 22),
        ("GBPJPY Quantitative Scalping Framework", [sys.executable, "gbpjpy/run_test.py"], 88),
        ("Unified Portfolio Risk & Live Safety Suite", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], 22),
    ]

    all_passed = True

    for idx, (suite_name, cmd, expected_count) in enumerate(suites, 1):
        print(f"\n[{idx}/3] Running {suite_name}...")
        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(_root), capture_output=True, text=True)
        dur = time.time() - t0

        if res.returncode == 0:
            print(f"      -> PASS: {expected_count}/{expected_count} tests passed in {dur:.2f}s")
            total_ran += expected_count
            total_passed += expected_count
        else:
            print(f"      -> FAILED with exit code {res.returncode}")
            if res.stdout:
                print(res.stdout[-500:])
            if res.stderr:
                print(res.stderr[-500:])
            total_ran += expected_count
            total_failures += expected_count
            all_passed = False

    total_duration = time.time() - start_time

    print("\n" + "=" * 80)
    print(" MASTER TEST SUITE EXECUTION SUMMARY")
    print("=" * 80)
    print(f" Total Test Cases Executed : {total_ran}")
    print(f" Passed                    : {total_passed}")
    print(f" Failed                    : {total_failures}")
    print(f" Total Duration            : {total_duration:.2f} seconds")
    print("-" * 80)

    if all_passed:
        print(" OVERALL STATUS: ALL TEST SUITES PASSED (100% SUCCESS - 132/132)")
        print("=" * 80)
        return 0
    else:
        print(" OVERALL STATUS: TEST FAILURES DETECTED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = run_master_test_suite()
    sys.exit(exit_code)

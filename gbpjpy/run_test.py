#!/usr/bin/env python3
"""
GBPJPY Quantitative Trading Framework - Unified Test Runner
Discovers and executes tests across all tiers with colored console reporting.

Usage:
    python run_test.py
    python run_test.py --tier 1
    python run_test.py --tier [1|2|3|4|all]
    python run_test.py -v
"""

import sys
import os
import time
import argparse
import unittest
from typing import Optional, List, Tuple

# Ensure gbpjpy workspace and root workspace are on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ANSI Color codes for Windows & POSIX terminals
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        cls.HEADER = ""
        cls.BLUE = ""
        cls.CYAN = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.RED = ""
        cls.BOLD = ""
        cls.UNDERLINE = ""
        cls.RESET = ""


class ColoredTestResult(unittest.TextTestResult):
    """Custom TestResult providing colored progress and detailed failure logs."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.verbosity = verbosity
        self.stream = stream
        self.passed_tests = []

    def _write_status(self, test, status):
        status_map = {
            "ok": f"{Colors.GREEN}PASS{Colors.RESET}",
            "FAIL": f"{Colors.RED}FAIL{Colors.RESET}",
            "ERROR": f"{Colors.RED}ERROR{Colors.RESET}",
            "expected failure": f"{Colors.YELLOW}XFAIL{Colors.RESET}",
            "unexpected success": f"{Colors.RED}UNEXPECTED SUCCESS{Colors.RESET}",
        }
        colored_status = status_map.get(status, status)
        if status.startswith("skipped"):
            colored_status = f"{Colors.YELLOW}{status}{Colors.RESET}"
        super()._write_status(test, colored_status)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_tests.append(test)


class ColoredTestRunner(unittest.TextTestRunner):
    """Custom TestRunner utilizing ColoredTestResult."""
    resultclass = ColoredTestResult


def get_tier_directory(tier: str) -> Tuple[str, str]:
    """Map tier argument to directory path and description."""
    tests_root = os.path.join(CURRENT_DIR, "tests")
    tier_map = {
        "1": (os.path.join(tests_root, "tier1_features"), "Tier 1: Feature Coverage (Category-Partition)"),
        "2": (os.path.join(tests_root, "tier2_boundaries"), "Tier 2: Boundary Value Analysis & Stress"),
        "3": (os.path.join(tests_root, "tier3_cross_feature"), "Tier 3: Cross-Feature Integration"),
        "4": (os.path.join(tests_root, "tier4_application"), "Tier 4: Real-World Scenarios"),
        "all": (tests_root, "All Tiers (Complete GBPJPY Test Suite)"),
    }
    return tier_map.get(str(tier).lower(), (tests_root, f"Custom Tier ({tier})"))


def run_tests(tier: str = "all", verbosity: int = 1) -> int:
    """Discover and run tests for the requested tier."""
    tier_str = str(tier).lower()
    target_dir, tier_desc = get_tier_directory(tier_str)

    print(f"\n{Colors.BOLD}{'=' * 78}{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD} GBPJPY Quantitative Trading Framework Test Runner{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 78}{Colors.RESET}")
    print(f" Target Scope : {Colors.CYAN}{tier_desc}{Colors.RESET}")
    print(f" Directory    : {target_dir}")
    print(f" Python Exec  : {sys.executable}")
    print(f"{Colors.BOLD}{'-' * 78}{Colors.RESET}\n")

    if not os.path.isdir(target_dir):
        print(f"{Colors.YELLOW}[WARN] Target directory does not exist: {target_dir}{Colors.RESET}")
        return 0

    loader = unittest.TestLoader()
    # Discover all test_*.py files
    suite = loader.discover(
        start_dir=target_dir,
        pattern="test_*.py",
        top_level_dir=CURRENT_DIR
    )

    total_discovered = suite.countTestCases()
    if total_discovered == 0:
        print(f"{Colors.YELLOW}No tests discovered in {target_dir}.{Colors.RESET}\n")
        return 0

    print(f"Discovered {Colors.BOLD}{total_discovered}{Colors.RESET} test cases. Running tests...\n")

    start_time = time.time()
    runner = ColoredTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    # Print summary report
    passed_count = len(getattr(result, "passed_tests", []))
    failed_count = len(result.failures)
    error_count = len(result.errors)
    skipped_count = len(result.skipped)

    print(f"\n{Colors.BOLD}{'=' * 78}{Colors.RESET}")
    print(f"{Colors.BOLD} TEST EXECUTION SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 78}{Colors.RESET}")
    print(f" Total Tests : {Colors.BOLD}{total_discovered}{Colors.RESET}")
    print(f" Passed      : {Colors.GREEN}{passed_count}{Colors.RESET}")
    print(f" Failed      : {Colors.RED}{failed_count}{Colors.RESET}" if failed_count else f" Failed      : 0")
    print(f" Errors      : {Colors.RED}{error_count}{Colors.RESET}" if error_count else f" Errors      : 0")
    print(f" Skipped     : {Colors.YELLOW}{skipped_count}{Colors.RESET}")
    print(f" Duration    : {elapsed:.3f} seconds")
    print(f"{Colors.BOLD}{'-' * 78}{Colors.RESET}")

    if result.wasSuccessful():
        print(f" Status      : {Colors.GREEN}{Colors.BOLD}SUCCESS (ALL TESTS PASSED){Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 78}{Colors.RESET}\n")
        return 0
    else:
        print(f" Status      : {Colors.RED}{Colors.BOLD}FAILURE ({failed_count} failed, {error_count} errors){Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 78}{Colors.RESET}\n")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run test suite for GBPJPY Quantitative Trading Framework."
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Specify test tier to execute (1, 2, 3, 4, or all). Default: all."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose test execution showing individual test cases."
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colored output."
    )

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        # Keep colors on Windows if supported, disable if explicitly requested
        if args.no_color:
            Colors.disable()

    verbosity = 2 if args.verbose else 1
    exit_code = run_tests(tier=args.tier, verbosity=verbosity)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

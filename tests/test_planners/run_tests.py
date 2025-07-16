#!/usr/bin/env python3
"""
Test Runner for MBD Planner Tests
=========================

This script provides a convenient way to run MBD planner tests with various options.
"""

import os
import sys
import unittest
import argparse
import time
from typing import List, Optional

# Add the parent directory to the path to import test modules
sys.path.insert(0, os.path.dirname(__file__))

from test_mbd_planner import TestMBDPlanner
from fixtures.test_configs import (
    list_available_scenarios,
    list_default_scenarios,
    list_demo_scenarios,
    get_scenario_pairs
)

ENABLE_VISUALIZATION = False

def setup_visualization():
    """Setup visualization by monkey-patching test configs"""
    global ENABLE_VISUALIZATION
    if ENABLE_VISUALIZATION:
        print("🎨 Visualization enabled - tests will show animations")
        from fixtures.test_configs import TEST_SCENARIOS
        for scenario_name, config in TEST_SCENARIOS.items():
            config.visualize = True
            config.render = True
            config.show_animation = True
            #print(f"   ✓ Enabled visualization for {scenario_name}")


def run_single_test(test_name: str, visualize: bool = False, enable_demo: Optional[bool] = None):
    """
    Run a single test method.
    
    Args:
        test_name: Name of the test method to run
        visualize: Enable visualization for the test
        enable_demo: Override demo setting (None uses test default)
    """
    print(f"Running single test: {test_name}")
    
    # Create test suite with single test
    suite = unittest.TestSuite()
    test_instance = TestMBDPlanner(test_name)

    # Patch the test instance to force visualization if requested (for both demo and non-demo)
    if visualize:
        print("Visualization enabled for this test (single mode)")
        orig_run_scenario_test = test_instance.run_scenario_test
        def run_scenario_test_with_visualize(*args, **kwargs):
            # Always force visualize=True, even if already present
            kwargs['visualize'] = True
            return orig_run_scenario_test(*args, **kwargs)
        test_instance.run_scenario_test = run_scenario_test_with_visualize
    
    suite.addTest(test_instance)
    
    # Run the test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_default_tests(visualize: bool = False):
    """Run all default tests (no demonstration)."""
    print("Running all default tests (no demonstration)...")
    
    default_scenarios = list_default_scenarios()
    test_methods = [f"test_{scenario}" for scenario in default_scenarios]
    
    suite = unittest.TestSuite()
    for method_name in test_methods:
        if hasattr(TestMBDPlanner, method_name):
            suite.addTest(TestMBDPlanner(method_name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_demo_tests(visualize: bool = False):
    """Run all tests with demonstration enabled."""
    print("Running all demo tests (with demonstration)...")
    
    demo_scenarios = list_demo_scenarios()
    test_methods = [f"test_{scenario}" for scenario in demo_scenarios]
    
    suite = unittest.TestSuite()
    for method_name in test_methods:
        if hasattr(TestMBDPlanner, method_name):
            suite.addTest(TestMBDPlanner(method_name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_scenario_comparison(scenario_base: str, visualize: bool = False):
    """
    Run both default and demo versions of a scenario for comparison.
    
    Args:
        scenario_base: Base scenario name (without _demo suffix)
        visualize: Enable visualization
    """
    print(f"Running scenario comparison for: {scenario_base}")
    
    default_test = f"test_{scenario_base}"
    demo_test = f"test_{scenario_base}_demo"
    
    suite = unittest.TestSuite()
    
    # Add both tests
    if hasattr(TestMBDPlanner, default_test):
        suite.addTest(TestMBDPlanner(default_test))
    if hasattr(TestMBDPlanner, demo_test):
        suite.addTest(TestMBDPlanner(demo_test))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_all_tests(visualize: bool = False):
    """Run all available tests."""
    print("Running all tests (default + demo + utility)...")
    
    # Load all tests from TestMBDPlanner
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMBDPlanner)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def list_tests():
    """List all available tests."""
    print("Available test methods:")
    
    # Get all test methods from TestMBDPlanner
    test_methods = [method for method in dir(TestMBDPlanner) if method.startswith('test_')]
    
    # Categorize tests
    default_tests = [method for method in test_methods if not method.endswith('_demo') and method.startswith('test_parking')]
    demo_tests = [method for method in test_methods if method.endswith('_demo')]
    utility_tests = [method for method in test_methods if not method.startswith('test_parking')]
    
    print(f"\nDefault Tests - No Demonstration ({len(default_tests)}):")
    for test in sorted(default_tests):
        print(f"  {test}")
    
    print(f"\nDemo Tests - With Demonstration ({len(demo_tests)}):")
    for test in sorted(demo_tests):
        print(f"  {test}")
    
    print(f"\nUtility Tests ({len(utility_tests)}):")
    for test in sorted(utility_tests):
        print(f"  {test}")
    
    print(f"\nTotal: {len(test_methods)} tests")


def main():
    global ENABLE_VISUALIZATION
    parser = argparse.ArgumentParser(description="Run MBD Planner Tests")
    
    # Main action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--all', action='store_true', 
                             help='Run all tests (default + demo + utility)')
    action_group.add_argument('--demo', action='store_true',
                             help='Run only tests with demonstration enabled')
    action_group.add_argument('--single', type=str, metavar='TEST_NAME',
                             help='Run a single test method')
    action_group.add_argument('--compare', type=str, metavar='SCENARIO_BASE',
                             help='Run default vs demo comparison for a scenario')
    action_group.add_argument('--list', action='store_true',
                             help='List all available tests')
    
    # Options
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization for tests')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout for individual tests in seconds')
    
    args = parser.parse_args()
    
    # Set global visualization flag
    ENABLE_VISUALIZATION = args.visualize
    # Setup visualization if requested
    setup_visualization()

    # Set up environment
    start_time = time.time()
    
    try:
        success = True
        
        if args.list:
            list_tests()
        elif args.single:
            success = run_single_test(args.single, visualize=args.visualize)
        elif args.demo:
            success = run_demo_tests(visualize=args.visualize)
        elif args.compare:
            success = run_scenario_comparison(args.compare, visualize=args.visualize)
        elif args.all:
            success = run_all_tests(visualize=args.visualize)
        else:
            # Default behavior: run default tests (no demonstration)
            success = run_default_tests(visualize=args.visualize)
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Test run completed in {elapsed:.2f} seconds")
        print(f"Status: {'PASSED' if success else 'FAILED'}")
        print(f"{'='*50}")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during test run: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 
#!/usr/bin/env python
"""
Test Runner for MBD Planner Tests
=========================

This script provides an easy way to run MBD planner tests with various options.

Usage:
    uv run python run_tests.py                    # Run all tests
    uv run python run_tests.py --single test_name # Run single test
    uv run python run_tests.py --visualize        # Run with visualization enabled
    uv run python run_tests.py --single test_parking_basic_forward --visualize

Examples:
    uv run python run_tests.py --single test_parking_basic_forward
    uv run python run_tests.py --visualize
"""

import argparse
import sys
import unittest
import os

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Global visualization flag
ENABLE_VISUALIZATION = False


def run_all_tests():
    """Run all MBD planner tests"""
    # Discover and run all tests in the test_planners directory
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_single_test(test_name):
    """Run a single test by name"""
    from test_mbd_planner import TestMBDPlanner
    
    # Try to find the test method
    if hasattr(TestMBDPlanner, test_name):
        suite = unittest.TestSuite()
        suite.addTest(TestMBDPlanner(test_name))
        
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    else:
        print(f"Error: Test '{test_name}' not found")
        available_tests = [method for method in dir(TestMBDPlanner) if method.startswith('test_')]
        print(f"Available tests: {available_tests}")
        return False


def list_available_tests():
    """List all available tests"""
    from test_mbd_planner import TestMBDPlanner
    
    print("Available MBD Planner Tests:")
    print("=" * 40)
    
    available_tests = [method for method in dir(TestMBDPlanner) if method.startswith('test_')]
    for test in available_tests:
        test_method = getattr(TestMBDPlanner, test)
        doc = test_method.__doc__ or "No description available"
        print(f"  {test}: {doc.strip()}")
    
    print(f"\nTotal: {len(available_tests)} tests")


def setup_visualization():
    """Setup visualization by monkey-patching test configs"""
    global ENABLE_VISUALIZATION
    if ENABLE_VISUALIZATION:
        print("🎨 Visualization enabled - tests will show animations")
        
        # Import the config module and patch the test scenarios
        from fixtures.test_configs import TEST_SCENARIOS
        
        # Enable visualization for all predefined scenarios
        for scenario_name, config in TEST_SCENARIOS.items():
            config.visualize = True
            config.render = True
            config.show_animation = True
            print(f"   ✓ Enabled visualization for {scenario_name}")


def main():
    global ENABLE_VISUALIZATION
    
    parser = argparse.ArgumentParser(
        description="Run MBD Planner Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--single', type=str,
                       help='Run single test by name')
    parser.add_argument('--list', action='store_true',
                       help='List available tests')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization (render and show_animation)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Set global visualization flag
    ENABLE_VISUALIZATION = args.visualize
    
    if args.list:
        list_available_tests()
        return True
    
    print("MBD Planner Test Runner")
    print("=" * 50)
    
    # Setup visualization if requested
    setup_visualization()
    
    success = False
    
    try:
        if args.single:
            print(f"Running single test: {args.single}")
            success = run_single_test(args.single)
        else:
            print("Running all tests...")
            success = run_all_tests()
            
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
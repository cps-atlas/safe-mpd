#!/usr/bin/env python3
"""
Test Script to Verify Enhanced Validation
==========================================

This script tests the enhanced test validation with constraint checking
and position-based success criteria.
"""

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from test_mbd_planner import TestMBDPlanner

def test_enhanced_validation():
    """Test that enhanced validation works correctly."""
    print("=" * 60)
    print("Testing Enhanced Validation Features")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestMBDPlanner()
    
    try:
        # Test kinematic dynamics
        print("\n1. Testing enhanced validation with kinematic dynamics...")
        reward, actions, states, timing = test_instance.run_scenario_test("parking_basic_forward", visualize=False)
        print("✓ Kinematic test with enhanced validation PASSED")
        
        # Test acceleration dynamics
        print("\n2. Testing enhanced validation with acceleration dynamics...")
        reward, actions, states, timing = test_instance.run_scenario_test("acc_parking_basic_forward", visualize=False)
        print("✓ Acceleration test with enhanced validation PASSED")
        
        print("\n" + "=" * 60)
        print("✅ Enhanced validation tests PASSED!")
        print("Key features tested:")
        print("  - Constraint violation checking (hitch + collision)")
        print("  - Position-based success criteria (tractor OR trailer)")
        print("  - Enhanced result logging with violation flags")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced validation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return False

if __name__ == '__main__':
    success = test_enhanced_validation()
    sys.exit(0 if success else 1) 
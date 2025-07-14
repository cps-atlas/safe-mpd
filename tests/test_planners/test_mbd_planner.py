"""
Basic Tests for MBD Planner
=========================

This module contains basic tests for the MBD planner functionality.
These tests verify core functionality with predefined configurations.

Tests
-----
- test_case2_basic_forward: Basic forward parking scenario  
- test_case2_basic_backward: Basic backward parking scenario  
- test_case2_no_preference: No motion preference scenario
- test_case2_enforce_forward: Strict forward enforcement scenario
- test_case2_enforce_backward: Strict backward enforcement scenario

Examples
--------
To run all tests:
    $ uv run python run_tests.py

To run a specific test:
    $ uv run python run_tests.py --single test_case2_basic_forward

To run with visualization:
    $ uv run python run_tests.py --single test_case2_basic_forward --visualize
"""

import unittest

try:
    from .test_base import BaseMBDTest
    from .fixtures.test_configs import get_test_config, create_custom_test_config
except ImportError:
    # Fallback for direct execution
    from test_base import BaseMBDTest
    from fixtures.test_configs import get_test_config, create_custom_test_config


class TestMBDPlanner(BaseMBDTest):
    """Basic tests for MBD planner functionality"""
    
    def test_case2_basic_forward(self):
        """Test basic forward parking in case2 scenario"""
        config = get_test_config("case2_basic_forward")
        
        reward, actions, states, timing = self.run_mbd_test(config)
        
        # Basic validation
        self.assertGreater(reward, 0.2, "Forward parking should achieve reasonable reward")
        print(f"✓ Test completed with reward: {reward:.4f}")
        
    def test_case2_basic_backward(self):
        """Test basic backward parking in case2 scenario"""
        config = get_test_config("case2_basic_backward")
        
        reward, actions, states, timing = self.run_mbd_test(config)
        
        # Basic validation
        self.assertGreater(reward, 0.2, "Backward parking should achieve reasonable reward")
        print(f"✓ Test completed with reward: {reward:.4f}")
        
    def test_case2_no_preference(self):
        """Test parking with no motion preference"""
        config = get_test_config("case2_no_preference")
        
        reward, actions, states, timing = self.run_mbd_test(config)
        
        # Basic validation
        self.assertGreater(reward, 0.2, "No preference parking should achieve reasonable reward")
        print(f"✓ Test completed with reward: {reward:.4f}")
        
    def test_case2_enforce_forward(self):
        """Test parking with strict forward motion enforcement"""
        config = get_test_config("case2_enforce_forward")
        
        reward, actions, states, timing = self.run_mbd_test(config)
        
        # Basic validation
        self.assertGreater(reward, 0.2, "Strict forward parking should achieve reasonable reward")
        print(f"✓ Test completed with reward: {reward:.4f}")
        
    def test_case2_enforce_backward(self):
        """Test parking with strict backward motion enforcement"""
        config = get_test_config("case2_enforce_backward")
        
        reward, actions, states, timing = self.run_mbd_test(config)
        
        # Basic validation
        self.assertGreater(reward, 0.2, "Strict backward parking should achieve reasonable reward")
        print(f"✓ Test completed with reward: {reward:.4f}")


if __name__ == "__main__":
    unittest.main() 
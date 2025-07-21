"""
Test Cases for MBD Planner
=========================

Integration tests for the MBD planner with various scenarios.
"""

import unittest
import time
import logging
from typing import Optional

try:
    from .test_base import BaseMBDTest
    from .fixtures.test_configs import (
        get_test_config, 
        create_demo_variant, 
        list_default_scenarios,
        list_demo_scenarios,
        list_kinematic_scenarios,
        list_acceleration_scenarios,
        get_scenario_pairs
    )
except ImportError:
    # Fallback for direct execution
    from test_base import BaseMBDTest
    from fixtures.test_configs import (
        get_test_config, 
        create_demo_variant, 
        list_default_scenarios,
        list_demo_scenarios,
        list_kinematic_scenarios,
        list_acceleration_scenarios,
        get_scenario_pairs
    )


class TestMBDPlanner(BaseMBDTest):
    """Test cases for MBD planner with various scenarios"""
    
    def run_scenario_test(self, scenario_name: str, enable_demo: Optional[bool] = None, visualize: Optional[bool] = None):
        """
        Helper method to run a test scenario with optional demo flag override.
        
        Args:
            scenario_name: Name of the scenario to test
            enable_demo: Optional override for demo setting (None uses scenario default)
            visualize: Optional override for visualization (None uses config default)
        """
        if enable_demo is not None:
            # Create variant with specified demo setting
            base_name = scenario_name.replace("_demo", "")
            config = create_demo_variant(base_name, enable_demo=enable_demo)
        else:
            # Use scenario as-is
            config = get_test_config(scenario_name)

        # If visualize is passed, set it on the config (this will set render/show_animation via __post_init__)
        if visualize is not None:
            config.visualize = visualize
            if visualize:
                config.render = True
                config.show_animation = True
            else:
                config.render = False
                config.show_animation = False
        
        reward, actions, states, timing = self.run_mbd_test(config)
        
        logging.debug(f"Test completed: {config.test_name}")
        logging.debug(f"Final reward: {reward:.4f}")
        logging.debug(f"Pure diffusion time: {timing['pure_diffusion_time']:.2f}s")
        logging.debug(f"Total time: {timing['total_time']:.2f}s")
        
        return reward, actions, states, timing

    # === Default Tests (No Demonstration) ===
    def test_parking_basic_forward(self, visualize: Optional[bool] = None):
        """Test basic forward parking scenario (no demonstration)"""
        self.run_scenario_test("parking_basic_forward", visualize=visualize)

    def test_parking_basic_backward(self, visualize: Optional[bool] = None):
        """Test basic backward parking scenario (no demonstration)"""
        self.run_scenario_test("parking_basic_backward", visualize=visualize)

    def test_parking_no_preference(self, visualize: Optional[bool] = None):
        """Test parking with no motion preference (no demonstration)"""
        self.run_scenario_test("parking_no_preference", visualize=visualize)

    def test_parking_enforce_forward(self, visualize: Optional[bool] = None):
        """Test parking with strict forward enforcement (no demonstration)"""
        self.run_scenario_test("parking_enforce_forward", visualize=visualize)

    def test_parking_enforce_backward(self, visualize: Optional[bool] = None):
        """Test parking with strict backward enforcement (no demonstration)"""
        self.run_scenario_test("parking_enforce_backward", visualize=visualize)

    # === Demo Tests (With Demonstration) ===
    def test_parking_basic_forward_demo(self, visualize: Optional[bool] = None):
        """Test basic forward parking scenario with demonstration"""
        self.run_scenario_test("parking_basic_forward_demo", visualize=visualize)

    def test_parking_basic_backward_demo(self, visualize: Optional[bool] = None):
        """Test basic backward parking scenario with demonstration"""
        self.run_scenario_test("parking_basic_backward_demo", visualize=visualize)

    def test_parking_no_preference_demo(self, visualize: Optional[bool] = None):
        """Test parking with no motion preference with demonstration"""
        self.run_scenario_test("parking_no_preference_demo", visualize=visualize)

    def test_parking_enforce_forward_demo(self, visualize: Optional[bool] = None):
        """Test parking with strict forward enforcement with demonstration"""
        self.run_scenario_test("parking_enforce_forward_demo", visualize=visualize)

    def test_parking_enforce_backward_demo(self, visualize: Optional[bool] = None):
        """Test parking with strict backward enforcement with demonstration"""
        self.run_scenario_test("parking_enforce_backward_demo", visualize=visualize)

    # === Acceleration Dynamics Tests (No Demonstration) ===
    def test_acc_parking_basic_forward(self, visualize: Optional[bool] = None):
        """Test basic forward parking scenario with acceleration dynamics (no demonstration)"""
        self.run_scenario_test("acc_parking_basic_forward", visualize=visualize)

    def test_acc_parking_basic_backward(self, visualize: Optional[bool] = None):
        """Test basic backward parking scenario with acceleration dynamics (no demonstration)"""
        self.run_scenario_test("acc_parking_basic_backward", visualize=visualize)

    def test_acc_parking_no_preference(self, visualize: Optional[bool] = None):
        """Test parking with no motion preference with acceleration dynamics (no demonstration)"""
        self.run_scenario_test("acc_parking_no_preference", visualize=visualize)

    def test_acc_parking_enforce_forward(self, visualize: Optional[bool] = None):
        """Test parking with strict forward enforcement with acceleration dynamics (no demonstration)"""
        self.run_scenario_test("acc_parking_enforce_forward", visualize=visualize)

    def test_acc_parking_enforce_backward(self, visualize: Optional[bool] = None):
        """Test parking with strict backward enforcement with acceleration dynamics (no demonstration)"""
        self.run_scenario_test("acc_parking_enforce_backward", visualize=visualize)

    # === Acceleration Dynamics Demo Tests (With Demonstration) ===
    def test_acc_parking_basic_forward_demo(self, visualize: Optional[bool] = None):
        """Test basic forward parking scenario with acceleration dynamics and demonstration"""
        self.run_scenario_test("acc_parking_basic_forward_demo", visualize=visualize)

    def test_acc_parking_basic_backward_demo(self, visualize: Optional[bool] = None):
        """Test basic backward parking scenario with acceleration dynamics and demonstration"""
        self.run_scenario_test("acc_parking_basic_backward_demo", visualize=visualize)

    def test_acc_parking_no_preference_demo(self, visualize: Optional[bool] = None):
        """Test parking with no motion preference with acceleration dynamics and demonstration"""
        self.run_scenario_test("acc_parking_no_preference_demo", visualize=visualize)

    def test_acc_parking_enforce_forward_demo(self, visualize: Optional[bool] = None):
        """Test parking with strict forward enforcement with acceleration dynamics and demonstration"""
        self.run_scenario_test("acc_parking_enforce_forward_demo", visualize=visualize)

    def test_acc_parking_enforce_backward_demo(self, visualize: Optional[bool] = None):
        """Test parking with strict backward enforcement with acceleration dynamics and demonstration"""
        self.run_scenario_test("acc_parking_enforce_backward_demo", visualize=visualize)

    # === Utility Tests ===
    def test_demo_flag_override(self, visualize: Optional[bool] = None):
        """Test that demo flag override works correctly"""
        # Test running a default scenario with demo enabled
        logging.debug("\n--- Testing demo flag override: default scenario with enable_demo=True ---")
        reward_with_demo, _, _, _ = self.run_scenario_test("parking_basic_forward", enable_demo=True, visualize=visualize)
        
        # Test running a demo scenario with demo disabled
        logging.debug("\n--- Testing demo flag override: demo scenario with enable_demo=False ---") 
        reward_no_demo, _, _, _ = self.run_scenario_test("parking_basic_forward", enable_demo=False, visualize=visualize)
        
        # Both should run without errors - exact reward comparison depends on randomness
        logging.debug(f"With-demo reward: {reward_with_demo:.4f}")
        logging.debug(f"No-demo reward: {reward_no_demo:.4f}")
        
        # Basic sanity check - both should be valid rewards
        self.assertGreater(reward_with_demo, -10.0)
        self.assertGreater(reward_no_demo, -10.0)

    def test_scenario_list_functions(self, visualize: Optional[bool] = None):
        """Test that scenario listing functions work correctly"""
        default_scenarios = list_default_scenarios()
        demo_scenarios = list_demo_scenarios()
        kinematic_scenarios = list_kinematic_scenarios()
        acceleration_scenarios = list_acceleration_scenarios()
        scenario_pairs = get_scenario_pairs()
        
        logging.debug(f"Default scenarios: {default_scenarios}")
        logging.debug(f"Demo scenarios: {demo_scenarios}")
        logging.debug(f"Kinematic scenarios: {kinematic_scenarios}")
        logging.debug(f"Acceleration scenarios: {acceleration_scenarios}")
        logging.debug(f"Scenario pairs: {scenario_pairs}")
        
        # Check that we have the expected number of scenarios
        self.assertEqual(len(default_scenarios), 10)  # 5 kinematic + 5 acceleration base scenarios (no demo)
        self.assertEqual(len(demo_scenarios), 10)  # 5 kinematic + 5 acceleration demo versions
        self.assertEqual(len(kinematic_scenarios), 10)  # 5 base + 5 demo kinematic scenarios
        self.assertEqual(len(acceleration_scenarios), 10)  # 5 base + 5 demo acceleration scenarios
        self.assertEqual(len(scenario_pairs), 10)  # 5 kinematic + 5 acceleration pairs
        
        # Check that all default scenarios don't end with _demo
        for scenario in default_scenarios:
            self.assertFalse(scenario.endswith("_demo"))
        
        # Check that all demo scenarios end with _demo
        for scenario in demo_scenarios:
            self.assertTrue(scenario.endswith("_demo"))
        
        # Check kinematic scenarios start with "parking_"
        for scenario in kinematic_scenarios:
            self.assertTrue(scenario.startswith("parking_"))
        
        # Check acceleration scenarios start with "acc_parking_"
        for scenario in acceleration_scenarios:
            self.assertTrue(scenario.startswith("acc_parking_"))
        
        # Check that pairs are correctly formed
        for default_name, demo_name in scenario_pairs:
            self.assertEqual(demo_name, f"{default_name}_demo")


if __name__ == '__main__':
    unittest.main() 
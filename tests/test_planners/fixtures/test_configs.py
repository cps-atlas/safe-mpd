"""
Test Configurations for MBD Planner Tests
=========================

This module defines the TestConfig dataclass and predefined test scenarios
for MBD planner testing.
"""

import copy
import numpy as np
import jax.numpy as jnp  # Use JAX arrays for proper dtype compatibility
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

# Add the MBD source path so we can import MBDConfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from mbd.planners.mbd_planner import MBDConfig


@dataclass
class TestConfig(MBDConfig):
    """Test configuration that extends MBDConfig with test-specific fields"""
    
    # Override only test-specific defaults (keep MBDConfig defaults for core parameters)
    seed: int = 0  # Fixed seed for reproducibility
    render: bool = False  # Turn off rendering for tests by default
    save_animation: bool = False
    show_animation: bool = False
    save_denoising_animation: bool = False
    
    # Test metadata (additional fields not in MBDConfig)
    test_name: str = ""
    description: str = ""
    expected_reward_min: float = -1.0
    expected_reward_max: float = 1.0
    timeout_seconds: float = 300
    visualize: bool = False  # If True, enable render and show_animation
    
    # Environment customization (test-specific) - using geometric positioning
    custom_circular_obstacles: Optional[List] = None
    custom_rectangular_obstacles: Optional[List] = None
    custom_parking_config: Optional[Dict] = None
    
    # Initial position using geometric parameters (relative to parking lot)
    init_dx: Optional[float] = None  # Distance from tractor front face to target parking space center (x-direction)
    init_dy: Optional[float] = None  # Distance from tractor to parking lot entrance line (y-direction)
    init_theta1: Optional[float] = None  # Initial tractor orientation
    init_theta2: Optional[float] = None  # Initial trailer orientation
    
    # Goal position (only orientations, position determined by target parking space)
    goal_theta1: Optional[float] = None  # Goal tractor orientation
    goal_theta2: Optional[float] = None  # Goal trailer orientation
    
    # Validation criteria (test-specific)
    min_final_distance_to_goal: float = 2.0
    
    def __post_init__(self):
        """Apply visualization settings after initialization"""
        if self.visualize:
            self.render = True
            self.show_animation = True


# Predefined test scenarios
TEST_SCENARIOS = {
    "parking_basic_forward": TestConfig(
        test_name="parking_basic_forward",
        description="Basic forward parking scenario",
        motion_preference=1,
        expected_reward_min=0.3,
        expected_reward_max=2.0,
        init_dx=-3.0,
        init_dy=4.0,
        init_theta1=0.0,
        init_theta2=0.0,
        goal_theta1=-jnp.pi/2,
        goal_theta2=-jnp.pi/2
    ),
    
    "parking_basic_backward": TestConfig(
        test_name="parking_basic_backward",
        description="Basic backward parking scenario",
        motion_preference=-1,
        expected_reward_min=0.3,
        expected_reward_max=2.0,
        init_dx=-3.0,
        init_dy=4.0,
        init_theta1=0.0,
        init_theta2=0.0,
        goal_theta1=jnp.pi/2,
        goal_theta2=jnp.pi/2
    ),
    
    "parking_no_preference": TestConfig(
        test_name="parking_no_preference",
        description="Parking with no motion preference",
        motion_preference=0,
        expected_reward_min=0.3,
        expected_reward_max=2.0,
        init_dx=-3.0,
        init_dy=4.0,
        init_theta1=0.0,
        init_theta2=0.0,
        goal_theta1=-jnp.pi/2,
        goal_theta2=-jnp.pi/2
    ),
    
    "parking_enforce_forward": TestConfig(
        test_name="parking_enforce_forward",
        description="Parking with strict forward enforcement",
        motion_preference=2,
        expected_reward_min=0.3,
        expected_reward_max=2.0,
        init_dx=-5.0,
        init_dy=5.0,
        init_theta1=0.0,
        init_theta2=0.0,
        goal_theta1=-jnp.pi/2,
        goal_theta2=-jnp.pi/2
    ),
    
    "parking_enforce_backward": TestConfig(
        test_name="parking_enforce_backward",
        description="Parking with strict backward enforcement",
        motion_preference=-2,
        expected_reward_min=0.3,
        expected_reward_max=2.0,
        init_dx=-12.0,
        init_dy=1.0,
        init_theta1=jnp.pi,
        init_theta2=jnp.pi,
        goal_theta1=jnp.pi/2,
        goal_theta2=jnp.pi/2
    ),
}


def get_test_config(scenario_name: str) -> TestConfig:
    """
    Get a test configuration by name.
    
    Args:
        scenario_name: Name of the predefined scenario
        
    Returns:
        TestConfig: Deep copy of the requested configuration
        
    Raises:
        KeyError: If scenario_name is not found
    """
    if scenario_name not in TEST_SCENARIOS:
        available = list(TEST_SCENARIOS.keys())
        raise KeyError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    
    return copy.deepcopy(TEST_SCENARIOS[scenario_name])


def create_custom_test_config(base_scenario: str, **overrides) -> TestConfig:
    """
    Create a custom test configuration based on a base scenario.
    
    Args:
        base_scenario: Name of base scenario to start from
        **overrides: Fields to override in the base configuration
        
    Returns:
        TestConfig: Modified configuration
    """
    config = get_test_config(base_scenario)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return config


def list_available_scenarios() -> List[str]:
    """Return list of available predefined scenario names."""
    return list(TEST_SCENARIOS.keys()) 
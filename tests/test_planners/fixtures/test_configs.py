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


# Predefined test scenarios for kinematic dynamics (tt2d)
TEST_SCENARIOS = {
    "parking_basic_forward": TestConfig(
        test_name="parking_basic_forward",
        description="Basic forward parking scenario",
        env_name="tt2d",
        enable_demo=False,
        motion_preference=1,
        expected_reward_min=0.3,
        expected_reward_max=2.0,
        init_dx=-2.0,
        init_dy=1.0,
        init_theta1=0.0,
        init_theta2=0.0,
        goal_theta1=-jnp.pi/2,
        goal_theta2=-jnp.pi/2
    ),
    
    "parking_basic_backward": TestConfig(
        test_name="parking_basic_backward",
        description="Basic backward parking scenario",
        env_name="tt2d",
        enable_demo=False,
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
        env_name="tt2d",
        enable_demo=False,
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
        env_name="tt2d",
        enable_demo=False,
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
        env_name="tt2d",
        enable_demo=False,
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

# Acceleration dynamics test scenarios (acc_tt2d)
ACC_TEST_SCENARIOS = {
    "acc_parking_basic_forward": TestConfig(
        test_name="acc_parking_basic_forward",
        description="Basic forward parking scenario (acceleration dynamics)",
        env_name="acc_tt2d",
        enable_demo=False,
        motion_preference=1,
        expected_reward_min=0.3,
        expected_reward_max=2.0,
        init_dx=-2.0,
        init_dy=1.0,
        init_theta1=0.0,
        init_theta2=0.0,
        goal_theta1=-jnp.pi/2,
        goal_theta2=-jnp.pi/2
    ),
    
    "acc_parking_basic_backward": TestConfig(
        test_name="acc_parking_basic_backward",
        description="Basic backward parking scenario (acceleration dynamics)",
        env_name="acc_tt2d",
        enable_demo=False,
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
    
    "acc_parking_no_preference": TestConfig(
        test_name="acc_parking_no_preference",
        description="Parking with no motion preference (acceleration dynamics)",
        env_name="acc_tt2d",
        enable_demo=False,
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
    
    "acc_parking_enforce_forward": TestConfig(
        test_name="acc_parking_enforce_forward",
        description="Parking with strict forward enforcement (acceleration dynamics)",
        env_name="acc_tt2d",
        enable_demo=False,
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
    
    "acc_parking_enforce_backward": TestConfig(
        test_name="acc_parking_enforce_backward",
        description="Parking with strict backward enforcement (acceleration dynamics)",
        env_name="acc_tt2d",
        enable_demo=False,
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

# Demo versions of all scenarios (automatically generated) - DEMONSTRATION IS SPECIAL CASE
DEMO_TEST_SCENARIOS = {}
ACC_DEMO_TEST_SCENARIOS = {}

def _create_demo_scenarios():
    """Create demo versions of all test scenarios"""
    # Create demo versions of kinematic scenarios
    for name, config in TEST_SCENARIOS.items():
        demo_name = f"{name}_demo"
        demo_config = copy.deepcopy(config)
        demo_config.test_name = demo_name
        demo_config.description = f"{config.description} (with demo)"
        demo_config.enable_demo = True
        # Higher expected reward range for demo (typically better performance)
        demo_config.expected_reward_min = config.expected_reward_min + 0.2
        demo_config.expected_reward_max = config.expected_reward_max
        DEMO_TEST_SCENARIOS[demo_name] = demo_config
    
    # Create demo versions of acceleration scenarios
    for name, config in ACC_TEST_SCENARIOS.items():
        demo_name = f"{name}_demo"
        demo_config = copy.deepcopy(config)
        demo_config.test_name = demo_name
        demo_config.description = f"{config.description} (with demo)"
        demo_config.enable_demo = True
        # Higher expected reward range for demo (typically better performance)
        demo_config.expected_reward_min = config.expected_reward_min + 0.2
        demo_config.expected_reward_max = config.expected_reward_max
        ACC_DEMO_TEST_SCENARIOS[demo_name] = demo_config

# Create the demo scenarios
_create_demo_scenarios()

# Combine all scenarios
ALL_TEST_SCENARIOS = {**TEST_SCENARIOS, **DEMO_TEST_SCENARIOS, **ACC_TEST_SCENARIOS, **ACC_DEMO_TEST_SCENARIOS}


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
    if scenario_name not in ALL_TEST_SCENARIOS:
        available = list(ALL_TEST_SCENARIOS.keys())
        raise KeyError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    
    return copy.deepcopy(ALL_TEST_SCENARIOS[scenario_name])


def create_demo_variant(scenario_name: str, enable_demo: bool = False) -> TestConfig:
    """
    Create a demo or no-demo variant of a base scenario.
    
    Args:
        scenario_name: Name of base scenario (without _demo suffix)
        enable_demo: Whether to enable demonstration (False by default)
        
    Returns:
        TestConfig: Configuration with demo setting applied
    """
    # Remove _demo suffix if present to get base scenario
    base_name = scenario_name.replace("_demo", "")
    
    if base_name not in TEST_SCENARIOS:
        available = list(TEST_SCENARIOS.keys())
        raise KeyError(f"Unknown base scenario '{base_name}'. Available: {available}")
    
    config = copy.deepcopy(TEST_SCENARIOS[base_name])
    
    if enable_demo:
        config.test_name = f"{base_name}_demo"
        config.description = f"{config.description} (with demo)"
        config.enable_demo = True
        # Higher expected reward range for demo
        config.expected_reward_min = config.expected_reward_min
        config.expected_reward_max = config.expected_reward_max
    
    return config


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
    return list(ALL_TEST_SCENARIOS.keys())


def list_default_scenarios() -> List[str]:
    """Return list of default scenarios (no demonstration)."""
    return [name for name in ALL_TEST_SCENARIOS.keys() if not name.endswith("_demo")]


def list_demo_scenarios() -> List[str]:
    """Return list of scenarios with demonstration enabled."""
    return [name for name in ALL_TEST_SCENARIOS.keys() if name.endswith("_demo")]


def list_kinematic_scenarios() -> List[str]:
    """Return list of kinematic (tt2d) scenarios (default + demo)."""
    return [name for name in ALL_TEST_SCENARIOS.keys() if name.startswith("parking_")]


def list_acceleration_scenarios() -> List[str]:
    """Return list of acceleration (acc_tt2d) scenarios (default + demo)."""
    return [name for name in ALL_TEST_SCENARIOS.keys() if name.startswith("acc_parking_")]


def get_scenario_pairs() -> List[Tuple[str, str]]:
    """
    Get pairs of (default_scenario, demo_scenario) for comparison.
    
    Returns:
        List of tuples with (default_scenario_name, demo_scenario_name)
    """
    pairs = []
    # Kinematic scenario pairs
    for base_name in TEST_SCENARIOS.keys():
        default_name = base_name
        demo_name = f"{base_name}_demo"
        pairs.append((default_name, demo_name))
    
    # Acceleration scenario pairs  
    for base_name in ACC_TEST_SCENARIOS.keys():
        default_name = base_name
        demo_name = f"{base_name}_demo"
        pairs.append((default_name, demo_name))
    
    return pairs 
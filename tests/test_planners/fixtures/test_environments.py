"""
Test Environment Setup Utilities
=========================

This module provides utilities for setting up test environments
for MBD planner testing, specifically focused on parking scenarios.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Add the MBD source path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import mbd
from mbd.envs.env import Env
from mbd.planners.mbd_planner import clear_jit_cache
try:
    from .test_configs import TestConfig
except ImportError:
    # Fallback for direct execution
    from test_configs import TestConfig


def setup_parking_environment(config: TestConfig) -> Env:
    """
    Set up a parking environment with custom obstacles and parking configuration.
    
    Args:
        config: TestConfig containing environment customizations
        
    Returns:
        Env: Configured environment object
    """
    # Create base parking environment
    parking_config = config.custom_parking_config or None
    env = Env(case="parking", parking_config=parking_config)
    
    # Add custom circular obstacles if specified
    if config.custom_circular_obstacles:
        existing_circles = env.obs_circle if len(env.obs_circle) > 0 else []
        # Filter out the default dummy obstacle (0,0,0) if it exists
        existing_circles = [obs for obs in existing_circles if obs[2] > 0]
        env.obs_circle = existing_circles + config.custom_circular_obstacles
        logging.debug(f"Added {len(config.custom_circular_obstacles)} custom circular obstacles")
        
    # Add custom rectangular obstacles if specified  
    if config.custom_rectangular_obstacles:
        existing_rects = env.obs_rectangle if len(env.obs_rectangle) > 0 else []
        env.obs_rectangle = existing_rects + config.custom_rectangular_obstacles
        logging.debug(f"Added {len(config.custom_rectangular_obstacles)} custom rectangular obstacles")
        
    return env


def create_test_tt2d_environment(config: TestConfig) -> Any:
    """
    Create a TractorTrailer2d environment configured for testing.
    
    Args:
        config: TestConfig with environment and MBD parameters
        
    Returns:
        TractorTrailer2d environment instance
    """
    # Set up the base environment configuration
    env_config = setup_parking_environment(config)
    
    # Create TT2D environment with test configuration 
    # Since TestConfig inherits from MBDConfig, pass all parameters from config
    env = mbd.envs.get_env(
        config.env_name,
        case=config.case,
        dt=config.dt,
        H=config.Hsample,
        motion_preference=config.motion_preference,
        collision_penalty=config.collision_penalty,
        enable_gated_rollout_collision=config.enable_gated_rollout_collision,
        hitch_penalty=config.hitch_penalty,
        enable_gated_rollout_hitch=config.enable_gated_rollout_hitch,
        reward_threshold=config.reward_threshold,
        ref_reward_threshold=config.ref_reward_threshold,
        max_w_theta=config.max_w_theta,
        hitch_angle_weight=config.hitch_angle_weight,
        l1=config.l1,
        l2=config.l2,
        lh=config.lh,
        tractor_width=config.tractor_width,
        trailer_width=config.trailer_width,
        v_max=config.v_max,
        delta_max_deg=config.delta_max_deg,
        # Acceleration control constraints (for acc_tt2d)
        a_max=config.a_max,
        omega_max=config.omega_max,
        d_thr_factor=config.d_thr_factor,
        k_switch=config.k_switch,
        steering_weight=config.steering_weight,
        preference_penalty_weight=config.preference_penalty_weight,
        heading_reward_weight=config.heading_reward_weight,
        terminal_reward_threshold=config.terminal_reward_threshold,
        terminal_reward_weight=config.terminal_reward_weight,
        ref_pos_weight=config.ref_pos_weight,
        ref_theta1_weight=config.ref_theta1_weight,
        ref_theta2_weight=config.ref_theta2_weight
    )
    
    # Now apply the custom environment configuration to the created environment
    # Replace the default environment with our customized one
    env.env = env_config
    
    # Update obstacle references to point to the custom environment
    obstacles = env_config.get_obstacles()
    env.obs_circles = obstacles['circles']
    env.obs_rectangles = obstacles['rectangles']
    env.obs = env.obs_circles  # For backward compatibility
    
    # Set initial position using geometric parameters (dx, dy, theta1, theta2)
    if (config.init_dx is not None and config.init_dy is not None and 
        config.init_theta1 is not None and config.init_theta2 is not None):
        env.set_init_pos(
            dx=config.init_dx, 
            dy=config.init_dy, 
            theta1=config.init_theta1, 
            theta2=config.init_theta2
        )
        logging.debug(f"Set initial position: dx={config.init_dx}, dy={config.init_dy}, "
              f"theta1={config.init_theta1:.3f}, theta2={config.init_theta2:.3f}")
    
    # Set goal position (only orientations, position determined by target parking space)
    if config.goal_theta1 is not None and config.goal_theta2 is not None:
        env.set_goal_pos(theta1=config.goal_theta1, theta2=config.goal_theta2)
        logging.debug(f"Set goal orientation: theta1={config.goal_theta1:.3f}, theta2={config.goal_theta2:.3f}")
    
    # Clear any existing demonstration trajectory to ensure fresh generation
    env.xref = None
    env.rew_xref = None
    if hasattr(env, 'angle_mask'):
        env.angle_mask = None
    logging.debug(f"Cleared demonstration trajectory for test: {config.test_name}")
    
    # Clear JIT function cache to ensure fresh compilation with new trajectory
    clear_jit_cache()
    logging.debug(f"Cleared JIT function cache for test: {config.test_name}")
    
    logging.debug(f"Final positions - x0: {env.x0}, xg: {env.xg}")
    
    return env


def get_default_parking_config() -> Dict[str, Any]:
    """
    Get the default parking configuration for parking scenario.
    
    Returns:
        Dict: Default parking configuration
    """
    return {
        'parking_rows': 2,
        'parking_cols': 8,
        'space_width': 3.5,
        'space_length': 7.0,
        'parking_y_offset': 4.0,
        'occupied_spaces': [1, 2, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15],
        'target_spaces': [3, 11],
        'obstacle_radius': 1.0,
    }


def create_tight_parking_config() -> Dict[str, Any]:
    """
    Create a tighter parking configuration for more challenging tests.
    
    Returns:
        Dict: Tight parking configuration
    """
    return {
        'parking_rows': 3,
        'parking_cols': 6,
        'space_width': 3.0,      # Tighter spaces
        'space_length': 6.5,
        'parking_y_offset': 3.0,
        'occupied_spaces': [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17],
        'target_spaces': [6, 12],
        'obstacle_radius': 1.0,
    } 
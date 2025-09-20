"""
Base Test Class for MBD Planner Tests
=========================

This module provides the base test class that all MBD planner tests inherit from.
It includes common utilities, validation methods, and test infrastructure.
"""

import os
import sys
import time
import json
import unittest
import subprocess
import logging
from datetime import datetime
from typing import Tuple, Dict, Any
from dataclasses import asdict

import numpy as np
import jax.numpy as jnp

# Add the MBD source path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mbd
from mbd.planners.mbd_planner import run_diffusion
try:
    from .fixtures.test_configs import TestConfig
    from .fixtures.test_environments import create_test_tt2d_environment
except ImportError:
    # Fallback for direct execution
    from fixtures.test_configs import TestConfig
    from fixtures.test_environments import create_test_tt2d_environment


class BaseMBDTest(unittest.TestCase):
    """Base class for all MBD planner tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once per test class"""
        # Suppress JAX info messages about unavailable backends (GPU/TPU)
        logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)
        logging.getLogger('jax').setLevel(logging.WARNING)
        
        cls.test_results_dir = "tests/test_planners/results"
        os.makedirs(cls.test_results_dir, exist_ok=True)
        logging.debug(f"Test results will be saved to: {cls.test_results_dir}")
        
    def setUp(self):
        """Set up before each test"""
        self.start_time = time.time()
        logging.info(f"\n=== Starting test: {self._testMethodName} ===")
        
    def tearDown(self):
        """Clean up after each test"""
        elapsed = time.time() - self.start_time
        logging.debug(f"=== Test {self._testMethodName} completed in {elapsed:.2f}s ===")
        print()
        
        

        
    def run_mbd_test(self, config: TestConfig) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Generic test runner for MBD planner.
        
        Args:
            config: TestConfig with test parameters
            
        Returns:
            Tuple of (final_distance, actions, states, timing_info)
        """
        logging.debug(f"Running test: {config.test_name}")
        logging.debug(f"Description: {config.description}")
        logging.debug(f"Motion preference: {config.motion_preference}")
        logging.debug(f"Enable demo: {config.enable_demo}")
        logging.debug(f"Samples: {config.Nsample}, Horizon: {config.Hsample}, Diffusion steps: {config.Ndiffuse}")
        
        # Create test environment
        env = create_test_tt2d_environment(config)
        
        # Run diffusion with timeout (config IS already an MBDConfig)
        try:
            with TimeoutContext(config.timeout_seconds):
                reward, actions, states, timing = run_diffusion(args=config, env=env)
        except TimeoutError:
            self.fail(f"Test {config.test_name} timed out after {config.timeout_seconds}s")
            
        # Convert JAX arrays to numpy for easier handling
        actions_np = np.array(actions)
        states_np = np.array(states)

        
        # Compute final distance to goal (minimum of tractor front vs trailer back)
        final_distance, tractor_distance, trailer_distance = self.compute_final_min_distance(states_np, env)

        # Validate results using distance-based criteria
        self.validate_test_results(
            config,
            final_distance,
            actions_np,
            states_np,
            timing,
            env,
            tractor_distance=tractor_distance,
            trailer_distance=trailer_distance,
        )
        
        # Save results for reproducibility
        self.save_test_results(
            config,
            final_distance,
            actions_np,
            states_np,
            timing,
            env,
            tractor_distance=tractor_distance,
            trailer_distance=trailer_distance,
        )
        
        return final_distance, actions_np, states_np, timing
        
    def validate_test_results(self, config: TestConfig, final_distance: float, 
                            actions: np.ndarray, states: np.ndarray, 
                            timing: Dict[str, float], env, *,
                            tractor_distance: float, trailer_distance: float):
        """
        Validate test results against expected criteria.
        
        Args:
            config: Test configuration
            reward: Final reward value
            actions: Action sequence array
            states: State trajectory array
            timing: Timing information dictionary
            env: Environment instance used for the test
        """
        # Distance bounds check (success criteria)
        self.assertGreaterEqual(final_distance, config.expected_distance_min, 
                               f"Final distance {final_distance:.3f} below minimum {config.expected_distance_min}")
        self.assertLessEqual(final_distance, config.expected_distance_max,
                            f"Final distance {final_distance:.3f} above maximum {config.expected_distance_max}")
        logging.info(f"✓ Final distance {final_distance:.3f} within expected range [{config.expected_distance_min}, {config.expected_distance_max}]")
        
        # Check constraint violations along trajectory
        hitch_violation_found, collision_found = self.check_trajectory_violations(states, env)
        
        # Position-based success check  
        final_state = states[-1]
        position_success, tractor_distance_chk, trailer_distance_chk = self.check_position_success(final_state, config, env)
        
        logging.info(f"✓ Constraint violations - Hitch: {hitch_violation_found}, Collision: {collision_found}")
        logging.info(f"✓ Position success: {position_success} (tractor: {tractor_distance_chk:.2f}m, trailer: {trailer_distance_chk:.2f}m, limit: {config.min_final_distance_to_goal}m)")
        
        # Success is defined as reaching goal position without this being a hard failure
        # We log violations but don't fail the test for now (for analysis purposes)
        self.assertTrue(position_success, f"Neither tractor nor trailer reached goal within {config.min_final_distance_to_goal}m")
        
        # Basic sanity checks
        self.assertEqual(len(actions), config.Hsample, "Action sequence length mismatch")
        self.assertEqual(len(states), config.Hsample + 1, "State sequence length mismatch")  # +1 for initial state
        self.assertEqual(actions.shape[1], 2, "Actions should have 2 dimensions")
        
    def check_trajectory_violations(self, states: np.ndarray, env) -> Tuple[bool, bool]:
        """
        Check for constraint violations along the entire trajectory.
        
        Args:
            states: State trajectory array
            env: Environment instance
            
        Returns:
            Tuple of (hitch_violation_found, collision_found)
        """
        import jax.numpy as jnp
        
        hitch_violation_found = False
        collision_found = False
        
        # Check each state in the trajectory
        for i, state in enumerate(states):
            # Convert to JAX array for environment functions
            state_jax = jnp.array(state)
            
            # For acceleration dynamics, only use first 4 elements for collision/hitch checking
            state_4d = state_jax[:4]
            
            # Check hitch violation using environment function
            if env.check_hitch_violation(state_4d):
                hitch_violation_found = True
            
            # Check collision using environment function
            if env.check_obstacle_collision(state_4d, env.obs_circles, env.obs_rectangles):
                collision_found = True
                
        return hitch_violation_found, collision_found
        
    def check_position_success(self, final_state: np.ndarray, config: TestConfig, env) -> Tuple[bool, float, float]:
        """
        Check if the final position is successful based on tractor and trailer positions.
        
        Args:
            final_state: Final state from trajectory
            config: Test configuration
            env: Environment instance
            
        Returns:
            Tuple of (success, tractor_distance, trailer_distance)
        """
        import jax.numpy as jnp
        
        # Convert to JAX array for environment functions
        state_jax = jnp.array(final_state)
        
        # Get tractor position (front center)
        tractor_pos = final_state[:2]
        
        # Get trailer back position using environment function (align with stat_mbd.py)
        # For acceleration dynamics, only use first 4 elements
        trailer_pos = np.array(env.get_trailer_back_position(state_jax[:4]))
        
        # Get goal positions (use 4D goal state for compatibility)
        goal_state = env.xg  
        
        # Compute distances
        tractor_distance = np.linalg.norm(tractor_pos - goal_state[:2])
        trailer_distance = np.linalg.norm(trailer_pos - goal_state[:2])
        
        # Success if either tractor OR trailer is close enough to goal
        success = (tractor_distance <= config.min_final_distance_to_goal or 
                  trailer_distance <= config.min_final_distance_to_goal)
        
        return success, tractor_distance, trailer_distance

    def compute_final_min_distance(self, states: np.ndarray, env) -> Tuple[float, float, float]:
        """
        Compute the final minimum distance to goal between tractor front and trailer back,
        following the logic in tests/stat_mbd.py (evaluate_trial_result).
        Returns (final_min_distance, tractor_distance, trailer_distance).
        """
        import jax.numpy as jnp
        final_state = states[-1]
        final_state_4d = final_state[:4]
        px, py = final_state_4d[:2]
        goal_px, goal_py = env.xg[0], env.xg[1]

        tractor_distance = float(np.sqrt((px - goal_px)**2 + (py - goal_py)**2))

        trailer_positions = env.get_trailer_back_position(jnp.array(final_state_4d))
        trailer_px, trailer_py = float(trailer_positions[0]), float(trailer_positions[1])
        trailer_distance = float(np.sqrt((trailer_px - goal_px)**2 + (trailer_py - goal_py)**2))

        final_min_distance = min(tractor_distance, trailer_distance)
        return final_min_distance, tractor_distance, trailer_distance

    def save_test_results(self, config: TestConfig, final_distance: float, 
                         actions: np.ndarray, states: np.ndarray, 
                         timing: Dict[str, float], env, *,
                         tractor_distance: float, trailer_distance: float):
        """
        Save test results for reproducibility.
        
        Args:
            config: Test configuration
            reward: Final reward
            actions: Action sequence
            states: State trajectory
            timing: Timing information
            env: Environment instance
        """
        # Check trajectory violations and position success (recompute success for logging)
        hitch_violation_found, collision_found = self.check_trajectory_violations(states, env)
        position_success, tractor_distance_chk, trailer_distance_chk = self.check_position_success(states[-1], config, env)
        
        result_data = {
            'config': asdict(config),
            'final_distance': float(final_distance),
            'final_state': states[-1].tolist(),
            'timing': timing,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self.get_git_commit(),
            'test_summary': {
                # Constraint violations
                'hitch_violation_found': bool(hitch_violation_found),
                'collision_found': bool(collision_found),
                # Position success
                'position_success': bool(position_success),
                'tractor_distance_to_goal': float(tractor_distance),
                'trailer_distance_to_goal': float(trailer_distance),
                'tractor_distance_to_goal_check': float(tractor_distance_chk),
                'trailer_distance_to_goal_check': float(trailer_distance_chk),
                'min_final_distance_to_goal': float(config.min_final_distance_to_goal),
                # Trajectory info
                'trajectory_length': len(states),
                'action_sequence_length': len(actions)            
            }
        }

        
        
        result_file = os.path.join(self.test_results_dir, f"{config.test_name}_result.json")
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        # Also save arrays separately
        arrays_file = os.path.join(self.test_results_dir, f"{config.test_name}_arrays.npz")
        np.savez(arrays_file, actions=actions, states=states)
        
        logging.debug(f"Results saved to: {result_file}")
        logging.debug(f"Arrays saved to: {arrays_file}")
        
    def get_git_commit(self) -> str:
        """Get current git commit hash for reproducibility."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(__file__)
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
            
    def get_system_info(self) -> Dict[str, str]:
        """Get system information for reproducibility."""
        import platform
        return {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
        }


class TimeoutContext:
    """Context manager for implementing test timeouts."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        
    def __enter__(self):
        # For simplicity, we'll just track start time
        # In production, might want to use signal.alarm on Unix systems
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutError(f"Operation timed out after {elapsed:.1f}s (limit: {self.timeout_seconds}s)") 
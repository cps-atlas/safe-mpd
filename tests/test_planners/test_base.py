"""
Base Test Class for MBD Planner Tests
=========================

This module provides the base test class that all MBD planner tests inherit from.
It includes common utilities, validation methods, and test infrastructure.
"""

# TODO: add without demonstratino test cases
# TODO: add obstacle on top (in some cases)
# TODO: modify test cases similar to the presentaiotn slides

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
        print(f"Test results will be saved to: {cls.test_results_dir}")
        
    def setUp(self):
        """Set up before each test"""
        self.start_time = time.time()
        print(f"\n=== Starting test: {self._testMethodName} ===")
        
    def tearDown(self):
        """Clean up after each test"""
        elapsed = time.time() - self.start_time
        print(f"=== Test {self._testMethodName} completed in {elapsed:.2f}s ===")
        

        
    def run_mbd_test(self, config: TestConfig) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Generic test runner for MBD planner.
        
        Args:
            config: TestConfig with test parameters
            
        Returns:
            Tuple of (reward, actions, states, timing_info)
        """
        print(f"Running test: {config.test_name}")
        print(f"Description: {config.description}")
        print(f"Motion preference: {config.motion_preference}")
        print(f"Enable demo: {config.enable_demo}")
        print(f"Samples: {config.Nsample}, Horizon: {config.Hsample}, Diffusion steps: {config.Ndiffuse}")
        
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

        
        # Validate results
        self.validate_test_results(config, reward, actions_np, states_np, timing, env)
        
        # Save results for reproducibility
        self.save_test_results(config, reward, actions_np, states_np, timing, env)
        
        return reward, actions_np, states_np, timing
        
    def validate_test_results(self, config: TestConfig, reward: float, 
                            actions: np.ndarray, states: np.ndarray, 
                            timing: Dict[str, float], env):
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
        # Reward bounds check
        self.assertGreaterEqual(reward, config.expected_reward_min, 
                               f"Reward {reward:.3f} below minimum {config.expected_reward_min}")
        self.assertLessEqual(reward, config.expected_reward_max,
                            f"Reward {reward:.3f} above maximum {config.expected_reward_max}")
        print(f"✓ Reward {reward:.3f} within expected range [{config.expected_reward_min}, {config.expected_reward_max}]")
        
        # # Goal distance check
        # final_state = states[-1]
        # goal_distance = self.compute_goal_distance(final_state, config, env)
        # self.assertLessEqual(goal_distance, config.min_final_distance_to_goal,
        #                     f"Final distance to goal {goal_distance:.3f} too large")
        # print(f"✓ Final goal distance: {goal_distance:.3f}m (limit: {config.min_final_distance_to_goal}m)")
        
        # Basic sanity checks
        self.assertEqual(len(actions), config.Hsample, "Action sequence length mismatch")
        self.assertEqual(len(states), config.Hsample + 1, "State sequence length mismatch")  # +1 for initial state
        self.assertEqual(actions.shape[1], 2, "Actions should have 2 dimensions (v, delta)")
        self.assertEqual(states.shape[1], 4, "States should have 4 dimensions (x, y, theta1, theta2)")
        #print("✓ Array dimensions are correct")
        
    def compute_goal_distance(self, final_state: np.ndarray, config: TestConfig, env) -> float:
        # Get the actual goal position from the environment
        goal_state = env.xg  # This is the actual goal state set in the environment
        goal_pos = goal_state[:2]  # x, y coordinates
        
        final_pos = final_state[:2]  # x, y only
        return np.linalg.norm(final_pos - goal_pos)
        
    def save_test_results(self, config: TestConfig, reward: float, 
                         actions: np.ndarray, states: np.ndarray, 
                         timing: Dict[str, float], env):
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
        result_data = {
            'config': asdict(config),
            'reward': float(reward),
            'final_state': states[-1].tolist(),
            'timing': timing,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self.get_git_commit(),
            'test_summary': {
                'goal_distance': float(self.compute_goal_distance(states[-1], config, env)),
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
        
        print(f"Results saved to: {result_file}")
        print(f"Arrays saved to: {arrays_file}")
        
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
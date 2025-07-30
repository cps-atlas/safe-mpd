"""
Statistical Evaluation for MBD Planner
=====================================

This module runs multiple diffusion trials with varied initial conditions
and reports comprehensive performance metrics for hyperparameter optimization.

OBSTACLE AND ENVIRONMENT CONFIGURATION:
======================================
To modify obstacles and parking configuration, edit the following functions:

1. get_default_parking_config() - Parking layout and occupied spaces
2. create_circular_obstacles() - Circular obstacles in occupied parking spaces  
3. create_rectangular_obstacles() - Rectangular boundary walls

These functions contain the default obstacle setup copied from the MBD environment
for easy local modification without changing the core MBD code.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import jax
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

# Add the MBD source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import mbd
from mbd.planners.mbd_planner import MBDConfig, run_diffusion, clear_jit_cache
from mbd.envs.env import Env

@dataclass
class StatisticalResults:
    """Results from statistical evaluation of diffusion planner"""
    success_rate: float
    avg_position_error: float
    collision_rate: float
    jackknife_rate: float
    avg_pure_diffusion_time: float
    std_pure_diffusion_time: float
    num_trials: int
    successful_trials: int
    individual_results: List[Dict]  # Detailed results for each trial


@dataclass 
class TrialConfig:
    """Configuration for a single trial"""
    dx: float  # Distance from target parking space center (x-direction)
    dy: float  # Distance from parking lot entrance line (y-direction)  
    theta1: float  # Initial tractor orientation
    theta2: float  # Initial trailer orientation
    trial_id: int


def sample_initial_conditions(num_trials: int, seed: int = 42) -> List[TrialConfig]:
    """
    Sample diverse initial conditions for statistical evaluation.
    Ensures sampled conditions don't violate constraints (collisions, jackknifing).
    
    Args:
        num_trials: Number of trials to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of TrialConfig objects with valid initial conditions
    """
    rng = np.random.RandomState(seed)
    
    configs = []
    attempts = 0
    max_attempts = num_trials * 10  # Prevent infinite loops
    
    while len(configs) < num_trials and attempts < max_attempts:
        attempts += 1
        
        # Sample dx from -12 to 12 (distance from target x to tractor front face)
        dx = rng.uniform(-12.0, 12.0)
        
        # Sample dy from 1.0 to 8.0 (distance from entrance line, must be >= 1.0)
        dy = rng.uniform(2.0, 7.0)
        
        # Sample orientations - use backwards-facing range around π
        # Sample from π - π/10 to π + π/10 for more realistic parking orientations
        theta1 = rng.uniform(-np.pi/10, np.pi/10) if rng.random() < 0.5 else rng.uniform(np.pi - np.pi/10, np.pi + np.pi/10)
        theta2 = theta1 + rng.uniform(-np.pi/10, np.pi/10)
        
        # Create trial config
        trial_config = TrialConfig(
            dx=dx, dy=dy, theta1=theta1, theta2=theta2, trial_id=len(configs)
        )
        
        # Check if this initial condition is valid (no constraint violations)
        if is_valid_initial_condition(trial_config):
            configs.append(trial_config)
        # If invalid, continue sampling (don't add to configs)
    
    if len(configs) < num_trials:
        print(f"Warning: Only found {len(configs)} valid initial conditions out of {num_trials} requested after {attempts} attempts")
    
    return configs


def get_default_parking_config():
    """
    Default parking configuration for parking scenario
    Copied from mbd.envs.env.Env.get_default_parking_config() for easy modification
    """
    return {
        'parking_rows': 2,
        'parking_cols': 8,
        'space_width': 3.5,     # Width of each parking space
        'space_length': 7.0,    # Length of each parking space
        'parking_y_offset': 4.0, # Distance from start area to parking lot
        'occupied_spaces': [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15],  # 1-indexed occupied spaces
        'target_spaces': [4, 12],  # Target spaces: tractor in 3, trailer in 11
        'obstacle_radius': 1.0,   # Radius of obstacles in occupied spaces
    }

def create_rectangular_obstacles():
    """
    Create rectangular obstacles for parking scenario
    Copied from mbd.envs.env.Env.set_obs_rectangle_parking() for easy modification
    
    Returns:
        List of rectangular obstacles [x_center, y_center, width, height, rotation]
    """
    obs_rectangles = [
        [0.0, -14.0, 30.0, 1.0, 0.0]   # Bottom boundary wall
    ]
    return obs_rectangles


def is_valid_initial_condition(trial_config: TrialConfig) -> bool:
    """
    Check if an initial condition violates any constraints.
    Uses local obstacle configuration for easy modification.
    
    Args:
        trial_config: Initial condition to validate
        
    Returns:
        True if initial condition is valid (no violations), False otherwise
    """
    try:
        # Set up obstacles using local configuration (easily modifiable)
        parking_config = get_default_parking_config()
        temp_parking_env = Env(case="parking", parking_config=parking_config)
        
        # Create a temporary environment to check constraints
        # Use minimal configuration for constraint checking
        temp_env = mbd.envs.get_env(
            env_name="tt2d",
            case="parking",
            env_config=temp_parking_env
        )
        
        
        # Create obstacles using local functions
        obs_rectangles = create_rectangular_obstacles()
        
        # Override environment obstacles with our local configuration
        temp_env.obs_rectangles = jnp.array(obs_rectangles)
        
        # Set the initial position
        temp_env.set_init_pos(
            dx=trial_config.dx,
            dy=trial_config.dy,
            theta1=trial_config.theta1,
            theta2=trial_config.theta2
        )
        
        # Get initial state
        initial_state = temp_env.x0  # [px, py, theta1, theta2]
        
        # Check obstacle collision
        obstacle_collision = temp_env.check_obstacle_collision(
            initial_state, temp_env.obs_circles, temp_env.obs_rectangles
        )
        
        # Check hitch violation (jackknifing)
        hitch_violation = temp_env.check_hitch_violation(initial_state)
        
        # Valid if no violations
        is_valid = not (obstacle_collision or hitch_violation)
        
        return is_valid
        
    except Exception as e:
        # If any error occurs during validation, consider invalid
        print(f"Warning: Error validating initial condition: {str(e)}")
        return False


def create_trial_environment(base_config: MBDConfig, trial_config: TrialConfig):
    """
    Create environment for a specific trial with given initial conditions.
    Uses local obstacle configuration for easy modification.
    
    Args:
        base_config: Base MBD configuration
        trial_config: Trial-specific initial conditions
        
    Returns:
        Configured environment ready for diffusion run
    """
    # Set up obstacles using local configuration (easily modifiable)
    parking_config = get_default_parking_config()
    temp_parking_env = Env(case="parking", parking_config=parking_config)
    # Create environment with base configuration
    env = mbd.envs.get_env(
        base_config.env_name,
        case=base_config.case,
        env_config=temp_parking_env,
        dt=base_config.dt,
        H=base_config.Hsample,
        motion_preference=base_config.motion_preference,
        collision_penalty=base_config.collision_penalty,
        hitch_penalty=base_config.hitch_penalty,
        enable_gated_rollout_collision=base_config.enable_gated_rollout_collision,
        enable_gated_rollout_hitch=base_config.enable_gated_rollout_hitch,
        enable_projection=base_config.enable_projection,
        enable_guidance=base_config.enable_guidance,
        reward_threshold=base_config.reward_threshold,
        ref_reward_threshold=base_config.ref_reward_threshold,
        max_w_theta=base_config.max_w_theta,
        hitch_angle_weight=base_config.hitch_angle_weight,
        # Physical parameters
        l1=base_config.l1,
        l2=base_config.l2,
        lh=base_config.lh,
        lf1=base_config.lf1,
        lr=base_config.lr,
        lf2=base_config.lf2,
        lr2=base_config.lr2,
        tractor_width=base_config.tractor_width,
        trailer_width=base_config.trailer_width,
        # Input constraints
        v_max=base_config.v_max,
        delta_max_deg=base_config.delta_max_deg,
        a_max=base_config.a_max,
        omega_max=base_config.omega_max,
        # Reward shaping parameters
        d_thr_factor=base_config.d_thr_factor,
        k_switch=base_config.k_switch,
        steering_weight=base_config.steering_weight,
        preference_penalty_weight=base_config.preference_penalty_weight,
        heading_reward_weight=base_config.heading_reward_weight,
        terminal_reward_threshold=base_config.terminal_reward_threshold,
        terminal_reward_weight=base_config.terminal_reward_weight,
        ref_pos_weight=base_config.ref_pos_weight,
        ref_theta1_weight=base_config.ref_theta1_weight,
        ref_theta2_weight=base_config.ref_theta2_weight,
    )
    
    
    # Create obstacles using local functions
    obs_rectangles = create_rectangular_obstacles()
    
    # Override environment obstacles with our local configuration
    env.obs_rectangles = jnp.array(obs_rectangles)
    env.env.obs_rectangles = jnp.array(obs_rectangles)
    
    # Set trial-specific initial conditions
    env.set_init_pos(
        dx=trial_config.dx,
        dy=trial_config.dy, 
        theta1=trial_config.theta1,
        theta2=trial_config.theta2
    )
    
    # Set goal position for parking (standard target)
    env.set_goal_pos(theta1=-jnp.pi/2, theta2=-jnp.pi/2)
    
    return env


def evaluate_trial_result(final_trajectory_state: jnp.ndarray, 
                         env, 
                         goal_position_threshold: float = 4.5) -> Dict:
    """
    Evaluate the results of a single diffusion trial.
    
    Args:
        final_trajectory_state: Final state from trajectory [px, py, theta1, theta2, ...]
        env: Environment used for the trial
        goal_position_threshold: Distance threshold for success determination
        
    Returns:
        Dictionary with trial evaluation metrics
    """
    # Extract final state components
    final_state_4d = final_trajectory_state[:4]
    px, py, theta1, theta2 = final_state_4d
    
    # Calculate position errors for both tractor and trailer
    goal_px, goal_py = env.xg[0], env.xg[1]
    
    # Tractor position (reference point)
    tractor_pos_error = np.sqrt((px - goal_px)**2 + (py - goal_py)**2)
    
    # Trailer back position (we care about the back when parking backward)
    trailer_positions = env.get_trailer_back_position(final_state_4d)
    trailer_px, trailer_py = trailer_positions[0], trailer_positions[1]
    trailer_pos_error = np.sqrt((trailer_px - goal_px)**2 + (trailer_py - goal_py)**2)
    
    # Use minimum position error (better of tractor/trailer positioning)
    final_position_error = min(tractor_pos_error, trailer_pos_error)
    
    # Check constraint violations
    obstacle_collision = env.check_obstacle_collision(final_state_4d, env.obs_circles, env.obs_rectangles)
    hitch_violation = env.check_hitch_violation(final_state_4d)
    
    # Determine success
    is_successful = (final_position_error <= goal_position_threshold and 
                    not obstacle_collision and 
                    not hitch_violation)
    
    return {
        'success': bool(is_successful),
        'position_error': float(final_position_error),
        'tractor_error': float(tractor_pos_error),
        'trailer_error': float(trailer_pos_error),
        'collision': bool(obstacle_collision),
        'jackknife': bool(hitch_violation),
        'final_state': final_trajectory_state
    }


def run_statistical_evaluation(config: MBDConfig, 
                              num_trials: int = 20,
                              seed: int = 42,
                              verbose: bool = True) -> StatisticalResults:
    """
    Run statistical evaluation with multiple trials and diverse initial conditions.
    
    Args:
        config: MBD configuration for diffusion planner
        num_trials: Number of trials to run
        seed: Random seed for reproducibility  
        verbose: Whether to print progress information
        
    Returns:
        StatisticalResults with comprehensive performance metrics
    """
    if verbose:
        print(f"Starting statistical evaluation with {num_trials} trials...")
        print(f"Configuration: {config.env_name}, case={config.case}")
    
    # Clear JIT cache to ensure fresh compilation
    #clear_jit_cache()
    
    # Sample initial conditions
    trial_configs = sample_initial_conditions(num_trials, seed)
    
    # Storage for results
    individual_results = []
    pure_diffusion_times = []
    successful_trials = 0
    
    # Run trials
    for i, trial_config in enumerate(trial_configs):
        if verbose:
            print(f"\nTrial {i+1}/{num_trials}: dx={trial_config.dx:.2f}, dy={trial_config.dy:.2f}, "
                  f"θ1={trial_config.theta1:.2f}, θ2={trial_config.theta2:.2f}")
        
        try:
            # Create environment for this trial
            env = create_trial_environment(config, trial_config)
            
            # Run diffusion
            rew_final, Y0, trajectory_states, timing_info = run_diffusion(args=config, env=env)
            
            # Evaluate trial result
            final_state = trajectory_states[-1]  # Last state in trajectory
            trial_result = evaluate_trial_result(final_state, env)
            
            # Add timing information
            trial_result['pure_diffusion_time'] = timing_info['pure_diffusion_time']
            trial_result['total_time'] = timing_info['total_time']
            trial_result['final_reward'] = float(rew_final)
            trial_result['trial_config'] = trial_config
            
            # Store results
            individual_results.append(trial_result)
            pure_diffusion_times.append(timing_info['pure_diffusion_time'])
            
            if trial_result['success']:
                successful_trials += 1
            
            if verbose:
                status = "SUCCESS" if trial_result['success'] else "FAILED"
                print(f"  Result: {status} | Pos Error: {trial_result['position_error']:.3f} | "
                      f"Time: {trial_result['pure_diffusion_time']:.3f}s | "
                      f"Collision: {trial_result['collision']} | Jackknife: {trial_result['jackknife']}")
                
        except Exception as e:
            if verbose:
                print(f"  ERROR in trial {i+1}: {str(e)}")
            # Record failed trial
            individual_results.append({
                'success': False,
                'position_error': float('inf'),
                'collision': True,  # Assume worst case
                'jackknife': True,
                'pure_diffusion_time': 0.0,
                'error': str(e),
                'trial_config': trial_config
            })
    
    # Compute aggregate statistics
    success_rate = successful_trials / num_trials
    
    # Position errors (only from successful trials to avoid inf values)
    successful_results = [r for r in individual_results if r['success']]
    if successful_results:
        avg_position_error = np.mean([r['position_error'] for r in successful_results])
    else:
        avg_position_error = float('inf')  # No successful trials
    
    # Constraint violation rates
    collision_rate = np.mean([r.get('collision', True) for r in individual_results])
    jackknife_rate = np.mean([r.get('jackknife', True) for r in individual_results])
    
    # Timing statistics (only from completed trials)
    completed_times = [t for t in pure_diffusion_times if t > 0]
    if completed_times:
        avg_pure_diffusion_time = np.mean(completed_times)
        std_pure_diffusion_time = np.std(completed_times)
    else:
        avg_pure_diffusion_time = 0.0
        std_pure_diffusion_time = 0.0
    
    results = StatisticalResults(
        success_rate=success_rate,
        avg_position_error=avg_position_error,
        collision_rate=collision_rate,
        jackknife_rate=jackknife_rate,
        avg_pure_diffusion_time=avg_pure_diffusion_time,
        std_pure_diffusion_time=std_pure_diffusion_time,
        num_trials=num_trials,
        successful_trials=successful_trials,
        individual_results=individual_results
    )
    
    if verbose:
        print(f"\n=== STATISTICAL EVALUATION RESULTS ===")
        print(f"Success Rate: {success_rate:.1%} ({successful_trials}/{num_trials})")
        print(f"Avg Position Error: {avg_position_error:.3f}m (successful trials only)")
        print(f"Collision Rate: {collision_rate:.1%}")
        print(f"Jackknife Rate: {jackknife_rate:.1%}")
        print(f"Avg Pure Diffusion Time: {avg_pure_diffusion_time:.3f}±{std_pure_diffusion_time:.3f}s")
        print("=" * 40)
    
    return results


def main():
    """Example usage of statistical evaluation"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create base configuration for parking scenario
    config = MBDConfig(
        # Core settings
        env_name="tt2d",
        case="parking", 
        motion_preference=0,  # No motion preference
        enable_demo=False,    # No demonstration
        # Reduce computational cost for testing
        Nsample=20000,
        Hsample=50,
        Ndiffuse=100,
        # Disable rendering for batch evaluation
        render=True,
        save_animation=False,
        show_animation=False,
        save_denoising_animation=False,
        verbose=False,
        # Algorithm
        enable_gated_rollout_collision=True,
        enable_gated_rollout_hitch=True,
        enable_projection=False,
        enable_guidance=False
    )
    
    # Run statistical evaluation
    results = run_statistical_evaluation(
        config=config,
        num_trials=10,  # Small number for testing
        seed=42,
        verbose=True
    )
    
    print(f"\nFinal Summary:")
    print(f"Overall success rate: {results.success_rate:.1%}")
    print(f"Average computation time: {results.avg_pure_diffusion_time:.3f}s")


if __name__ == "__main__":
    main() 
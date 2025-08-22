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
import matplotlib.pyplot as plt

# Add the MBD source path - go up one level from tests to reach mbd module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
        #'occupied_spaces': [3, 5, 11, 13],  # 1-indexed occupied spaces
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
    # obs_rectangles = [
    # ]
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
        enable_shielded_rollout_collision=base_config.enable_shielded_rollout_collision,
        enable_shielded_rollout_hitch=base_config.enable_shielded_rollout_hitch,
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
        num_trailers=base_config.num_trailers,
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


def evaluate_trial_result(trajectory_states: jnp.ndarray,
                         env,
                         goal_position_threshold: float = 4.0) -> Dict:
    """
    Evaluate the results of a single diffusion trial.
    
    Args:
        trajectory_states: Full trajectory states array of shape (T, state_dim)
        env: Environment used for the trial
        goal_position_threshold: Distance threshold for success determination
        
    Returns:
        Dictionary with trial evaluation metrics
    """
    # Use final state for position error computation
    final_state = trajectory_states[-1]
    final_state_4d = final_state[:4]
    px, py = final_state_4d[:2]
    
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
    
    # Check constraint violations across the entire trajectory
    obstacle_collision = False
    hitch_violation = False
    for t in range(trajectory_states.shape[0]):
        state_4d = trajectory_states[t][:4]
        # If either violation occurs at any timestep, mark as True
        if env.check_obstacle_collision(state_4d, env.obs_circles, env.obs_rectangles):
            obstacle_collision = True
        if env.check_hitch_violation(state_4d):
            hitch_violation = True
        if obstacle_collision and hitch_violation:
            break
    
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
        'final_state': final_state
    }


def run_statistical_evaluation(config: MBDConfig, 
                              num_trials: int = 20,
                              seed: int = 42,
                              verbose: bool = True,
                              show_heat_map: bool = False) -> StatisticalResults:
    """
    Run statistical evaluation with multiple trials and diverse initial conditions.
    
    Args:
        config: MBD configuration for diffusion planner
        num_trials: Number of trials to run
        seed: Random seed for reproducibility  
        verbose: Whether to print progress information
        show_heat_map: Whether to display heat map of initial positions with success/failure markers
        
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
        
        # Create environment for this trial
        env = create_trial_environment(config, trial_config)
        
        # Run diffusion
        rew_final, Y0, trajectory_states, timing_info = run_diffusion(args=config, env=env)
        
        # Evaluate trial result using entire trajectory for constraint checks
        trial_result = evaluate_trial_result(trajectory_states, env)
        
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
    
    # Generate heat map visualization if requested
    if show_heat_map:
        create_heat_map_visualization(individual_results, env, verbose)
    
    return results


def create_heat_map_visualization(individual_results: List[Dict], env, verbose: bool = True, return_fig: bool = False):
    """
    Create heat map visualization showing initial positions and success/failure outcomes.
    
    Args:
        individual_results: List of trial results with success/failure info
        env: Environment used for rendering
        verbose: Whether to print status messages
        return_fig: If True, return the figure object instead of showing it
        
    Returns:
        matplotlib.figure.Figure if return_fig=True, otherwise None
    """
    if verbose:
        print("Generating heat map visualization...")
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Render environment (obstacles, parking spaces, etc.) with empty trajectory
    env.render(ax, jnp.array([]).reshape(0, 4))
    
    # Extract initial positions and success status
    successful_positions = []
    failed_positions = []
    
    for result in individual_results:
        if 'trial_config' in result and 'error' not in result:
            trial_config = result['trial_config']
            
            # Set the initial position for this trial to get correct x, y coordinates
            env.set_init_pos(
                dx=trial_config.dx,
                dy=trial_config.dy,
                theta1=trial_config.theta1,
                theta2=trial_config.theta2
            )
            
            # Get the actual x, y coordinates from environment
            initial_x, initial_y = float(env.x0[0]), float(env.x0[1])
            
            if result['success']:
                successful_positions.append((initial_x, initial_y))
            else:
                failed_positions.append((initial_x, initial_y))
    
    # Plot successful trials as green circles
    if successful_positions:
        success_x, success_y = zip(*successful_positions)
        ax.scatter(success_x, success_y, c='green', marker='o', s=120, alpha=0.7, 
                    edgecolors='darkgreen', linewidth=2, label=f'Success ({len(successful_positions)})')
    
    # Plot failed trials as red crosses  
    if failed_positions:
        fail_x, fail_y = zip(*failed_positions)
        ax.scatter(fail_x, fail_y, c='red', marker='x', s=120, alpha=0.7,
                    linewidth=3, label=f'Failed ({len(failed_positions)})')
    
    # Handle case where no positions were extracted
    if not successful_positions and not failed_positions:
        ax.text(0.5, 0.5, 'No valid trial positions found', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Customize plot
    if not return_fig:
        ax.set_title('MBD Planner Performance Heat Map\nInitial Positions vs Success/Failure', 
                    fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    # Add summary statistics as text
    total_trials = len(individual_results)
    success_rate = len(successful_positions) / total_trials if total_trials > 0 else 0
    ax.text(0.02, 0.98, f'Success Rate: {success_rate:.1%}\nTotal Trials: {total_trials}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if return_fig:
        # Return the figure object for external use (e.g., W&B upload)
        if verbose:
            print(f"Heat map created: {len(successful_positions)} successful, {len(failed_positions)} failed trials")
        return fig
    else:
        # Show the plot as before
        plt.savefig(f"heat_map.png")
        plt.savefig(f"heat_map.svg")
        plt.show()
        if verbose:
            print(f"Heat map displayed: {len(successful_positions)} successful, {len(failed_positions)} failed trials")
        return None
        


def main():
    """Example usage of statistical evaluation"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create base configuration for parking scenario
    config = MBDConfig(
        # Core settings
        env_name="acc_tt2d",
        case="parking", 
        motion_preference=0,  # No motion preference
        enable_demo=False,    # No demonstration
        # Reduce computational cost for testing
        Nsample=20000,
        Hsample=50,
        Ndiffuse=100,
        # Disable rendering for batch evaluation
        render=False,
        save_animation=False,
        show_animation=False,
        save_denoising_animation=False,
        verbose=False,
        # Algorithm
        enable_shielded_rollout_collision=True,
        enable_shielded_rollout_hitch=True,
        enable_projection=False,
        enable_guidance=False,
        terminal_reward_weight=5.78395,
        terminal_reward_threshold=10.0,
        temp_sample=0.0001,
        steering_weight=0.01,
        reward_threshold=50.0,
        k_switch=0.1,
        hitch_angle_weight=0.01,
        d_thr_factor=0.5,
        num_trailers=1
    )
    
    # hyperparmeters found for tt2d, 250802
        # terminal_reward_weight=5.78395,
        # terminal_reward_threshold=10.0,
        # temp_sample=0.0001,
        # steering_weight=0.01,
        # reward_threshold=50.0,
        # k_switch=0.1,
        # hitch_angle_weight=0.01,
        # d_thr_factor=0.5
        
    # hyperparmeters found for acc_tt2d, 250804
        # terminal_reward_weight=10.0,
        # terminal_reward_threshold=10.0,
        # temp_sample=0.0001,
        # steering_weight=0.01,
        # reward_threshold=10.0,
        # k_switch=5.0,
        # hitch_angle_weight=0.01,
        # d_thr_factor=5.0
        
    
    # Run statistical evaluation
    results = run_statistical_evaluation(
        config=config,
        num_trials=100,  # Small number for testing
        seed=42,
        verbose=True,
        show_heat_map=True  # Enable heat map visualization
    )
    
    print(f"\nFinal Summary:")
    print(f"Overall success rate: {results.success_rate:.1%}")
    print(f"Average computation time: {results.avg_pure_diffusion_time:.3f}s")


if __name__ == "__main__":
    main() 
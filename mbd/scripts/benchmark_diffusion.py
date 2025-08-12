#!/usr/bin/env python3
"""
Benchmark script for diffusion-based planning.

This script demonstrates proper timing measurement that excludes JIT compilation time.
"""

import time
import jax.numpy as jnp
import sys
import os

# Add the parent directory to path to import mbd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mbd
from mbd.planners.mbd_planner import MBDConfig, run_diffusion
import numpy as np


def benchmark_diffusion(args=None, env=None, num_trials=5, warmup_trials=1):
    """
    Run multiple diffusion trials for benchmarking.
    
    Args:
        args: Configuration for diffusion
        env: Environment object
        num_trials: Number of trials to run for benchmarking
        warmup_trials: Number of warmup trials (results not counted)
    
    Returns:
        benchmark_results: Dictionary with timing statistics
    """
    print(f"\n=== DIFFUSION BENCHMARK: {warmup_trials} warmup + {num_trials} trials ===")
    
    all_timing_info = []
    all_rewards = []
    
    # Warmup trials (results discarded)
    print(f"Running {warmup_trials} warmup trial(s)...")
    for i in range(warmup_trials):
        print(f"\nWarmup trial {i+1}/{warmup_trials}")
        rew, _, _, timing = run_diffusion(args=args, env=env)
        print(f"Warmup {i+1} - Pure diffusion time: {timing['pure_diffusion_time']:.3f}s, Reward: {rew:.3e}")
    
    # Actual benchmark trials
    print(f"\nRunning {num_trials} benchmark trial(s)...")
    for i in range(num_trials):
        print(f"\nBenchmark trial {i+1}/{num_trials}")
        rew, _, _, timing = run_diffusion(args=args, env=env)
        all_timing_info.append(timing)
        all_rewards.append(rew)
        print(f"Trial {i+1} - Pure diffusion time: {timing['pure_diffusion_time']:.3f}s, Reward: {rew:.3e}")
    
    # Compute statistics (excluding rendering time)
    pure_diffusion_times = [t['pure_diffusion_time'] for t in all_timing_info]
    total_times = [t['total_time'] for t in all_timing_info]
    overhead_times = [t['overhead_time'] for t in all_timing_info]
    compilation_times = [t['compilation_time'] for t in all_timing_info]
    
    benchmark_results = {
        'num_trials': num_trials,
        'pure_diffusion_time': {
            'mean': np.mean(pure_diffusion_times),
            'std': np.std(pure_diffusion_times),
            'min': np.min(pure_diffusion_times),
            'max': np.max(pure_diffusion_times),
            'median': np.median(pure_diffusion_times),
        },
        'total_time': {
            'mean': np.mean(total_times),
            'std': np.std(total_times),
            'min': np.min(total_times),
            'max': np.max(total_times),
            'median': np.median(total_times),
        },
        'overhead_time': {
            'mean': np.mean(overhead_times),
            'std': np.std(overhead_times),
            'min': np.min(overhead_times),
            'max': np.max(overhead_times),
            'median': np.median(overhead_times),
        },
        'compilation_time': {
            'mean': np.mean(compilation_times),
            'std': np.std(compilation_times),
            'min': np.min(compilation_times),
            'max': np.max(compilation_times),
            'median': np.median(compilation_times),
        },
        'rewards': {
            'mean': np.mean(all_rewards),
            'std': np.std(all_rewards),
            'min': np.min(all_rewards),
            'max': np.max(all_rewards),
            'median': np.median(all_rewards),
        },
        'all_timing_info': all_timing_info,
        'all_rewards': all_rewards,
    }
    
    # Print benchmark summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Configuration: Nsample={args.Nsample}, Hsample={args.Hsample}, Ndiffuse={args.Ndiffuse}")
    print(f"Number of trials: {num_trials}")
    print()
    print("PURE DIFFUSION TIME (excluding compilation & overhead):")
    print(f"  Mean:   {benchmark_results['pure_diffusion_time']['mean']:.3f} ± {benchmark_results['pure_diffusion_time']['std']:.3f} seconds")
    print(f"  Median: {benchmark_results['pure_diffusion_time']['median']:.3f} seconds")
    print(f"  Min:    {benchmark_results['pure_diffusion_time']['min']:.3f} seconds")
    print(f"  Max:    {benchmark_results['pure_diffusion_time']['max']:.3f} seconds")
    print()
    print("TOTAL TIME (including all overhead):")
    print(f"  Mean:   {benchmark_results['total_time']['mean']:.3f} ± {benchmark_results['total_time']['std']:.3f} seconds")
    print(f"  Median: {benchmark_results['total_time']['median']:.3f} seconds")
    print()
    print("OVERHEAD TIME:")
    print(f"  Mean:   {benchmark_results['overhead_time']['mean']:.3f} ± {benchmark_results['overhead_time']['std']:.3f} seconds")
    print(f"  % of total: {benchmark_results['overhead_time']['mean']/benchmark_results['total_time']['mean']*100:.1f}%")
    print()
    print("COMPILATION TIME (part of overhead):")
    print(f"  Mean:   {benchmark_results['compilation_time']['mean']:.3f} ± {benchmark_results['compilation_time']['std']:.3f} seconds")
    print(f"  % of total: {benchmark_results['compilation_time']['mean']/benchmark_results['total_time']['mean']*100:.1f}%")
    print()
    print("FINAL REWARDS:")
    print(f"  Mean:   {benchmark_results['rewards']['mean']:.3e} ± {benchmark_results['rewards']['std']:.3e}")
    print(f"  Median: {benchmark_results['rewards']['median']:.3e}")
    print("="*60)
    
    return benchmark_results


def benchmark_parking_scenario():
    """Benchmark the parking scenario with different configurations"""
    
    print("Benchmarking Tractor-Trailer Parking")
    print("=" * 50)
    
    # Base configuration for parking scenario
    base_config = MBDConfig(
        env_name="tt2d",
        case="parking", 
        motion_preference=-1,  # backward parking
        enable_demo=True,
        render=False,  # Disable rendering for benchmarking
        save_animation=False,
        show_animation=False,
        save_denoising_animation=False,
        dt=0.25
    )
    
    # Test configurations with different parameters
    test_configs = [
        # Small test case
        {
            "name": "Small (Fast)",
            "Nsample": 5000,
            "Hsample": 30,
            "Ndiffuse": 100,
        },
        # Medium test case  
        {
            "name": "Medium",
            "Nsample": 10000,
            "Hsample": 40,
            "Ndiffuse": 120,
        },
        # Large test case (original default)
        {
            "name": "Large (Original)",
            "Nsample": 20000,
            "Hsample": 50,
            "Ndiffuse": 150,
        }
    ]
    
    results = {}
    
    for test_config in test_configs:
        print(f"\n{'='*20} {test_config['name']} {'='*20}")
        
        # Create config for this test
        config = MBDConfig(
            **{**vars(base_config), **{k: v for k, v in test_config.items() if k != 'name'}}
        )
        
        # Create environment
        env = mbd.envs.get_env(
            config.env_name,
            case=config.case,
            dt=config.dt,
            H=config.Hsample,
            motion_preference=config.motion_preference,
            collision_penalty=config.collision_penalty,
                    enable_shielded_rollout_collision=config.enable_shielded_rollout_collision,
        hitch_penalty=config.hitch_penalty,
        enable_shielded_rollout_hitch=config.enable_shielded_rollout_hitch,
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
            d_thr_factor=config.d_thr_factor,
            k_switch=config.k_switch,
            steering_weight=config.steering_weight,
            preference_penalty_weight=config.preference_penalty_weight,
            heading_reward_weight=config.heading_reward_weight,
            ref_pos_weight=config.ref_pos_weight,
            ref_theta1_weight=config.ref_theta1_weight,
            ref_theta2_weight=config.ref_theta2_weight
        )
        
        # Set initial position using geometric parameters
        env.set_init_pos(dx=-3.0, dy=1.0, theta1=0, theta2=0)
        env.set_goal_pos(theta1=jnp.pi/2, theta2=jnp.pi/2)  # backward parking
        
        # Run benchmark
        benchmark_result = benchmark_diffusion(
            args=config, 
            env=env, 
            num_trials=3,  # Run 3 trials for each configuration
            warmup_trials=1  # 1 warmup trial
        )
        
        results[test_config['name']] = benchmark_result
    
    # Print comparison summary
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Config':<15} {'Nsample':<8} {'Hsample':<8} {'Ndiffuse':<9} {'Pure Time':<12} {'Total Time':<12} {'Reward':<12}")
    print("-" * 80)
    
    for config_name, result in results.items():
        config = next(c for c in test_configs if c['name'] == config_name)
        print(f"{config_name:<15} {config['Nsample']:<8} {config['Hsample']:<8} {config['Ndiffuse']:<9} "
              f"{result['pure_diffusion_time']['mean']:.2f}±{result['pure_diffusion_time']['std']:.2f}s  "
              f"{result['total_time']['mean']:.2f}±{result['total_time']['std']:.2f}s  "
              f"{result['rewards']['mean']:.2e}")
    
    print("="*80)
    print("Pure Time = Actual diffusion computation (excludes JIT compilation)")
    print("Total Time = Everything including overhead, rendering, etc.")
    
    return results


if __name__ == "__main__":
    print("Model-Based Diffusion Benchmarking Suite")
    print("This benchmark measures pure diffusion computation time")
    print("(excluding JAX JIT compilation overhead)")
    
    # Run comparison benchmark
    print("\n" + "="*60) 
    print("MULTI-CONFIGURATION COMPARISON")
    print("="*60)
    comparison_results = benchmark_parking_scenario()
    
    print(f"\nBenchmarking completed!")
    print(f"Key takeaway: Use 'pure_diffusion_time' for performance comparisons")
    print(f"This excludes JIT compilation and other overhead.") 
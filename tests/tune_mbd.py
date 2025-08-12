"""
Hyperparameter Optimization for MBD Planner
==========================================

This module uses Optuna with Gaussian Process optimization and MedianPruner to tune 
diffusion planner hyperparameters based on statistical evaluation results.

Key Features:
- Gaussian Process (GP) sampler for efficient hyperparameter exploration
- MedianPruner for early termination of unpromising trials
- Comprehensive W&B logging with heat maps and pruning statistics
- Success rate optimization (single objective)
- Automatic heat map generation showing spatial success/failure patterns
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Third-party imports
import optuna
from optuna.samplers import GPSampler
from optuna.pruners import MedianPruner
import wandb
import matplotlib.pyplot as plt

# Add the MBD source path - go up one level from tests to reach mbd module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mbd.planners.mbd_planner import MBDConfig, clear_jit_cache
from stat_mbd import run_statistical_evaluation, StatisticalResults, create_trial_environment, create_heat_map_visualization


class DiffusionOptimizer:
    """Hyperparameter optimizer for MBD diffusion planner"""
    
    def __init__(self, 
                 base_config: MBDConfig,
                 num_trials_per_eval: int = 20,
                 study_name: str = "diffusion_optimization",
                 use_wandb: bool = True):
        """
        Initialize the optimizer.
        
        Args:
            base_config: Base configuration with fixed parameters
            num_trials_per_eval: Number of statistical trials per hyperparameter evaluation
            study_name: Name for the Optuna study
            use_wandb: Whether to use Weights & Biases logging
        """
        self.base_config = base_config
        self.num_trials_per_eval = num_trials_per_eval
        self.study_name = study_name
        self.use_wandb = use_wandb
        
        # Initialize W&B if requested
        if self.use_wandb:
            wandb.init(
                project="mbd-diffusion-optimization",
                name=f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self._config_to_dict(base_config)
            )
        
        # Storage for results
        self.optimization_results = []
        self.best_trial_config = None
        self.best_trial_env = None
    
    def _config_to_dict(self, config: MBDConfig) -> Dict[str, Any]:
        """Convert MBDConfig to dictionary for logging"""
        return {
            'env_name': config.env_name,
            'case': config.case,
            'Nsample': config.Nsample,
            'Hsample': config.Hsample,
            'Ndiffuse': config.Ndiffuse,
            'motion_preference': config.motion_preference,
            'enable_demo': config.enable_demo,
            # Physical parameters (fixed)
            'l1': config.l1, 'l2': config.l2, 'lh': config.lh,
            'lf1': config.lf1, 'lr': config.lr, 'lf2': config.lf2, 'lr2': config.lr2,
            'tractor_width': config.tractor_width,
            'trailer_width': config.trailer_width,
            'v_max': config.v_max,
            'delta_max_deg': config.delta_max_deg,
        }
    
    def create_heat_map(self, results: StatisticalResults, config: MBDConfig, trial_number: int):
        """
        Create heat map visualization.
        
        Args:
            results: Statistical evaluation results
            config: Configuration used for the trial
            trial_number: Current trial number for logging
        """
        if not self.use_wandb:
            return None
        
        # Create a representative environment for rendering
        sample_trial_config = results.individual_results[0]['trial_config'] if results.individual_results else None
        if sample_trial_config is None:
            return None
            
        env = create_trial_environment(config, sample_trial_config)
        
        # Use the heat map creation function from stat_mbd.py
        fig = create_heat_map_visualization(
            individual_results=results.individual_results,
            env=env,
            verbose=False,  # Quiet during optimization
            return_fig=True  # Return figure instead of showing
        )
        
        if fig is None:
            print(f"Warning: Failed to create heat map for trial {trial_number}")
            return
        
        # Update the title to include trial number
        fig.suptitle(f'Trial {trial_number} Performance Heat Map', 
                    fontsize=14, fontweight='bold')
        
        # Convert matplotlib figure to numpy array for W&B
        fig.canvas.draw()
        
        print(f"Heat map for trial {trial_number} uploaded to W&B")
        return fig
            
    
    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for optimization.
        
        Based on user requirements:
        - Tune: temp_sample, reward_threshold, terminal_reward_threshold, 
                terminal_reward_weight, steering_weight, d_thr_factor, 
                k_switch, hitch_angle_weight
        - Don't tune: beta0, betaT, collision_penalty, hitch_penalty,
                      preference_penalty_weight, ref_* weights, heading_reward_weight
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {}
        
        # Diffusion parameters  
        params['temp_sample'] = trial.suggest_float('temp_sample', 0.0001, 10.0, log=True)
        
        # Reward thresholds
        params['reward_threshold'] = trial.suggest_float('reward_threshold', 10.0, 50.0)
        params['terminal_reward_threshold'] = trial.suggest_float('terminal_reward_threshold', 0.5, 10.0) 
        params['terminal_reward_weight'] = trial.suggest_float('terminal_reward_weight', 0.1, 10.0)
        
        # Cost weights
        params['steering_weight'] = trial.suggest_float('steering_weight', 0.01, 0.2)
        
        # Reward shaping parameters
        params['d_thr_factor'] = trial.suggest_float('d_thr_factor', 0.5, 5.0)
        params['k_switch'] = trial.suggest_float('k_switch', 0.1, 5.0)
        params['hitch_angle_weight'] = trial.suggest_float('hitch_angle_weight', 0.01, 0.5)
        
        return params
    
    def create_config_with_params(self, params: Dict[str, Any]) -> MBDConfig:
        """
        Create MBDConfig with optimized parameters while keeping fixed ones unchanged.
        
        Args:
            params: Dictionary of hyperparameters to optimize
            
        Returns:
            MBDConfig with updated hyperparameters
        """
        # Start with base config
        config_dict = self._config_to_dict(self.base_config)
        
        # Update with optimized parameters
        config_dict.update(params)
        
        # Create new config object (keeping all original fields)
        return MBDConfig(
            # Core settings (fixed)
            seed=self.base_config.seed,
            env_name=self.base_config.env_name,
            case=self.base_config.case,
            verbose=self.base_config.verbose,
            # Diffusion settings (fixed except temp_sample)
            Nsample=self.base_config.Nsample,
            Hsample=self.base_config.Hsample,
            Ndiffuse=self.base_config.Ndiffuse,
            temp_sample=params['temp_sample'],
            beta0=self.base_config.beta0,  # Fixed
            betaT=self.base_config.betaT,  # Fixed
            enable_demo=self.base_config.enable_demo,  # Fixed
            # Movement preference (fixed)
            motion_preference=self.base_config.motion_preference,  # Fixed at 0
            # Collision handling (fixed)
            collision_penalty=self.base_config.collision_penalty,  # Fixed
            hitch_penalty=self.base_config.hitch_penalty,  # Fixed
            enable_shielded_rollout_collision=self.base_config.enable_shielded_rollout_collision,
            enable_shielded_rollout_hitch=self.base_config.enable_shielded_rollout_hitch,
            enable_projection=self.base_config.enable_projection,
            enable_guidance=self.base_config.enable_guidance,
            # Reward thresholds (tunable)
            reward_threshold=params['reward_threshold'],
            ref_reward_threshold=self.base_config.ref_reward_threshold,  # Fixed (no demo)
            max_w_theta=self.base_config.max_w_theta,
            hitch_angle_weight=params['hitch_angle_weight'],
            # Physical parameters (fixed)
            l1=self.base_config.l1,
            l2=self.base_config.l2,
            lh=self.base_config.lh,
            lf1=self.base_config.lf1,
            lr=self.base_config.lr,
            lf2=self.base_config.lf2,
            lr2=self.base_config.lr2,
            tractor_width=self.base_config.tractor_width,
            trailer_width=self.base_config.trailer_width,
            # Input constraints (fixed)
            v_max=self.base_config.v_max,
            delta_max_deg=self.base_config.delta_max_deg,
            a_max=self.base_config.a_max,
            omega_max=self.base_config.omega_max,
            # Terminal reward (tunable)
            terminal_reward_threshold=params['terminal_reward_threshold'],
            terminal_reward_weight=params['terminal_reward_weight'],
            # Reward shaping parameters (tunable/fixed mix)
            d_thr_factor=params['d_thr_factor'],
            k_switch=params['k_switch'],
            steering_weight=params['steering_weight'],
            preference_penalty_weight=self.base_config.preference_penalty_weight,  # Fixed
            heading_reward_weight=self.base_config.heading_reward_weight,  # Fixed at 0.5
            # Reference weights (fixed - no demo)
            ref_pos_weight=self.base_config.ref_pos_weight,  # Fixed
            ref_theta1_weight=self.base_config.ref_theta1_weight,  # Fixed
            ref_theta2_weight=self.base_config.ref_theta2_weight,  # Fixed
            # Animation (fixed for optimization)
            render=False,
            save_animation=False,
            show_animation=False,
            save_denoising_animation=False,
            frame_skip=self.base_config.frame_skip,
            dt=self.base_config.dt,
        )
    
    def objective_function(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Runs statistical evaluation with sampled hyperparameters and returns
        the success rate as the objective to maximize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Success rate (0.0 to 1.0) - higher is better
        """
        # Sample hyperparameters
        params = self.define_hyperparameter_space(trial)
        
        # Create configuration with sampled parameters
        config = self.create_config_with_params(params)

        # Clear JIT cache for fresh compilation with new parameters
        clear_jit_cache()
        
        # Run statistical evaluation
        results = run_statistical_evaluation(
            config=config,
            num_trials=self.num_trials_per_eval,
            seed=42,  # Same seed for all trials to ensure fair comparison
            verbose=False  # Quiet during optimization
        )
        
        # Calculate objective score components  
        success_rate = results.success_rate
        compute_time = results.avg_pure_diffusion_time  # Lower is better
        safety = 1.0 - (results.collision_rate + results.jackknife_rate) / 2.0
        
        # Report intermediate values for pruning
        # Use success rate as the only metric for pruning since it's our objective
        trial.report(success_rate, 0)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned based on success rate: {success_rate:.3f}")
            raise optuna.TrialPruned()
        
        # Objective is success rate only (higher is better)
        objective_score = success_rate
        
        # Create and upload heat map to W&B
        fig_heat_map = self.create_heat_map(results, config, trial.number)
        
        # Store detailed results
        trial_result = {
            'trial_number': trial.number,
            'params': params,
            'success_rate': success_rate,
            'avg_position_error': results.avg_position_error,
            'collision_rate': results.collision_rate,
            'jackknife_rate': results.jackknife_rate,
            'avg_pure_diffusion_time': results.avg_pure_diffusion_time,
            'compute_time': compute_time,
            'safety': safety,
            'objective_score': objective_score,
            'successful_trials': results.successful_trials,
            'num_trials': results.num_trials
        }
        
        self.optimization_results.append(trial_result)
        
        # Log to W&B if enabled
        if self.use_wandb:
            wandb.log({
                'trial_number': trial.number,
                'objective_score': objective_score,
                'success_rate': success_rate,
                'avg_position_error': results.avg_position_error,
                'collision_rate': results.collision_rate,
                'jackknife_rate': results.jackknife_rate,
                'avg_pure_diffusion_time': results.avg_pure_diffusion_time,
                'compute_time': compute_time,
                'safety': safety,
                f"heat_map_trial_{trial.number}": wandb.Image(fig_heat_map, caption=f"Trial {trial.number} Performance Heat Map"),
                **{f'param_{k}': v for k, v in params.items()}
            })
            plt.close(fig_heat_map)
        
        print(f"Trial {trial.number}: Score={objective_score:.3f}, "
              f"Success={success_rate:.1%}, Time={results.avg_pure_diffusion_time:.2f}s")
        
        return objective_score
    
    def run_optimization(self, 
                        n_trials: int = 50,
                        timeout: Optional[int] = None,
                        n_jobs: int = 1) -> optuna.Study:
        """
        Run hyperparameter optimization using Optuna.
        
        Args:
            n_trials: Number of optimization trials to run
            timeout: Timeout in seconds (None for no timeout)
            n_jobs: Number of parallel jobs (1 for sequential)
            
        Returns:
            Completed Optuna study object
        """
        print(f"Starting hyperparameter optimization...")
        print(f"Configuration: {n_trials} trials, {self.num_trials_per_eval} evaluations per trial")
        
        # Create Optuna study with Gaussian Process sampler and MedianPruner
        sampler = GPSampler(seed=42)
        
        # MedianPruner Configuration Guide for Production (100+ trials):
        # ============================================================
        # 
        # Current settings (good for testing with 20-50 trials):
        # - n_startup_trials=5: Don't prune first 5 trials (need baseline data)
        # - n_warmup_steps=1: Start pruning after step 1 (after success rate report)
        # - interval_steps=1: Check for pruning at every step
        #
        # Recommended settings for 100+ trials:
        # - n_startup_trials=10-15: More trials to establish median baseline
        # - n_warmup_steps=1: Keep at 1 since we only have 1 metric (success rate)
        # - interval_steps=1: Keep at 1 for immediate pruning decisions
        #
        # Advanced tuning tips:
        # 1. Increase n_startup_trials for more stable pruning (10-20% of total trials)
        # 2. For very expensive trials, use SuccessiveHalvingPruner instead
        # 3. Monitor pruning efficiency: aim for 30-50% pruned trials
        # 4. If too aggressive: increase n_startup_trials
        # 5. If too conservative: decrease n_startup_trials or use PercentilePruner
        
        pruner = MedianPruner(
            n_startup_trials=15,  # Don't prune first 5 trials
            n_warmup_steps=1,    # Start pruning after step 1 (after success rate)
            interval_steps=1     # Check for pruning at every step
        )
        study = optuna.create_study(
            direction='maximize',  # Maximize objective score
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name
        )
        
        # Run optimization
        study.optimize(
            self.objective_function,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # Print results
        print(f"\n=== OPTIMIZATION COMPLETED ===")
        print(f"Total trials: {len(study.trials)}")
        print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best objective score: {study.best_value:.3f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Log final results to W&B
        if self.use_wandb:
            total_trials = len(study.trials)
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            
            wandb.log({
                'optimization_completed': True,
                'total_trials': total_trials,
                'completed_trials': completed_trials,
                'pruned_trials': pruned_trials,
                'failed_trials': failed_trials,
                'pruning_efficiency': pruned_trials / total_trials if total_trials > 0 else 0,
                'best_objective_score': study.best_value,
                'best_trial_number': study.best_trial.number,
                **{f'best_{k}': v for k, v in study.best_params.items()}
            })
        
        return study
    
    def save_results(self, study: optuna.Study, filepath: str = "optimization_results.json"):
        """Save optimization results to file"""
        results = {
            'study_name': self.study_name,
            'best_trial_number': study.best_trial.number,
            'best_objective_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'base_config': self._config_to_dict(self.base_config),
            'all_results': self.optimization_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {filepath}")


def create_base_config() -> MBDConfig:
    """Create base configuration for optimization"""
    return MBDConfig(
        # Core settings
        env_name="acc_tt2d",
        case="parking",
        motion_preference=0,  # No motion preference as requested
        enable_demo=False,    # No demonstration as requested
        
        # Algorithm settings (fixed)
        Nsample=20000,   # Reasonable for optimization
        Hsample=50,      # Reasonable horizon
        Ndiffuse=100,     # Reasonable diffusion steps
        
        # Fixed parameters per user requirements
        beta0=1e-5,      # Don't tune
        betaT=1e-2,      # Don't tune
        collision_penalty=0.15,  # Don't tune
        hitch_penalty=0.10,      # Don't tune
        preference_penalty_weight=0.5,  # Don't tune (motion_preference=0)
        heading_reward_weight=0.5,       # Don't tune (keep at 0.5)
        
        # Default values for parameters that will be optimized
        # (These will be overridden by optimizer)
        temp_sample=0.01,
        reward_threshold=25.0,
        terminal_reward_threshold=1.0,
        terminal_reward_weight=1.0,
        steering_weight=0.05,
        d_thr_factor=1.0,
        k_switch=2.5,
        hitch_angle_weight=0.05,
        
        # Disable rendering for batch optimization
        render=False,
        save_animation=False,
        show_animation=False,
        save_denoising_animation=False,
        verbose=False
    )


def main():
    """Example usage of hyperparameter optimization"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create base configuration
    base_config = create_base_config()
    
    # Create optimizer
    optimizer = DiffusionOptimizer(
        base_config=base_config,
        num_trials_per_eval=100,  # Statistical trials per hyperparameter evaluation
        study_name="mbd_parking_optimization",
        use_wandb=True
    )
    
    # Run optimization with pruning enabled
    study = optimizer.run_optimization(
        n_trials=100,     # Number of hyperparameter trials (reduced for testing)
        timeout=72000,    # 20 hour timeout
        n_jobs=1         # Sequential execution (required for pruning to work properly)
    )
    
    # Save results
    optimizer.save_results(study, "mbd_optimization_results.json")
    
    # Close W&B
    if optimizer.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 
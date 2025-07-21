from functools import partial
import os
import jax
from jax import numpy as jnp
from dataclasses import dataclass
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import logging

import mbd
from mbd.utils import (
    rollout_us, 
    rollout_us_with_terminal,
    create_animation, 
    create_denoising_animation
)

# NOTE: enable this if you want higher precision
# config.update("jax_enable_x64", True)

# Enable JAX compilation logging
jax.config.update('jax_log_compiles', False)
    
# Global compilation counter for debugging
compilation_count = 0

def count_compilation(f):
    """Decorator to count JIT compilations"""
    def wrapped(*args, **kwargs):
        global compilation_count
        compilation_count += 1
        logging.debug(f"    [COMPILATION #{compilation_count}] Compiling {f.__name__}")
        return f(*args, **kwargs)
    return wrapped

# Global cache for JIT functions to avoid recompilation across runs
_jit_function_cache = {}


def clear_jit_cache():
    """Clear the JIT function cache - useful for tests with different configurations"""
    global _jit_function_cache
    _jit_function_cache.clear()
    logging.debug("JIT function cache cleared")


@dataclass
class MBDConfig:
    # exp
    seed: int = 0
    # env
    env_name: str = "acc_tt2d"  # "tt2d" for kinematic, "acc_tt2d" for acceleration
    case: str = "parking" # "parking" for parking scenario, "navigation" for navigation scenario
    verbose: bool = False
    # diffusion
    Nsample: int = 20000  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 100 # number of diffusion steps
    temp_sample: float = 0.01  # temperature for sampling
    beta0: float = 1e-5  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False
    # movement preference
    motion_preference: int = 0  # 0=none, 1=forward, -1=backward
    # collision handling
    collision_penalty: float = 0.15  # penalty applied for obstacle collisions
    enable_gated_rollout_collision: bool = True  # whether to use gated rollout for obstacle collision
    hitch_penalty: float = 0.10  # penalty applied for hitch angle violations
    enable_gated_rollout_hitch: bool = True  # whether to use gated rollout for hitch violation
    # physical parameters
    l1: float = 3.23  # tractor wheelbase
    l2: float = 2.9   # trailer length
    lh: float = 1.15  # hitch length
    tractor_width: float = 2.0  # tractor width
    trailer_width: float = 2.5  # trailer width
    # input constraints
    v_max: float = 3.0  # velocity limit
    delta_max_deg: float = 55.0  # steering angle limit in degrees
    # acceleration control constraints (for acc_tt2d)
    a_max: float = 2.0  # acceleration limit [m/s²]
    omega_max: float = 1.0  # steering rate limit [rad/s]
    # reward thresholds
    reward_threshold: float = 25.0  # position error threshold for main reward function
    ref_reward_threshold: float = 10.0  # position error threshold for demonstration evaluation
    max_w_theta: float = 0.5  # maximum weight for heading reward vs position reward
    hitch_angle_weight: float = 0.05  # weight for hitch angle (articulation) reward
    # terminal reward
    terminal_reward_threshold: float = 1.0  # position error threshold for terminal reward
    terminal_reward_weight: float = 1.0  # weight for terminal reward at final state
    # reward shaping parameters
    d_thr_factor: float = 1.0  # multiplier for distance threshold (multiplied by rig length)
    k_switch: float = 2.5  # slope of logistic switch for position/heading reward blending
    steering_weight: float = 0.05  # weight for trajectory-level steering cost
    preference_penalty_weight: float = 0.5  # penalty weight for movement preference
    heading_reward_weight: float = 0.5  # (should be 0.5 always) weight for heading reward calculation
    # demonstration evaluation weights
    ref_pos_weight: float = 0.8  # position weight in demo evaluation
    ref_theta1_weight: float = 0.1  # theta1 weight in demo evaluation  
    ref_theta2_weight: float = 0.1 # theta2 weight in demo evaluation
    # animation
    render: bool = True
    save_animation: bool = False # flag to enable animation saving
    show_animation: bool = True  # flag to show animation during creation
    save_denoising_animation: bool = False  # flag to enable denoising process visualization
    frame_skip: int = 1  # skip every other frame for denoising animation
    dt: float = 0.25


def dict_to_config_obj(config_dict):
    """
    Convert dictionary to MBDConfig dataclass for JAX compatibility.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        MBDConfig: Dataclass with configuration values
    """
    # Create dataclass with values from dict - all parameters must be provided
    # Explicitly convert types to ensure proper typing
    return MBDConfig(
        seed=int(config_dict["seed"]),
        render=bool(config_dict["render"]),
        env_name=str(config_dict["env_name"]),
        case=str(config_dict["case"]),
        Nsample=int(config_dict["Nsample"]),
        Hsample=int(config_dict["Hsample"]),
        Ndiffuse=int(config_dict["Ndiffuse"]),
        temp_sample=float(config_dict["temp_sample"]),
        beta0=float(config_dict["beta0"]),
        betaT=float(config_dict["betaT"]),
        enable_demo=bool(config_dict["enable_demo"]),
        motion_preference=int(config_dict.get("motion_preference", 0)),
        collision_penalty=float(config_dict.get("collision_penalty", 0.15)),
        enable_gated_rollout_collision=bool(config_dict.get("enable_gated_rollout_collision", True)),
        hitch_penalty=float(config_dict.get("hitch_penalty", 0.10)),
        enable_gated_rollout_hitch=bool(config_dict.get("enable_gated_rollout_hitch", True)),
        reward_threshold=float(config_dict.get("reward_threshold", 25.0)),
        ref_reward_threshold=float(config_dict.get("ref_reward_threshold", 5.0)),
        max_w_theta=float(config_dict.get("max_w_theta", 0.75)),
        hitch_angle_weight=float(config_dict.get("hitch_angle_weight", 0.2)),
        # physical parameters
        l1=float(config_dict.get("l1", 3.23)),
        l2=float(config_dict.get("l2", 2.9)),
        lh=float(config_dict.get("lh", 1.15)),
        tractor_width=float(config_dict.get("tractor_width", 2.0)),
        trailer_width=float(config_dict.get("trailer_width", 2.5)),
        # input constraints
        v_max=float(config_dict.get("v_max", 3.0)),
        delta_max_deg=float(config_dict.get("delta_max_deg", 55.0)),
        # acceleration control constraints
        a_max=float(config_dict.get("a_max", 2.0)),
        omega_max=float(config_dict.get("omega_max", 1.0)),
        # reward shaping parameters
        d_thr_factor=float(config_dict.get("d_thr_factor", 1.0)),
        k_switch=float(config_dict.get("k_switch", 2.5)),
        steering_weight=float(config_dict.get("steering_weight", 0.05)),
        preference_penalty_weight=float(config_dict.get("preference_penalty_weight", 0.05)),
        heading_reward_weight=float(config_dict.get("heading_reward_weight", 0.5)),
        terminal_reward_threshold=float(config_dict.get("terminal_reward_threshold", 10.0)),
        terminal_reward_weight=float(config_dict.get("terminal_reward_weight", 1.0)),
        ref_pos_weight=float(config_dict.get("ref_pos_weight", 0.3)),
        ref_theta1_weight=float(config_dict.get("ref_theta1_weight", 0.5)),
        ref_theta2_weight=float(config_dict.get("ref_theta2_weight", 0.2)),
        save_animation=bool(config_dict["save_animation"]),
        show_animation=bool(config_dict["show_animation"]),
        save_denoising_animation=bool(config_dict["save_denoising_animation"]),
        dt=float(config_dict["dt"]),
    )


def run_diffusion(args=None, env=None):
    """
    Run the diffusion-based planning algorithm.
    
    Args:
        args: Configuration dictionary with diffusion parameters.
        env: Environment object
    
    Returns:
        rew_final: Final reward value
        Y0: Final action sequence
        trajectory_states: Trajectory states
        timing_info: Dictionary with detailed timing information
    """
    # Convert dictionary to dataclass if needed
    if isinstance(args, dict):
        args = dict_to_config_obj(args)
    elif args is None:
        raise ValueError("args parameter is required and cannot be None")
    
    logging.debug("=== DIFFUSION TIMING ANALYSIS ===")
    total_start_time = time.time()
    
    rng = jax.random.PRNGKey(seed=args.seed)
    Nx = env.observation_size
    Nu = env.action_size
    
    # Generate demonstration trajectory if enabled
    demo_start_time = time.time()
    reward_compile_time = 0.0
    if args.enable_demo:
        # Generate demonstration trajectory
        env.generate_demonstration_trajectory(search_direction="horizontal", motion_preference=args.motion_preference)
        # Compile the reward function with the demonstration (time this separately)
        reward_compile_start = time.time()
        env.compute_demonstration_reward()
        reward_compile_time = time.time() - reward_compile_start
        logging.info(f"Demo trajectory generated with reward: {env.rew_xref:.3f}")
    demo_time = time.time() - demo_start_time
    
    # Setup JIT compiled functions (with simple caching to avoid recompilation)
    jit_setup_start_time = time.time()
    
    # Simple cache key based on environment type only
    cache_key = f"{type(env).__name__}_env_funcs"
    
    if cache_key in _jit_function_cache:
        logging.debug(f"Using cached environment JIT functions")
        step_env_jit, reset_env_jit, rollout_us_jit, rollout_us_with_terminal_jit = _jit_function_cache[cache_key]
    else:
        logging.info(f"Creating new environment JIT functions")
        step_env_jit = jax.jit(env.step)
        reset_env_jit = jax.jit(env.reset)
        rollout_us_jit = jax.jit(partial(rollout_us, step_env_jit))
        rollout_us_with_terminal_jit = jax.jit(partial(rollout_us_with_terminal, step_env_jit, env))
        _jit_function_cache[cache_key] = (step_env_jit, reset_env_jit, rollout_us_jit, rollout_us_with_terminal_jit)
    
    # NOTE: a, b = jax.random.split(b) is a standard way to use random. always use a as random variable, not b. 
    rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
    state_init = reset_env_jit(rng_reset) # NOTE: in car2d, just reset with pre-defined x0. currently no randomization.
    jit_setup_time = time.time() - jit_setup_start_time
    
    # Simple cache key for reverse_once function
    reverse_once_cache_key = "reverse_once_function"
    
    # Warm-up phase: compile JIT functions with correct arguments (only if using new functions)
    need_warmup = cache_key not in _jit_function_cache or reverse_once_cache_key not in _jit_function_cache
    
    # Define YN regardless of warmup status (needed for actual computation later)
    YN = jnp.zeros([args.Hsample, Nu])
    
    if need_warmup:
        logging.debug("Warming up JIT compiled functions...")
        #print(f"Environment shapes: Nx={Nx}, Nu={Nu}")
        logging.debug(f"Diffusion shapes: Nsample={args.Nsample}, Hsample={args.Hsample}, Ndiffuse={args.Ndiffuse}")
        warmup_start_time = time.time()
        
        # Warm up step_env_jit and reset_env_jit (already done above, but let's be explicit)
        logging.debug("  Warming up reset_env_jit...")
        _ = reset_env_jit(rng_reset)
        logging.debug("  Warming up step_env_jit...")
        # Use the SAME action shape AND creation pattern that will be used in post-processing
        # Create a dummy Y0 array and extract actions the same way as in post-processing
        dummy_Y0_warmup = jnp.zeros([args.Hsample, Nu])
        dummy_action_raw = dummy_Y0_warmup[0]  # Same as Y0[t] operation
        dummy_action = jnp.array(dummy_action_raw)  # Same as action = jnp.array(action_raw)
        #print(f"    dummy_action shape: {dummy_action.shape}, dtype: {dummy_action.dtype}, type: {type(dummy_action)}")
        
        # CRITICAL: Warm up with a state that has the same computational history as the loop
        # First call creates a state similar to what we'll use in iterations 1, 2, 3...
        warmup_state_step1 = step_env_jit(state_init, dummy_action)
        # Second call uses a state that went through step_env_jit (like the actual loop)
        _ = step_env_jit(warmup_state_step1, dummy_action)
        #print("    step_env_jit warmed up with both initial state and evolved state")
        
        # Warm up rollout_us_jit with correct input shapes
        logging.debug("  Warming up rollout_us_jit...")
        dummy_Y0 = jnp.zeros([args.Hsample, Nu])  # Correct shape for action sequence
        #print(f"    dummy_Y0 shape: {dummy_Y0.shape}")
        _, _ = rollout_us_jit(state_init, dummy_Y0)
        
        # Warm up rollout_us_with_terminal_jit
        logging.debug("  Warming up rollout_us_with_terminal_jit...")
        _, _ = rollout_us_with_terminal_jit(state_init, dummy_Y0)
    else:
        logging.debug("Skipping warmup - using cached JIT functions")
        warmup_start_time = time.time()
    

    
    # Set up diffusion parameters that will be used in reverse_once
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    
    # Define the steering reward function that will be used in reverse_once
    @jax.jit
    def compute_steering_reward(Y0s):
        """
        Compute trajectory-level steering reward for action sequences.
        Y0s: (Nsample, Hsample, Nu) where Nu=2 (v, delta)
        Returns: (Nsample,) steering rewards
        """
        # Extract steering angles (second action dimension)
        deltas = Y0s[:, :, 1]  # Shape: (Nsample, Hsample)
        
        # Scale to actual steering angles for reward computation
        deltas_scaled = deltas * env.delta_max
        
        # 1. Steering magnitude reward: r_delta = 1.0 - (delta_t / delta_max)^2
        r_delta = 1.0 - (jnp.abs(deltas) ** 2)  # deltas already normalized to [-1,1]
        
        # 2. Steering rate reward: r_deltadot = 1.0 - ((delta_t - delta_{t-1}) / (0.1 * delta_max))^2
        delta_diffs = jnp.diff(deltas_scaled, axis=1)  # Shape: (Nsample, Hsample-1)
        delta_rate_threshold = 0.1 * env.delta_max
        r_deltadot = 1.0 - jnp.clip((jnp.abs(delta_diffs) / delta_rate_threshold) ** 2, 0.0, 1.0)
        
        # 3. Combined steering reward: r_steer = 0.5 * (r_delta + r_deltadot)
        # For r_deltadot, pad with 1.0 at the beginning (no rate reward for first timestep)
        r_deltadot_padded = jnp.concatenate([
            jnp.ones((Y0s.shape[0], 1)), r_deltadot
        ], axis=1)  # Shape: (Nsample, Hsample)
        
        r_steer = 0.5 * (r_delta + r_deltadot_padded)
        
        # Average over trajectory horizon
        r_steer_avg = jnp.mean(r_steer, axis=1)  # Shape: (Nsample,)
        
        return r_steer_avg

    # Define the main reverse_once function that will be used for both warmup and actual computation
    # Check if we already have a cached version
    if reverse_once_cache_key in _jit_function_cache:
        logging.debug(f"Using cached reverse_once function")
        reverse_once = _jit_function_cache[reverse_once_cache_key]
    else:
        logging.debug(f"Creating new reverse_once function")
        @partial(jax.jit, static_argnums=(1,))
        def reverse_once(carry, config_params_tuple):
            """
            Single reverse diffusion step with configuration parameters as arguments.
            
            Args:
                carry: (i, rng, Ybar_i) tuple
                config_params_tuple: Tuple of (motion_preference, temp_sample, enable_demo, Nsample, Hsample, Nu)
            """
            i, rng, Ybar_i = carry
            
            # Extract config parameters from tuple
            motion_preference, temp_sample, enable_demo, Nsample, Hsample, Nu = config_params_tuple
            
            Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

            # sample from q_i
            rng, Y0s_rng = jax.random.split(rng)
            eps_u = jax.random.normal(Y0s_rng, (Nsample, Hsample, Nu)) # NOTE: Sample from N(0, I) 
            Y0s = eps_u * sigmas[i]/jnp.sqrt(alphas_bar[i-1]) + Ybar_i # TODO: changed this based on the paper (it seems the original code is wrong)
            Y0s = jnp.clip(Y0s, -1.0, 1.0) # NOTE: clip action to [-1, 1] (it is defined in dynamics)
            
            # Apply strict motion enforcement for motion_preference = 2 or -2
            # For 2: enforce forward only (clip velocity to [0, 1])
            # For -2: enforce backward only (clip velocity to [-1, 0])
            velocity_component = Y0s[:, :, 0]
            
            # Use JAX conditionals to avoid recompilation
            # Forward only enforcement (motion_preference = 2)
            velocity_forward_only = jnp.clip(velocity_component, 0.0, 1.0)
            
            # Backward only enforcement (motion_preference = -2)
            velocity_backward_only = jnp.clip(velocity_component, -1.0, 0.0)
            
            # Select appropriate velocity clipping based on motion_preference
            velocity_final = jnp.where(
                motion_preference == 2,
                velocity_forward_only,
                jnp.where(
                    motion_preference == -2,
                    velocity_backward_only,
                    velocity_component  # No additional clipping for other values
                )
            )
            
            # Update Y0s with the modified velocity component
            Y0s = Y0s.at[:, :, 0].set(velocity_final)

            # esitimate mu_0tm1
            rewss, qs = jax.vmap(rollout_us_with_terminal_jit, in_axes=(None, 0))(state_init, Y0s)
            rews = rewss  # rollout_us_with_terminal now returns total reward directly (mean + terminal)
            
            # Compute steering cost and blend with geometric rewards
            r_steer = compute_steering_reward(Y0s)  # Shape: (Nsample,)
            rews_combined = rews + env.steering_weight * r_steer  # Blend steering reward
            
            rew_std = rews_combined.std() # NOTE: scalar
            rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std) # NOTE: at early stage it is near 0, and increase to 0.1 ish after.
            rew_mean = rews_combined.mean()
            logp0 = (rews_combined - rew_mean) / rew_std / temp_sample

            # evalulate demo
            if enable_demo:
                xref_logpds = jax.vmap(lambda q: env.eval_xref_logpd(q, motion_preference=motion_preference))(qs)
                xref_logpds = xref_logpds - xref_logpds.max()
                logpdemo = (
                    (xref_logpds + env.rew_xref - rew_mean) / rew_std / temp_sample
                )
                demo_mask = logpdemo > logp0
                logp0 = jnp.where(demo_mask, logpdemo, logp0)
                #logp0 = (logp0 - logp0.mean()) / logp0.std() / temp_sample # FIXME: I commented it out since I don't know why this is necessary.

            weights = jax.nn.softmax(logp0)
            Ybar = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

            score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
            Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

            Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

            return (i - 1, rng, Ybar_im1), rews_combined.mean()
        
        # Cache the function for future use
        _jit_function_cache[reverse_once_cache_key] = reverse_once
    
        # Warm up reverse_once function that will be used in actual computation
        # This ensures JIT compilation happens now with correct arguments, not during timing
        logging.debug("  Warming up reverse_once...")
        warmup_i = args.Ndiffuse - 1  # Valid index for warmup
        warmup_carry = (warmup_i, rng, YN)
        
        # Create config parameters tuple for warmup (order must match function signature)
        warmup_config_params = (
            args.motion_preference,
            args.temp_sample,
            args.enable_demo,
            args.Nsample,
            args.Hsample,
            Nu
        )
        
        logging.debug(f"    warmup_carry shapes: i={warmup_i}, rng.shape={rng.shape}, YN.shape={YN.shape}")
        logging.debug(f"    Using indices: alphas_bar[{warmup_i}], sigmas[{warmup_i}]")
        _ = reverse_once(warmup_carry, warmup_config_params)  # This compiles the function with correct shapes
        logging.debug("  reverse_once warmup completed")
        
    warmup_time = time.time() - warmup_start_time
    if need_warmup:
        logging.debug(f"JIT warmup completed in {warmup_time:.2f} seconds")
    else:
        logging.debug(f"JIT warmup skipped in {warmup_time:.2f} seconds")

    ## Run diffusion (actual computation)
    diffusion_setup_start_time = time.time()

    Sigmas_cond = (
        (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    )
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)
    logging.debug(f"init sigma = {sigmas[-1]:.2e}")

    diffusion_setup_time = time.time() - diffusion_setup_start_time

    # Actual diffusion computation (this is what we want to time)
    logging.debug("Starting diffusion computation...")
    #print(f"About to use YN.shape={YN.shape}, rng.shape={rng.shape}")
    pure_diffusion_start_time = time.time()
    
    # run reverse
    def reverse(YN, rng):
        Yi = YN
        # Pre-allocate Ybars array to avoid dynamic concatenation recompilation
        # Shape: (Ndiffuse-1, Hsample, Nu) where Ndiffuse-1 is the number of iterations
        num_iterations = args.Ndiffuse - 1
        Ybars = jnp.zeros((num_iterations, args.Hsample, Nu))
        
        # Create config parameters tuple for reverse_once (order must match function signature)
        config_params = (
            args.motion_preference,
            args.temp_sample,
            args.enable_demo,
            args.Nsample,
            args.Hsample,
            Nu
        )
        
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for idx, i in enumerate(pbar):
                carry_once = (i, rng, Yi)
                # Check if this call triggers recompilation
                if i == args.Ndiffuse - 1:
                    #print(f"    First diffusion step: carry shapes i={i}, rng.shape={rng.shape}, Yi.shape={Yi.shape}")
                    pass
                (i, rng, Yi), rew = reverse_once(carry_once, config_params)  # Pass config_params tuple
                # Use index assignment instead of append to avoid recompilation
                Ybars = Ybars.at[idx].set(Yi)
                # Update the progress bar's suffix to show the current reward
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return Ybars, Yi  # Return both all Ybars and final Yi

    rng_exp, rng = jax.random.split(rng)
    Ybars, Y0 = reverse(YN, rng_exp) # NOTE: YN: all zeros, one trajectory of actions 
    
    pure_diffusion_time = time.time() - pure_diffusion_start_time
    logging.debug(f"Diffusion computation completed in {pure_diffusion_time:.3f}s")
    
    # Post-processing time
    post_processing_start_time = time.time()
    
    # Store the final trajectory for potential extraction
    final_trajectory_actions = Y0  # This is the optimized action sequence
    
    # Compute trajectory states (needed for path extraction)
    #print("Computing trajectory states for post-processing...")
    
    # Pre-allocate the trajectory array to avoid dynamic concatenation recompilation
    trajectory_length = Y0.shape[0] + 1  # +1 for initial state
    xs = jnp.zeros((trajectory_length, state_init.pipeline_state.shape[0]))
    xs = xs.at[0].set(state_init.pipeline_state)  # Set initial state
    
    state = state_init
    
    # Process trajectory states using pre-allocated arrays
    for t in range(Y0.shape[0]):
        action_raw = Y0[t]
        # Create a fresh array to avoid JAX type inconsistencies from array slicing
        action = jnp.array(action_raw)
        
        state = step_env_jit(state, action)
        # Use index assignment instead of concatenation to avoid recompilation
        xs = xs.at[t + 1].set(state.pipeline_state)
    
    trajectory_states = xs
    #print("Post-processing trajectory computation completed")
    
    post_processing_time = time.time() - post_processing_start_time
    
    # Calculate total time here (before optional rendering and final reward computation)
    total_time = time.time() - total_start_time
    
    # Group compilation times together
    total_compilation_time = jit_setup_time + warmup_time + reward_compile_time
    
    # Create detailed timing information (core algorithm only)
    timing_info = {
        'total_time': total_time,
        'demo_generation_time': demo_time - reward_compile_time,  # Demo generation without compilation
        'compilation_time': total_compilation_time,  # All compilation times grouped
        'jit_setup_time': jit_setup_time,
        'warmup_time': warmup_time,
        'reward_compile_time': reward_compile_time,
        'diffusion_setup_time': diffusion_setup_time,
        'pure_diffusion_time': pure_diffusion_time,  # This is the main metric we care about
        'post_processing_time': post_processing_time,
        'overhead_time': total_time - pure_diffusion_time,  # Everything except pure diffusion
    }
    
    # Rendering and visualization (optional, not included in timing analysis)
    if args.render:
        path = f"{mbd.__path__[0]}/../results/{args.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        jnp.save(f"{path}/mu_0ts.npy", Ybars)
        
        # Store final actions for path extraction
        jnp.save(f"{path}/final_actions.npy", Y0)
        
        # Store the trajectory states for path extraction
        jnp.save(f"{path}/trajectory_states.npy", trajectory_states)
        
        #if args.env_name == "car2d":
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # rollout (trajectory already computed above)
        env.render(ax, trajectory_states)
        if args.enable_demo and hasattr(env, 'xref') and env.xref is not None:
            ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", linewidth=2, label="Demonstration path", alpha=0.7)
        ax.legend()
        
        plt.switch_backend('TkAgg')  # Switch to interactive backend
        plt.draw()  # Ensure the plot is fully rendered
        plt.show()  # Show blocking so user can examine the static plot
        
        plt.savefig(f"{path}/rollout.png")
        
        # Close the static plot before showing animations
        if args.show_animation or args.save_denoising_animation:
            plt.close(fig)
        
        # Create animation if requested
        if args.save_animation or args.show_animation:
            # Prepare trajectory data for animation
            trajectory_states_list = [state_init.pipeline_state]
            trajectory_actions = []
            state = state_init
            for t in range(Y0.shape[0]):  # Use Y0
                action = Y0[t]
                trajectory_actions.append(action)
                state = step_env_jit(state, action)
                trajectory_states_list.append(state.pipeline_state)
            
            # Add final state with no action
            trajectory_actions.append(None)
            
            # Create animation
            create_animation(env, trajectory_states_list, trajectory_actions, args)
            
                # Create denoising animation if requested
        if args.save_denoising_animation:
            create_denoising_animation(env, Ybars, args, step_env_jit, state_init, frame_skip=args.frame_skip)

    # Optional final reward computation (for display only, not included in timing)
    rew_final, _ = rollout_us_with_terminal_jit(state_init, Y0)  # Use Y0
    # rew_final is already the total reward (mean + terminal), no need to call .mean()
    
    # Print detailed timing report
    logging.debug("\n=== TIMING REPORT ===")
    logging.debug(f"Total time:              {timing_info['total_time']:.3f}s")
    logging.info(f"Pure diffusion time:     {timing_info['pure_diffusion_time']:.3f}s ({timing_info['pure_diffusion_time']/timing_info['total_time']*100:.1f}%)")
    logging.debug(f"Overhead time:           {timing_info['overhead_time']:.3f}s ({timing_info['overhead_time']/timing_info['total_time']*100:.1f}%)")
    logging.debug(f"  - Demo generation:     {timing_info['demo_generation_time']:.3f}s")
    logging.debug(f"  - Compilation total:   {timing_info['compilation_time']:.3f}s")
    logging.debug(f"    * JIT setup:         {timing_info['jit_setup_time']:.3f}s")
    logging.debug(f"    * JIT warmup:        {timing_info['warmup_time']:.3f}s")
    logging.debug(f"    * Reward compile:    {timing_info['reward_compile_time']:.3f}s")
    logging.debug(f"  - Diffusion setup:     {timing_info['diffusion_setup_time']:.3f}s")
    logging.debug(f"  - Post-processing:     {timing_info['post_processing_time']:.3f}s")
    logging.debug(f"Final reward:            {rew_final:.3e}")
    logging.debug("=====================")

    return rew_final, Y0, trajectory_states, timing_info




if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # For standalone testing, use default config with demonstration enabled
    config = MBDConfig()
    
    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Suppress JAX info messages about unavailable backends
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)
    logging.getLogger('jax').setLevel(logging.WARNING)
    
    # Create environment
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
        # physical parameters
        l1=config.l1,
        l2=config.l2,
        lh=config.lh,
        tractor_width=config.tractor_width,
        trailer_width=config.trailer_width,
        # input constraints
        v_max=config.v_max,
        delta_max_deg=config.delta_max_deg,
        # acceleration control constraints
        a_max=config.a_max,
        omega_max=config.omega_max,
        # reward shaping parameters
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
    
    # Set initial position using geometric parameters relative to parking lot
    # dx: distance from tractor front face to target parking space center
    # dy: distance from tractor to parking lot entrance line
    env.set_init_pos(dx=9.0, dy=2.0, theta1=0, theta2=0)
    if config.motion_preference == -2:
        env.set_init_pos(dx=-12.0, dy=1.0, theta1=jnp.pi, theta2=jnp.pi)
    # Set goal angles based on motion preference
    if config.motion_preference in [-1, -2]:  # backward parking
        env.set_goal_pos(theta1=jnp.pi/2, theta2=jnp.pi/2)  # backward parking
    
    rew_final, Y0, trajectory_states, timing_info = run_diffusion(args=config, env=env)
    end_time = time.time()

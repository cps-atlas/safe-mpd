import functools
import os
import jax
from jax import numpy as jnp
from dataclasses import dataclass
from tqdm import tqdm
from matplotlib import pyplot as plt
import time

import mbd
from mbd.utils import (
    rollout_us, 
    create_animation, 
    create_denoising_animation
)

# NOTE: enable this if you want higher precision
# config.update("jax_enable_x64", True)


@dataclass
class MBDConfig:
    # exp
    seed: int = 0
    # env
    env_name: str = "tt2d"
    case: str = "case2" # "case1" for original obstacles, "case2" for parking scenario
    # diffusion
    Nsample: int = 20000  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 150 # number of diffusion steps
    temp_sample: float = 0.01  # temperature for sampling
    beta0: float = 1e-5  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = True
    # movement preference
    movement_preference: int = 1  # 0=none, 1=forward, -1=backward
    # collision handling
    collision_penalty: float = 0.15  # penalty applied for obstacle collisions
    enable_collision_projection: bool = True  # whether to project state back on obstacle collision
    hitch_penalty: float = 0.10  # penalty applied for hitch angle violations
    enable_hitch_projection: bool = True  # whether to project state back on hitch violation
    # reward thresholds
    reward_threshold: float = 25.0  # position error threshold for main reward function
    ref_reward_threshold: float = 10.0  # position error threshold for demonstration evaluation
    max_w_theta: float = 0.75  # maximum weight for heading reward vs position reward
    hitch_angle_weight: float = 0.2  # weight for hitch angle (articulation) reward
    # physical parameters
    l1: float = 3.23  # tractor wheelbase
    l2: float = 2.9   # trailer length
    lh: float = 1.15  # hitch length
    tractor_width: float = 2.0  # tractor width
    trailer_width: float = 2.5  # trailer width
    # input constraints
    v_max: float = 3.0  # velocity limit
    delta_max_deg: float = 55.0  # steering angle limit in degrees
    # reward shaping parameters
    d_thr_factor: float = 1.0  # multiplier for distance threshold (multiplied by rig length)
    k_switch: float = 2.5  # slope of logistic switch for position/heading reward blending
    steering_weight: float = 0.05  # weight for trajectory-level steering cost
    preference_penalty_weight: float = 0.05  # penalty weight for movement preference
    heading_reward_weight: float = 0.5  # (should be 0.5 always) weight for heading reward calculation
    # demonstration evaluation weights
    ref_pos_weight: float = 0.3  # position weight in demo evaluation
    ref_theta1_weight: float = 0.35  # theta1 weight in demo evaluation  
    ref_theta2_weight: float = 0.35 # theta2 weight in demo evaluation
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
        movement_preference=int(config_dict.get("movement_preference", 0)),
        collision_penalty=float(config_dict.get("collision_penalty", 0.15)),
        enable_collision_projection=bool(config_dict.get("enable_collision_projection", False)),
        hitch_penalty=float(config_dict.get("hitch_penalty", 0.10)),
        enable_hitch_projection=bool(config_dict.get("enable_hitch_projection", True)),
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
        # reward shaping parameters
        d_thr_factor=float(config_dict.get("d_thr_factor", 1.0)),
        k_switch=float(config_dict.get("k_switch", 2.5)),
        steering_weight=float(config_dict.get("steering_weight", 0.05)),
        preference_penalty_weight=float(config_dict.get("preference_penalty_weight", 0.05)),
        heading_reward_weight=float(config_dict.get("heading_reward_weight", 0.5)),
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
    """
    # Convert dictionary to dataclass if needed
    if isinstance(args, dict):
        args = dict_to_config_obj(args)
    elif args is None:
        raise ValueError("args parameter is required and cannot be None")
    
    rng = jax.random.PRNGKey(seed=args.seed)
    Nx = env.observation_size
    Nu = env.action_size
    
    # Generate demonstration trajectory if enabled
    if args.enable_demo:
        # Generate demonstration trajectory
        env.generate_demonstration_trajectory(search_direction="horizontal", movement_preference=args.movement_preference)
        # Compile the reward function with the demonstration
        env.compile_reward_function()
        print(f"Demo trajectory generated with reward: {env.rew_xref:.3f}")
    
    # env functions
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    # eval_us = jax.jit(functools.partial(mbd.utils.eval_us, step_env_jit))
    rollout_us_jit = jax.jit(functools.partial(rollout_us, step_env_jit))

    # NOTE: a, b = jax.random.split(b) is a standard way to use random. always use a as random variable, not b. 
    rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
    state_init = reset_env_jit(rng_reset) # NOTE: in car2d, just reset with pre-defined x0. currently no randomization.

    ## run diffusion
    start_time = time.time()

    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    Sigmas_cond = (
        (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    )
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)
    print(f"init sigma = {sigmas[-1]:.2e}")

    YN = jnp.zeros([args.Hsample, Nu])

    @jax.jit
    def compute_steering_cost(Y0s):
        """
        Compute trajectory-level steering cost for action sequences.
        Y0s: (Nsample, Hsample, Nu) where Nu=2 (v, delta)
        Returns: (Nsample,) steering costs
        """
        # Extract steering angles (second action dimension)
        deltas = Y0s[:, :, 1]  # Shape: (Nsample, Hsample)
        
        # Scale to actual steering angles for cost computation
        deltas_scaled = deltas * env.delta_max
        
        # 1. Steering magnitude cost: r_delta = 1.0 - (delta_t / delta_max)^2
        r_delta = 1.0 - (jnp.abs(deltas) ** 2)  # deltas already normalized to [-1,1]
        
        # 2. Steering rate cost: r_deltadot = 1.0 - ((delta_t - delta_{t-1}) / (0.1 * delta_max))^2
        delta_diffs = jnp.diff(deltas_scaled, axis=1)  # Shape: (Nsample, Hsample-1)
        delta_rate_threshold = 0.1 * env.delta_max
        r_deltadot = 1.0 - jnp.clip((jnp.abs(delta_diffs) / delta_rate_threshold) ** 2, 0.0, 1.0)
        
        # 3. Combined steering cost: r_steer = 0.5 * (r_delta + r_deltadot)
        # For r_deltadot, pad with 1.0 at the beginning (no rate cost for first timestep)
        r_deltadot_padded = jnp.concatenate([
            jnp.ones((Y0s.shape[0], 1)), r_deltadot
        ], axis=1)  # Shape: (Nsample, Hsample)
        
        r_steer = 0.5 * (r_delta + r_deltadot_padded)
        
        # Average over trajectory horizon
        r_steer_avg = jnp.mean(r_steer, axis=1)  # Shape: (Nsample,)
        
        return r_steer_avg

    @jax.jit
    def reverse_once(carry, unused):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (args.Nsample, args.Hsample, Nu)) # NOTE: Sample from N(0, I) 
        Y0s = eps_u * sigmas[i]/jnp.sqrt(alphas_bar[i-1]) + Ybar_i # TODO: changed this based on the paper (it seems the original code is wrong)
        Y0s = jnp.clip(Y0s, -1.0, 1.0) # NOTE: clip action to [-1, 1] (it is defined in dynamics)

        # esitimate mu_0tm1
        rewss, qs = jax.vmap(rollout_us_jit, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=-1)
        
        # Compute steering cost and blend with geometric rewards
        r_steer = compute_steering_cost(Y0s)  # Shape: (Nsample,)
        rews_combined = rews + env.steering_weight * r_steer  # Blend steering cost
        
        rew_std = rews_combined.std() # NOTE: scalar
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std) # NOTE: at early stage it is near 0, and increase to 0.1 ish after.
        rew_mean = rews_combined.mean()
        logp0 = (rews_combined - rew_mean) / rew_std / args.temp_sample

        # evalulate demo
        if args.enable_demo:
            xref_logpds = jax.vmap(lambda q: env.eval_xref_logpd(q, movement_preference=args.movement_preference))(qs)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = (
                (xref_logpds + env.rew_xref - rew_mean) / rew_std / args.temp_sample
            )
            demo_mask = logpdemo > logp0
            logp0 = jnp.where(demo_mask, logpdemo, logp0)
            #logp0 = (logp0 - logp0.mean()) / logp0.std() / args.temp_sample # FIXME: I commented it out since I don't know why this is necessary.

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        return (i - 1, rng, Ybar_im1), rews_combined.mean()

    # run reverse
    def reverse(YN, rng):
        Yi = YN
        Ybars = []
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                carry_once = (i, rng, Yi)
                (i, rng, Yi), rew = reverse_once(carry_once, None) # NOTE: just to maintain similar style with scan, here no xs. 
                Ybars.append(Yi)
                # Update the progress bar's suffix to show the current reward
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return jnp.array(Ybars), Yi  # Return both all Ybars and final Yi

    rng_exp, rng = jax.random.split(rng)
    Ybars, Y0 = reverse(YN, rng_exp) # NOTE: YN: all zeros, one trajectory of actions 
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Store the final trajectory for potential extraction
    final_trajectory_actions = Y0  # This is the optimized action sequence
    
    # Compute trajectory states (needed for path extraction)
    xs = jnp.array([state_init.pipeline_state])
    state = state_init
    for t in range(Y0.shape[0]):
        state = step_env_jit(state, Y0[t])
        xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
    trajectory_states = xs
    
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
            
        

    rewss_final, _ = rollout_us_jit(state_init, Y0)  # Use Y0
    rew_final = rewss_final.mean()

    return rew_final, Y0, trajectory_states  # Return additional data for path extraction


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # For standalone testing, use default config with demonstration enabled
    config = MBDConfig()
    
    # Create environment
    env = mbd.envs.get_env(
        config.env_name, 
        case=config.case, 
        dt=config.dt, 
        H=config.Hsample,
        movement_preference=config.movement_preference,
        collision_penalty=config.collision_penalty,
        enable_collision_projection=config.enable_collision_projection,
        hitch_penalty=config.hitch_penalty,
        enable_hitch_projection=config.enable_hitch_projection,
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
        # reward shaping parameters
        d_thr_factor=config.d_thr_factor,
        k_switch=config.k_switch,
        steering_weight=config.steering_weight,
        preference_penalty_weight=config.preference_penalty_weight,
        heading_reward_weight=config.heading_reward_weight,
        ref_pos_weight=config.ref_pos_weight,
        ref_theta1_weight=config.ref_theta1_weight,
        ref_theta2_weight=config.ref_theta2_weight
    )
    
    # Set initial position using geometric parameters relative to parking lot
    # dx: distance from tractor front face to target parking space center
    # dy: distance from tractor to parking lot entrance line
    env.set_init_pos(dx=4.0, dy=6.0, theta1=0, theta2=0)
    # Set goal angles based on movement preference
    if config.movement_preference == -1:  # backward parking
        env.set_goal_pos(theta1=jnp.pi/2, theta2=jnp.pi/2)  # backward parking
    
    rew_final, Y0, trajectory_states = run_diffusion(args=config, env=env)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"final reward = {rew_final:.2e}")

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
    Ndiffuse: int = 100 # number of diffusion steps
    temp_sample: float = 0.01  # temperature for sampling
    beta0: float = 1e-5  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False
    # animation
    render: bool = True
    save_animation: bool = True # flag to enable animation saving
    show_animation: bool = True  # flag to show animation during creation
    save_denoising_animation: bool = True  # flag to enable denoising process visualization
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
        env.generate_demonstration_trajectory(search_direction="horizontal")
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
            xref_logpds = jax.vmap(env.eval_xref_logpd)(qs)
            xref_logpds = xref_logpds - xref_logpds.max() # FIXME: without - max, it can deviate from the demo if necessary !!
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
        H=config.Hsample
    )
    
    # Set initial position using geometric parameters relative to parking lot
    # dx: distance from tractor front face to target parking space center
    # dy: distance from tractor to parking lot entrance line
    env.set_init_pos(dx=14.0, dy=6.0, theta1=0, theta2=0)
    
    rew_final, Y0, trajectory_states = run_diffusion(args=config, env=env)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"final reward = {rew_final:.2e}")

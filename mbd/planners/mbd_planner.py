import functools
import os
import jax
from jax import numpy as jnp
from dataclasses import dataclass
import tyro
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


## load config
@dataclass
class Args:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    not_render: bool = False
    # env
    env_name: str = (
        "tt2d"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d", "car2d"
    )
    case: str = "case2"  # "case1" for original obstacles, "case2" for parking scenario
    # diffusion
    Nsample: int = 4000  # number of samples
    Hsample: int = 60  # horizon
    Ndiffuse: int = 100  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False
    # animation
    save_animation: bool = False  # flag to enable animation saving
    show_animation: bool = True  # flag to show animation during creation
    save_denoising_animation: bool = True  # flag to enable denoising process visualization
    dt: float = 0.25


def run_diffusion(args: Args):

    rng = jax.random.PRNGKey(seed=args.seed)
    env = mbd.envs.get_env(args.env_name, case=args.case, dt=args.dt, H=args.Hsample)
    Nx = env.observation_size
    Nu = env.action_size
    
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
    def reverse_once(carry, unused):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (args.Nsample, args.Hsample, Nu)) # NOTE: Sample from N(0, I) 
        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0) # NOTE: clip action to [-1, 1] (it is defined in dynamics)

        # esitimate mu_0tm1
        rewss, qs = jax.vmap(rollout_us_jit, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=-1)
        rew_std = rews.std() # NOTE: scalar
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std) # NOTE: at early stage it is near 0, and increase to 0.1 ish after.
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample

        # evalulate demo
        if args.enable_demo:
            xref_logpds = jax.vmap(env.eval_xref_logpd)(qs)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = (
                (xref_logpds + env.rew_xref - rew_mean) / rew_std / args.temp_sample
            )
            demo_mask = logpdemo > logp0
            logp0 = jnp.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        return (i - 1, rng, Ybar_im1), rews.mean()

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
        return jnp.array(Ybars)

    rng_exp, rng = jax.random.split(rng)
    Yi = reverse(YN, rng_exp) # NOTE: YN: all zeros, one trajectory of actions 
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    if not args.not_render:
        path = f"{mbd.__path__[0]}/../results/{args.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        jnp.save(f"{path}/mu_0ts.npy", Yi)
        
        #if args.env_name == "car2d":
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # rollout
        xs = jnp.array([state_init.pipeline_state])
        state = state_init
        for t in range(Yi.shape[1]):
            state = step_env_jit(state, Yi[-1, t]) # NOTE: Yi[-1, :] is the action at the last denoised step. 
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
        env.render(ax, xs)
        if args.enable_demo:
            ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", label="RRT path")
        ax.legend()
        plt.switch_backend('TkAgg')  # Switch to interactive backend
        plt.show()
        plt.savefig(f"{path}/rollout.png")
        
        # Create denoising animation if requested
        if args.save_denoising_animation:
            create_denoising_animation(env, Yi, args, step_env_jit, state_init)
            
        # Create animation if requested
        if args.save_animation or args.show_animation:
            # Prepare trajectory data for animation
            trajectory_states = [state_init.pipeline_state]
            trajectory_actions = []
            state = state_init
            for t in range(Yi.shape[1]):
                action = Yi[-1, t]
                trajectory_actions.append(action)
                state = step_env_jit(state, action)
                trajectory_states.append(state.pipeline_state)
            
            # Add final state with no action
            trajectory_actions.append(None)
            
            # Create animation
            create_animation(env, trajectory_states, trajectory_actions, args)
        

    rewss_final, _ = rollout_us_jit(state_init, Yi[-1])
    rew_final = rewss_final.mean()

    return rew_final


if __name__ == "__main__":
    import time
    start_time = time.time()
    rew_final = run_diffusion(args=tyro.cli(Args))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"final reward = {rew_final:.2e}")

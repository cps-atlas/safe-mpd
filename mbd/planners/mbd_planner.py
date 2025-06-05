import functools
import os
import jax
from jax import numpy as jnp
from jax import config
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt
import subprocess
import glob
import time
import numpy as np

import mbd

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
    # diffusion
    Nsample: int = 4000  # number of samples
    Hsample: int = 50  # horizon
    Ndiffuse: int = 500  # number of diffusion steps
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = True
    # animation
    save_animation: bool = True  # flag to enable animation saving
    show_animation: bool = True  # flag to show animation during creation


def setup_animation_saving(env_name):
    """Setup directories for animation saving"""
    current_directory_path = os.getcwd()
    animation_path = f"{mbd.__path__[0]}/../results/{env_name}/animations"
    if not os.path.exists(animation_path):
        os.makedirs(animation_path)
    # if file exists, delete all
    if os.path.exists(animation_path):
        for file_name in glob.glob(f"{animation_path}/*.png"):
            os.remove(file_name)
    return animation_path


def export_video(env_name):
    """Convert image sequence to video using ffmpeg"""
    # Use the same path structure as setup_animation_saving
    animation_path = f"{mbd.__path__[0]}/../results/{env_name}/animations"
    
    # Debug: Check if frames exist
    frame_files = glob.glob(f"{animation_path}/frame_*.png")
    print(f"Animation path: {animation_path}")
    print(f"Found {len(frame_files)} frame files")
    if len(frame_files) > 0:
        print(f"First frame: {frame_files[0]}")
        print(f"Last frame: {frame_files[-1]}")
    else:
        print("ERROR: No frame files found!")
        return
    
    # Create video using ffmpeg
    result = subprocess.call(['ffmpeg', '-y',  # -y to overwrite existing files
                     '-framerate', '10',  # Input framerate
                     '-i', f'{animation_path}/frame_%04d.png',
                     '-vf', 'scale=1920:1080,fps=30',  # Scale and set output framerate
                     '-pix_fmt', 'yuv420p',
                     f'{animation_path}/tractor_trailer_animation.mp4'])
    
    if result == 0:
        print("Video created successfully!")
        # Clean up individual frames
        for file_name in glob.glob(f"{animation_path}/*.png"):
            os.remove(file_name)
        print(f"Animation saved to: {animation_path}/tractor_trailer_animation.mp4")
    else:
        print(f"FFmpeg failed with return code: {result}")


def create_animation(env, trajectory_states, trajectory_actions, args):
    """Create animation of the tractor-trailer trajectory"""
    print("Creating animation...")
    
    # Setup animation saving if enabled
    if args.save_animation:
        animation_path = setup_animation_saving(args.env_name)
    
    # Setup interactive plotting
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Get plot limits from environment
    x_range, y_range = env.env.get_plot_limits()
    
    # Set up plot properties
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Tractor-Trailer Animation")
    
    # Add obstacles
    obs = env.env.get_obstacles()
    for i in range(obs.shape[0]):
        circle = plt.Circle(
            obs[i, :2], obs[i, 2], color="k", fill=True, alpha=0.5
        )
        ax.add_artist(circle)
    
    # Add reference trajectory if available
    if hasattr(env, 'xref'):
        ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", alpha=0.5, label="Reference path")
    
    # Add goal
    ax.scatter(env.xg[0], env.xg[1], c='red', s=100, marker='*', label='Goal')
    
    # Setup animation patches
    env.setup_animation_patches(ax)
    
    # Add trajectory trace
    trajectory_line, = ax.plot([], [], 'b-', alpha=0.6, linewidth=2, label='Trajectory')
    
    ax.legend()
    fig.tight_layout()
    
    # Animation loop
    traj_x = []
    traj_y = []
    
    for frame_idx, (state, action) in enumerate(zip(trajectory_states, trajectory_actions)):
        # Convert jax arrays to numpy for matplotlib
        state_np = np.array(state)
        action_np = np.array(action) if action is not None else None
        
        # Update robot visualization
        env.update_animation_patches(state_np, action_np)
        
        # Update trajectory trace
        traj_x.append(state_np[0])
        traj_y.append(state_np[1])
        trajectory_line.set_data(traj_x, traj_y)
        
        # Update plot
        if args.show_animation:
            plt.pause(0.01)  # Small pause for animation effect
        else:
            plt.draw()
        
        # Save frame if animation saving is enabled
        if args.save_animation:
            frame_filename = f"{animation_path}/frame_{frame_idx:04d}.png"
            plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
    
    # Show final result
    if args.show_animation:
        plt.show()
    
    plt.ioff()
    plt.close()
    
    # Export video if saving animation
    if args.save_animation:
        export_video(args.env_name)


def run_diffusion(args: Args):

    rng = jax.random.PRNGKey(seed=args.seed)
    env = mbd.envs.get_env(args.env_name)
    Nx = env.observation_size
    Nu = env.action_size
    
    # env functions
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    # eval_us = jax.jit(functools.partial(mbd.utils.eval_us, step_env_jit))
    rollout_us = jax.jit(functools.partial(mbd.utils.rollout_us, step_env_jit))

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
        rewss, qs = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
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

    rewss_final, _ = rollout_us(state_init, Yi[-1])
    rew_final = rewss_final.mean()

    return rew_final


if __name__ == "__main__":
    import time
    start_time = time.time()
    rew_final = run_diffusion(args=tyro.cli(Args))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"final reward = {rew_final:.2e}")

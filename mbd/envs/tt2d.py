import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt

import mbd

"""
Created on June 3rd, 2025
@author: Taekyung Kim

@description: 
Implement kinematic tractor-trailer dynamics.  
"""

# Runge-Kutta 4th order method
def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray  # NOTE: just the state, pipeline is a term from Brax.
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

    
class TractorTrailer2d:
    """
    State: [px, py, theta1, theta2]
    Action: [v, delta] - velocity and steering angle
    Input constraint: v ∈ [-1, 1], delta ∈ [-55°, 55°]
    """
    def __init__(self):
        self.dt = 0.1
        self.H = 50
        
        # Tractor-trailer parameters
        self.l1 = 3.23  # tractor wheelbase
        self.l2 = 2.9   # trailer length
        self.lh = 1.15  # hitch length
        self.tractor_width = 2.0
        self.trailer_width = 2.5
        
        # Input constraints
        self.v_max = 2.0  # velocity limit
        self.delta_max = 75.0 * jnp.pi / 180.0  # steering angle limit in radians
        
        # Obstacles
        r_obs = 0.3
        self.obs_center = jnp.array(
            [
                [-r_obs * 3, r_obs * 2],
                [-r_obs * 2, r_obs * 2],
                [-r_obs * 1, r_obs * 2],
                [0.0, r_obs * 2],
                [0.0, r_obs * 1],
                [0.0, 0.0],
                [0.0, -r_obs * 1],
                [-r_obs * 3, -r_obs * 2],
                [-r_obs * 2, -r_obs * 2],
                [-r_obs * 1, -r_obs * 2],
                [0.0, -r_obs * 2],
            ]
        )
        self.obs_radius = r_obs  # Radius of the obstacle
        
        # Initial and goal states: [px, py, theta1, theta2]
        self.x0 = jnp.array([-0.5, 0.0, 0.0, 0.0])
        self.xg = jnp.array([0.5, 0.0, 0.0, 0.0])
        
        # Load and process reference trajectory
        xref_original = jnp.load(f"{mbd.__path__[0]}/assets/car2d_xref.npy")
        
        # interpolate each two points in xref to make it 100 points
        # Interpolate xref to get 100 points by averaging consecutive points
        xref_interp = []
        for i in range(xref_original.shape[0]-1):
            xref_interp.append(xref_original[i,:])
            xref_interp.append((xref_original[i,:] + xref_original[i+1,:])/2)
        xref_interp.append(xref_original[-1,:])
        xref_2d = jnp.array(xref_interp)
        
        xref_2d = xref_original # TODO: use the original trajectory for now.
        
        # Extend 2D reference to 4D state space [px, py, theta1, theta2]
        xref_diff = jnp.diff(xref_2d, axis=0)
        theta = jnp.arctan2(xref_diff[:, 1], xref_diff[:, 0])  # Note: y,x order for atan2
        theta = jnp.append(theta, theta[-1])
        
        # Create 4D reference trajectory
        self.xref = jnp.zeros((xref_2d.shape[0], 4))
        self.xref = self.xref.at[:, :2].set(xref_2d)  # px, py
        self.xref = self.xref.at[:, 2].set(theta)  # theta1
        self.xref = self.xref.at[:, 3].set(theta)  # theta2 (assume aligned initially)
        
        print(f"Reference trajectory shape: {self.xref.shape}")
        self.rew_xref = jax.vmap(self.get_reward)(self.xref).mean()

    def reset(self, rng: jax.Array):
        """Resets the environment to an initial state."""
        return State(self.x0, self.x0, 0.0, 0.0)
    
    @partial(jax.jit, static_argnums=(0,))
    def input_scaler(self, u_normalized):
        """Scale normalized inputs [-1, 1] to actual input ranges"""
        v = u_normalized[0] * self.v_max  # velocity
        delta = u_normalized[1] * self.delta_max  # steering angle
        return jnp.array([v, delta])
    
    @partial(jax.jit, static_argnums=(0,))
    def tractor_trailer_dynamics(self, x, u):
        """
        Tractor-trailer dynamics
        State: [px, py, theta1, theta2]
        Input: [v, delta] (already scaled)
        """
        px, py, theta1, theta2 = x
        v, delta = u
        
        # Dynamics equations
        px_dot = v * jnp.cos(theta1)
        py_dot = v * jnp.sin(theta1)
        theta1_dot = (v / self.l1) * jnp.tan(delta)
        theta2_dot = (v / self.l2) * (
            jnp.sin(theta1 - theta2) - 
            (self.lh / self.l1) * jnp.cos(theta1 - theta2) * jnp.tan(delta)
        )
        
        return jnp.array([px_dot, py_dot, theta1_dot, theta2_dot])
        
    @partial(jax.jit, static_argnums=(0,))
    def check_collision(self, x, obs_center, obs_radius):
        """Check collision using tractor position for now (FIXME: will be updated later for full geometry)"""
        dist2objs = jnp.linalg.norm(x[:2] - obs_center, axis=1)
        return jnp.any(dist2objs < obs_radius)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        action = jnp.clip(action, -1.0, 1.0)
        
        # Scale inputs from normalized [-1, 1] to actual ranges
        u_scaled = self.input_scaler(action)
        
        q = state.pipeline_state
        q_new = rk4(self.tractor_trailer_dynamics, state.pipeline_state, u_scaled, self.dt)
        
        # Check collision (using tractor position for now)
        collide = self.check_collision(q_new, self.obs_center, self.obs_radius)
        
        # If collision, don't update the state
        q = jnp.where(collide, q, q_new)
        reward = self.get_reward(q)
        return state.replace(pipeline_state=q, obs=q, reward=reward, done=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        """Reward based on distance to goal position"""
        reward = (
            1.0 - (jnp.clip(jnp.linalg.norm(q[:2] - self.xg[:2]), 0.0, 0.2) / 0.2) ** 2
        )
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def eval_xref_logpd(self, xs):
        """Evaluate log probability density with respect to reference trajectory"""
        xs_err = xs[:, :2] - self.xref[:, :2]
        logpd = 0.0-(
            (jnp.clip(jnp.linalg.norm(xs_err, axis=-1), 0.0, 0.5) / 0.5) ** 2
        ).mean(axis=-1)
        return logpd # NOTE: unnormalized logpd, log p_demo(Y0) in the paper.

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 4  # [px, py, theta1, theta2]

    def render(self, ax, xs: jnp.ndarray):
        """Render the tractor-trailer system"""
        # obstacles
        for i in range(self.obs_center.shape[0]):
            circle = plt.Circle(
                self.obs_center[i, :], self.obs_radius, color="k", fill=True, alpha=0.5
            )
            ax.add_artist(circle)
        
        # Plot trajectory
        ax.scatter(xs[:, 0], xs[:, 1], c=range(self.H + 1), cmap="Reds")
        ax.plot(xs[:, 0], xs[:, 1], "r-", label="Tractor path")
        
        # Plot tractor and trailer orientations (optional)
        # ax.quiver(
        #     xs[::5, 0], xs[::5, 1],
        #     jnp.cos(xs[::5, 3]), jnp.sin(xs[::5, 3]),
        #     color='blue', alpha=0.7, scale=20, label='Tractor orientation'
        # )
        # ax.quiver(
        #     xs[::5, 0], xs[::5, 1], 
        #     jnp.cos(xs[::5, 4]), jnp.sin(xs[::5, 4]),
        #     color='green', alpha=0.7, scale=20, label='Trailer orientation'
        # )
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True)
        # ax.set_title("Tractor-Trailer 2D")

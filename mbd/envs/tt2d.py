import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D

import mbd
from .env import Env

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
    def __init__(self, x0=None, xg=None, env_config=None, case="case1", dt=0.2, H=50):
        # Time parameters
        self.dt = dt
        self.H = H
        
        # Tractor-trailer parameters
        self.l1 = 3.23  # tractor wheelbase
        self.l2 = 2.9   # trailer length
        self.lh = 1.15  # hitch length
        self.tractor_width = 2.0
        self.trailer_width = 2.5
        
        # Input constraints
        self.v_max = 3.0  # velocity limit
        self.delta_max = 55.0 * jnp.pi / 180.0  # steering angle limit in radians
        
        # Reward and reference thresholds (hyperparameters)
        if case == "case1":
            self.reward_threshold = 1.2  # was 0.2 for original scale, scaled by 6x for [-3, 3]
            self.ref_threshold = 3.0     # was 0.5 for original scale, scaled by 6x for [-3, 3]
        elif case == "case2":
            self.reward_threshold = 25.0  # Larger threshold for parking scenario
            self.ref_threshold = 3.0     # Larger threshold for parking scenario
        
        # Environment setup
        if env_config is None:
            self.env = Env(case=case)
        else:
            self.env = env_config
        
        # Get obstacles from environment
        obstacles = self.env.get_obstacles()
        self.obs_circles = obstacles['circles']
        self.obs_rectangles = obstacles['rectangles']
        
        # For backward compatibility with existing collision checking
        self.obs = self.obs_circles
        
        # Set initial and goal states based on case and user input
        if x0 is None:
            if case == "case1":
                self.x0 = self.set_initial_pos()
            elif case == "case2":
                self.x0 = self.env.get_initial_position_case2()
        else:
            self.x0 = x0
            
        if xg is None:
            if case == "case1":
                self.xg = self.set_goal_pos()
            elif case == "case2":
                self.xg = self.env.get_goal_position_case2()
        else:
            self.xg = xg
        
            
        print(f"x0: {self.x0}")
        print(f"xg: {self.xg}")
        
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
        #print(f"xref_2d: {xref_2d}")
        xref_2d = xref_2d * 6
        #print(f"xref_2d: {xref_2d}")
        
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

        # Animation-related attributes
        self.tractor_body = None
        self.trailer_body = None
        self.tractor_wheels = []
        self.trailer_wheels = []
        self.hitch_line = None
        
        # Print environment info
        if hasattr(self.env, 'print_parking_layout'):
            self.env.print_parking_layout()
        
    def set_initial_pos(self, x=-3.0, y=0.0, theta1=np.pi, theta2=np.pi):
        """Set initial position for tractor-trailer (case1 default)"""
        return jnp.array([x, y, theta1, theta2])

    def set_goal_pos(self, x=3.0, y=0.0, theta1=np.pi, theta2=np.pi):
        """Set goal position for tractor-trailer (case1 default)"""
        return jnp.array([x, y, theta1, theta2]) 

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
    def get_tractor_trailer_rectangles(self, x):
        """Get the corner points of tractor and trailer rectangles for collision checking"""
        px, py, theta1, theta2 = x
        
        # Tractor rectangle (centered at tractor center)
        tractor_center_x = px + (self.l1/2) * jnp.cos(theta1)
        tractor_center_y = py + (self.l1/2) * jnp.sin(theta1)
        
        # Tractor corners in local coordinates (before rotation)
        tractor_half_length = self.l1 / 2
        tractor_half_width = self.tractor_width / 2
        tractor_local_corners = jnp.array([
            [-tractor_half_length, -tractor_half_width],  # rear left
            [-tractor_half_length, tractor_half_width],   # rear right  
            [tractor_half_length, tractor_half_width],    # front right
            [tractor_half_length, -tractor_half_width]    # front left
        ])
        
        # Rotate and translate tractor corners
        cos_theta1, sin_theta1 = jnp.cos(theta1), jnp.sin(theta1)
        rotation_matrix_1 = jnp.array([[cos_theta1, -sin_theta1], [sin_theta1, cos_theta1]])
        tractor_corners = (tractor_local_corners @ rotation_matrix_1.T) + jnp.array([tractor_center_x, tractor_center_y])
        
        # Trailer rectangle
        hitch_x = px - self.lh * jnp.cos(theta1)
        hitch_y = py - self.lh * jnp.sin(theta1)
        trailer_center_x = hitch_x - (self.l2/2) * jnp.cos(theta2)
        trailer_center_y = hitch_y - (self.l2/2) * jnp.sin(theta2)
        
        # Trailer corners in local coordinates
        trailer_half_length = self.l2 / 2
        trailer_half_width = self.trailer_width / 2
        trailer_local_corners = jnp.array([
            [-trailer_half_length, -trailer_half_width],  # rear left
            [-trailer_half_length, trailer_half_width],   # rear right
            [trailer_half_length, trailer_half_width],    # front right  
            [trailer_half_length, -trailer_half_width]    # front left
        ])
        
        # Rotate and translate trailer corners
        cos_theta2, sin_theta2 = jnp.cos(theta2), jnp.sin(theta2)
        rotation_matrix_2 = jnp.array([[cos_theta2, -sin_theta2], [sin_theta2, cos_theta2]])
        trailer_corners = (trailer_local_corners @ rotation_matrix_2.T) + jnp.array([trailer_center_x, trailer_center_y])
        
        return {
            'tractor_corners': tractor_corners,
            'tractor_center': jnp.array([tractor_center_x, tractor_center_y]),
            'tractor_angle': theta1,
            'trailer_corners': trailer_corners, 
            'trailer_center': jnp.array([trailer_center_x, trailer_center_y]),
            'trailer_angle': theta2
        }

    @partial(jax.jit, static_argnums=(0,))
    def check_rectangle_circle_collision(self, rect_corners, circle_center, circle_radius):
        """Check collision between a rectangle (given by corners) and a circle"""
        # Find the closest point on the rectangle to the circle center
        # Get rectangle edges
        edges = jnp.roll(rect_corners, -1, axis=0) - rect_corners
        
        # For each edge, find closest point to circle center
        edge_starts = rect_corners
        to_centers = circle_center[None, :] - edge_starts  # Shape: (4, 2)
        
        # Project circle center onto each edge
        edge_lengths_sq = jnp.sum(edges ** 2, axis=1)  # Shape: (4,)
        edge_lengths_sq = jnp.where(edge_lengths_sq < 1e-8, 1e-8, edge_lengths_sq)  # Avoid division by zero
        
        projections = jnp.sum(to_centers * edges, axis=1) / edge_lengths_sq  # Shape: (4,)
        t_values = jnp.clip(projections, 0.0, 1.0)  # Shape: (4,)
        
        # Closest points on edges
        closest_points = edge_starts + t_values[:, None] * edges  # Shape: (4, 2)
        
        # Distances squared from circle center to closest points
        dist_sqs = jnp.sum((circle_center[None, :] - closest_points) ** 2, axis=1)  # Shape: (4,)
        min_dist_sq = jnp.min(dist_sqs)
        
        return min_dist_sq <= circle_radius ** 2

    @partial(jax.jit, static_argnums=(0,))
    def check_rectangle_rectangle_collision(self, rect1_corners, rect2_corners):
        """Check collision between two rectangles using Separating Axis Theorem"""
        
        def get_axes(corners):
            """Get the two axes (edge normals) for a rectangle"""
            edges = jnp.roll(corners, -1, axis=0) - corners
            # Get normals (perpendicular to edges) - take first two edges only
            normals = jnp.array([[-edges[0, 1], edges[0, 0]], 
                                [-edges[1, 1], edges[1, 0]]])
            # Normalize
            norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
            norms = jnp.where(norms < 1e-8, 1.0, norms)  # Avoid division by zero
            normals = normals / norms
            return normals
        
        def project_rectangle(corners, axis):
            """Project rectangle corners onto an axis"""
            projections = jnp.dot(corners, axis)
            return jnp.min(projections), jnp.max(projections)
        
        # Get axes for both rectangles
        axes1 = get_axes(rect1_corners)
        axes2 = get_axes(rect2_corners)
        all_axes = jnp.vstack([axes1, axes2])  # Shape: (4, 2)
        
        # Check separation on each axis using vectorized operations
        def check_axis_separation(axis):
            min1, max1 = project_rectangle(rect1_corners, axis)
            min2, max2 = project_rectangle(rect2_corners, axis)
            # Return True if separated, False if overlapping
            return (max1 < min2) | (max2 < min1)
        
        # Check all axes
        separations = jax.vmap(check_axis_separation)(all_axes)  # Shape: (4,)
        
        # If any axis shows separation, no collision
        return ~jnp.any(separations)

    @partial(jax.jit, static_argnums=(0,))
    def check_collision(self, x, obs_circles, obs_rectangles):
        """Check collision with full tractor-trailer geometry against all obstacles"""
        # Get tractor and trailer geometry
        geometry = self.get_tractor_trailer_rectangles(x)
        tractor_corners = geometry['tractor_corners']
        trailer_corners = geometry['trailer_corners']
        
        collision = False
        
        # Check circular obstacles
        if obs_circles.shape[0] > 0:
            def check_circle_collision(circle_obs):
                circle_center = circle_obs[:2]
                circle_radius = circle_obs[2]
                
                # Check tractor-circle collision
                tractor_collision = self.check_rectangle_circle_collision(
                    tractor_corners, circle_center, circle_radius
                )
                
                # Check trailer-circle collision  
                trailer_collision = self.check_rectangle_circle_collision(
                    trailer_corners, circle_center, circle_radius
                )
                
                return tractor_collision | trailer_collision
            
            # Vectorize over all circular obstacles
            circle_collisions = jax.vmap(check_circle_collision)(obs_circles)
            collision = collision | jnp.any(circle_collisions)
        
        # Check rectangular obstacles
        if obs_rectangles.shape[0] > 0:
            def check_rect_collision(rect_obs):
                obs_x, obs_y, obs_width, obs_height, obs_angle = rect_obs
                
                # Get obstacle rectangle corners
                obs_half_width = obs_width / 2
                obs_half_height = obs_height / 2
                obs_local_corners = jnp.array([
                    [-obs_half_width, -obs_half_height],
                    [-obs_half_width, obs_half_height], 
                    [obs_half_width, obs_half_height],
                    [obs_half_width, -obs_half_height]
                ])
                
                # Rotate and translate obstacle corners
                cos_obs, sin_obs = jnp.cos(obs_angle), jnp.sin(obs_angle)
                obs_rotation = jnp.array([[cos_obs, -sin_obs], [sin_obs, cos_obs]])
                obs_corners = (obs_local_corners @ obs_rotation.T) + jnp.array([obs_x, obs_y])
                
                # Check tractor-rectangle collision
                tractor_collision = self.check_rectangle_rectangle_collision(
                    tractor_corners, obs_corners
                )
                
                # Check trailer-rectangle collision
                trailer_collision = self.check_rectangle_rectangle_collision(
                    trailer_corners, obs_corners
                )
                
                return tractor_collision | trailer_collision
            
            # Vectorize over all rectangular obstacles
            rect_collisions = jax.vmap(check_rect_collision)(obs_rectangles)
            collision = collision | jnp.any(rect_collisions)
        
        return collision

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        action = jnp.clip(action, -1.0, 1.0)
        
        # Scale inputs from normalized [-1, 1] to actual ranges
        u_scaled = self.input_scaler(action)
        
        q = state.pipeline_state
        q_new = rk4(self.tractor_trailer_dynamics, state.pipeline_state, u_scaled, self.dt)
        
        # Check collision with full geometry
        collide = self.check_collision(q_new, self.obs_circles, self.obs_rectangles)
        
        # If collision, don't update the state
        q = jnp.where(collide, q, q_new)
        reward = self.get_reward(q)
        return state.replace(pipeline_state=q, obs=q, reward=reward, done=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        """Reward based on distance to goal position"""
        reward = (
            1.0 - (jnp.clip(jnp.linalg.norm(q[:2] - self.xg[:2]), 0.0, self.reward_threshold) / self.reward_threshold) ** 2
        )
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def eval_xref_logpd(self, xs):
        """Evaluate log probability density with respect to reference trajectory"""
        xs_err = xs[:, :2] - self.xref[:, :2]
        logpd = 0.0-(
            (jnp.clip(jnp.linalg.norm(xs_err, axis=-1), 0.0, self.ref_threshold) / self.ref_threshold) ** 2
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
        # Render circular obstacles
        if self.obs_circles.shape[0] > 0:
            for i in range(self.obs_circles.shape[0]):
                circle = plt.Circle(
                    self.obs_circles[i, :2], self.obs_circles[i, 2], color="k", fill=True, alpha=0.5
                )
                ax.add_artist(circle)
        
        # Render rectangular obstacles
        if self.obs_rectangles.shape[0] > 0:
            for i in range(self.obs_rectangles.shape[0]):
                x_center, y_center, width, height, angle = self.obs_rectangles[i]
                
                # Create rectangle patch
                rect = plt.Rectangle((-width/2, -height/2), width, height,
                                   linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                
                # Apply rotation and translation
                transform = (Affine2D()
                           .rotate(angle)
                           .translate(x_center, y_center) + ax.transData)
                rect.set_transform(transform)
                ax.add_patch(rect)
        
        # Plot trajectory
        ax.scatter(xs[:, 0], xs[:, 1], c=range(self.H + 1), cmap="Reds", s=45)
        ax.plot(xs[:, 0], xs[:, 1], "r-", linewidth=1.5, label="Tractor path")
        
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
        
        # Get plot limits from environment
        x_range, y_range = self.env.get_plot_limits()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect("equal")
        #ax.grid(True)
        # ax.set_title("Tractor-Trailer 2D")

    def setup_animation_patches(self, ax, tractor_color='blue', trailer_color='red'):
        """Initialize the patches for tractor-trailer visualization"""
        
        # Tractor body (rectangle)
        self.tractor_body = ax.add_patch(
            plt.Rectangle((-self.l1/2, -self.tractor_width/2),
                         self.l1, self.tractor_width,
                         linewidth=2, edgecolor='black', facecolor=tractor_color, alpha=0.7)
        )
        
        # Trailer body (rectangle)
        self.trailer_body = ax.add_patch(
            plt.Rectangle((-self.l2/2, -self.trailer_width/2),
                         self.l2, self.trailer_width,
                         linewidth=2, edgecolor='black', facecolor=trailer_color, alpha=0.7)
        )
        
        # Wheels
        wheel_width = 0.3
        wheel_length = 0.15
        
        # Tractor wheels (front and rear)
        for i in range(2):
            wheel = ax.add_patch(
                plt.Rectangle((-wheel_length/2, -wheel_width/2),
                             wheel_length, wheel_width,
                             edgecolor='black', facecolor='gray', alpha=0.9)
            )
            self.tractor_wheels.append(wheel)
        
        # Trailer wheels
        for i in range(2):
            wheel = ax.add_patch(
                plt.Rectangle((-wheel_length/2, -wheel_width/2),
                             wheel_length, wheel_width,
                             edgecolor='black', facecolor='gray', alpha=0.9)
            )
            self.trailer_wheels.append(wheel)
        
        # Hitch connection line
        self.hitch_line, = ax.plot([], [], 'k-', linewidth=3, alpha=0.8)

    def get_tractor_trailer_positions(self, x):
        """Calculate tractor and trailer positions and orientations"""
        px, py, theta1, theta2 = x
        
        # Tractor position (rear axle center)
        tractor_rear_x = px
        tractor_rear_y = py
        
        # Tractor front axle
        tractor_front_x = px + self.l1 * np.cos(theta1)
        tractor_front_y = py + self.l1 * np.sin(theta1)
        
        # Hitch point (at rear of tractor)
        hitch_x = px - self.lh * np.cos(theta1)
        hitch_y = py - self.lh * np.sin(theta1)
        
        # Trailer rear axle (at hitch + trailer length)
        trailer_rear_x = hitch_x - self.l2 * np.cos(theta2)
        trailer_rear_y = hitch_y - self.l2 * np.sin(theta2)
        
        # Trailer front (at hitch point)
        trailer_front_x = hitch_x
        trailer_front_y = hitch_y
        
        return {
            'tractor_rear': (tractor_rear_x, tractor_rear_y),
            'tractor_front': (tractor_front_x, tractor_front_y),
            'trailer_rear': (trailer_rear_x, trailer_rear_y),
            'trailer_front': (trailer_front_x, trailer_front_y),
            'hitch': (hitch_x, hitch_y),
            'theta1': theta1,
            'theta2': theta2
        }

    def render_rigid_body(self, x, u=None):
        """Return the transforms to render the tractor-trailer system"""
        px, py, theta1, theta2 = x
        
        # Get positions
        positions = self.get_tractor_trailer_positions(x)
        
        # Tractor body transform (centered at tractor center)
        tractor_center_x = px + (self.l1/2) * np.cos(theta1)
        tractor_center_y = py + (self.l1/2) * np.sin(theta1)
        transform_tractor_body = (Affine2D()
                                 .rotate(theta1)
                                 .translate(tractor_center_x, tractor_center_y) + plt.gca().transData)
        
        # Trailer body transform (centered at trailer center)
        trailer_center_x = positions['hitch'][0] - (self.l2/2) * np.cos(theta2)
        trailer_center_y = positions['hitch'][1] - (self.l2/2) * np.sin(theta2)
        transform_trailer_body = (Affine2D()
                                 .rotate(theta2)
                                 .translate(trailer_center_x, trailer_center_y) + plt.gca().transData)
        
        # Wheel transforms
        transforms_tractor_wheels = []
        # Tractor rear wheel
        transforms_tractor_wheels.append(
            Affine2D().rotate(theta1).translate(*positions['tractor_rear']) + plt.gca().transData
        )
        # Tractor front wheel (with steering angle if u is provided)
        steering_angle = 0.0
        if u is not None:
            v, delta = u
            steering_angle = delta
        transforms_tractor_wheels.append(
            Affine2D().rotate(theta1 + steering_angle).translate(*positions['tractor_front']) + plt.gca().transData
        )
        
        transforms_trailer_wheels = []
        # Trailer wheels
        transforms_trailer_wheels.append(
            Affine2D().rotate(theta2).translate(*positions['trailer_rear']) + plt.gca().transData
        )
        transforms_trailer_wheels.append(
            Affine2D().rotate(theta2).translate(*positions['trailer_front']) + plt.gca().transData
        )
        
        return {
            'tractor_body': transform_tractor_body,
            'trailer_body': transform_trailer_body,
            'tractor_wheels': transforms_tractor_wheels,
            'trailer_wheels': transforms_trailer_wheels,
            'hitch_line_data': (positions['tractor_rear'], positions['hitch'])
        }

    def update_animation_patches(self, x, u=None):
        """Update all patches with new state"""
        transforms = self.render_rigid_body(x, u)
        
        # Update body transforms
        self.tractor_body.set_transform(transforms['tractor_body'])
        self.trailer_body.set_transform(transforms['trailer_body'])
        
        # Update wheel transforms
        for i, wheel in enumerate(self.tractor_wheels):
            wheel.set_transform(transforms['tractor_wheels'][i])
        for i, wheel in enumerate(self.trailer_wheels):
            wheel.set_transform(transforms['trailer_wheels'][i])
        
        # Update hitch line
        hitch_data = transforms['hitch_line_data']
        self.hitch_line.set_data([hitch_data[0][0], hitch_data[1][0]], 
                                [hitch_data[0][1], hitch_data[1][1]])

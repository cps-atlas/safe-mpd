import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.transforms import Affine2D

from .tt2d import TractorTrailer2d, State, rk4

"""
Created on August 4th, 2025
@description: 
Kinematic bicycle model derived from tractor-trailer dynamics.
State: [px, py, theta1]
Action: [v, delta]
"""

class KinematicBicycle2d(TractorTrailer2d):
    """
    Kinematic bicycle model
    State: [px, py, theta1] - position and heading of tractor only
    Action: [v, delta] - velocity and steering angle
    
    Inherits most functionality from TractorTrailer2d but overrides
    functions related to trailer dynamics and collision checking.
    """
    
    def __init__(self, **kwargs):
        # Initialize parent class with all parameters
        super().__init__(**kwargs)
        
        # Override initial and goal states to be 3D
        if hasattr(self, 'x0') and len(self.x0) > 3:
            self.x0 = self.x0[:3]  # Keep only [px, py, theta1]
        if hasattr(self, 'xg') and len(self.xg) > 3:
            self.xg = self.xg[:3]  # Keep only [px, py, theta1]

    def set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0):
        """
        Set initial position for kinematic bicycle (3D state).
        Note: theta2 parameter is ignored since we don't have a trailer.
        """
        if hasattr(self.env, 'case') and self.env.case == "parking" and dx is not None and dy is not None:
            # Use same parking logic as parent but only keep 3D state
            super().set_init_pos(x=x, y=y, dx=dx, dy=dy, theta1=theta1, theta2=0.0)
            self.x0 = self.x0[:3]  # Convert to 3D
        elif x is not None and y is not None:
            self.x0 = jnp.array([x, y, theta1])
            logging.debug(f"overwrite x0: {self.x0}")
        else:
            # Default case
            x = x if x is not None else -3.0
            y = y if y is not None else 0.0
            self.x0 = jnp.array([x, y, theta1])

    def set_goal_pos(self, x=None, y=None, theta1=None, theta2=None):
        """
        Set goal position for kinematic bicycle).
        Note: theta2 parameter is ignored since we don't have a trailer.
        """
        if hasattr(self, 'xg'):
            # Update only specified values while keeping others unchanged
            new_x = x if x is not None else (self.xg[0] if len(self.xg) > 0 else 3.0)
            new_y = y if y is not None else (self.xg[1] if len(self.xg) > 1 else 0.0)
            new_theta1 = theta1 if theta1 is not None else (self.xg[2] if len(self.xg) > 2 else np.pi)
        else:
            # Use defaults for unspecified values
            new_x = x if x is not None else 3.0
            new_y = y if y is not None else 0.0
            new_theta1 = theta1 if theta1 is not None else np.pi
            
        self.xg = jnp.array([new_x, new_y, new_theta1])
        logging.debug(f"overwrite xg: {self.xg}")

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array):
        """Resets the environment to an initial state (3D)."""
        return State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(2), jnp.array(False))
    
    @partial(jax.jit, static_argnums=(0,))
    def bicycle_dynamics(self, x, u):
        """
        Kinematic bicycle dynamics (no trailer)
        State: [px, py, theta1]
        Input: [v, delta] (already scaled)
        """
        px, py, theta1 = x
        v, delta = u
        
        # Bicycle dynamics equations
        px_dot = v * jnp.cos(theta1)
        py_dot = v * jnp.sin(theta1)
        theta1_dot = (v / self.l1) * jnp.tan(delta)
        
        return jnp.array([px_dot, py_dot, theta1_dot])
    
    @partial(jax.jit, static_argnums=(0,))
    def tractor_trailer_dynamics(self, x, u):
        """Override to use bicycle dynamics instead of tractor-trailer dynamics."""
        return self.bicycle_dynamics(x, u)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_tractor_rectangle(self, x):
        """Get only the tractor rectangle for collision checking"""
        px, py, theta1 = x[:3]
        
        # Tractor rectangle (centered at tractor center)
        tractor_center_x = px + 0.5 * (self.l1 + self.lf1 - self.lr) * jnp.cos(theta1)
        tractor_center_y = py + 0.5 * (self.l1 + self.lf1 - self.lr) * jnp.sin(theta1)

        # Tractor corners in local coordinates
        tractor_half_length = (self.l1 + self.lf1 + self.lr) / 2
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
        
        return {
            'tractor_corners': tractor_corners,
            'tractor_center': jnp.array([tractor_center_x, tractor_center_y]),
            'tractor_angle': theta1
        }

    @partial(jax.jit, static_argnums=(0,))
    def check_obstacle_collision(self, x, obs_circles, obs_rectangles):
        """Check collision with only tractor geometry against all obstacles"""
        # Get only tractor geometry
        geometry = self.get_tractor_rectangle(x)
        tractor_corners = geometry['tractor_corners']
        
        collision = False  # Start with no collision
        
        # Check circular obstacles
        if obs_circles.shape[0] > 0:
            def check_circle_collision(circle_obs):
                circle_center = circle_obs[:2]
                circle_radius = circle_obs[2]
                
                # Check only tractor-circle collision
                tractor_collision = self.check_rectangle_circle_collision(
                    tractor_corners, circle_center, circle_radius
                )
                
                return tractor_collision
            
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
                
                # Check only tractor-rectangle collision
                tractor_collision = self.check_rectangle_rectangle_collision(
                    tractor_corners, obs_corners
                )
                
                return tractor_collision
            
            # Vectorize over all rectangular obstacles
            rect_collisions = jax.vmap(check_rect_collision)(obs_rectangles)
            collision = collision | jnp.any(rect_collisions)
        
        return collision
    
    @partial(jax.jit, static_argnums=(0,))
    def check_hitch_violation(self, x):
        """No hitch angle violation for bicycle model"""
        return False  # No trailer, so no hitch violation possible

    @partial(jax.jit, static_argnums=(0,))
    def check_collision(self, x, obs_circles, obs_rectangles):
        """Check collision with only tractor geometry (no hitch violations)"""
        return self.check_obstacle_collision(x, obs_circles, obs_rectangles)
    
    @partial(jax.jit, static_argnums=(0, 3))
    def _step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        # Persist backup mode across steps; once entered, stay in backup
        backup_active_prev = state.in_backup_mode
        u_backup_norm = jnp.array([0.0, 0.0])
        action_eff_norm = jnp.where(backup_active_prev, u_backup_norm, action)
        u_scaled = self.input_scaler(action_eff_norm)
        q = state.pipeline_state
        if self.enable_projection:
            u_safe_normalized = self.project_control_to_safe_set(q, action)
            u_safe_scaled = self.input_scaler(u_safe_normalized)
            q_final = rk4(self.tractor_trailer_dynamics, q, u_safe_scaled, self.dt)
            q_reward = q_final
            obstacle_collision = self.check_obstacle_collision(q_reward, self.obs_circles, self.obs_rectangles)
            hitch_violation = self.check_hitch_violation(q_reward)
            applied_action = u_safe_normalized
            backup_active_next = backup_active_prev
        elif self.enable_guidance:
            q_proposed = rk4(self.tractor_trailer_dynamics, q, u_scaled, self.dt)
            q_reward = self.apply_guidance(q_proposed)
            q_final = q_reward if visualization_mode else q_proposed
            obstacle_collision = False
            hitch_violation = False
            applied_action = action_eff_norm
            backup_active_next = backup_active_prev
        else:
            q_proposed = rk4(self.tractor_trailer_dynamics, q, u_scaled, self.dt)
            def use_shielded_rollout_fn(args):
                q_prop, = args
                return self._step_with_shielded_rollout((q, q_prop))
            def use_naive_penalty_fn(args):
                q_prop, = args
                return self._step_without_shielded_rollout(q_prop)
            use_shielded_rollout = self.enable_shielded_rollout_collision | self.enable_shielded_rollout_hitch
            q_final, obstacle_collision, hitch_violation = jax.lax.cond(
                use_shielded_rollout,
                use_shielded_rollout_fn,
                use_naive_penalty_fn,
                (q_proposed,)
            )
            q_reward = q_final
            should_fallback = (self.enable_shielded_rollout_collision & obstacle_collision) | \
                              (self.enable_shielded_rollout_hitch & hitch_violation)
            backup_active_next = backup_active_prev | should_fallback
            applied_action = jnp.where(backup_active_next, u_backup_norm, action)
        reward = self.get_reward(q_reward)
        reward = jnp.where(obstacle_collision, reward - self.collision_penalty, reward)
        reward = jnp.where(hitch_violation, reward - self.hitch_penalty, reward)
        # Use actually applied control for preference penalty
        u_applied_scaled = self.input_scaler(applied_action)
        preference_penalty = self.get_preference_penalty(u_applied_scaled)
        has_preference = self.motion_preference != 0
        reward = jnp.where(has_preference, reward - preference_penalty, reward)
        # NOTE: since apply fallback for all time steps drop the sample efficiency too much, we allow shielded 
        return state.replace(pipeline_state=q_final, obs=q_final, reward=reward, done=0.0,
                              applied_action=applied_action, in_backup_mode=backup_active_prev)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_trailer_back_position(self, x):
        """Note: there is no trailer in the bicycle model, so we return the tractor back position"""
        px, py, theta1 = x[:3]
        
        tractor_back_x = px - self.lh * jnp.cos(theta1)
        tractor_back_y = py - self.lh * jnp.sin(theta1)
        
        return jnp.array([tractor_back_x, tractor_back_y])
    

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        """
        Reward function for bicycle model - only track tractor position and heading.
        """
        # Convenience shorthands
        px, py, theta1 = q
        pgx, pgy, thetag = self.xg[:3]  # goal pos, goal heading
        tractor_pos = jnp.array([px + self.l1 * jnp.cos(theta1), py + self.l1 * jnp.sin(theta1)])
        tractor_goal = self.xg[:2]
        
        # Positional reward
        d_pos = jnp.linalg.norm(tractor_pos - tractor_goal)
        r_pos = 1.0 - (jnp.clip(d_pos, 0., self.reward_threshold) / self.reward_threshold) ** 2

        # Heading-to-goal alignment
        wrap_pi = lambda a: (a + jnp.pi) % (2.*jnp.pi) - jnp.pi
        
        # Handle angle errors based on motion preference
        e_theta1_direct = wrap_pi(theta1 - thetag)
        
        # For no preference case, also try opposite orientation
        e_theta1_offset = wrap_pi(theta1 - thetag - jnp.pi)
        
        # Choose orientation with smaller error (for no preference case)
        use_direct = jnp.abs(e_theta1_direct) < jnp.abs(e_theta1_offset)
        e_theta1_wrapped = jnp.where(use_direct, e_theta1_direct, e_theta1_offset)
        
        # Select final angle: use wrapping only when no preference, direct otherwise
        no_preference = self.motion_preference == 0
        e_theta1 = jnp.where(no_preference, e_theta1_wrapped, e_theta1_direct)

        r_hdg = 1.0 - (jnp.abs(e_theta1) / self.theta_max) ** 2

        # No articulation term for bicycle model (r_art = 0)
        r_art = 0.0

        # Logistic distance switch
        sigmoid = lambda z: 1.0 / (1.0 + jnp.exp(-z))
        w_theta = sigmoid((self.d_thr - d_pos) / self.k_switch)
        w_theta = jnp.clip(w_theta, 0.0, self.max_w_theta)

        # Weighted stage cost (no hitch angle component)
        reward = (1.0 - w_theta) * r_pos + w_theta * r_hdg
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def get_terminal_reward(self, q):
        """Terminal reward for bicycle model - only position tracking."""
        px, py, theta1 = q
        tractor_pos = jnp.array([px + self.l1 * jnp.cos(theta1), py + self.l1 * jnp.sin(theta1)])
        tractor_goal = self.xg[:2]
        
        # Positional reward only
        d_pos = jnp.linalg.norm(tractor_pos - tractor_goal)
        r_pos = 1.0 - (jnp.clip(d_pos, 0., self.terminal_reward_threshold) / self.terminal_reward_threshold) ** 2
        
        return r_pos

    @property
    def observation_size(self):
        return 3  # [px, py, theta1]

    def generate_demonstration_trajectory(self, search_direction="horizontal", num_waypoints_max=4, motion_preference=0):
        """Generate demonstration trajectory for bicycle model (3D states)."""
        import numpy as np
        
        logging.debug(f"Generating bicycle demo trajectory with motion_preference={motion_preference}")
        
        # Use parent method but convert to 3D states
        # Temporarily set a 4D goal for parent method
        original_xg = self.xg
        temp_xg = jnp.concatenate([self.xg, jnp.array([self.xg[2]])])  # theta2 = theta1
        self.xg = temp_xg
        
        # Call parent method
        xref_4d = super().generate_demonstration_trajectory(search_direction, num_waypoints_max, motion_preference)
        
        # Convert to 3D and restore original goal
        self.xref = xref_4d[:, :3]  # Keep only [px, py, theta1]
        self.xg = original_xg
        
        # Update angle mask for 3D trajectory
        if hasattr(self, 'angle_mask'):
            # Angle mask remains the same since it's per time step
            pass
        
        logging.debug(f"Bicycle demo trajectory updated: start=({self.xref[0,0]:.2f},{self.xref[0,1]:.2f}), end=({self.xref[-1,0]:.2f},{self.xref[-1,1]:.2f})")
        
        return self.xref

    def eval_xref_logpd(self, xs, motion_preference=None):
        """Evaluate log probability density with respect to reference trajectory (3D states)."""
        if self.xref is None or not hasattr(self, 'xref'):
            return 0.0
        
        if motion_preference is None:
            motion_preference = self.motion_preference
        
        @jax.jit
        def _eval_xref_logpd(xs, xref, angle_mask, ref_threshold, phi_max, motion_preference, ref_pos_weight, ref_theta1_weight):
            # Position error - use tractor position
            xs_tractor = xs[:, :2]
            ref_pos = xref[:, :2]
            
            xs_pos_err = xs_tractor - ref_pos
            pos_err_norm = jnp.linalg.norm(xs_pos_err, axis=-1)
            pos_logpd = -(jnp.clip(pos_err_norm, 0.0, ref_threshold) / ref_threshold) ** 2
            
            # Angle error handling
            wrap_pi = lambda a: (a + jnp.pi) % (2.*jnp.pi) - jnp.pi
            
            # For has preference (direct difference)
            theta1_err_direct = wrap_pi(xs[:, 2] - xref[:, 2])
            
            # Try offset by π (opposite orientation)
            theta1_err_offset = wrap_pi(xs[:, 2] - xref[:, 2] - jnp.pi)
            
            # Choose orientation with smaller error
            use_direct = jnp.abs(theta1_err_direct) < jnp.abs(theta1_err_offset)
            theta1_err_wrapped = jnp.where(use_direct, theta1_err_direct, theta1_err_offset)
            
            # Select final angle
            no_preference = motion_preference == 0
            theta1_err = jnp.where(no_preference, theta1_err_wrapped, theta1_err_direct)
            
            # Apply angle mask
            theta1_err = jnp.where(angle_mask, theta1_err, 0.0)
            
            theta1_logpd = -(jnp.clip(jnp.abs(theta1_err), 0.0, phi_max) / phi_max) ** 2
            
            # Combined logpd (no theta2 component)
            combined_logpd = (ref_pos_weight * pos_logpd + 
                            ref_theta1_weight * theta1_logpd)
            
            return combined_logpd.mean(axis=-1)
        
        # Check if angle mask is available
        if hasattr(self, 'angle_mask'):
            angle_mask = self.angle_mask
        else:
            angle_mask = jnp.ones(self.xref.shape[0], dtype=bool)
        
        return _eval_xref_logpd(xs, self.xref, angle_mask, self.ref_reward_threshold, self.phi_max, motion_preference, 
                               self.ref_pos_weight, self.ref_theta1_weight)

    def render(self, ax, xs: jnp.ndarray):
        """Render the bicycle system (tractor only, no trailer)"""
        # Use parent render method but xs should be 3D
        # Add parking space boundaries for parking scenario
        if hasattr(self.env, 'case') and self.env.case == "parking" and hasattr(self.env, 'parking_config'):
            config = self.env.parking_config
            rows = config['parking_rows']
            cols = config['parking_cols']
            space_width = config['space_width']
            space_length = config['space_length']
            y_offset = config['parking_y_offset']
            
            # Calculate parking lot position
            parking_lot_width = cols * space_width
            parking_lot_height = rows * space_length
            parking_start_x = -parking_lot_width / 2
            parking_start_y = self.env.y_range[0] + y_offset
            
            # Draw parking space boundaries
            for row in range(rows + 1):
                y = parking_start_y + row * space_length
                ax.plot([parking_start_x, parking_start_x + parking_lot_width], [y, y], 'k-', alpha=0.3, linewidth=1)
            
            for col in range(cols + 1):
                x = parking_start_x + col * space_width
                ax.plot([x, x], [parking_start_y, parking_start_y + parking_lot_height], 'k-', alpha=0.3, linewidth=1)
            
            # Add parking space numbers
            for row in range(rows):
                for col in range(cols):
                    space_num = row * cols + col + 1
                    space_center_x = parking_start_x + (col + 0.5) * space_width
                    space_center_y = parking_start_y + (row + 0.5) * space_length
                    
                    if space_num in config['target_spaces']:
                        color = 'lightgreen'
                        text_color = 'black'
                    elif space_num in config['occupied_spaces']:
                        color = 'lightcoral'
                        text_color = 'white'
                    else:
                        color = 'lightblue'
                        text_color = 'black'
                    
                    if space_num not in config['occupied_spaces']:
                        ax.text(space_center_x, space_center_y, str(space_num), 
                               ha='center', va='center', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                               color=text_color)
        
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
                
                rect = plt.Rectangle((-width/2, -height/2), width, height,
                                   linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                
                transform = (Affine2D()
                           .rotate(angle)
                           .translate(x_center, y_center) + ax.transData)
                rect.set_transform(transform)
                ax.add_patch(rect)
        
        # Plot trajectory (only tractor path)
        if xs.shape[0] > 0:
            ax.scatter(xs[:, 0], xs[:, 1], c=range(len(xs)), cmap="Reds", s=45)
            ax.plot(xs[:, 0], xs[:, 1], "r-", linewidth=1.5, label="Bicycle path")
            
        # Get plot limits from environment
        x_range, y_range = self.env.get_plot_limits()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect("equal")

    def setup_animation_patches(self, ax, tractor_color='blue'):
        """Initialize patches for bicycle visualization (tractor only, no trailer)"""
        
        # Tractor body (rectangle)
        tractor_full_length = self.l1 + self.lf1 + self.lr
        self.tractor_body = ax.add_patch(
            plt.Rectangle((-tractor_full_length/2, -self.tractor_width/2),
                         tractor_full_length, self.tractor_width,
                         linewidth=2, edgecolor='black', facecolor=tractor_color, alpha=0.7)
        )
        
        # Wheels (front and rear)
        wheel_width = 0.45
        wheel_length = 0.8
        
        self.tractor_wheels = []
        for i in range(2):  # front and rear wheels
            wheel = ax.add_patch(
                plt.Rectangle((-wheel_length/2, -wheel_width/2),
                             wheel_length, wheel_width,
                             edgecolor='black', facecolor='gray', alpha=0.9)
            )
            self.tractor_wheels.append(wheel)
        
        # No trailer or hitch lines for bicycle model

    def get_bicycle_positions(self, x):
        """Calculate bicycle positions and orientations"""
        px, py, theta1 = x[:3]
        
        # Tractor position (rear axle center)
        tractor_rear_x = px
        tractor_rear_y = py
        
        # Tractor front axle
        tractor_front_x = px + self.l1 * np.cos(theta1)
        tractor_front_y = py + self.l1 * np.sin(theta1)
        
        return {
            'tractor_rear': (tractor_rear_x, tractor_rear_y),
            'tractor_front': (tractor_front_x, tractor_front_y),
            'theta1': theta1
        }

    def render_rigid_body(self, x, u=None):
        """Return transforms to render the bicycle system (tractor only)"""
        px, py, theta1 = x[:3]
        
        # Get positions
        positions = self.get_bicycle_positions(x)
        
        # Tractor body transform
        tractor_center_x = px + 0.5 * (self.l1 + self.lf1 - self.lr) * np.cos(theta1)
        tractor_center_y = py + 0.5 * (self.l1 + self.lf1 - self.lr) * np.sin(theta1)

        transform_tractor_body = (Affine2D()
                                 .rotate(theta1)
                                 .translate(tractor_center_x, tractor_center_y) + plt.gca().transData)
        
        # Wheel transforms
        transforms_tractor_wheels = []
        # Rear wheel (no steering)
        transforms_tractor_wheels.append(
            Affine2D().rotate(theta1).translate(*positions['tractor_rear']) + plt.gca().transData
        )
        # Front wheel (with steering angle)
        steering_angle = 0.0
        if u is not None:
            v, delta = u
            steering_angle = delta
        front_wheel_transform = (Affine2D()
                               .rotate(theta1 + steering_angle)
                               .translate(*positions['tractor_front']) + plt.gca().transData)
        transforms_tractor_wheels.append(front_wheel_transform)
        
        return {
            'tractor_body': transform_tractor_body,
            'tractor_wheels': transforms_tractor_wheels
        }

    def update_animation_patches(self, x, u=None):
        """Update patches with new state (bicycle only)"""
        transforms = self.render_rigid_body(x, u)
        
        # Update body transform
        self.tractor_body.set_transform(transforms['tractor_body'])
        
        # Update wheel transforms
        for i, wheel in enumerate(self.tractor_wheels):
            wheel.set_transform(transforms['tractor_wheels'][i])
        
        # No trailer or hitch lines to update

    def _numpy_projection_function(self, args_tuple):
        """
        Pure NumPy projection function for host callback (bicycle model).
        Minimizes ||u - u_original||² subject to safety constraints on resulting state.
        Uses tractor-only geometry (no trailer, no hitch constraint).
        """
        import numpy as np
        from scipy.optimize import minimize
        
        # Unpack arguments
        q_current_np, u_original_normalized_np = args_tuple
        
        # Store environment parameters for use in nested functions
        l1, lf1, lr = self.l1, self.lf1, self.lr
        tractor_width = self.tractor_width
        v_max, delta_max = self.v_max, self.delta_max
        dt = self.dt
        obs_circles = self.obs_circles
        obs_rectangles = self.obs_rectangles
        
        def bicycle_dynamics_np(x, u_scaled):
            """NumPy version of bicycle dynamics"""
            px, py, theta1 = x
            v, delta = u_scaled
            px_dot = v * np.cos(theta1)
            py_dot = v * np.sin(theta1)
            theta1_dot = (v / l1) * np.tan(delta)
            return np.array([px_dot, py_dot, theta1_dot])
        
        def rk4_np(dynamics, x, u, dt):
            """NumPy version of RK4 integration"""
            k1 = dynamics(x, u)
            k2 = dynamics(x + dt / 2 * k1, u)
            k3 = dynamics(x + dt / 2 * k2, u)
            k4 = dynamics(x + dt * k3, u)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        def input_scaler_np(u_normalized):
            """NumPy version of input scaler"""
            v = u_normalized[0] * v_max
            delta = u_normalized[1] * delta_max
            return np.array([v, delta])
        
        def get_tractor_rectangle_np(x):
            """Get tractor oriented box params: center, half_sizes, angle"""
            px, py, theta1 = x[:3]
            tractor_center_x = px + 0.5 * (l1 + lf1 - lr) * np.cos(theta1)
            tractor_center_y = py + 0.5 * (l1 + lf1 - lr) * np.sin(theta1)
            half_len = (l1 + lf1 + lr) / 2.0
            half_wid = tractor_width / 2.0
            center = np.array([tractor_center_x, tractor_center_y])
            half_sizes = np.array([half_len, half_wid])
            angle = theta1
            return center, half_sizes, angle
        
        def _sdf_oriented_box_np(point, center, half_size, angle):
            """
            Signed distance from point to oriented rectangle.
            Positive outside, negative inside.
            """
            # box axes
            u = np.array([np.cos(angle), np.sin(angle)])
            v = np.array([-np.sin(angle), np.cos(angle)])
            
            # vector from box center to query point
            d = np.array([np.dot(point - center, u), np.dot(point - center, v)])
            
            # how far outside along each local axis
            q = np.abs(d) - half_size
            
            # outside term: Euclidean norm of positive parts
            outside = np.linalg.norm(np.maximum(q, 0.0))
            
            # inside term: max of the two inside distances (≤0)
            inside = min(max(q[0], q[1]), 0.0)
            
            return outside + inside
        
        def _sdf_two_boxes_np(c1, h1, a1, c2, h2, a2):
            """Signed distance (gap) between two oriented boxes."""
            # box axes
            u1 = np.array([np.cos(a1), np.sin(a1)])
            v1 = np.array([-np.sin(a1), np.cos(a1)])
            u2 = np.array([np.cos(a2), np.sin(a2)])
            v2 = np.array([-np.sin(a2), np.cos(a2)])
            axes = [u1, v1, u2, v2]
            
            def gap_on_axis(axis):
                # projection centers
                p1 = np.dot(c1, axis)
                p2 = np.dot(c2, axis)
                # projection radii
                r1 = h1[0] * abs(np.dot(u1, axis)) + h1[1] * abs(np.dot(v1, axis))
                r2 = h2[0] * abs(np.dot(u2, axis)) + h2[1] * abs(np.dot(v2, axis))
                # signed gap: positive if disjoint, negative if overlapping
                return max((p2 - r2) - (p1 + r1), (p1 - r1) - (p2 + r2))
            
            # take the worst (largest) gap over all separating axes
            gaps = [gap_on_axis(a) for a in axes]
            return max(gaps)
        
        def _signed_dist_rect_circle_np(x, circle):
            """Signed distance between tractor rectangle and circle."""
            center, half_sizes, angle = get_tractor_rectangle_np(x)
            center_c, r = circle[:2], circle[2]
            d = _sdf_oriented_box_np(center_c, center, half_sizes, angle)
            return d - r
        
        def _signed_dist_rect_rect_np(x, rect_obs):
            """Signed distance between tractor rectangle and rectangular obstacle."""
            ox, oy, w, h, oa = rect_obs
            c_obs = np.array([ox, oy])
            hs_obs = np.array([w/2.0, h/2.0])
            c_t, hs_t, a_t = get_tractor_rectangle_np(x)
            return _sdf_two_boxes_np(c_t, hs_t, a_t, c_obs, hs_obs, oa)
        
        def constraint_function_np(q):
            """Constraint evaluation: returns array with elements >= 0 for feasibility."""
            cs = []
            
            # Circular obstacles
            if len(obs_circles) > 0:
                for i in range(len(obs_circles)):
                    circle = obs_circles[i]
                    d = _signed_dist_rect_circle_np(q, circle)
                    cs.append(float(d))
            
            # Rectangular obstacles
            if len(obs_rectangles) > 0:
                for i in range(len(obs_rectangles)):
                    rect = obs_rectangles[i]
                    d = _signed_dist_rect_rect_np(q, rect)
                    cs.append(float(d))
            
            return np.array(cs) if len(cs) > 0 else np.array([1.0])
        
        def objective(u_normalized):
            """Objective: minimize distance to original control input"""
            return np.sum((u_normalized - u_original_normalized_np)**2)
        
        def constraint_function(u_normalized):
            """Constraint: evaluate feasibility at next state from RK4 step."""
            u_scaled = input_scaler_np(u_normalized)
            q_new = rk4_np(bicycle_dynamics_np, q_current_np, u_scaled, dt)
            constraints = constraint_function_np(q_new)
            return constraints
        
        # Optimization setup
        u0 = u_original_normalized_np
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        constraints = {
            'type': 'ineq',
            'fun': constraint_function
        }
        
        try:
            # Use trust-constr which handles nonlinear constraints well
            result = minimize(
                objective,
                u0,
                method='trust-constr',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 50,
                    'gtol': 1e-4,
                    'verbose': 0
                }
            )
            
            if result.success:
                return result.x
            else:
                return u_original_normalized_np
        except Exception:
            return u_original_normalized_np

    @partial(jax.jit, static_argnums=(0,))
    def _guidance_function(self, x):
        """Guidance function for bicycle model"""
        px, py, theta1 = x[:3]
        
        # Get tractor position only
        tractor_pos = jnp.array([px, py])
        
        total_cost = 0.0
        safety_margin = 2.0
        
        # Penalty for circular obstacles
        if self.obs_circles.shape[0] > 0:
            for i in range(self.obs_circles.shape[0]):
                circle = self.obs_circles[i]
                circle_center = circle[:2]
                circle_radius = circle[2]
                
                # Distance from tractor to circle center
                tractor_dist = jnp.linalg.norm(tractor_pos - circle_center)
                tractor_violation = jnp.maximum(0.0, circle_radius + safety_margin - tractor_dist)
                
                # Add squared penalty
                total_cost += tractor_violation ** 2
        
        # No hitch angle penalty for bicycle model
        
        return total_cost 
    
    @partial(jax.jit, static_argnums=(0, 3))  # Make max_steps static (position 3)
    def apply_guidance(self, q_proposed, step_size=0.05, max_steps=5):
        """
        Apply gradient descent guidance to move the proposed state away from constraint violations.
        
        Args:
            q_proposed: Proposed state that may violate constraints
            step_size: Gradient descent step size
            max_steps: Number of gradient descent steps (default 1 as requested) - must be static
            
        Returns:
            q_guided: State after applying guidance
        """
        if not self.enable_guidance:
            return q_proposed
        
        def guidance_step(x, _):
            # Compute gradient of guidance function
            grad = jax.grad(self._guidance_function)(x)
            
            # Check for NaN or inf in gradient
            grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
            
            # Gradient descent step (move in negative gradient direction to minimize cost)
            x_new = x - step_size * grad
            
            # Apply reasonable bounds to keep state valid
            x_new = x_new.at[0].set(jnp.clip(x_new[0], -100.0, 100.0))  # px bounds
            x_new = x_new.at[1].set(jnp.clip(x_new[1], -100.0, 100.0))  # py bounds
            x_new = x_new.at[2].set(jnp.clip(x_new[2], -2*jnp.pi, 2*jnp.pi))  # theta1
            
            # Safety check: if result has NaN/inf, return previous state
            x_new = jnp.where(jnp.isfinite(x_new), x_new, x)
            
            return x_new, None
        
        # Apply gradient descent steps
        x_final, _ = jax.lax.scan(guidance_step, q_proposed, None, length=max_steps)
        
        return x_final
import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.transforms import Affine2D

import mbd
from ..envs.env import Env

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
    applied_action: jnp.ndarray
    in_backup_mode: jnp.ndarray

    
class TractorTrailer2d:
    """
    State: [px, py, theta1, theta2]
    Action: [v, delta] - velocity and steering angle
    Input constraint: v ∈ [-1, 1], delta ∈ [-55°, 55°]
    """
    def __init__(self, x0=None, xg=None, env_config=None, case="parking", dt=0.2, H=50, motion_preference=0, 
                 collision_penalty=0.15, enable_shielded_rollout_collision=False, hitch_penalty=0.10, 
                 enable_shielded_rollout_hitch=True, enable_projection=False, enable_guidance=False, reward_threshold=25.0, ref_reward_threshold=5.0,
                 max_w_theta=0.75, hitch_angle_weight=0.2, l1=3.23, l2=2.9, lh=1.15, lf1=0.9, lr=1.848, lf2=1.42, lr2=3.225, 
                 tractor_width=2.0, trailer_width=2.5, v_max=3.0, delta_max_deg=55.0,
                 d_thr_factor=1.0, k_switch=2.5, steering_weight=0.05, preference_penalty_weight=0.05,
                 heading_reward_weight=0.5, terminal_reward_threshold=10.0, terminal_reward_weight=1.0, ref_pos_weight=0.3, ref_theta1_weight=0.5, ref_theta2_weight=0.2,
                 num_trailers: int = 1):
        # Time parameters
        self.dt = dt
        self.H = H
        
        # scalability parameter
        self.num_trailers = int(num_trailers)
        
        # motion preference: 0=none, 1=forward, -1=backward
        self.motion_preference = motion_preference
        
        # Collision handling parameters
        self.collision_penalty = collision_penalty
        self.enable_shielded_rollout_collision = enable_shielded_rollout_collision
        self.hitch_penalty = hitch_penalty
        self.enable_shielded_rollout_hitch = enable_shielded_rollout_hitch
        self.enable_projection = enable_projection
        self.enable_guidance = enable_guidance
        
        # Mode flag: True during final visualization (use guided states), False during denoising (kinematic consistency)
        self.visualization_mode = False
        
        # Reward thresholds
        self.reward_threshold = reward_threshold
        self.ref_reward_threshold = ref_reward_threshold
        self.max_w_theta = max_w_theta
        self.hitch_angle_weight = hitch_angle_weight
        
        # Demonstration evaluation weights
        self.ref_pos_weight = ref_pos_weight
        self.ref_theta1_weight = ref_theta1_weight
        self.ref_theta2_weight = ref_theta2_weight
        
        # Tractor-trailer parameters (configurable)
        self.l1 = l1  # tractor wheelbase
        self.l2 = l2   # trailer length
        self.lh = lh  # hitch length
        self.lf1 = lf1
        self.lr = lr
        self.lf2 = lf2
        self.lr2 = lr2
        self.tractor_width = tractor_width
        self.trailer_width = trailer_width
        
        # Input constraints (configurable)
        self.v_max = v_max  # velocity limit
        self.delta_max = delta_max_deg * jnp.pi / 180.0  # steering angle limit in radians
        
        # --- reward-shaping hyper-parameters (configurable) ----------------------
        self.theta_max = jnp.pi / 2          # 90° cap in rad
        self.phi_max = jnp.deg2rad(40.0)     # 40° cap for articulation
        self.d_thr = d_thr_factor * (self.l1 + self.lh + self.l2)  # 'one rig length'
        self.k_switch = k_switch             # [m] slope of logistic switch
        self.steering_weight = steering_weight  # weight for trajectory-level steering cost
        self.preference_penalty_weight = preference_penalty_weight  # penalty weight for motion preference
        self.heading_reward_weight = heading_reward_weight  # weight for heading reward calculation
        self.terminal_reward_threshold = terminal_reward_threshold  # position error threshold for terminal reward
        self.terminal_reward_weight = terminal_reward_weight  # weight for terminal reward at final state
        
        
        
        # Environment setup
        if env_config is None:
            self.env = Env(case=case)
        else:
            self.env = env_config
        
        # Get obstacles from environment
        obstacles = self.env.get_obstacles()
        self.obs_circles = obstacles['circles']
        self.obs_rectangles = obstacles['rectangles']
        
        # Set initial and goal states based on case and user input
        if x0 is None:
            if case == "parking":
                self.x0 = self.env.get_default_init_pos(case=case)
            elif case == "navigation":
                # For navigation, use default but will be overridden by set_init_pos/set_goal_pos calls
                self.x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.x0 = x0
            
        if xg is None:
            if case == "parking":
                self.xg = self.env.get_default_goal_pos(case=case)
            elif case == "navigation":
                # For navigation, use default but will be overridden by set_goal_pos calls
                self.xg = jnp.array([0.0, 0.0, jnp.pi, jnp.pi])
        else:
            self.xg = xg
        
        
        # print(f"x0: {self.x0}")
        # print(f"xg: {self.xg}")
        
        # Demonstration trajectory will be generated later using generate_demonstration_trajectory()
        # This allows setting x0, xg, and obstacles before creating the demo
        self.xref = None
        self.rew_xref = None
        
        # Animation-related attributes
        self.tractor_body = None
        self.trailer_body = None
        self.tractor_wheels = []
        self.trailer_wheels = []
        self.hitch_line = None
        
        # Print environment info
        if hasattr(self.env, 'print_parking_layout'):
            self.env.print_parking_layout()
        
        
        

    def set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0):
        """
        Set initial position for tractor-trailer.
        
        Args:
            x, y: Direct coordinates (for manual positioning)
            dx, dy: Geometric positioning relative to parking lot (for parking scenario)
                dx: Distance from tractor front face to target parking space center (x-direction)
                dy: Distance from tractor to parking lot entrance line (y-direction)
            theta1, theta2: Initial orientations for tractor and trailer
        """
        if hasattr(self.env, 'case') and self.env.case == "parking" and dx is not None and dy is not None:
            # Get parking configuration
            config = self.env.parking_config
            target_spaces = config['target_spaces']
            space_length = config['space_length']
            
            # Get centers of both target parking spaces
            target_centers = []
            for space_num in target_spaces:
                center_x, center_y = self.env.get_parking_space_center(space_num)
                target_centers.append((center_x, center_y))
            
            # Find the target space with the larger y coordinate (higher up)
            target_centers = sorted(target_centers, key=lambda p: p[1])  # Sort by y
            higher_target_x, higher_target_y = target_centers[-1]  # Take the one with larger y
            
            # Calculate parking lot entrance line (above the higher target space)
            entrance_y = higher_target_y + space_length / 2
            
            # Calculate initial tractor position
            # dx: positive distance from target x to tractor front face
            # The front face is at (px + l1 * cos(theta1), py + l1 * sin(theta1))
            # We want: front_face_x = target_x + dx
            # So: px + l1 * cos(theta1) = target_x + dx
            # Therefore: px = target_x + dx - l1 * cos(theta1)
            target_x = target_centers[0][0]  # Use first target (tractor target) for x reference
            px = target_x + dx - self.l1 * np.cos(theta1)
            
            # dy: positive distance from entrance line to tractor
            py = entrance_y + dy
            
            self.x0 = jnp.array([px, py, theta1, theta2])
            
        elif x is not None and y is not None:
            # Direct coordinate specification
            self.x0 = jnp.array([x, y, theta1, theta2])
            logging.debug(f"overwrite x0: {self.x0}")
        else:
            # Default case1 values
            x = x if x is not None else -3.0
            y = y if y is not None else 0.0
            self.x0 = jnp.array([x, y, theta1, theta2])
        
        # Invalidate JIT cache when initial conditions change
        #mbd.planners.mbd_planner.clear_jit_cache()

    def set_goal_pos(self, x=None, y=None, theta1=None, theta2=None):
        """
        Set goal position for tractor-trailer.
        If self.xg exists, only update specified values.
        If self.xg doesn't exist, use defaults for unspecified values.
        
        Default values:
        x = 3.0, y = 0.0, theta1 = pi, theta2 = pi
        """
        if hasattr(self, 'xg'):
            # Update only specified values while keeping others unchanged
            new_x = x if x is not None else self.xg[0]
            new_y = y if y is not None else self.xg[1] 
            new_theta1 = theta1 if theta1 is not None else self.xg[2]
            new_theta2 = theta2 if theta2 is not None else self.xg[3]
        else:
            # Use defaults for unspecified values
            new_x = x if x is not None else 3.0
            new_y = y if y is not None else 0.0
            new_theta1 = theta1 if theta1 is not None else np.pi
            new_theta2 = theta2 if theta2 is not None else np.pi
            
        self.xg = jnp.array([new_x, new_y, new_theta1, new_theta2])
        logging.debug(f"overwrite xg: {self.xg}")
        
        # Invalidate JIT cache when goal conditions change
        #mbd.planners.mbd_planner.clear_jit_cache()

    def set_rectangle_obs(self, rectangles, coordinate_mode="left-top", padding=0.0):
        """Set rectangular obstacles"""
        self.obs_rectangles = self.env.set_rectangle_obs(rectangles, coordinate_mode=coordinate_mode, padding=padding)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array):
        """Resets the environment to an initial state."""
        return State(self.x0, self.x0, 0.0, 0.0, jnp.zeros(2), jnp.array(False))
    
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
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        return self._step_internal(state, action, self.visualization_mode)
    
    @partial(jax.jit, static_argnums=(0, 3))  # visualization_mode is static argument 3
    def _step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State:
        """Internal step function with explicit visualization_mode parameter."""
        action = jnp.clip(action, -1.0, 1.0)
        # Persist backup mode across steps; once entered, stay in backup for remainder
        backup_active_prev = state.in_backup_mode
        u_backup_norm = jnp.array([0.0, 0.0])  # zero velocity and steering as backup for kinematic models
        action_eff_norm = jnp.where(backup_active_prev, u_backup_norm, action)
        
        # Scale inputs from normalized [-1, 1] to actual ranges
        u_scaled = self.input_scaler(action_eff_norm)
        
        q = state.pipeline_state
        
        # Handle projection case separately to avoid vmap-of-cond + io_callback issue
        if self.enable_projection:
            # Use control-based projection: optimize control input, then compute resulting state
            u_safe_normalized = self.project_control_to_safe_set(q, action)  # action is normalized
            u_safe_scaled = self.input_scaler(u_safe_normalized)  # Scale to actual ranges
            q_final = rk4(self.tractor_trailer_dynamics, q, u_safe_scaled, self.dt)
            q_reward = q_final  # For projection, q_reward = q_final
            obstacle_collision = False  # Guaranteed safe by projection
            hitch_violation = False     # Guaranteed safe by projection
            applied_action = u_safe_normalized
            backup_active_next = backup_active_prev
        elif self.enable_guidance:
            # Use guidance: compute proposed state, apply guidance for reward computation
            q_proposed = rk4(self.tractor_trailer_dynamics, q, u_scaled, self.dt)
            q_reward = self.apply_guidance(q_proposed)  # Guided state for reward computation
            
            # During denoising: use q_proposed for kinematic consistency
            # During visualization: use q_reward to show actual guided trajectory
            q_final = q_reward if visualization_mode else q_proposed
            
            # Collision flags are handled by guidance, not through naive penalty
            obstacle_collision = False  # constraints are handled by guidance, not through naive penalty
            hitch_violation = False     # constraints are handled by guidance, not through naive penalty
            applied_action = action_eff_norm
            backup_active_next = backup_active_prev
        else:
            # Use original conditional logic for non-projection/non-guidance methods
            q_proposed = rk4(self.tractor_trailer_dynamics, q, u_scaled, self.dt)
            
            def use_shielded_rollout_fn(args):
                q_prop, = args
                return self._step_with_shielded_rollout((q, q_prop))
            
            def use_naive_penalty_fn(args):
                q_prop, = args
                return self._step_without_shielded_rollout(q_prop)
            
            # Select method based on shielded rollout flags
            use_shielded_rollout = self.enable_shielded_rollout_collision | self.enable_shielded_rollout_hitch
            
            q_final, obstacle_collision, hitch_violation = jax.lax.cond(
                use_shielded_rollout,
                use_shielded_rollout_fn,
                use_naive_penalty_fn,
                (q_proposed,)
            )
            q_reward = q_final  # For other methods, q_reward = q_final

            # Determine if fallback should start at this step
            should_fallback = (self.enable_shielded_rollout_collision & obstacle_collision) | \
                              (self.enable_shielded_rollout_hitch & hitch_violation)
            backup_active_next = backup_active_prev | should_fallback
            applied_action = jnp.where(backup_active_next, u_backup_norm, action)
        
        # Compute reward on q_reward (the state we want to evaluate)
        reward = self.get_reward(q_reward)
        
        # Apply penalties using flags (only if violations were checked)
        reward = jnp.where(obstacle_collision, reward - self.collision_penalty, reward)
        reward = jnp.where(hitch_violation, reward - self.hitch_penalty, reward)
        
        # Add preference penalty based on motion direction (use actually applied control)
        u_applied_scaled = self.input_scaler(applied_action)
        preference_penalty = self.get_preference_penalty(u_applied_scaled)
        has_preference = self.motion_preference != 0
        reward = jnp.where(has_preference, reward - preference_penalty, reward)
        
        # Return q_final as the next state (ensures kinematic consistency)\
        # NOTE: since apply fallback for all time steps drop the sample efficiency too much, we allow shielded 
        return state.replace(pipeline_state=q_final, obs=q_final, reward=reward, done=0.0,
                              applied_action=applied_action, in_backup_mode=backup_active_prev)

    @partial(jax.jit, static_argnums=(0,))
    def _step_with_shielded_rollout(self, data):
        """Step function with shielded rollout enabled"""
        q, q_proposed = data
        return self.shielded_rollout(q, q_proposed)
    
    @partial(jax.jit, static_argnums=(0,))
    def _step_without_shielded_rollout(self, q_proposed):
        """Step function with simple forward dynamics (no shielded rollout)"""
        # Check collision and hitch violations on q_proposed (for penalty computation)
        obstacle_collision = self.check_obstacle_collision(q_proposed, self.obs_circles, self.obs_rectangles)
        hitch_violation = self.check_hitch_violation(q_proposed)
        
        return q_proposed, obstacle_collision, hitch_violation

    @partial(jax.jit, static_argnums=(0,))
    def shielded_rollout(self, q, q_proposed):
        """
        Shielded rollout for kinematic dynamics.
        This extracts the existing safety logic into a reusable function.
        Returns both final state and collision/violation flags to avoid recomputation.
        """
        # Check obstacle collision and hitch angle violation separately
        obstacle_collision = self.check_obstacle_collision(q_proposed, self.obs_circles, self.obs_rectangles)
        hitch_violation = self.check_hitch_violation(q_proposed)
        
        q_shielded = jnp.where(self.enable_shielded_rollout_collision & obstacle_collision, q, q_proposed)
        q_final = jnp.where(self.enable_shielded_rollout_hitch & hitch_violation, q, q_shielded)
        
        return q_final, obstacle_collision, hitch_violation

    @partial(jax.jit, static_argnums=(0,))
    def get_preference_penalty(self, x_or_u, u=None):
        """
        Calculate preference penalty based on movement direction.
        
        Args:
            x_or_u: For kinematic model, this is u (scaled input [v, delta]).
                   For acceleration model, this is x (state) and u is provided separately.
            u: Optional scaled control input, used when x_or_u is state x.
            
        Returns:
            penalty: small penalty for non-preferred movement direction
        """
        # Determine if this is kinematic (u only) or acceleration dynamics (x and u)
        if u is None:
            # Kinematic model: x_or_u is actually u (scaled input)
            v, delta = x_or_u
        else:
            # Acceleration model: x_or_u is state x, extract velocity from state
            # For kinematic model fallback, extract from control input
            if len(x_or_u) >= 5:  # 6D state (acceleration model)
                v = x_or_u[4]  # Extract velocity from state
            else:  # 4D state, use control input
                v, delta = u
        
        penalty_weight = self.preference_penalty_weight  # Configurable penalty weight for stronger preference enforcement
        
        # Forward preference penalty (penalize backward movement)
        forward_penalty = jnp.where(v < 0, penalty_weight * jnp.abs(v), 0.0)
        
        # Backward preference penalty (penalize forward movement)
        backward_penalty = jnp.where(v > 0, penalty_weight * jnp.abs(v), 0.0)
        
        # No preference penalty
        no_penalty = 0.0
        
        # Select penalty based on preference using JAX conditionals
        is_forward_pref = self.motion_preference == 1
        is_backward_pref = self.motion_preference == -1
        is_forward_enforce = self.motion_preference == 2
        is_backward_enforce = self.motion_preference == -2
        
        # For strict enforcement (2, -2), no penalty is needed since motion is already constrained
        penalty = jnp.where(is_forward_pref, forward_penalty,
                           jnp.where(is_backward_pref, backward_penalty,
                           jnp.where(is_forward_enforce | is_backward_enforce, no_penalty, no_penalty)))
            
        return penalty

    def generate_demonstration_trajectory(self, search_direction="horizontal", num_waypoints_max=4, motion_preference=0):
        """
        Generate a simple demonstration trajectory from x0 to xg avoiding obstacles.
        
        Args:
            search_direction: "horizontal" or "vertical" - initial search direction
            num_waypoints_max: maximum number of waypoints to use
            
        Returns:
            xref: (H+1, 4) array of reference states [px, py, theta1, theta2]
        """
        import numpy as np
        
        logging.debug(f"Generating NEW demonstration trajectory with motion_preference={motion_preference}")
        
        # Extract start and goal positions
        x0_pos = self.x0[:2]
        xg_pos = self.xg[:2]
        logging.debug(f"Demo trajectory: x0_pos={x0_pos}, xg_pos={xg_pos}")
        
        # List to store waypoints
        waypoints = [x0_pos]
        
        # Helper function to check if a line segment collides with obstacles
        def check_line_collision(p1, p2, num_samples=50):
            """Check if line from p1 to p2 collides with any obstacle (point-based check)"""
            # Sample points along the line
            t_values = np.linspace(0, 1, num_samples)
            for t in t_values:
                point = p1 + t * (p2 - p1)
                
                # Check if point collides with any circular obstacles
                for obs in self.obs_circles:
                    obs_center = obs[:2]
                    obs_radius = obs[2]
                    dist = np.linalg.norm(point - obs_center)
                    if dist <= obs_radius:
                        #print(f"Point collision detected with circle at {point}")
                        return True
                
                # Check if point collides with any rectangular obstacles
                for obs in self.obs_rectangles:
                    obs_x, obs_y, obs_width, obs_height, obs_angle = obs
                    
                    # Transform point to obstacle's local coordinate system
                    dx = point[0] - obs_x
                    dy = point[1] - obs_y
                    
                    # Rotate point by -obs_angle to align with obstacle's axes
                    cos_angle = np.cos(-obs_angle)
                    sin_angle = np.sin(-obs_angle)
                    local_x = dx * cos_angle - dy * sin_angle
                    local_y = dx * sin_angle + dy * cos_angle
                    
                    # Check if point is inside rectangle bounds
                    if (abs(local_x) <= obs_width / 2 and abs(local_y) <= obs_height / 2):
                        #print(f"Point collision detected with rectangle at {point}")
                        return True
            
            return False
        
        # Try direct path first
        if not check_line_collision(x0_pos, xg_pos):
            # Direct path is clear
            #print("Direct path is clear")
            waypoints.append(xg_pos)
        else:
            # Need intermediate waypoints - use perpendicular approach
            if search_direction == "horizontal":
                # Go horizontally first (change x, keep y), then vertically (change y, keep x)
                intermediate = jnp.array([xg_pos[0], x0_pos[1]])
            else:
                # Go vertically first (change y, keep x), then horizontally (change x, keep y)
                intermediate = jnp.array([x0_pos[0], xg_pos[1]])
            
            # Check if this perpendicular path is collision-free
            if not check_line_collision(x0_pos, intermediate) and \
               not check_line_collision(intermediate, xg_pos):
                waypoints.append(intermediate)
                waypoints.append(xg_pos)
                #print("Perpendicular path is clear")
            else:
                # Try the other direction
                if search_direction == "horizontal":
                    intermediate = jnp.array([x0_pos[0], xg_pos[1]])
                else:
                    intermediate = jnp.array([xg_pos[0], x0_pos[1]])
                
                if not check_line_collision(x0_pos, intermediate) and \
                   not check_line_collision(intermediate, xg_pos):
                    waypoints.append(intermediate)
                    waypoints.append(xg_pos)
                else:
                    # If both perpendicular paths fail, just go direct (will have collision)
                    waypoints.append(xg_pos)
        
        # Limit number of waypoints
        if len(waypoints) > num_waypoints_max:
            # Keep start, goal, and evenly spaced intermediate points
            indices = np.linspace(0, len(waypoints)-1, num_waypoints_max, dtype=int)
            waypoints = [waypoints[i] for i in indices]
        
        # Convert waypoints to numpy array for easier manipulation
        waypoints = np.array(waypoints)
        #print(f"waypoints: {waypoints}")
        
        # Calculate total path length
        path_lengths = []
        total_length = 0.0
        for i in range(len(waypoints)-1):
            segment_length = np.linalg.norm(waypoints[i+1] - waypoints[i])
            path_lengths.append(segment_length)
            total_length += segment_length
        
        # Discretize path into H+1 points
        num_points = self.H 
        points_2d = []
        segment_indices = []  # Track which segment each point belongs to
        
        # Calculate how many points per segment based on length
        if total_length > 0:
            points_per_segment = []
            for length in path_lengths:
                n_points = max(2, int(np.round(num_points * length / total_length)))
                points_per_segment.append(n_points)
            
            # Adjust to ensure we have exactly H+1 points
            current_total = sum(points_per_segment)
            if current_total != num_points:
                # Adjust the longest segment
                max_idx = np.argmax(points_per_segment)
                points_per_segment[max_idx] += num_points - current_total
            
            # Generate points along each segment
            for i in range(len(waypoints)-1):
                n_pts = points_per_segment[i]
                # Don't include the last point of each segment (except the final one)
                # to avoid duplicates
                if i < len(waypoints)-2:
                    t_values = np.linspace(0, 1, n_pts, endpoint=False)
                else:
                    t_values = np.linspace(0, 1, n_pts, endpoint=True)
                
                for t in t_values:
                    point = waypoints[i] + t * (waypoints[i+1] - waypoints[i])
                    points_2d.append(point)
                    segment_indices.append(i)  # Track which segment this point belongs to
        else:
            # All waypoints are the same, just repeat
            points_2d = [waypoints[0] for _ in range(num_points)]
            segment_indices = [0] * num_points
        
        # Ensure we have exactly H+1 points
        points_2d = np.array(points_2d)
        segment_indices = np.array(segment_indices)
        
        if len(points_2d) > num_points:
            # Downsample
            indices = np.linspace(0, len(points_2d)-1, num_points, dtype=int)
            points_2d = points_2d[indices]
            segment_indices = segment_indices[indices]
        elif len(points_2d) < num_points:
            # Pad with goal position
            padding = num_points - len(points_2d)
            points_2d = np.vstack([points_2d, np.tile(xg_pos, (padding, 1))])
            # Pad segment indices with the last segment index (final segment)
            last_segment_idx = len(waypoints) - 2 if len(waypoints) > 1 else 0
            segment_indices = np.concatenate([segment_indices, np.full(padding, last_segment_idx)])
        
        # Calculate heading angles based on path direction
        xref = np.zeros((num_points, 4))
        xref[:, :2] = points_2d  # px, py
        
        # Calculate heading angles from consecutive points
        # Only apply angle constraints on the final segment (last waypoint to goal)
        
        # Find which segment is the final one (last waypoint to goal)
        final_segment_idx = len(waypoints) - 2 if len(waypoints) > 1 else 0
        logging.debug(f"Final segment index: {final_segment_idx} out of {len(waypoints)-1} segments")
        
        # Create angle constraint mask: True only for points in the final segment
        angle_mask = segment_indices == final_segment_idx
        logging.debug(f"Angle constraints will be applied to {np.sum(angle_mask)} out of {len(angle_mask)} points")
        
        for i in range(num_points-1):
            dx = points_2d[i+1, 0] - points_2d[i, 0]
            dy = points_2d[i+1, 1] - points_2d[i, 1]
            if np.abs(dx) > 1e-6 or np.abs(dy) > 1e-6:
                theta = np.arctan2(dy, dx)
                
                # Adjust angle based on motion preference
                if motion_preference in [-1, -2]:  # Backward preference
                    theta = theta + np.pi  # Add π for backward orientation
                # For forward preference (1, 2) or no preference (0), use direct angle
                
            else:
                # No movement, keep previous angle or use goal angle
                theta = xref[i-1, 2] if i > 0 else self.xg[2]
            
            # Only set angles for final segment, use 0.0 for earlier segments
            if angle_mask[i]:
                xref[i, 2] = theta  # theta1
                xref[i, 3] = theta  # theta2 (assume aligned)
            else:
                xref[i, 2] = 0.0  # Will be ignored in eval_xref_logpd
                xref[i, 3] = 0.0  # Will be ignored in eval_xref_logpd
        
        # Last point uses goal angles (as manually set)
        xref[-1, 2] = self.xg[2]
        xref[-1, 3] = self.xg[3]
        
        # Convert to jax array
        self.xref = jnp.array(xref)
        self.angle_mask = jnp.array(angle_mask)  # Store the angle constraint mask
        
        logging.debug(f"Demo trajectory UPDATED: start=({self.xref[0,0]:.2f},{self.xref[0,1]:.2f}), end=({self.xref[-1,0]:.2f},{self.xref[-1,1]:.2f}), motion_pref={motion_preference}")
        
        return self.xref
    
    def compute_demonstration_reward(self):
        """
        Compile the reward function after setting x0, xg, and obstacles.
        This should be called after initialization and any updates to goal/start positions.
        """
        if hasattr(self, 'xref') and self.xref is not None:
            # Compute reference trajectory reward
            #jax.debug.print("xref: {xref}", xref=self.xref)
            stage_rewards = jax.vmap(self.get_reward)(self.xref)
            
            # Compute mean of stage rewards first
            stage_reward_mean = stage_rewards.mean()
            
            # Add separate terminal reward
            terminal_reward = self.get_terminal_reward(self.xref[-1])
            self.rew_xref = stage_reward_mean + self.terminal_reward_weight * terminal_reward
            #logging.debug(f"Reference trajectory reward compiled: {self.rew_xref:.3f}")
        else:
            logging.warning("Warning: No reference trajectory set. Call generate_demonstration_trajectory first.")

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        """
        Stage-weighted reward:
          – position early,
          – heading + articulation near goal,
          – optional steering-effort term handled at trajectory level.
        """
        # ---------------------------------------------------------------
        # 0. convenience shorthands
        px, py, theta1, theta2 = q
        pgx, pgy, thetag      = self.xg[:3]        # goal pos, goal heading
        # modify the tractor_pos to be the center of the tractor
        tractor_pos           = jnp.array([px + self.l1 * jnp.cos(theta1), py + self.l1 * jnp.sin(theta1)])
        tractor_goal          = self.xg[:2]
        
        # Compute trailer position from current state
        trailer_pos = self.get_trailer_position(q)
        
        # Compute trailer goal position from goal state
        trailer_goal = self.get_trailer_position(self.xg[:4])
        
        trailer_dist = jnp.linalg.norm(trailer_pos - trailer_goal)
        trailer_reward = (
            1.0 - (jnp.clip(trailer_dist, 0.0, self.reward_threshold) / self.reward_threshold) ** 2
        ) # TODO: currently not using

        # ---------------------------------------------------------------
        # 1. positional reward - use different position based on preference
        # Forward preference/enforcement: track tractor position
        # Backward/None preference: track trailer position
        use_tractor_tracking = (self.motion_preference == 1) | (self.motion_preference == 2)
        
        # Compute distances for both cases
        d_pos_tractor = jnp.linalg.norm(tractor_pos - tractor_goal)
        d_pos_trailer = jnp.linalg.norm(trailer_pos - trailer_goal)
        
        # Select distance based on preference
        d_pos = jnp.where(use_tractor_tracking, d_pos_tractor, d_pos_trailer)
        r_pos = 1.0 - (jnp.clip(d_pos, 0., self.reward_threshold)
                       / self.reward_threshold) ** 2

        # ---------------------------------------------------------------
        # 2. heading-to-goal alignment  r_hdg
        wrap_pi = lambda a: (a + jnp.pi) % (2.*jnp.pi) - jnp.pi
        
        # Handle angle errors based on motion preference:
        # - No preference (0): Use angle wrapping to find best orientation (direct vs π-offset)
        # - Has preference/enforcement (±1, ±2): Goal angles are set correctly, use direct difference
        
        # Compute direct angle differences
        e_theta1_direct = wrap_pi(theta1 - thetag)
        e_theta2_direct = wrap_pi(theta2 - thetag)
        
        # For no preference case, also try opposite orientation
        e_theta1_offset = wrap_pi(theta1 - thetag - jnp.pi)
        e_theta2_offset = wrap_pi(theta2 - thetag - jnp.pi)
        total_err_direct = jnp.abs(e_theta1_direct) + jnp.abs(e_theta2_direct)
        total_err_offset = jnp.abs(e_theta1_offset) + jnp.abs(e_theta2_offset)
        
        # Choose orientation with smaller total error (for no preference case)
        use_direct = total_err_direct < total_err_offset
        e_theta1_wrapped = jnp.where(use_direct, e_theta1_direct, e_theta1_offset)
        e_theta2_wrapped = jnp.where(use_direct, e_theta2_direct, e_theta2_offset)
        
        # Select final angles: use wrapping only when no preference, direct otherwise
        no_preference = self.motion_preference == 0
        e_theta1 = jnp.where(no_preference, e_theta1_wrapped, e_theta1_direct)
        e_theta2 = jnp.where(no_preference, e_theta2_wrapped, e_theta2_direct)

        r_hdg = self.heading_reward_weight * ( 1.0 - (jnp.abs(e_theta1) / self.theta_max) ** 2
                      + 1.0 - (jnp.abs(e_theta2) / self.theta_max) ** 2 )

        # ---------------------------------------------------------------
        # 3. articulation regulariser  r_art
        e_phi  = wrap_pi(theta1 - theta2)
        r_art  = 1.0 - (jnp.abs(e_phi) / self.phi_max) ** 2

        # ---------------------------------------------------------------
        # 4. logistic distance switch  w_theta(d)
        sigmoid  = lambda z: 1.0 / (1.0 + jnp.exp(-z))
        w_theta = sigmoid((self.d_thr - d_pos) / self.k_switch)      # ∈ (0,1)
        # set upper bound on w_theta
        w_theta = jnp.clip(w_theta, 0.0, self.max_w_theta) # prevent dropping off all the position reward

        # ---------------------------------------------------------------
        # 5. weighted stage cost
        reward = (1.0 - self.hitch_angle_weight)*((1.0 - w_theta) * r_pos + w_theta * r_hdg) + self.hitch_angle_weight * r_art
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def get_terminal_reward(self, q):
        """
        Terminal reward computed at final state without logistic switch.
        Equal weighting (0.5) for position and heading rewards.
        """
        # ---------------------------------------------------------------
        # 0. convenience shorthands
        px, py, theta1, theta2 = q
        thetag = self.xg[2]  # goal heading
        tractor_pos = jnp.array([px + self.l1 * jnp.cos(theta1), py + self.l1 * jnp.sin(theta1)])
        tractor_goal = self.xg[:2]
        
        # Compute trailer position from current state
        trailer_pos = self.get_trailer_position(q)
        
        # Compute trailer goal position from goal state
        trailer_goal = self.get_trailer_position(self.xg[:4])
        
        # ---------------------------------------------------------------
        # 1. positional reward - use different position based on preference
        use_tractor_tracking = (self.motion_preference == 1) | (self.motion_preference == 2)
        
        # Compute distances for both cases
        d_pos_tractor = jnp.linalg.norm(tractor_pos - tractor_goal)
        d_pos_trailer = jnp.linalg.norm(trailer_pos - trailer_goal)
        
        # Select distance based on preference
        d_pos = jnp.where(use_tractor_tracking, d_pos_tractor, d_pos_trailer)
        r_pos = 1.0 - (jnp.clip(d_pos, 0., self.terminal_reward_threshold) / self.terminal_reward_threshold) ** 2
        
        # # ---------------------------------------------------------------
        # # 2. heading-to-goal alignment
        # wrap_pi = lambda a: (a + jnp.pi) % (2.*jnp.pi) - jnp.pi
        
        # # Handle angle errors based on motion preference
        # e_theta1_direct = wrap_pi(theta1 - thetag)
        # e_theta2_direct = wrap_pi(theta2 - thetag)
        # total_err_direct = jnp.abs(e_theta1_direct) + jnp.abs(e_theta2_direct)
        
        # # Try offset by π (opposite orientation) 
        # e_theta1_offset = wrap_pi(theta1 - thetag - jnp.pi)
        # e_theta2_offset = wrap_pi(theta2 - thetag - jnp.pi)
        # total_err_offset = jnp.abs(e_theta1_offset) + jnp.abs(e_theta2_offset)
        
        # # Choose orientation with smaller total error (for no preference case)
        # use_direct = total_err_direct < total_err_offset
        # e_theta1_wrapped = jnp.where(use_direct, e_theta1_direct, e_theta1_offset)
        # e_theta2_wrapped = jnp.where(use_direct, e_theta2_direct, e_theta2_offset)
        
        # r_hdg = self.heading_reward_weight * (1.0 - (jnp.abs(e_theta1) / self.theta_max) ** 2
        #               + 1.0 - (jnp.abs(e_theta2) / self.theta_max) ** 2)
        
        # ---------------------------------------------------------------
        # 3. terminal reward, currently only using position reward
        terminal_reward = r_pos
        return terminal_reward

    
    @partial(jax.jit, static_argnums=(0,))
    def get_trailer_position(self, x):
        """Get the trailer center position of the wheel axis. Used for cost function"""
        px, py, theta1, theta2 = x[:4]
        
        # Hitch point (at rear of tractor)
        hitch_x = px - self.lh * jnp.cos(theta1)
        hitch_y = py - self.lh * jnp.sin(theta1)
        
        # Trailer center position of the wheel axis
        trailer_frame_x = hitch_x - self.l2 * jnp.cos(theta2)
        trailer_frame_y = hitch_y - self.l2 * jnp.sin(theta2)
        
        return jnp.array([trailer_frame_x, trailer_frame_y])
    
    @partial(jax.jit, static_argnums=(0,))
    def get_trailer_back_position(self, x):
        """Get the trailer back position of the wheel axis. Used for cost function"""
        px, py, theta1, theta2 = x[:4]
        
        # Hitch point (at rear of tractor)
        hitch_x = px - self.lh * jnp.cos(theta1)
        hitch_y = py - self.lh * jnp.sin(theta1)
        
        # Trailer back position of the wheel axis
        trailer_back_x = hitch_x - (self.l2 + self.lr2) * jnp.cos(theta2)
        trailer_back_y = hitch_y - (self.l2 + self.lr2) * jnp.sin(theta2)
        
        return jnp.array([trailer_back_x, trailer_back_y])
    
    @partial(jax.jit, static_argnums=(0,))
    def get_tractor_trailer_rectangles(self, x):
        """Get the corner points of tractor and trailer rectangles for collision checking"""
        px, py, theta1, theta2 = x[:4]
        
        # Tractor rectangle (centered at tractor center)
        tractor_center_x = px + 0.5 * (self.l1 + self.lf1 - self.lr) * jnp.cos(theta1)
        tractor_center_y = py + 0.5 * (self.l1 + self.lf1 - self.lr) * jnp.sin(theta1)

        # Tractor corners in local coordinates (before rotation)
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
        
        # Trailer rectangle
        hitch_x = px - self.lh * jnp.cos(theta1)
        hitch_y = py - self.lh * jnp.sin(theta1)
        trailer_center_x = hitch_x + (0.5 * (self.lf2 - self.lr2) - self.l2) * jnp.cos(theta2)
        trailer_center_y = hitch_y + (0.5 * (self.lf2 - self.lr2) - self.l2) * jnp.sin(theta2)

        # Trailer corners in local coordinates
        trailer_half_length = (self.lf2 + self.lr2) / 2
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
    def check_hitch_violation(self, x):
        """Check hitch angle violation (jack-knifing prevention)"""
        px, py, theta1, theta2 = x
        wrap_pi = lambda a: (a + jnp.pi) % (2.*jnp.pi) - jnp.pi
        hitch_angle = wrap_pi(theta1 - theta2)
        return jnp.abs(hitch_angle) > self.phi_max

    @partial(jax.jit, static_argnums=(0,))
    def check_obstacle_collision(self, x, obs_circles, obs_rectangles):
        """Check collision with full tractor-trailer geometry against all obstacles"""
        # Get tractor and trailer geometry
        geometry = self.get_tractor_trailer_rectangles(x)
        tractor_corners = geometry['tractor_corners']
        trailer_corners = geometry['trailer_corners']
        
        collision = False  # Start with no collision
        
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
    def check_collision(self, x, obs_circles, obs_rectangles):
        """Check collision with full tractor-trailer geometry against all obstacles and hitch angle violations (legacy function)"""
        obstacle_collision = self.check_obstacle_collision(x, obs_circles, obs_rectangles)
        hitch_violation = self.check_hitch_violation(x)
        return obstacle_collision | hitch_violation

    def eval_xref_logpd(self, xs, motion_preference=None):
        """Evaluate log probability density with respect to reference trajectory"""
        # Check if reference trajectory is set
        if self.xref is None or not hasattr(self, 'xref'):
            return 0.0
        
        # Use provided preference or default to instance preference
        if motion_preference is None:
            motion_preference = self.motion_preference
        
        # JIT-compile the actual computation
        @jax.jit
        def _eval_xref_logpd(xs, xref, angle_mask, ref_threshold, phi_max, motion_preference, ref_pos_weight, ref_theta1_weight, ref_theta2_weight):
            # Position error - use different reference points based on preference
            # Use JAX conditional operations instead of Python if statements
            
            # For tractor position (forward preference)
            xs_tractor = xs[:, :2]
            ref_pos = xref[:, :2]
            
            # For trailer position (backward/none preference)
            xs_trailer = jax.vmap(self.get_trailer_position)(xs)
            
            # Select position based on preference using jnp.where
            use_tractor = (motion_preference == 1) | (motion_preference == 2)
            xs_pos = jnp.where(use_tractor, xs_tractor, xs_trailer)
            
            xs_pos_err = xs_pos - ref_pos
            pos_err_norm = jnp.linalg.norm(xs_pos_err, axis=-1)
            pos_logpd = -(jnp.clip(pos_err_norm, 0.0, ref_threshold) / ref_threshold) ** 2
            
            # Angle error handling based on preference:
            # - No preference (0): Use angle wrapping to find best orientation (direct vs π-offset)
            # - Has preference/enforcement (±1, ±2): Goal angles are manually set correctly, use direct difference
            wrap_pi = lambda a: (a + jnp.pi) % (2.*jnp.pi) - jnp.pi
            
            # For has preference (direct difference)
            theta1_err_direct = wrap_pi(xs[:, 2] - xref[:, 2])
            theta2_err_direct = wrap_pi(xs[:, 3] - xref[:, 3])
            total_err_direct = jnp.abs(theta1_err_direct) + jnp.abs(theta2_err_direct)
            
            # Try offset by π (opposite orientation)
            theta1_err_offset = wrap_pi(xs[:, 2] - xref[:, 2] - jnp.pi)
            theta2_err_offset = wrap_pi(xs[:, 3] - xref[:, 3] - jnp.pi)
            total_err_offset = jnp.abs(theta1_err_offset) + jnp.abs(theta2_err_offset)
            
            # Choose orientation with smaller total error (for no preference case)
            use_direct = total_err_direct < total_err_offset
            theta1_err_wrapped = jnp.where(use_direct, theta1_err_direct, theta1_err_offset)
            theta2_err_wrapped = jnp.where(use_direct, theta2_err_direct, theta2_err_offset)
            
            # Select final angles: use wrapping only when no preference, direct otherwise
            no_preference = motion_preference == 0
            theta1_err = jnp.where(no_preference, theta1_err_wrapped, theta1_err_direct)
            theta2_err = jnp.where(no_preference, theta2_err_wrapped, theta2_err_direct)
            
            # Apply angle mask: set angle errors to 0 where mask is False (no constraint)
            theta1_err = jnp.where(angle_mask, theta1_err, 0.0)
            theta2_err = jnp.where(angle_mask, theta2_err, 0.0)
            
            theta1_logpd = -(jnp.clip(jnp.abs(theta1_err), 0.0, phi_max) / phi_max) ** 2
            theta2_logpd = -(jnp.clip(jnp.abs(theta2_err), 0.0, phi_max) / phi_max) ** 2
            
            # Combined logpd with configurable weights
            combined_logpd = (ref_pos_weight * pos_logpd + 
                            ref_theta1_weight * theta1_logpd + 
                            ref_theta2_weight * theta2_logpd)
            
            return combined_logpd.mean(axis=-1)
        
        # Check if angle mask is available, if not create a default mask (all True)
        if hasattr(self, 'angle_mask'):
            angle_mask = self.angle_mask
        else:
            angle_mask = jnp.ones(self.xref.shape[0], dtype=bool)  # Default: all points have angle constraints
        
        return _eval_xref_logpd(xs, self.xref, angle_mask, self.ref_reward_threshold, self.phi_max, motion_preference, 
                               self.ref_pos_weight, self.ref_theta1_weight, self.ref_theta2_weight)

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 4  # [px, py, theta1, theta2]

    def render(self, ax, xs: jnp.ndarray):
        """Render the tractor-trailer system"""
        # Add parking space boundaries for parking scenario
        if hasattr(self.env, 'case') and self.env.case == "parking" and hasattr(self.env, 'parking_config'):
            config = self.env.parking_config
            rows = config['parking_rows']
            cols = config['parking_cols']
            space_width = config['space_width']
            space_length = config['space_length']
            y_offset = config['parking_y_offset']
            
            # Calculate parking lot position (using same logic as env.py)
            parking_lot_width = cols * space_width
            parking_lot_height = rows * space_length
            parking_start_x = -parking_lot_width / 2
            parking_start_y = self.env.y_range[0] + y_offset  # Bottom of parking area
            
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
                    
                    # Color code: target spaces in green, occupied in red, empty in white
                    if space_num in config['target_spaces']:
                        color = 'lightgreen'
                        text_color = 'black'
                    elif space_num in config['occupied_spaces']:
                        color = 'lightcoral'
                        text_color = 'white'
                    else:
                        color = 'lightblue'
                        text_color = 'black'
                    
                    # Add colored background for space number
                    if space_num not in config['occupied_spaces']:  # Don't show numbers on occupied spaces (they have obstacles)
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
        if xs.shape[0] > 0:
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
        
        # Tractor body (rectangle) - spans from rear axle to front axle
        # Tractor body - use correct geometry including front and rear overhangs
        tractor_full_length = self.l1 + self.lf1 + self.lr
        self.tractor_body = ax.add_patch(
            plt.Rectangle((-tractor_full_length/2, -self.tractor_width/2),
                         tractor_full_length, self.tractor_width,
                         linewidth=2, edgecolor='black', facecolor=tractor_color, alpha=0.7)
        )
        
        # Trailer body - use correct geometry including front and rear overhangs
        trailer_full_length = self.lf2 + self.lr2
        self.trailer_body = ax.add_patch(
            plt.Rectangle((-trailer_full_length/2, -self.trailer_width/2),
                         trailer_full_length, self.trailer_width,
                         linewidth=2, edgecolor='black', facecolor=trailer_color, alpha=0.7)
        )
        
        # Wheels - positioned at (0,0) in local coordinates, will be translated to axle positions
        wheel_width = 0.45
        wheel_length = 0.8
        
        # Tractor wheels (front and rear)
        for i in range(2):
            wheel = ax.add_patch(
                plt.Rectangle((-wheel_length/2, -wheel_width/2),
                             wheel_length, wheel_width,
                             edgecolor='black', facecolor='gray', alpha=0.9)
            )
            self.tractor_wheels.append(wheel)
        
        # Trailer wheels
        for i in range(1):
            wheel = ax.add_patch(
                plt.Rectangle((-wheel_length/2, -wheel_width/2),
                             wheel_length, wheel_width,
                             edgecolor='black', facecolor='gray', alpha=0.9)
            )
            self.trailer_wheels.append(wheel)
        
        # Hitch connection lines
        self.tractor_hitch_line, = ax.plot([], [], 'k-', linewidth=3, alpha=0.8)
        self.trailer_hitch_line, = ax.plot([], [], 'k-', linewidth=3, alpha=0.8)

    def get_tractor_trailer_positions(self, x):
        """Calculate tractor and trailer positions and orientations"""
        px, py, theta1, theta2 = x[:4]
        
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
        
        # Trailer front face center
        trailer_front_x = hitch_x - (self.l2 - self.lf2) * np.cos(theta2)
        trailer_front_y = hitch_y - (self.l2 - self.lf2) * np.sin(theta2)
        
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
        px, py, theta1, theta2 = x[:4]
        
        # Get positions
        positions = self.get_tractor_trailer_positions(x)
        
        # Tractor body transform - use correct geometry from get_tractor_trailer_rectangles
        # Tractor center position accounting for front and rear overhangs
        tractor_center_x = px + 0.5 * (self.l1 + self.lf1 - self.lr) * np.cos(theta1)
        tractor_center_y = py + 0.5 * (self.l1 + self.lf1 - self.lr) * np.sin(theta1)
        transform_tractor_body = (Affine2D()
                                 .rotate(theta1)
                                 .translate(tractor_center_x, tractor_center_y) + plt.gca().transData)
        
        # Trailer body transform - use correct geometry from get_tractor_trailer_rectangles
        # Hitch position
        hitch_x = px - self.lh * np.cos(theta1)
        hitch_y = py - self.lh * np.sin(theta1)
        # Trailer center position accounting for front and rear overhangs
        trailer_center_x = hitch_x + (0.5 * (self.lf2 - self.lr2) - self.l2) * np.cos(theta2)
        trailer_center_y = hitch_y + (0.5 * (self.lf2 - self.lr2) - self.l2) * np.sin(theta2)
        transform_trailer_body = (Affine2D()
                                 .rotate(theta2)
                                 .translate(trailer_center_x, trailer_center_y) + plt.gca().transData)
        
        # Wheel transforms
        transforms_tractor_wheels = []
        # Tractor rear wheel (no steering)
        transforms_tractor_wheels.append(
            Affine2D().rotate(theta1).translate(*positions['tractor_rear']) + plt.gca().transData
        )
        # Tractor front wheel (with steering angle)
        # The wheel should rotate around its center at the front axle position
        steering_angle = 0.0
        
        # Extract steering angle based on state dimensionality
        if len(x) >= 6:
            # 6D state (acceleration dynamics): delta is at index 5
            steering_angle = x[5]
        elif u is not None:
            # 4D state (kinematic dynamics): delta from control input
            v, delta = u
            steering_angle = delta
        front_wheel_transform = (Affine2D()
                               .rotate(theta1 + steering_angle)
                               .translate(*positions['tractor_front']) + plt.gca().transData)
        transforms_tractor_wheels.append(front_wheel_transform)
        
        transforms_trailer_wheels = []
        # Trailer wheels
        transforms_trailer_wheels.append(
            Affine2D().rotate(theta2).translate(*positions['trailer_rear']) + plt.gca().transData
        )
        # transforms_trailer_wheels.append(
        #     Affine2D().rotate(theta2).translate(*positions['trailer_front']) + plt.gca().transData
        # )
        
        return {
            'tractor_body': transform_tractor_body,
            'trailer_body': transform_trailer_body,
            'tractor_wheels': transforms_tractor_wheels,
            'trailer_wheels': transforms_trailer_wheels,
            'tractor_hitch_line_data': (positions['tractor_rear'], positions['hitch']),
            'trailer_hitch_line_data': (positions['trailer_front'], positions['hitch'])
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
        
        # Update hitch lines
        tractor_hitch_data = transforms['tractor_hitch_line_data']
        self.tractor_hitch_line.set_data([tractor_hitch_data[0][0], tractor_hitch_data[1][0]], 
                                        [tractor_hitch_data[0][1], tractor_hitch_data[1][1]])
        
        trailer_hitch_data = transforms['trailer_hitch_line_data']
        self.trailer_hitch_line.set_data([trailer_hitch_data[0][0], trailer_hitch_data[1][0]], 
                                        [trailer_hitch_data[0][1], trailer_hitch_data[1][1]])

    def _numpy_projection_function(self, args_tuple):
        """
        Pure NumPy projection function for host callback.
        Minimizes ||u - u_original||² subject to safety constraints on resulting state.
        """
        import numpy as np
        from scipy.optimize import minimize
        
        # Unpack arguments
        q_current_np, u_original_normalized_np = args_tuple
        
        # Store environment parameters for use in nested functions
        l1, l2, lh, lf1, lr, lf2, lr2 = self.l1, self.l2, self.lh, self.lf1, self.lr, self.lf2, self.lr2
        tractor_width, trailer_width = self.tractor_width, self.trailer_width
        phi_max = self.phi_max
        v_max, delta_max = self.v_max, self.delta_max
        dt = self.dt
        obs_circles = self.obs_circles
        obs_rectangles = self.obs_rectangles
        
        def tractor_trailer_dynamics_np(x, u_scaled):
            """NumPy version of tractor trailer dynamics"""
            px, py, theta1, theta2 = x
            v, delta = u_scaled  # u_scaled is in actual units
            
            px_dot = v * np.cos(theta1)
            py_dot = v * np.sin(theta1)
            theta1_dot = (v / l1) * np.tan(delta)
            theta2_dot = (v / l2) * (
                np.sin(theta1 - theta2) - 
                (lh / l1) * np.cos(theta1 - theta2) * np.tan(delta)
            )
            
            return np.array([px_dot, py_dot, theta1_dot, theta2_dot])
        
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
        
        def get_tractor_trailer_rectangles_np(x):
            """NumPy version of get_tractor_trailer_rectangles"""
            px, py, theta1, theta2 = x[:4]
            
            # Tractor rectangle (centered at tractor center)
            tractor_center_x = px + 0.5 * (l1 + lf1 - lr) * np.cos(theta1)
            tractor_center_y = py + 0.5 * (l1 + lf1 - lr) * np.sin(theta1)
            
            # Trailer rectangle
            hitch_x = px - lh * np.cos(theta1)
            hitch_y = py - lh * np.sin(theta1)
            trailer_center_x = hitch_x + (0.5 * (lf2 - lr2) - l2) * np.cos(theta2)
            trailer_center_y = hitch_y + (0.5 * (lf2 - lr2) - l2) * np.sin(theta2)
            
            return {
                'tractor_center': np.array([tractor_center_x, tractor_center_y]),
                'tractor_angle': theta1,
                'trailer_center': np.array([trailer_center_x, trailer_center_y]),
                'trailer_angle': theta2
            }
        
        def _sdf_oriented_box_np(point, center, half_size, angle):
            """
            NumPy version of signed distance from point to oriented rectangle.
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
            """
            NumPy version of signed distance between two oriented boxes.
            """
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
            """
            NumPy version of signed distance between tractor+trailer rectangles and circle.
            """
            geom = get_tractor_trailer_rectangles_np(x)
            center_c, r = circle[:2], circle[2]
            
            # tractor SDF
            hd_trac = np.array([l1/2, tractor_width/2])
            dt = _sdf_oriented_box_np(center_c, geom['tractor_center'], hd_trac, geom['tractor_angle'])
            
            # trailer SDF
            hd_tral = np.array([l2/2, trailer_width/2])
            dtr = _sdf_oriented_box_np(center_c, geom['trailer_center'], hd_tral, geom['trailer_angle'])
            
            # subtract circle radius, then take the tighter (minimum)
            return min(dt, dtr) - r
        
        def _signed_dist_rect_rect_np(x, rect_obs):
            """
            NumPy version of signed distance between tractor+trailer and rectangular obstacle.
            """
            ox, oy, w, h, oa = rect_obs
            c_obs = np.array([ox, oy])
            hs_obs = np.array([w/2, h/2])
            
            geom = get_tractor_trailer_rectangles_np(x)
            
            # tractor
            c_t = geom['tractor_center']
            ht = np.array([l1/2, tractor_width/2])
            at = geom['tractor_angle']
            d1 = _sdf_two_boxes_np(c_t, ht, at, c_obs, hs_obs, oa)
            
            # trailer  
            c_r = geom['trailer_center']
            hr = np.array([l2/2, trailer_width/2])
            ar = geom['trailer_angle']
            d2 = _sdf_two_boxes_np(c_r, hr, ar, c_obs, hs_obs, oa)
            
            return min(d1, d2)
        
        def constraint_function_np(q):
            """NumPy version of constraint evaluation"""
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
            
            # Hitch angle constraint
            wrap_pi = lambda a: (a + np.pi) % (2*np.pi) - np.pi
            hitch_angle = wrap_pi(q[2] - q[3])
            hitch_constraint = phi_max - np.abs(hitch_angle)
            cs.append(float(hitch_constraint))
            
            return np.array(cs) if len(cs) > 0 else np.array([1.0])
        
        def objective(u_normalized):
            """Objective: minimize distance to original control input"""
            return np.sum((u_normalized - u_original_normalized_np)**2)
        
        def constraint_function(u_normalized):
            """Constraint function: returns array where each element >= 0 for feasibility"""
            # Scale control input to actual ranges
            u_scaled = input_scaler_np(u_normalized)
            
            # Compute resulting state
            q_new = rk4_np(tractor_trailer_dynamics_np, q_current_np, u_scaled, dt)
            
            # Evaluate constraints on resulting state
            constraints = constraint_function_np(q_new)
            
            return constraints
        
        # Set up optimization problem
        u0 = u_original_normalized_np  # Start from original normalized control
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]  # Normalized control bounds
        
        # Constraint dict for SciPy
        constraints = {
            'type': 'ineq',
            'fun': constraint_function
        }
        
        try:
            # Use trust-constr which handles nonlinear constraints well
            print("WIP... Running Trust-Region Constrained Method for Projection in CPU...")
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
                # If optimization fails, return original control
                return u_original_normalized_np
                
        except Exception as e:
            # Fallback to original on any error
            return u_original_normalized_np

    @partial(jax.jit, static_argnums=(0,))
    def project_control_to_safe_set(self, q_current, u_original_normalized):
        """
        Project control input to safe set using external callback to SciPy optimization.
        
        Args:
            q_current: Current state
            u_original_normalized: Original normalized control input [-1,1]
            
        Returns:
            u_safe_normalized: Safe normalized control input [-1,1]
        """
        # Check if projection is enabled
        if not self.enable_projection:
            return u_original_normalized
        
        # Use external callback to call NumPy-based optimization
        import jax.experimental
        
        # Prepare arguments
        args_tuple = (q_current, u_original_normalized)
        
        # Call the NumPy function via external callback
        u_projected = jax.experimental.io_callback(
            self._numpy_projection_function,
            u_original_normalized,  # result shape and dtype
            args_tuple
        )
        
        return u_projected

    @partial(jax.jit, static_argnums=(0,))
    def _guidance_function(self, x):
        """
        Simplified guidance function for gradient descent.
        Uses basic point-to-point distances instead of complex SDFs to ensure differentiability.
        """
        px, py, theta1, theta2 = x[:4]
        
        # Get tractor and trailer positions (simplified)
        tractor_pos = jnp.array([px, py])
        
        # Simple trailer position approximation
        trailer_x = px - self.lh * jnp.cos(theta1) - self.l2 * jnp.cos(theta2)
        trailer_y = py - self.lh * jnp.sin(theta1) - self.l2 * jnp.sin(theta2)
        trailer_pos = jnp.array([trailer_x, trailer_y])
        
        total_cost = 0.0
        safety_margin = 2.0  # Safety margin around obstacles
        
        # Penalty for circular obstacles (simple point-to-circle distance)
        # NOTE: There is no straightforward method to compute differentiable SDF between a non-convex and disjoint polygon
        #       and disjoint polygons. So, simply apply a simple point-to-circle distance penalty.
        #       Also, ignore the rectangular obstacles for here.
        if self.obs_circles.shape[0] > 0:
            for i in range(self.obs_circles.shape[0]):
                circle = self.obs_circles[i]
                circle_center = circle[:2]
                circle_radius = circle[2]
                
                # Distance from tractor to circle center
                tractor_dist = jnp.linalg.norm(tractor_pos - circle_center)
                tractor_violation = jnp.maximum(0.0, circle_radius + safety_margin - tractor_dist)
                
                # Distance from trailer to circle center
                trailer_dist = jnp.linalg.norm(trailer_pos - circle_center)
                trailer_violation = jnp.maximum(0.0, circle_radius + safety_margin - trailer_dist)
                
                # Add squared penalties
                total_cost += tractor_violation ** 2 + trailer_violation ** 2
    
        hitch_angle_raw = theta1 - theta2
        
        # Use multiple periodic penalties to handle angle wrapping smoothly
        # This creates a smooth, differentiable function that penalizes large hitch angles
        hitch_penalties = []
        for offset in [0.0, 2*jnp.pi, -2*jnp.pi]:  # Consider multiple wrapping points
            adjusted_angle = hitch_angle_raw + offset
            angle_magnitude = jnp.abs(adjusted_angle)
            hitch_violation = jnp.maximum(0.0, angle_magnitude - self.phi_max)
            hitch_penalties.append(hitch_violation ** 2)
        
        # Use the minimum penalty (most favorable interpretation)
        hitch_penalty = jnp.min(jnp.array(hitch_penalties))
        
        # Scale by number of obstacles to balance against collision constraints
        num_total_obstacles = (self.obs_circles.shape[0] + self.obs_rectangles.shape[0])
        hitch_weight = jnp.maximum(1.0, num_total_obstacles * 0.05)  # At least 5x weight
        
        total_cost += hitch_weight * hitch_penalty
        
        # Small regularization for numerical stability
        #total_cost +=  jnp.sum(x ** 2)
        
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
            x_new = x_new.at[3].set(jnp.clip(x_new[3], -2*jnp.pi, 2*jnp.pi))  # theta2
            
            # Safety check: if result has NaN/inf, return previous state
            x_new = jnp.where(jnp.isfinite(x_new), x_new, x)
            
            return x_new, None
        
        # Apply gradient descent steps
        x_final, _ = jax.lax.scan(guidance_step, q_proposed, None, length=max_steps)
        
        return x_final
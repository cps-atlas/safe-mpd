import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import numpy as np

from .tt2d import TractorTrailer2d, State, rk4

"""
Created on July 21st, 2025
@author: Taekyung Kim

@description: 
Implement acceleration-controlled tractor-trailer dynamics.
State: [px, py, theta1, theta2, v, delta] (6D)
Control: [a, omega] where a = v_dot, omega = delta_dot
"""

class AccTractorTrailer2d(TractorTrailer2d):
    """
    Acceleration-controlled Tractor-Trailer 2D
    State: [px, py, theta1, theta2, v, delta]
    Action: [a, omega] - acceleration and steering rate
    """
    
    def __init__(self, **kwargs):
        # Add acceleration-specific parameters with reasonable defaults
        self.a_max = kwargs.pop('a_max', 2.0)  # maximum acceleration [m/s²]
        self.omega_max = kwargs.pop('omega_max', 1.0)  # maximum steering rate [rad/s]
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Pre-compute constants for JIT optimization
        # Use larger time step for predictive rollout to reduce computation
        self._rollout_dt = 1.0  # seconds - larger dt for rollout prediction only
        self._v_threshold = 0.5  # velocity threshold for rollout decision
        self._max_rollout_steps = int(2.0 * (self.v_max-self._v_threshold) / (self.a_max * self._rollout_dt))
        
        # Override initial state to include v and delta (initialized to zero)
        if hasattr(self, 'x0'):
            # Extend existing x0 with v=0, delta=0
            self.x0 = jnp.concatenate([self.x0, jnp.array([0.0, 0.0])])
        else:
            # Default 6D initial state
            self.x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        if hasattr(self, 'xg'):
            # Extend existing xg with v=0, delta=0 (goal velocities)
            self.xg = jnp.concatenate([self.xg, jnp.array([0.0, 0.0])])
        else:
            # Default 6D goal state
            self.xg = jnp.array([3.0, 0.0, np.pi, np.pi, 0.0, 0.0])

    def set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0, v=0.0, delta=0.0):
        """
        Set initial position for acceleration-controlled tractor-trailer.
        Extended to include initial velocity and steering angle.
        """
        # Call parent method to set position and angles
        super().set_init_pos(x, y, dx, dy, theta1, theta2)
        
        # Extend to 6D state with initial v and delta
        self.x0 = jnp.concatenate([self.x0, jnp.array([v, delta])])

    def set_goal_pos(self, x=None, y=None, theta1=None, theta2=None, v=0.0, delta=0.0):
        """
        Set goal position for acceleration-controlled tractor-trailer.
        Extended to include goal velocity and steering angle.
        """
        # Call parent method to set position and angles
        super().set_goal_pos(x, y, theta1, theta2)
        
        # Extend to 6D state with goal v and delta
        self.xg = jnp.concatenate([self.xg, jnp.array([v, delta])])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array):
        """Resets the environment to an initial state."""
        return State(self.x0, self.x0, 0.0, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def input_scaler(self, u_normalized):
        """Scale normalized inputs [-1, 1] to actual input ranges for acceleration control"""
        a = u_normalized[0] * self.a_max  # acceleration
        omega = u_normalized[1] * self.omega_max  # steering rate
        return jnp.array([a, omega])

    @partial(jax.jit, static_argnums=(0,))
    def tractor_trailer_dynamics(self, x, u):
        """
        Acceleration-controlled tractor-trailer dynamics
        State: [px, py, theta1, theta2, v, delta]
        Input: [a, omega] (already scaled)
        """
        px, py, theta1, theta2, v, delta = x
        a, omega = u
        
        # Clip velocity and steering to physical limits
        v = jnp.clip(v, -self.v_max, self.v_max)
        delta = jnp.clip(delta, -self.delta_max, self.delta_max)
        
        # Position dynamics (same as kinematic model)
        px_dot = v * jnp.cos(theta1)
        py_dot = v * jnp.sin(theta1)
        theta1_dot = (v / self.l1) * jnp.tan(delta)
        theta2_dot = (v / self.l2) * (
            jnp.sin(theta1 - theta2) - 
            (self.lh / self.l1) * jnp.cos(theta1 - theta2) * jnp.tan(delta)
        )
        
        # Velocity dynamics
        v_dot = a
        delta_dot = omega
        
        return jnp.array([px_dot, py_dot, theta1_dot, theta2_dot, v_dot, delta_dot])

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        action = jnp.clip(action, -1.0, 1.0)
        
        # Scale inputs from normalized [-1, 1] to actual ranges
        u_scaled = self.input_scaler(action)
        
        q = state.pipeline_state
        
        # Propose next state via RK4
        q_proposed = rk4(self.tractor_trailer_dynamics, q, u_scaled, self.dt)
        
        # Choose between three methods: projection, gated rollout, or naive penalty
        def use_projection_fn(args):
            q_prop, = args
            q_safe = self.project_to_safe_set(q_prop)
            return q_safe, False, False  # No penalties since guaranteed safe
        
        def use_gated_rollout_fn(args):
            q_prop, = args
            return self._step_with_gated_rollout((q, q_prop))
        
        def use_naive_penalty_fn(args):
            q_prop, = args
            return self._step_without_gated_rollout(q_prop)
        
        # Select method based on flags (three equal options)
        use_gated_rollout = self.enable_gated_rollout_collision | self.enable_gated_rollout_hitch
        
        q_final, obstacle_collision, hitch_violation = jax.lax.cond(
            self.enable_projection,
            use_projection_fn,
            lambda args: jax.lax.cond(
                use_gated_rollout,
                use_gated_rollout_fn,
                use_naive_penalty_fn,
                args
            ),
            (q_proposed,)
        )
        
        # Compute reward using only position/angle states
        reward = self.get_reward(q_final[:4])
        
        # Apply penalties using flags (only if violations were checked)
        reward = jnp.where(obstacle_collision, reward - self.collision_penalty, reward)
        reward = jnp.where(hitch_violation, reward - self.hitch_penalty, reward)
        
        # Add preference penalty based on motion direction (use state velocity)
        preference_penalty = self.get_preference_penalty(q_final, u_scaled)
        has_preference = self.motion_preference != 0
        reward = jnp.where(has_preference, reward - preference_penalty, reward)
        
        return state.replace(pipeline_state=q_final, obs=q_final, reward=reward, done=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def _step_with_gated_rollout(self, data):
        """Step function with gated rollout enabled"""
        q, q_proposed = data
        return self.gated_rollout(q, q_proposed)
    
    @partial(jax.jit, static_argnums=(0,))
    def _step_without_gated_rollout(self, q_proposed):
        """Step function with simple forward dynamics (no gated rollout)"""
        # Check collision and hitch violations on q_proposed (for penalty computation)
        obstacle_collision = self.check_obstacle_collision(q_proposed[:4], self.obs_circles, self.obs_rectangles)
        hitch_violation = self.check_hitch_violation(q_proposed[:4])
        
        return q_proposed, obstacle_collision, hitch_violation

    @partial(jax.jit, static_argnums=(0,))
    def gated_rollout(self, q, q_proposed):
        """
        Gated rollout for acceleration dynamics with correct safety checking logic.
        
        Logic:
        1. Use pre-computed q_proposed as the desired next state
        2. If velocity is high, do additional rollout until stop → q_candidate_final (for safety checking only)
        3. Check safety during the entire rollout (from q_proposed to q_candidate_final)
        4. If safe → return q_proposed; if unsafe → return q (stick to previous state)
        """
        # Step 1: Use pre-computed proposed state (already computed via RK4 in step function)
        q_new = q_proposed
        
        # Step 2: Check if we need additional rollout for safety checking
        v_new = q_new[4]
        
        # Choose safety checking strategy based on velocity
        # For very low velocities, use simple single-step check
        # For higher velocities, use predictive rollout checking
        obstacle_collision, hitch_violation = jax.lax.cond(
            jnp.abs(v_new) <= self._v_threshold,
            self._check_safety_single_step,
            self._check_safety_with_rollout,
            (q_new, v_new)
        )
        
        # Step 4: Return q_new if safe, otherwise stick to previous q
        # Apply projections based on settings
        q_gated = jnp.where(self.enable_gated_rollout_collision & obstacle_collision, q, q_new)
        q_final = jnp.where(self.enable_gated_rollout_hitch & hitch_violation, q, q_gated)
        
        return q_final, obstacle_collision, hitch_violation

    @partial(jax.jit, static_argnums=(0,))
    def _check_safety_single_step(self, data):
        """Check safety for just the single step case"""
        q_new, v_new = data
        obstacle_collision = self.check_obstacle_collision(q_new[:4], self.obs_circles, self.obs_rectangles)
        hitch_violation = self.check_hitch_violation(q_new[:4])
        return obstacle_collision, hitch_violation

    @partial(jax.jit, static_argnums=(0,))
    def _rollout_step(self, carry, _):
        """Single step of the rollout-to-stop process (JIT-optimized)"""
        q_curr, stopped, obs_collision_any, hitch_violation_any, u_stop = carry
        v_curr = q_curr[4]
        
        # Continue rollout if not stopped
        should_continue = jnp.abs(v_curr) > self._v_threshold
        q_next = jnp.where(should_continue, rk4(self.tractor_trailer_dynamics, q_curr, u_stop, self._rollout_dt), q_curr)
        
        # Check safety at this step
        obs_collision_step = self.check_obstacle_collision(q_next[:4], self.obs_circles, self.obs_rectangles)
        hitch_violation_step = self.check_hitch_violation(q_next[:4])
        
        # Accumulate any violations found during rollout
        obs_collision_any = obs_collision_any | obs_collision_step
        hitch_violation_any = hitch_violation_any | hitch_violation_step
        
        # Update stopped flag (using same threshold as rollout decision)
        stopped_next = stopped | (jnp.abs(q_next[4]) <= self._v_threshold)
        
        return (q_next, stopped_next, obs_collision_any, hitch_violation_any, u_stop), None

    @partial(jax.jit, static_argnums=(0,))
    def _check_safety_with_rollout(self, data):
        """Check safety during rollout until stop (for safety prediction)"""
        q_new, v_new = data
        
        # Check safety of q_new first
        obstacle_collision_new = self.check_obstacle_collision(q_new[:4], self.obs_circles, self.obs_rectangles)
        hitch_violation_new = self.check_hitch_violation(q_new[:4])
        
        # Determine deceleration direction from q_new
        decel_sign = jnp.where(v_new > 0, -1.0, jnp.where(v_new < 0, 1.0, 0.0))
        u_stop = jnp.array([decel_sign * self.a_max, 0.0])
        
        # Run rollout with pre-computed maximum steps using JIT-optimized rollout step
        init_carry = (q_new, False, obstacle_collision_new, hitch_violation_new, u_stop)
        (q_final, _, obstacle_collision_any, hitch_violation_any, _), _ = jax.lax.scan(
            self._rollout_step, init_carry, None, length=self._max_rollout_steps
        )
        
        return obstacle_collision_any, hitch_violation_any

    @partial(jax.jit, static_argnums=(0,))
    def project_to_safe_set(self, q_prop):
        """
        Project 6D proposed state to safe set using parent 4D projection functions.
        Only the first 4 elements (position/angle) are constrained for safety.
        """
        # Check if projection is enabled (via parent class)
        if not self.enable_projection:
            # If projection is disabled, just return the proposed state
            return q_prop
        
        # Extract first 4 elements (position/angle states)
        q_4d = q_prop[:4]
        
        # Project 4D state to safe set using parent function
        q_4d_safe = super().project_to_safe_set(q_4d)
        
        # Reconstruct 6D state by keeping velocity and steering unchanged
        q_safe = jnp.concatenate([q_4d_safe, q_prop[4:]])
        
        return q_safe

    def get_preference_penalty(self, x, u):
        """
        Calculate preference penalty using parent method.
        Parent method handles both kinematic and acceleration dynamics.
        """
        return super().get_preference_penalty(x, u)

    def get_reward(self, q):
        """Calculate reward using parent method with 4D state slicing."""
        return super().get_reward(q[:4])
    
    def get_terminal_reward(self, q):
        """Calculate terminal reward using parent method with 4D state slicing."""
        return super().get_terminal_reward(q[:4])

    @property
    def action_size(self):
        return 2  # [a, omega]

    @property  
    def observation_size(self):
        return 6  # [px, py, theta1, theta2, v, delta]

    def generate_demonstration_trajectory(self, search_direction="horizontal", num_waypoints_max=4, motion_preference=0):
        """
        Generate demonstration trajectory for acceleration dynamics.
        Uses parent method for 4D trajectory, ignores velocity/steering for demonstration.
        """
        # Use parent method for 4D trajectory
        xref_4d = super().generate_demonstration_trajectory(search_direction, num_waypoints_max, motion_preference)
        
        # For acceleration dynamics, we only use position/angle states in demonstration
        # Store as 4D trajectory (consistent with user requirement)
        self.xref = xref_4d
        
        return self.xref

    def compute_demonstration_reward(self):
        """
        Compute demonstration reward using parent method.
        Reference trajectory is stored as 4D, so parent method works directly.
        """
        super().compute_demonstration_reward()

    def eval_xref_logpd(self, xs, motion_preference=None):
        """
        Evaluate log probability density with respect to reference trajectory.
        Extract first 4 elements from 6D state and use parent method.
        """
        # Extract only first 4 states (position/angle) for evaluation
        xs_4d = xs[:, :4]
        
        # Use parent method with 4D states
        return super().eval_xref_logpd(xs_4d, motion_preference) 
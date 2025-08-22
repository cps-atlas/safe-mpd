import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from .tt2d import TractorTrailer2d, State, rk4

class NTrailer2d(TractorTrailer2d):
    """
    Kinematic N-trailer model (N >= 2).
    State: [px, py, theta0, theta1..thetaN]
    Action: [v, delta]
    Geometry arrays are size N for each trailer.
    """
    def __init__(self, *, num_trailers: int,
                 trailer_lengths: jnp.ndarray | None = None,
                 trailer_front_offsets: jnp.ndarray | None = None,
                 trailer_rear_offsets: jnp.ndarray | None = None,
                 **kwargs):
        assert num_trailers is not None and num_trailers >= 2, "use other dynamics"
        super().__init__(num_trailers=num_trailers, **kwargs)
        self.num_trailers = int(num_trailers)
        # Build per-trailer parameters arrays (size = N)
        default_l2 = float(self.l2)
        default_lf2 = float(self.lf2)
        default_lr2 = float(self.lr2)
        self.trailer_lengths = jnp.full((self.num_trailers,), default_l2) if trailer_lengths is None else jnp.asarray(trailer_lengths, dtype=jnp.float32)
        self.trailer_front_offsets = jnp.full((self.num_trailers,), default_lf2) if trailer_front_offsets is None else jnp.asarray(trailer_front_offsets, dtype=jnp.float32)
        if trailer_rear_offsets is None:
            # Copy from tractor lr for all, then set last trailer's rear offset to original lr2
            rear = jnp.full((self.num_trailers,), float(self.lr))
            rear = rear.at[-1].set(default_lr2)
            self.trailer_rear_offsets = rear
        else:
            self.trailer_rear_offsets = jnp.asarray(trailer_rear_offsets, dtype=jnp.float32)

        # Expand x0/xg from 4D to 3+N by repeating the base trailer angle for all trailers
        def _augment_state(x4):
            theta0 = x4[2]
            theta1 = x4[3]
            # build [theta1, theta2..thetaN] all equal to theta1 initially
            trailer_thetas = jnp.concatenate([jnp.array([theta1]), jnp.full((self.num_trailers - 1,), theta1)])
            return jnp.concatenate([x4[:3], trailer_thetas])
        self.x0 = _augment_state(self.x0)
        self.xg = _augment_state(self.xg)

        # Animation elements for multi-trailer
        self.trailer_bodies = []
        self.trailer_wheels = []
        # Two sets of hitch lines: front (prev hitch -> current trailer front), back (current trailer rear -> next hitch)
        self.trailer_hitch_lines_front = []
        self.trailer_hitch_lines_back = []

    @partial(jax.jit, static_argnums=(0,))
    def input_scaler(self, u_normalized):
        return super().input_scaler(u_normalized)

    @partial(jax.jit, static_argnums=(0,))
    def n_trailer_dynamics(self, x, u):
        v, delta = u
        px, py = x[0], x[1]
        theta0 = x[2]
        thetas = x[3: 3 + self.num_trailers]

        px_dot = v * jnp.cos(theta0)
        py_dot = v * jnp.sin(theta0)
        theta0_dot = (v / self.l1) * jnp.tan(delta)

        # Prepare output array for trailer angle rates
        trailer_dots = jnp.zeros((self.num_trailers,), dtype=x.dtype)

        def loop(i, acc):
            prev_theta, prev_dot, prev_axle_speed, dots = acc
            theta_i = thetas[i]
            L_i = self.trailer_lengths[i]
            dtheta = prev_theta - theta_i
            # First trailer (i==0): use tractor inputs with minus sign on hitch term
            theta_i_dot = jnp.where(
                i == 0,
                (v / L_i) * (jnp.sin(dtheta) - (self.lh / self.l1) * jnp.cos(dtheta) * jnp.tan(delta)),
                (prev_axle_speed / L_i) * jnp.sin(dtheta) - (self.lh / L_i) * prev_dot * jnp.cos(dtheta)
            )
            # Axle speed at this trailer (for next stage): project hitch velocity along body
            axle_speed_i = jnp.where(
                i == 0,
                v * jnp.cos(dtheta) + self.lh * theta0_dot * jnp.sin(dtheta),
                prev_axle_speed * jnp.cos(dtheta) + self.lh * prev_dot * jnp.sin(dtheta)
            )
            dots = dots.at[i].set(theta_i_dot)
            return (theta_i, theta_i_dot, axle_speed_i, dots)

        init = (theta0, theta0_dot, v, trailer_dots)
        _, _, _, trailer_dots = jax.lax.fori_loop(0, self.num_trailers, loop, init)

        theta_dots = jnp.concatenate([jnp.array([theta0_dot]), trailer_dots])
        x_dot = jnp.concatenate([jnp.array([px_dot, py_dot]), theta_dots])
        return x_dot

    def reset(self, rng: jax.Array):
        return State(self.x0, self.x0, jnp.array(0.0, dtype=self.x0.dtype), jnp.array(0.0, dtype=self.x0.dtype), jnp.zeros(2), jnp.array(False))

    def set_init_pos(self, x=None, y=None, dx=None, dy=None, theta1=0.0, theta2=0.0):
        # Reuse base logic to set tractor (theta1) and first trailer (theta2)
        super().set_init_pos(x=x, y=y, dx=dx, dy=dy, theta1=theta1, theta2=theta2)
        # Augment to 3+N with remaining trailers equal to first trailer angle
        theta0, theta_first = self.x0[2], self.x0[3]
        extra = jnp.full((self.num_trailers - 1,), theta_first)
        self.x0 = jnp.concatenate([self.x0[:3], jnp.concatenate([jnp.array([theta_first]), extra])])

    def set_goal_pos(self, x=None, y=None, theta1=None, theta2=None):
        super().set_goal_pos(x=x, y=y, theta1=theta1, theta2=theta2)
        # Augment to 3+N with remaining trailers equal to first trailer goal angle
        theta0_g, theta_first_g = self.xg[2], self.xg[3]
        extra = jnp.full((self.num_trailers - 1,), theta_first_g)
        self.xg = jnp.concatenate([self.xg[:3], jnp.concatenate([jnp.array([theta_first_g]), extra])])

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        return self._step_internal(state, action, self.visualization_mode)
    
    @partial(jax.jit, static_argnums=(0, 3))
    def _step_internal(self, state: State, action: jax.Array, visualization_mode: bool) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        backup_active_prev = state.in_backup_mode
        u_backup_norm = jnp.array([0.0, 0.0])
        action_eff_norm = jnp.where(backup_active_prev, u_backup_norm, action)
        u_scaled = self.input_scaler(action_eff_norm)
        q = state.pipeline_state
        assert q.shape[0] == 3 + self.num_trailers, "state length mismatch in NTrailer2d._step_internal"
        if self.enable_projection:
            raise AssertionError("Projection not implemented for NTrailer2d")
        elif self.enable_guidance:
            q_proposed = rk4(self.n_trailer_dynamics, q, u_scaled, self.dt)
            q_reward = self.apply_guidance(q_proposed)
            q_final = q_reward if visualization_mode else q_proposed
            obstacle_collision = self.check_obstacle_collision(q_reward, self.obs_circles, self.obs_rectangles)
            hitch_violation = self.check_hitch_violation(q_reward)
            applied_action = action_eff_norm
            backup_active_next = backup_active_prev
        else:
            q_proposed = rk4(self.n_trailer_dynamics, q, u_scaled, self.dt)
            use_shielded_rollout = self.enable_shielded_rollout_collision | self.enable_shielded_rollout_hitch
            def use_shielded(args):
                q_prop, = args
                return self._step_with_shielded_rollout((q, q_prop))
            def use_naive(args):
                q_prop, = args
                return self._step_without_shielded_rollout(q_prop)
            q_final, obstacle_collision, hitch_violation = jax.lax.cond(use_shielded_rollout, use_shielded, use_naive, (q_proposed,))
            q_reward = q_final
            should_fallback = (self.enable_shielded_rollout_collision & obstacle_collision) | \
                              (self.enable_shielded_rollout_hitch & hitch_violation)
            backup_active_next = backup_active_prev | should_fallback
            applied_action = jnp.where(backup_active_next, u_backup_norm, action)

        reward = self.get_reward(q_reward)
        reward = jnp.where(obstacle_collision, reward - self.collision_penalty, reward)
        reward = jnp.where(hitch_violation, reward - self.hitch_penalty, reward)
        preference_penalty = self.get_preference_penalty(u_scaled)
        has_pref = self.motion_preference != 0
        reward = jnp.where(has_pref, reward - preference_penalty, reward)
        # NOTE: since apply fallback for all time steps drop the sample efficiency too much, we allow shielded 
        return state.replace(pipeline_state=q_final, obs=q_final, reward=reward, done=0.0,
                             applied_action=applied_action, in_backup_mode=backup_active_prev)

    @partial(jax.jit, static_argnums=(0,))
    def _step_with_shielded_rollout(self, data):
        q, q_proposed = data
        return self.shielded_rollout(q, q_proposed)

    @partial(jax.jit, static_argnums=(0,))
    def _step_without_shielded_rollout(self, q_proposed):
        obstacle_collision = self.check_obstacle_collision(q_proposed, self.obs_circles, self.obs_rectangles)
        hitch_violation = self.check_hitch_violation(q_proposed)
        return q_proposed, obstacle_collision, hitch_violation

    @partial(jax.jit, static_argnums=(0,))
    def shielded_rollout(self, q, q_proposed):
        obstacle_collision = self.check_obstacle_collision(q_proposed, self.obs_circles, self.obs_rectangles)
        hitch_violation = self.check_hitch_violation(q_proposed)
        q_shielded = jnp.where(self.enable_shielded_rollout_collision & obstacle_collision, q, q_proposed)
        q_final = jnp.where(self.enable_shielded_rollout_hitch & hitch_violation, q, q_shielded)
        return q_final, obstacle_collision, hitch_violation

    @partial(jax.jit, static_argnums=(0,))
    def get_preference_penalty(self, x_or_u, u=None):
        if u is None:
            v, delta = x_or_u
        else:
            if len(x_or_u) >= 5:
                v = x_or_u[4]
            else:
                v, delta = u
        penalty_weight = self.preference_penalty_weight
        forward_penalty = jnp.where(v < 0, penalty_weight * jnp.abs(v), 0.0)
        backward_penalty = jnp.where(v > 0, penalty_weight * jnp.abs(v), 0.0)
        is_forward_pref = self.motion_preference == 1
        is_backward_pref = self.motion_preference == -1
        is_forward_enforce = self.motion_preference == 2
        is_backward_enforce = self.motion_preference == -2
        penalty = jnp.where(is_forward_pref, forward_penalty,
                           jnp.where(is_backward_pref, backward_penalty,
                           jnp.where(is_forward_enforce | is_backward_enforce, 0.0, 0.0)))
        return penalty

    @partial(jax.jit, static_argnums=(0, 3))
    def apply_guidance(self, q_proposed, step_size=0.05, max_steps=5):
        if not self.enable_guidance:
            return q_proposed
        def guidance_step(x, _):
            grad = jax.grad(self._guidance_function)(x)
            grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
            x_new = x - step_size * grad
            x_new = x_new.at[0].set(jnp.clip(x_new[0], -100.0, 100.0))
            x_new = x_new.at[1].set(jnp.clip(x_new[1], -100.0, 100.0))
            # Clip all angles conservatively
            x_new = x_new.at[2:].set(jnp.clip(x_new[2:], -2*jnp.pi, 2*jnp.pi))
            x_new = jnp.where(jnp.isfinite(x_new), x_new, x)
            return x_new, None
        x_final, _ = jax.lax.scan(guidance_step, q_proposed, None, length=max_steps)
        return x_final

    @partial(jax.jit, static_argnums=(0,))
    def get_last_trailer_position(self, x):
        px, py, theta0 = x[0], x[1], x[2]
        thetas = x[3: 3 + self.num_trailers]
        # Initial hitch at tractor rear
        hitch = jnp.array([px - self.lh * jnp.cos(theta0), py - self.lh * jnp.sin(theta0)])
        # Carry: (prev_hitch_x, prev_hitch_y, last_center_x, last_center_y)
        def body(i, carry):
            prev_hx, prev_hy, _, _ = carry
            theta_k = thetas[i]
            Lk = self.trailer_lengths[i]
            lf_k = self.trailer_front_offsets[i]
            # center at distance Lk behind hitch
            cx = prev_hx - Lk * jnp.cos(theta_k)
            cy = prev_hy - Lk * jnp.sin(theta_k)
            # next hitch at front offset lf_k from center
            next_hx = cx + lf_k * jnp.cos(theta_k)
            next_hy = cy + lf_k * jnp.sin(theta_k)
            return (next_hx, next_hy, cx, cy)
        # Initialize carry
        init = (hitch[0], hitch[1], hitch[0], hitch[1])
        final = jax.lax.fori_loop(0, self.num_trailers, body, init)
        last_cx, last_cy = final[2], final[3]
        return jnp.array([last_cx, last_cy])

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q):
        # Position-only cost, choose tractor vs last trailer based on preference (0=min of both)
        px, py, theta0 = q[0], q[1], q[2]
        tractor_pos = jnp.array([px + self.l1 * jnp.cos(theta0), py + self.l1 * jnp.sin(theta0)])
        last_trailer_pos = self.get_last_trailer_position(q)
        tractor_goal = self.xg[:2]
        last_trailer_goal = self.get_last_trailer_position(self.xg)
        d_tr = jnp.linalg.norm(tractor_pos - tractor_goal)
        d_la = jnp.linalg.norm(last_trailer_pos - last_trailer_goal)
        if self.motion_preference == 1 or self.motion_preference == 2:
            d = d_tr
        elif self.motion_preference == -1 or self.motion_preference == -2:
            d = d_la
        else:
            d = jnp.minimum(d_tr, d_la)
        return 1.0 - (jnp.clip(d, 0., self.reward_threshold) / self.reward_threshold) ** 2

    @partial(jax.jit, static_argnums=(0,))
    def check_hitch_violation(self, x):
        # Check all consecutive angle differences against phi_max
        theta0 = x[2]
        thetas = x[3: 3 + self.num_trailers]
        wrap_pi = lambda a: (a + jnp.pi) % (2.*jnp.pi) - jnp.pi
        # First hitch (tractor-trailer1)
        h0 = jnp.abs(wrap_pi(theta0 - thetas[0])) > self.phi_max
        def body(carry, data):
            prev_theta, violation = carry
            theta_k = data
            v = violation | (jnp.abs(wrap_pi(prev_theta - theta_k)) > self.phi_max)
            return (theta_k, v), v
        (last_theta, v_end), _ = jax.lax.scan(lambda c, d: body(c, d), (thetas[0], h0), thetas[1:])
        return v_end

    @partial(jax.jit, static_argnums=(0,))
    def get_tractor_trailer_rectangles_all(self, x):
        # Compute tractor rectangle and all trailer rectangles
        px, py, theta0 = x[0], x[1], x[2]
        # Tractor
        tractor_center_x = px + 0.5 * (self.l1 + self.lf1 - self.lr) * jnp.cos(theta0)
        tractor_center_y = py + 0.5 * (self.l1 + self.lf1 - self.lr) * jnp.sin(theta0)
        tractor_half_length = (self.l1 + self.lf1 + self.lr) / 2
        tractor_half_width = self.tractor_width / 2
        tractor_local = jnp.array([[-tractor_half_length, -tractor_half_width],
                                   [-tractor_half_length, tractor_half_width],
                                   [tractor_half_length,  tractor_half_width],
                                   [tractor_half_length, -tractor_half_width]])
        cos0, sin0 = jnp.cos(theta0), jnp.sin(theta0)
        R0 = jnp.array([[cos0, -sin0], [sin0, cos0]])
        tractor_corners = (tractor_local @ R0.T) + jnp.array([tractor_center_x, tractor_center_y])
        # Hitches and trailer centers iteratively from front hitch
        thetas = x[3: 3 + self.num_trailers]
        hitch = jnp.array([px - self.lh * jnp.cos(theta0), py - self.lh * jnp.sin(theta0)])
        def loop(i, acc):
            prev_h, corners_list = acc
            theta_k = thetas[i]
            Lk = self.trailer_lengths[i]
            lf_k = self.trailer_front_offsets[i]
            lr_k = self.trailer_rear_offsets[i]
            s_center = 0.5 * (lf_k - lr_k) - Lk
            cx = prev_h[0] + s_center * jnp.cos(theta_k)
            cy = prev_h[1] + s_center * jnp.sin(theta_k)
            half_len = 0.5 * (lf_k + lr_k)
            half_w = self.trailer_width / 2
            local = jnp.array([[-half_len, -half_w], [-half_len, half_w], [half_len, half_w], [half_len, -half_w]])
            c, s = jnp.cos(theta_k), jnp.sin(theta_k)
            R = jnp.array([[c, -s], [s, c]])
            corners = (local @ R.T) + jnp.array([cx, cy])
            # next hitch at rear face minus lh along -theta
            rear_x = cx - half_len * jnp.cos(theta_k)
            rear_y = cy - half_len * jnp.sin(theta_k)
            next_h = jnp.array([rear_x - self.lh * jnp.cos(theta_k), rear_y - self.lh * jnp.sin(theta_k)])
            corners_list = corners_list.at[i].set(corners)
            return (next_h, corners_list)
        corners_out = jnp.zeros((self.num_trailers, 4, 2))
        _, corners_out = jax.lax.fori_loop(0, self.num_trailers, loop, (hitch, corners_out))
        return tractor_corners, corners_out

    @partial(jax.jit, static_argnums=(0,))
    def check_obstacle_collision(self, x, obs_circles, obs_rectangles):
        tractor_corners, trailer_corners_all = self.get_tractor_trailer_rectangles_all(x)
        collision = False
        # Circles
        if obs_circles.shape[0] > 0:
            def check_circle_for_trailer(corners):
                def per_circle(circle_obs):
                    center = circle_obs[:2]
                    radius = circle_obs[2]
                    return self.check_rectangle_circle_collision(corners, center, radius)
                return jnp.any(jax.vmap(per_circle)(obs_circles))
            # tractor against circles
            tractor_hit = jnp.any(jax.vmap(lambda c: self.check_rectangle_circle_collision(tractor_corners, c[:2], c[2]))(obs_circles))
            # any trailer against circles
            trailers_hit = jnp.any(jax.vmap(check_circle_for_trailer)(trailer_corners_all))
            collision = collision | tractor_hit | trailers_hit
        # Rectangles
        if obs_rectangles.shape[0] > 0:
            def check_rect_for_trailer(corners):
                def per_rect(rect_obs):
                    ox, oy, w, h, ang = rect_obs
                    half_w, half_h = w/2, h/2
                    local = jnp.array([[-half_w, -half_h], [-half_w, half_h], [half_w, half_h], [half_w, -half_h]])
                    c, s = jnp.cos(ang), jnp.sin(ang)
                    R = jnp.array([[c, -s], [s, c]])
                    obs_corners = (local @ R.T) + jnp.array([ox, oy])
                    return self.check_rectangle_rectangle_collision(corners, obs_corners)
                return jnp.any(jax.vmap(per_rect)(obs_rectangles))
            tractor_hit = jnp.any(jax.vmap(lambda r: check_rect_for_trailer(tractor_corners))(jnp.arange(1)))
            trailers_hit = jnp.any(jax.vmap(check_rect_for_trailer)(trailer_corners_all))
            collision = collision | tractor_hit | trailers_hit
        return collision

    @partial(jax.jit, static_argnums=(0,))
    def get_terminal_reward(self, q):
        px, py, theta0 = q[0], q[1], q[2]
        tractor_pos = jnp.array([px + self.l1 * jnp.cos(theta0), py + self.l1 * jnp.sin(theta0)])
        last_trailer_pos = self.get_last_trailer_position(q)
        tractor_goal = self.xg[:2]
        last_trailer_goal = self.get_last_trailer_position(self.xg)
        d_tr = jnp.linalg.norm(tractor_pos - tractor_goal)
        d_la = jnp.linalg.norm(last_trailer_pos - last_trailer_goal)
        d = jnp.where((self.motion_preference == 1) | (self.motion_preference == 2), d_tr,
                      jnp.where((self.motion_preference == -1) | (self.motion_preference == -2), d_la,
                                jnp.minimum(d_tr, d_la)))
        r_pos = 1.0 - (jnp.clip(d, 0., self.terminal_reward_threshold) / self.terminal_reward_threshold) ** 2
        return r_pos

    @property
    def observation_size(self):
        return 3 + self.num_trailers  # [px, py, theta0, ..., thetaN]

    @property
    def action_size(self):
        return 2
    @partial(jax.jit, static_argnums=(0,))
    def get_trailer_position(self, x):
        # Alias to last trailer
        return self.get_last_trailer_position(x) 

    # Rendering for all trailers
    def setup_animation_patches(self, ax):
        tractor_color = '#66CCFF'
        trailer_color = '#FFCC66'
        # Tractor
        tractor_full_length = self.l1 + self.lf1 + self.lr
        self.tractor_body = ax.add_patch(
            plt.Rectangle((-tractor_full_length/2, -self.tractor_width/2),
                         tractor_full_length, self.tractor_width,
                         linewidth=2, edgecolor='black', facecolor=tractor_color, alpha=0.7)
        )
        # Trailers
        self.trailer_bodies = []
        self.trailer_wheels = []
        # Two sets of hitch lines: front (prev hitch -> current trailer front), back (current trailer rear -> next hitch)
        self.trailer_hitch_lines_front = []
        self.trailer_hitch_lines_back = []
        wheel_width = 0.45
        wheel_length = 0.8
        for i in range(self.num_trailers):
            full_len = float(self.trailer_front_offsets[i] + self.trailer_rear_offsets[i])
            body = ax.add_patch(
                plt.Rectangle((-full_len/2, -self.trailer_width/2),
                             full_len, self.trailer_width,
                             linewidth=2, edgecolor='black', facecolor=trailer_color, alpha=0.7)
            )
            self.trailer_bodies.append(body)
            wheel = ax.add_patch(
                plt.Rectangle((-wheel_length/2, -wheel_width/2),
                             wheel_length, wheel_width,
                             edgecolor='black', facecolor='gray', alpha=0.9)
            )
            self.trailer_wheels.append(wheel)
        # Create hitch lines: N forward lines, N-1 back lines
        for _ in range(self.num_trailers):
            line, = ax.plot([], [], 'k-', linewidth=3, alpha=0.8)
            self.trailer_hitch_lines_front.append(line)
        for _ in range(max(self.num_trailers - 1, 0)):
            line, = ax.plot([], [], 'k-', linewidth=3, alpha=0.8)
            self.trailer_hitch_lines_back.append(line)
        # Tractor wheels
        self.tractor_wheels = []
        for _ in range(2):
            wheel = ax.add_patch(
                plt.Rectangle((-wheel_length/2, -wheel_width/2),
                             wheel_length, wheel_width,
                             edgecolor='black', facecolor='gray', alpha=0.9)
            )
            self.tractor_wheels.append(wheel)
        self.tractor_hitch_line, = ax.plot([], [], 'k-', linewidth=3, alpha=0.8)

    def render_rigid_body(self, x, u=None):
        px, py, theta0 = x[0], x[1], x[2]
        # Tractor body
        tractor_center_x = px + 0.5 * (self.l1 + self.lf1 - self.lr) * np.cos(theta0)
        tractor_center_y = py + 0.5 * (self.l1 + self.lf1 - self.lr) * np.sin(theta0)
        tform_tractor = (Affine2D().rotate(theta0).translate(tractor_center_x, tractor_center_y) + plt.gca().transData)
        # Tractor wheels
        tforms_tractor_wheels = [
            Affine2D().rotate(theta0).translate(px, py) + plt.gca().transData,
        ]
        steering_angle = 0.0
        if u is not None:
            steering_angle = u[1]
        tforms_tractor_wheels.append(
            Affine2D().rotate(theta0 + steering_angle).translate(px + self.l1*np.cos(theta0), py + self.l1*np.sin(theta0)) + plt.gca().transData
        )
        # Trailers positions
        thetas_raw = x[3:]
        if len(thetas_raw) < self.num_trailers:
            pad_val = thetas_raw[-1] if len(thetas_raw) > 0 else theta0
            thetas = np.concatenate([thetas_raw, np.full((self.num_trailers - len(thetas_raw),), pad_val)])
        else:
            thetas = thetas_raw[:self.num_trailers]
        n_act = self.num_trailers
        hitch_x = px - self.lh * np.cos(theta0)
        hitch_y = py - self.lh * np.sin(theta0)
        tforms_trailer_bodies = []
        tforms_trailer_wheels = []
        hitch_lines_front = []
        hitch_lines_back = []
        prev_hitch = np.array([hitch_x, hitch_y])
        for i in range(n_act):
            Lk = float(self.trailer_lengths[i])
            lf_k = float(self.trailer_front_offsets[i])
            lr_k = float(self.trailer_rear_offsets[i])
            th = thetas[i]
            # center from front hitch: s_center = (lf - lr)/2 - Lk
            s_center = 0.5 * (lf_k - lr_k) - Lk
            center_x = prev_hitch[0] + s_center * np.cos(th)
            center_y = prev_hitch[1] + s_center * np.sin(th)
            tforms_trailer_bodies.append(
                Affine2D().rotate(th).translate(center_x, center_y) + plt.gca().transData
            )
            tforms_trailer_wheels.append(
                Affine2D().rotate(th).translate(center_x, center_y) + plt.gca().transData
            )
            # next hitch at rear face minus lh
            half_len = 0.5 * (lf_k + lr_k)
            rear_x = center_x - half_len * np.cos(th)
            rear_y = center_y - half_len * np.sin(th)
            next_hx = rear_x - self.lh * np.cos(th)
            next_hy = rear_y - self.lh * np.sin(th)
            # Forward hitch line: prev hitch -> current trailer front
            front_x = center_x + half_len * np.cos(th)
            front_y = center_y + half_len * np.sin(th)
            hitch_lines_front.append(((prev_hitch[0], prev_hitch[1]), (front_x, front_y)))
            # Back hitch line: current trailer rear -> next hitch (only for non-last trailers)
            if i < n_act - 1:
                hitch_lines_back.append(((rear_x, rear_y), (next_hx, next_hy)))
            prev_hitch = np.array([next_hx, next_hy])
        # Tractor hitch line (rear axle to first hitch)
        tractor_hitch_line = ((px, py), (hitch_x, hitch_y))
        return {
            'tractor_body': tform_tractor,
            'tractor_wheels': tforms_tractor_wheels,
            'tractor_hitch_line': tractor_hitch_line,
            'trailer_bodies': tforms_trailer_bodies,
            'trailer_wheels': tforms_trailer_wheels,
            'trailer_hitch_lines_front': hitch_lines_front,
            'trailer_hitch_lines_back': hitch_lines_back,
        }

    def update_animation_patches(self, x, u=None):
        tf = self.render_rigid_body(x, u)
        # Tractor
        self.tractor_body.set_transform(tf['tractor_body'])
        for i, w in enumerate(self.tractor_wheels):
            w.set_transform(tf['tractor_wheels'][i])
        self.tractor_hitch_line.set_data([tf['tractor_hitch_line'][0][0], tf['tractor_hitch_line'][1][0]],
                                         [tf['tractor_hitch_line'][0][1], tf['tractor_hitch_line'][1][1]])
        # Trailers (iterate bodies/wheels)
        n_act = len(tf['trailer_bodies'])
        for i in range(n_act):
            self.trailer_bodies[i].set_transform(tf['trailer_bodies'][i])
            self.trailer_wheels[i].set_transform(tf['trailer_wheels'][i])
        # Forward hitch lines for each trailer (N lines)
        n_front = len(self.trailer_hitch_lines_front)
        for i in range(n_front):
            line_data = tf['trailer_hitch_lines_front'][i]
            self.trailer_hitch_lines_front[i].set_data([line_data[0][0], line_data[1][0]],
                                                       [line_data[0][1], line_data[1][1]])
        # Back hitch lines for non-last trailers (N-1 lines)
        n_back = len(self.trailer_hitch_lines_back)
        for i in range(n_back):
            line_data = tf['trailer_hitch_lines_back'][i]
            self.trailer_hitch_lines_back[i].set_data([line_data[0][0], line_data[1][0]],
                                                      [line_data[0][1], line_data[1][1]]) 
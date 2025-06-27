from .car2d import Car2d
from .tt2d import TractorTrailer2d

def get_env(env_name: str, case: str = "case1", dt=0.2, H=50, motion_preference=0, 
            collision_penalty=0.15, enable_collision_projection=False, hitch_penalty=0.10, 
            enable_hitch_projection=True, reward_threshold=25.0, ref_reward_threshold=5.0,
            max_w_theta=0.75, hitch_angle_weight=0.2, l1=3.23, l2=2.9, lh=1.15, 
            tractor_width=2.0, trailer_width=2.5, v_max=3.0, delta_max_deg=55.0,
            d_thr_factor=1.0, k_switch=2.5, steering_weight=0.05, preference_penalty_weight=0.05,
            heading_reward_weight=0.5, terminal_reward_threshold=10.0, terminal_reward_weight=1.0, ref_pos_weight=0.3, ref_theta1_weight=0.5, ref_theta2_weight=0.2):
    if env_name == "car2d":
        return Car2d()
    elif env_name == "tt2d":
        return TractorTrailer2d(case=case, dt=dt, H=H, motion_preference=motion_preference, 
                               collision_penalty=collision_penalty, enable_collision_projection=enable_collision_projection,
                               hitch_penalty=hitch_penalty, enable_hitch_projection=enable_hitch_projection,
                               reward_threshold=reward_threshold, ref_reward_threshold=ref_reward_threshold,
                               max_w_theta=max_w_theta, hitch_angle_weight=hitch_angle_weight, l1=l1, l2=l2, lh=lh,
                               tractor_width=tractor_width, trailer_width=trailer_width, v_max=v_max, delta_max_deg=delta_max_deg,
                               d_thr_factor=d_thr_factor, k_switch=k_switch, steering_weight=steering_weight,
                               preference_penalty_weight=preference_penalty_weight, heading_reward_weight=heading_reward_weight,
                               terminal_reward_threshold=terminal_reward_threshold, terminal_reward_weight=terminal_reward_weight, ref_pos_weight=ref_pos_weight, ref_theta1_weight=ref_theta1_weight, ref_theta2_weight=ref_theta2_weight)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

from .car2d import Car2d
from .tt2d import TractorTrailer2d

def get_env(env_name: str, case: str = "case1", dt=0.2, H=50, movement_preference=0, 
            collision_penalty=0.15, enable_collision_projection=False, hitch_penalty=0.10, 
            enable_hitch_projection=True, reward_threshold=25.0, ref_reward_threshold=5.0,
            max_w_theta=0.75, ref_pos_weight=0.3, ref_theta1_weight=0.5, ref_theta2_weight=0.2):
    if env_name == "car2d":
        return Car2d()
    elif env_name == "tt2d":
        return TractorTrailer2d(case=case, dt=dt, H=H, movement_preference=movement_preference, 
                               collision_penalty=collision_penalty, enable_collision_projection=enable_collision_projection,
                               hitch_penalty=hitch_penalty, enable_hitch_projection=enable_hitch_projection,
                               reward_threshold=reward_threshold, ref_reward_threshold=ref_reward_threshold,
                               max_w_theta=max_w_theta, ref_pos_weight=ref_pos_weight, ref_theta1_weight=ref_theta1_weight,
                               ref_theta2_weight=ref_theta2_weight)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

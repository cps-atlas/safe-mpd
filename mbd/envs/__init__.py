from brax import envs as brax_envs

from .car2d import Car2d
from .tt2d import TractorTrailer2d

def get_env(env_name: str, case: str = "case1", dt=0.2, H=50):
    if env_name == "car2d":
        return Car2d()
    elif env_name == "tt2d":
        return TractorTrailer2d(case=case, dt=dt, H=H)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

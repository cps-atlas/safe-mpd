from brax import envs as brax_envs

from .car2d import Car2d
from .tt2d import TractorTrailer2d

def get_env(env_name: str):
    if env_name == "car2d":
        return Car2d()
    elif env_name == "tt2d":
        return TractorTrailer2d()
    else:
        raise ValueError(f"Unknown environment: {env_name}")

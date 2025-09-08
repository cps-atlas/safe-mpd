
## Installation
1. Install the docker and nvidia container toolkit (cuda), and check whether the installation is complete using the below command.

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

2. Build the docker container.

```bash
# give sudo permission to docker (once need to run once)
sudo usermod -aG docker $USER (then, reboot your computer)
# build docker container
docker compose build
# give x11 (screen) access to docker
xhost +local:docker
docker compose up -d
```

3. Execute docker container and run any code.
```bash
docker exec -it jax_dev bash
python $your_code$.py
```

4. (Optional, only when you run `python tests/tune_mbd.py`) Run this line inside the docer. Since optuna's GP sampler requires torch (which deafult to require numpy==1.x), and jax[cuda12] requires numpy>=2, this is the lazy solution I found to fix the version issue. If you run `uv add` separately, then the version breaks. But note, this will result in version incompatible issue between the cuda and jax, slowing down the computation speed. 
```bash
uv add torch==2.3.1 "jax[cuda12]"
```

## Getting Started

### First run (quick start)

1) Enter the container/shell (see Installation), then from the repo root:

```bash
python mbd/planners/mbd_planner.py
```

This launches the diffusion planner with the default `MBDConfig`, renders a rollout, and saves arrays under `src/safe_mbd/results/{env_name}/`.

### Supported dynamics and how to select them

| Environment | Description | n_state | n_control |
|-------------|-------------|---------|-----------|
| **kinematic_bicycle2d** | simple bicycle model | 3 | 2 |
| **tt2d** | kinematic tractor–trailer (single trailer) | 4 | 2 |
| **acc_tt2d** | acceleration-controlled tractor–trailer (`u`: acceleration and steering rate) | 6 | 2 |

Select via `MBDConfig.env_name` and `MBDConfig.num_trailers`:

```python
# example
from mbd.planners.mbd_planner import MBDConfig, run_diffusion
import mbd

config = MBDConfig(
    env_name="tt2d",      # "kinematic_bicycle2d", "tt2d", "acc_tt2d", or "n_trailer2d"
    case="parking",       # "parking" or "navigation"
    num_trailers=1,        # 0 → bicycle; 1 → single trailer; >=2 → n-trailer
)

env = mbd.envs.get_env(
    config.env_name,
    case=config.case,
    dt=config.dt,
    H=config.Hsample,
    num_trailers=config.num_trailers,
)

run_diffusion(args=config, env=env)
```

Notes:
- For the kinematic bicycle, either set `env_name="kinematic_bicycle2d"` or `env_name="tt2d"` with `num_trailers=0`.
- For n‑trailer, use `env_name="tt2d"` with `num_trailers >= 2`, or `env_name="n_trailer2d"`.

### Setting start/goal (parking scenario)

With `case="parking"`, set start and goal succinctly:

```python
import jax.numpy as jnp

# Initial pose using geometric convenience params
env.set_init_pos(dx=2.0, dy=1.0, theta1=0.0, theta2=0.0)

# Goal pose: update orientation only (position comes from parking target)
env.set_goal_pos(theta1=-jnp.pi/2, theta2=-jnp.pi/2)
```

- **dx**: distance from the target parking space center to the tractor front face (x‑direction)
- **dy**: distance from the parking lot entrance line (y‑direction)
- **theta1/theta2**: tractor/trailer headings

Parking targets are defined in `mbd/envs/env.py` via `Env.get_default_parking_config()['target_spaces']`. To choose a different space pair:

```python
from mbd.envs.env import Env

parking = Env(case="parking")
parking.parking_config['target_spaces'] = [4, 12]  # example pair
env = mbd.envs.get_env(config.env_name, case="parking", env_config=parking, dt=config.dt, H=config.Hsample)
```

The custom environment for general navigation can be set easily with `case="navigation"`. For custom maps, add rectangles with `env.set_rectangle_obs([...], coordinate_mode="left-top"|"center", padding=...)`, and set initial and goal position with `env.set_init_pos(x=x, yy=y, theta1=theta1, theta2=theta2)` and `env.set_goal_pos(x=x, yy=y, theta1=theta1, theta2=theta2)`.

### Safety methods and baselines

Toggle safety strategies in `MBDConfig`:

- **Shielded rollout (default)**: using shield algorithm to guarantee safety
  - `enable_shielded_rollout_collision=True`
  - `enable_shielded_rollout_hitch=True`
- **Guidance** (gradient-based correction): `enable_guidance=True`
- **Projection** (control projection): `enable_projection=True`
- **Original MBD w/ naive penalty**: set all three off:
  - `enable_shielded_rollout_collision=False`
  - `enable_shielded_rollout_hitch=False`
  - `enable_guidance=False`, `enable_projection=False`

### Running tests

```bash
# Default tests
python tests/test_planners/run_tests.py

# Single test with visualization
python tests/test_planners/run_tests.py --single test_parking_basic_forward --visualize

# Acceleration-controlled TT or kinematic TT
python tests/test_planners/run_tests.py --acc
python tests/test_planners/run_tests.py --kinematic
```

See [tests/test_planners/README.md](`tests/test_planners/README.md`) for details and scenario lists.

### Utilities: statistics and tuning

- **tests/stat_mbd.py**: batch evaluation over diverse initial conditions; reports success rates, errors, and computation time. Optionally visualize heat maps.
  - Run: `python tests/stat_mbd.py`
- **tests/tune_mbd.py**: `Optuna`-based hyperparameter optimization (GP sampler + MedianPruner), optionally logs to W&B; optimizes success rate using the statistical evaluator.
  - Run: `python tests/tune_mbd.py`

## Reference
- Original repository: https://github.com/LeCAR-Lab/model-based-diffusion

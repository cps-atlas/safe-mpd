# safe-mpd

This repository contains the implementation of the Safe Model Predictive Diffusion (Safe MPD) algorithm, a training-free diffusion planner for generating provably safe and kinodynamically feasible trajectories. By enforcing feasibility and safety on every sample throughout the denoising process, our method avoids the common pitfalls of post-processing corrections, such as computational intractability and loss of feasibility. Through a parallelization in GPU, our method achieves sub-second planning times even on challenging, non-convex problems. Please see our paper ["Safe Model Predictive Diffusion with Shielding"](https://www.taekyung.me/safe-mpd) for more details.


<div align="center">

  <img src="https://github.com/user-attachments/assets/0560a3cc-ec7b-4d83-8f26-27c61ade3a23"  height="250px">  <img src="https://github.com/user-attachments/assets/a89e9580-5169-4a0a-9814-98ba844ba87e"  height="250px"> 

<div align="center">

[[Homepage]](https://www.taekyung.me/safe-mpd)
[[Arxiv]]()
[[Video]](https://youtu.be/DQBeybU7EYI)
[[Research Group1]](https://dasc-lab.github.io/)
[[Research Group2]](https://amrd.toyota.com/division/trina/)

</div>
</div>


## Features

- **Training-free model-based diffusion planner** that generates provably safe and kinodynamically feasible trajectories without requiring offline training or demonstration data.
- **GPU-accelerated parallelization** using JAX, achieving sub-second planning times even for challenging non-convex problems with thousands of samples.
- **Shielded rollout mechanism** that enforces safety constraints (collision avoidance and hitch angle limits) at every diffusion step, guaranteeing feasibility throughout the denoising process.
- **Multiple robot dynamics support**, including kinematic bicycle, single and multi-trailer systems (n-trailer), and dynamic tractor-trailer models with configurable physical parameters.
- **Parking and navigation scenarios** with configurable obstacle environments, supporting complex maneuver planning tasks like parallel parking and obstacle avoidance.



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

<div align="center">

  <img src="https://github.com/user-attachments/assets/f71e5f45-88aa-4cd9-873b-56ee7f40bc6d"  height="250px">  <img src="https://github.com/user-attachments/assets/b841be83-db95-4df0-998f-b653d0f17b03"  height="250px"> 

</div>
    
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

### Custom navigation environment

For custom navigation scenarios with user-defined obstacles, use `case="navigation"`. A complete minimal example is available at `tests/test_mbd.py`. Run it with:

```bash
python tests/test_mbd.py
```

Here's a quick overview of the key steps:

```python
import jax.numpy as jnp
import mbd
from mbd.planners.mbd_planner import MBDConfig, run_diffusion
from mbd.envs.env import Env

# 1. Create configuration
config = MBDConfig(
    $config_info$
)

# 2. Create base environment
env_config = Env(case="navigation")

# 3. Create robot environment
env = mbd.envs.get_env(
    config.env_name,
    case=config.case,
    env_config=env_config
)

# 4. Add obstacles - rectangles:[x_center, y_center, width, height, rotation]
#                    circles: [x_center, y_center, radius]
env.set_rectangle_obs([
      $obs_info$
    ], coordinate_mode="center", padding=0.0)

env.set_circle_obs([
      $obs_info$
    ], padding=0.0)

# 5. Set initial and goal positions
env.set_init_pos($init_pose$)
env.set_goal_pos($goal_pose$)

# 6. Run the planner
_ = run_diffusion(args=config, env=env)
```


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
  
|      Benchmark results form `stat_mbd.py`            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/4e55e8e5-b9a5-41a2-b7d8-13b2e1ec1bd8"  height="350px"> |

- **tests/tune_mbd.py**: `Optuna`-based hyperparameter optimization (GP sampler + MedianPruner), optionally logs to W&B; optimizes success rate using the statistical evaluator.
  - Run: `python tests/tune_mbd.py`
    
 |      W&B dashboard example when running `tune_mbd.py`            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/559f00cb-446a-4d48-9f50-68bd864d2d15"  height="350px"> |
 
## Citing

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2025safempd, 
    author    = {Kim, Taekyung and Majd, Keyvan and Okamoto, Hideki and Hoxha, Bardh and Panagou, Dimitra and Fainekos, Georgios},
    title     = {Safe Model Predictive Diffusion with Shielding},
    booktitle = {arXiv preprint arXiv.25},
    shorttitle = {Safe MPD},
    year      = {2025}
}
```

## Reference
This repository was built based on the [implementation of model-based diffusion](https://github.com/LeCAR-Lab/model-based-diffusion). Thanks for the great work of [jc-bao](https://github.com/jc-bao) and [iscoyizj](https://github.com/iscoyizj).

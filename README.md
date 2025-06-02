# Introduction 
The focus of this project is on the control and planning of tractor-trailer systems, which have significant applications ranging from towing trucks to warehouse mobile trailers. These systems present substantial control challenges due to their nonlinear dynamics and, in particular, the inherent instability associated with backward motion. Ensuring safe and effective operation of such systems requires addressing three core challenges: (A) rapidly generating kinodynamically feasible global paths, (B) reliably tracking reference paths while guaranteeing safety under complex dynamic constraints, and (C) preventing jack-knifing behavior.

During Summer 2025, I will lead a project addressing the first challenge: developing fast and kinodynamically feasible path planning methods that enable existing sampling-based controllers to effectively track planned trajectories. The primary objective is to develop a path planner that has a significantly better computational efficiency compared to the current internal implementation for the trailer systems, which is hybrid A* planner. This project will also explore the theoretical connections between generative model-based method (ex. model-based diffusion and flow matching) and set invariance theory (including control barrier functions). Furthermore and more importantly, we aim to implement these algorithms using the JAX library in Python to achieve real-time, sub-second execution on on-board computing hardware.

# Getting Started
## Installation
Note, jax is sensitive to CUDA version. In summer 2025, I tested everything with jax==0.6.1 and CUDA 12.8 with docker. 
In the future, this docker build might not work with latest jax version.

1. After Installing docker (somehow, just follow the website), check the nvidia container has been setup properly. If below cmd success, then you good.

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

## Run examples
To be updated. But refer to the below readme (from the original repo)

To run model-based diffusion to optimize a trajectory, run the following command:

```bash
cd mbd/planners
python mbd_planner.py --env_name $ENV_NAME
```

where `$ENV_NAME` is the name of the environment, you can choose from `hopper`, `halfcheetah`, `walker2d`, `ant`, `humanoidrun`, `humanoidstandup`, `humanoidtrack`, `car2d`, `pushT`.

To run model-based diffusion combined with demonstrations, run the following command:

```bash
cd mbd/planners
python mbd_planner.py --env_name $ENV_NAME --enable_demo
```

Currently, only the `humanoidtrack`, `car2d` support demonstrations.

To run multiple seeds, run the following command:

```bash
cd mbd/scripts
python run_mbd.py --env_name $ENV_NAME
```

To visualize the diffusion process, run the following command:

```bash
cd mbd/scripts
python vis_diffusion.py --env_name $ENV_NAME
```

Please make sure you have run the planner first to generate the data.

### Other Baselines

To run RL-based baselines, run the following command:

```bash
cd mbd/rl
python train_brax.py --env_name $ENV_NAME
```

To run other zeroth order trajectory optimization baselines, run the following command:

```bash
cd mbd/planners
python path_integral.py --env_name $ENV_NAME --mode $MODE
```

where `$MODE` is the mode of the planner, you can choose from `mppi`, `cem`, `cma-es`.

## Reference
- Original repository: https://github.com/LeCAR-Lab/model-based-diffusion

FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    YOUR_ENV=default_value \
    PATH="/root/.local/bin:${PATH}"
    
# Basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends\
    curl \
    wget \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 
    

# Set the default Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
COPY pyproject.toml ./ 
COPY mbd/ mbd/

# ENV PYTHONPATH="/home:/home/cbfkit:/home/cbfkit/src:${PYTHONPATH}"

# Install dependencies
ENV PATH="/workspace/.venv/bin:${PATH}"
RUN uv venv && uv pip install .
RUN uv sync

RUN uv add shapely
RUN uv add "jax[cuda12]"
RUN uv add matplotlib gymnasium ipykernel gym pandas seaborn imageio control tqdm tyro meshcat sympy gymnax jax distrax gputil optuna wandb

RUN uv sync

# enable display
RUN apt update && apt install -y \
    x11-apps libgl1 libx11-6 libxext6 libxrender1 libsm6 libxft2 \
    python3-tk \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /workspace/.venv/bin/activate" >> /root/.bashrc

# # JAX and JAXlib with CUDA
# #RUN pip install --upgrade pip --break-system-packages
# RUN pip install --upgrade "jax[cuda12]" --break-system-packages # jaxlib will match CUDA 12.8 

# # More packages
# RUN pip install matplotlib gymnasium ipykernel brax --break-system-packages

# # packages for model-based diffusion
# RUN pip install gym pandas seaborn matplotlib imageio control tqdm tyro meshcat sympy gymnax jax distrax gputil jaxopt --break-system-packages
WORKDIR /workspace
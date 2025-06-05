FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04

# Basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-tk \
    git \
    wget \
    nano \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    x11-apps \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Set the default Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# JAX and JAXlib with CUDA
#RUN pip install --upgrade pip --break-system-packages
RUN pip install --upgrade "jax[cuda12]" --break-system-packages # jaxlib will match CUDA 12.8 

# More packages
RUN pip install matplotlib gymnasium ipykernel brax --break-system-packages

# packages for model-based diffusion
RUN pip install gym pandas seaborn matplotlib imageio control tqdm tyro meshcat sympy gymnax jax distrax gputil jaxopt --break-system-packages

WORKDIR /workspace
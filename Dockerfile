# FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04
FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    YOUR_ENV=default_value \
    PATH="/workspace/.venv/bin:/root/.local/bin:${PATH}"

# 3.13 has compatibility issues
# FROM python:3.11
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ffmpeg \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the default Python
# -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv /usr/local/bin/

WORKDIR /workspace

# Copy metadata first for better Docker layer caching
COPY pyproject.toml ./

# Ensure JAX CUDA wheels are discoverable
ENV PIP_FIND_LINKS=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Create virtualenv and install dependencies using uv sync.
# This will generate uv.lock if it doesn't exist and install packages.
RUN uv venv && uv sync

# Copy the project code and install the package in editable mode
COPY mbd/ mbd/
RUN uv pip install -e .

# # enable display
# Install GUI-related dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    x11-apps libgl1 libx11-6 libxext6 libxrender1 libsm6 libxft2 \
    python3-tk \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add venv to bashrc for interactive sessions
RUN echo "source /workspace/.venv/bin/activate" >> /root/.bashrc

WORKDIR /workspace
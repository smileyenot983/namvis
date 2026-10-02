FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL description="Docker container for NAMVIS with dependencies installed. CUDA VERSION"

ENV DEVICE="cuda"
ENV PYTHONNOUSERSITE=1
ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        git \
        wget \
        ninja-build \
        python3-pip \
        python3-dev \
        python3-packaging \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip


RUN python3 -m pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

COPY ./docker/requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir \
    -r /workspace/requirements.txt

RUN MAX_JOBS=4 python3 -m pip install --no-cache-dir \
    flash-attn==2.8.3.post1 \
    --no-build-isolation

ARG UID
ARG GID

# Create a group and user with your exact IDs
RUN groupadd -g ${GID} ramil && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash ramil

# (Optional) Switch to this user automatically
USER ramil

WORKDIR /workspace


# to install smth as root from inside container:
# docker exec -u root -it rkhafizov.infinity_container bash
# python3 -m pip install some-package
# python -m pip install -U trackio huggingface_hub
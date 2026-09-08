# syntax=docker/dockerfile:1.6
#
# RecGen inference image.
#
# Build:
#   docker build -t recgen-inference .
#
# Run (requires NVIDIA Container Toolkit):
#   docker run --rm --gpus all -v $PWD/data:/data recgen-inference \
#       python -c "from recgen_inference import build_recgen; print(build_recgen.list_checkpoints())"

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: builder — compiles spconv, flash-attn, diff-gaussian-rasterization
# ──────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        git build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools

# Torch first so downstream CUDA extensions link against it.
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Project dependencies (without the source — leverage layer cache).
COPY pyproject.toml README.md /tmp/recgen/
RUN pip install --no-deps --no-build-isolation -e /tmp/recgen || true \
    && pip install \
        numpy scipy Pillow trimesh safetensors huggingface-hub easydict tqdm \
        plyfile opencv-python-headless

# Training dependencies (recgen_training package)
RUN pip install \
        "webdataset>=0.2" braceexpand "kornia>=0.7,<0.8" "einops>=0.7" \
        "tensorboard>=2.14" "wandb>=0.16"

# Evaluation dependencies (recgen_eval package: HB / ArtVIP paper evals)
RUN pip install \
        "pandas>=2.0" "PyYAML>=6.0" "scikit-learn>=1.3" \
        "open3d>=0.17,<0.18" "imageio>=2.30" "imageio-ffmpeg>=0.5" pyrender

# CUDA extensions. flash-attn is optional — SDPA fallback exists.
RUN pip install spconv-cu120
RUN pip install flash-attn==2.6.3 --no-build-isolation || \
    echo "flash-attn build failed — falling back to PyTorch SDPA"

RUN git clone --recursive --depth 1 \
        https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
        /tmp/dgr \
    && sed -i 's|#include <iostream>|#include <iostream>\n#include <cstdint>|' \
        /tmp/dgr/cuda_rasterizer/rasterizer_impl.h \
    && pip install --no-build-isolation /tmp/dgr \
    && rm -rf /tmp/dgr

# GLB texture export extras: xatlas, pyvista, pymeshfix, igraph, utils3d, open3d, pyrender
RUN pip install \
        xatlas pyvista pymeshfix igraph open3d pyrender \
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# nvdiffrast (textured GLB export) — built from source against the installed torch
RUN git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
    && pip install --no-build-isolation /tmp/nvdiffrast \
    && rm -rf /tmp/nvdiffrast

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — CUDA devel base (JIT extension builds) + venv from builder
# ──────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 ca-certificates libgomp1 libglib2.0-0 \
        libgl1 libegl1 libgles2 libglvnd0 \
        libx11-6 libxext6 libxrender1 libxi6 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Non-root user; /cache and /data are meant to be mounted.
RUN useradd --create-home --uid 1000 recgen \
    && mkdir -p /cache/huggingface /data /workspace \
    && chown -R recgen:recgen /cache /data /workspace

WORKDIR /workspace
COPY --chown=recgen:recgen . /workspace
RUN pip install --no-deps -e /workspace

USER recgen
CMD ["python", "-c", "from recgen_inference import build_recgen; print(build_recgen.list_checkpoints())"]

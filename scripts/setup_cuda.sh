#!/bin/bash
# Install CUDA-dependent packages for RecGen inference.
#
# Usage:
#   bash setup_cuda.sh              # spconv + flash-attn + diff-gaussian-rasterization
#   bash setup_cuda.sh --nvdiffrast # build nvdiffrast from source
#   bash setup_cuda.sh --all        # everything

set -e

EXTDIR="${RECGEN_EXT_DIR:-/tmp/recgen_extensions}"
mkdir -p "$EXTDIR"

# ── Parse arguments ──────────────────────────────────────────────────────
BUILD_NVDIFFRAST=false
BUILD_DIFF_GAUSSIAN=false
ALL=false

if [ "$#" -eq 0 ]; then
    # Default: just spconv + flash-attn
    true
fi

for arg in "$@"; do
    case "$arg" in
        --nvdiffrast)     BUILD_NVDIFFRAST=true ;;
        --diff-gaussian)  BUILD_DIFF_GAUSSIAN=true ;;
        --all)            ALL=true ;;
        -h|--help)
            echo "Usage: $0 [--nvdiffrast] [--diff-gaussian] [--all]"
            exit 0 ;;
    esac
done

# ── Build diff_gaussian_rasterization from source ───────────────────────
_build_diff_gaussian() {
    echo ""
    echo "Building diff_gaussian_rasterization from source (mip-splatting fork)..."
    if [ ! -d "$EXTDIR/mip-splatting" ]; then
        git clone --recursive https://github.com/autonomousvision/mip-splatting.git \
            "$EXTDIR/mip-splatting"
    fi
    pip install --no-build-isolation \
        "$EXTDIR/mip-splatting/submodules/diff-gaussian-rasterization"
    echo "diff_gaussian_rasterization installed."
}

if [ "$BUILD_DIFF_GAUSSIAN" = true ] && [ "$ALL" = false ]; then
    _build_diff_gaussian
    exit 0
fi

# ── nvdiffrast ───────────────────────────────────────────────────────────
if [ "$ALL" = true ] || [ "$BUILD_NVDIFFRAST" = true ]; then
    echo "── Building nvdiffrast ──"
    if [ ! -d "$EXTDIR/nvdiffrast" ]; then
        git clone https://github.com/NVlabs/nvdiffrast.git "$EXTDIR/nvdiffrast"
    fi
    pip install --no-build-isolation "$EXTDIR/nvdiffrast"
    echo "nvdiffrast installed successfully."
    # If only --nvdiffrast was requested, exit
    if [ "$ALL" = false ]; then
        exit 0
    fi
fi

# ── Detect CUDA version ─────────────────────────────────────────────────
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "nvcc not found. Trying to detect from torch..."
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo "PyTorch CUDA: $CUDA_VERSION"
fi

# ── Install spconv ───────────────────────────────────────────────────────
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

if [ "$CUDA_MAJOR" = "11" ]; then
    echo "Installing spconv-cu118..."
    pip install spconv-cu118
elif [ "$CUDA_MAJOR" = "12" ]; then
    echo "Installing spconv-cu120..."
    pip install spconv-cu120
else
    echo "WARNING: Unknown CUDA version $CUDA_VERSION, trying spconv-cu120..."
    pip install spconv-cu120
fi

# ── Install flash-attn (optional) ───────────────────────────────────────
echo ""
echo "Installing flash-attn (optional, for faster inference)..."
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "flash-attn installation failed (GPU arch may not be supported)."
    echo "Falling back to PyTorch native SDPA attention."
}

_build_diff_gaussian

echo ""
echo "Done! Core CUDA dependencies installed."

#!/bin/bash -l

# ==============================================================================
# 🛠️  TopoBench Environment Setup Script
# ==============================================================================
# usage: bash uv_env_setup.sh [CUDA_VERSION]
#
# Arguments:
#   CUDA_VERSION (Optional): Specifies the compute platform.
#
# Available Options:
#   cpu    : Installs CPU-only versions (Default). Best for macOS or non-GPU machines.
#   cu118  : Installs CUDA 11.8 binaries.
#   cu121  : Installs CUDA 12.1 binaries.
#
# Examples:
#   bash uv_env_setup.sh          # Installs CPU version
#   bash uv_env_setup.sh cu121    # Installs CUDA 12.1 version
# ==============================================================================

# --- Configuration ---
TORCH_VERSION="2.3.0"
CUDA_PLATFORM="${1:-cpu}" # Defaults to 'cpu' if no argument is provided

# Visual Header
echo ""
echo "======================================================="
echo "🚀 Initializing TopoBench Environment Setup"
echo "======================================================="
echo "📌 Torch Version : ${TORCH_VERSION}"
echo "📌 Platform      : ${CUDA_PLATFORM}"
echo "======================================================="
echo ""

# Clean up
echo "🧹 Cleaning up old environment artifacts..."
rm -rf .venv uv.lock ~/.cache/pre-commit
echo "✅ Cleanup complete."

echo "🐍 Creating virtual environment (Python 3.11)..."
uv venv --python 3.11
source .venv/bin/activate

# ------------------------------------------------------------------------------
# Step 1: Install PyTorch Core
# ------------------------------------------------------------------------------
echo ""
echo "📦 [Step 1/4] Installing PyTorch Core..."
echo "   Source: https://download.pytorch.org/whl/${CUDA_PLATFORM}"
uv pip install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_PLATFORM}"

# ------------------------------------------------------------------------------
# Step 2: Install PyTorch Geometric Extensions
# ------------------------------------------------------------------------------
echo ""
echo "📦 [Step 2/4] Installing PyG Extensions (Scatter, Sparse, Cluster)..."
echo "   Source: https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_PLATFORM}.html"
uv pip install torch-scatter torch-sparse torch-cluster \
    --find-links "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_PLATFORM}.html"

# ------------------------------------------------------------------------------
# Step 3: Install TopoBench (Editable Mode)
# ------------------------------------------------------------------------------
# Strategy: We lock the torch version we just installed to prevent uv from 
# seeing the 'torch==2.3.0' requirement in pyproject.toml and upgrading 
# it to the standard PyPI version (which defaults to CUDA on Linux).
echo ""
echo "📦 [Step 3/4] Installing TopoBench Project..."

# 1. Detect current version
INSTALLED_TORCH=$(python -c 'import torch; print(torch.__version__)')
echo "🔒 Locking Torch to currently installed version: ${INSTALLED_TORCH}"

# 2. Create constraint file
echo "torch==${INSTALLED_TORCH}" > torch_constraint.txt

# 3. Install with constraints
uv pip install -e '.[all]' \
    --constraint torch_constraint.txt \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_PLATFORM}"

rm torch_constraint.txt
echo "✅ Project installed successfully."

# ------------------------------------------------------------------------------
# Step 4: Configure Git Hooks
# ------------------------------------------------------------------------------
echo ""
echo "🔧 [Step 4/4] Configuring Development Hooks..."
uv pip install pre-commit
# Important: We run 'pre-commit' directly (not 'uv run') to avoid auto-sync triggers
pre-commit install

# ------------------------------------------------------------------------------
# Final Summary
# ------------------------------------------------------------------------------
echo ""
echo "======================================================="
echo "🎉 Setup Complete!"
echo "======================================================="
python -c "import torch; print(f'✅ Torch Version : {torch.__version__}'); print(f'✅ CUDA Available: {torch.cuda.is_available()}'); print(f'✅ CUDA Version  : {torch.version.cuda}')"
echo "======================================================="
echo "To activate this environment later, run:"
echo "   source .venv/bin/activate"
echo ""
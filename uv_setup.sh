#!/bin/bash -l
# Setup script for TopoBench using uv package manager
# This script installs dependencies and configures PyTorch with the appropriate CUDA version

set -e  # Exit on error

echo "🚀 Setting up TopoBench with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ uv version: $(uv --version)"

# Configuration for PyTorch and CUDA
# Note that not all combinations of torch and CUDA are available
# See https://github.com/pyg-team/pyg-lib to check the configuration that works for you

# Auto-detect CUDA version if not specified
if [ -z "$CUDA_VERSION" ]; then
    echo "🔍 Auto-detecting CUDA version..."
    
    # Try to detect CUDA via nvcc
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
        echo "   Found CUDA ${NVCC_VERSION} via nvcc"
        
        # Map CUDA version to PyTorch CUDA version suffix
        case "${NVCC_VERSION}" in
            13.0*)
                CUDA_VERSION="cu130"
                ;;
            12.9*)
                CUDA_VERSION="cu129"
                ;;
            12.8*)
                CUDA_VERSION="cu128"
                ;;
            12.6*)
                CUDA_VERSION="cu126"
                ;;
            12.4*)
                CUDA_VERSION="cu124"
                ;;
            12.1*|12.2*|12.3*|12.5*|12.7*)
                CUDA_VERSION="cu121"
                ;;
            11.8*)
                CUDA_VERSION="cu118"
                ;;
            *)
                echo "   ⚠️  CUDA version ${NVCC_VERSION} is not supported (requires 11.8+)"
                echo "   Falling back to CPU version"
                CUDA_VERSION="cpu"
                ;;
        esac
    # Try to detect CUDA via nvidia-smi
    elif command -v nvidia-smi &> /dev/null; then
        NVIDIA_SMI_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
        echo "   Found CUDA ${NVIDIA_SMI_VERSION} via nvidia-smi"
        
        # Map CUDA version to PyTorch CUDA version suffix
        case "${NVIDIA_SMI_VERSION}" in
            13.0*)
                CUDA_VERSION="cu130"
                ;;
            12.9*)
                CUDA_VERSION="cu129"
                ;;
            12.8*)
                CUDA_VERSION="cu128"
                ;;
            12.6*)
                CUDA_VERSION="cu126"
                ;;
            12.4*)
                CUDA_VERSION="cu124"
                ;;
            12.1*|12.2*|12.3*|12.5*|12.7*)
                CUDA_VERSION="cu121"
                ;;
            11.8*)
                CUDA_VERSION="cu118"
                ;;
            *)
                echo "   ⚠️  CUDA version ${NVIDIA_SMI_VERSION} is not supported (requires 11.8+)"
                echo "   Falling back to CPU version"
                CUDA_VERSION="cpu"
                ;;
        esac
    else
        echo "   No CUDA installation detected, using CPU version"
        CUDA_VERSION="cpu"
    fi
    
    echo "   Selected CUDA variant: ${CUDA_VERSION}"
else
    echo "🔧 Using manually specified CUDA version: ${CUDA_VERSION}"
fi

# Auto-select PyTorch version based on CUDA if not specified
# Compatibility matrix from https://github.com/pyg-team/pyg-lib
if [ -z "$TORCH_VERSION" ]; then
    echo "🔍 Selecting compatible PyTorch version for CUDA ${CUDA_VERSION}..."
    
    case "${CUDA_VERSION}" in
        cu130|cu128|cu126)
            TORCH_VERSION="2.9.0"
            ;;
        cu129)
            TORCH_VERSION="2.8.0"
            ;;
        cu124)
            TORCH_VERSION="2.6.0"
            ;;
        cu121|cu118)
            TORCH_VERSION="2.5.0"
            ;;
        cpu)
            # CPU version - use latest stable
            TORCH_VERSION="2.8.0"
            ;;
        *)
            echo "   ⚠️  CUDA version ${CUDA_VERSION} is not supported"
            echo "   Supported CUDA versions: cu118, cu121, cu124, cu126, cu128, cu129, cu130"
            echo "   Please use: CUDA_VERSION=cpu bash uv_setup.sh"
            exit 1
            ;;
    esac
    
    echo "   Selected PyTorch version: ${TORCH_VERSION}"
else
    echo "🔧 Using manually specified PyTorch version: ${TORCH_VERSION}"
fi

echo "📦 Installing TopoBench with PyTorch ${TORCH_VERSION} and CUDA ${CUDA_VERSION}..."

# Create/sync the virtual environment with dependencies
uv sync --all-extras

# Install PyTorch with specific CUDA version
if [ "$CUDA_VERSION" = "cpu" ]; then
    echo "🔧 Installing CPU version of PyTorch..."
    uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu
else
    echo "🔧 Installing CUDA ${CUDA_VERSION} version of PyTorch..."
    uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
fi

# Install PyTorch Geometric dependencies
echo "🔧 Installing PyTorch Geometric dependencies..."
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
uv run pre-commit install

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or run commands directly with:"
echo "  uv run python <script>"
echo ""

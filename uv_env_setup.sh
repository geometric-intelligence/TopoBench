#!/bin/bash -l

# 1. Clean up old mismatched environment
echo "Cleaning up..."
rm -rf .venv uv.lock
rm -rf ~/.cache/pre-commit  # Clear pre-commit cache to remove old python paths

# 2. Create the environment explicitly with Python 3.11
# This replaces 'conda create -n tb python=3.11'
echo "Creating environment with Python 3.11..."
uv venv --python 3.11

source .venv/bin/activate

# 3. Define Versions
CUDA="cu121"
TORCH_VER="2.3.0"
TORCH_URL="https://download.pytorch.org/whl/${CUDA}"
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA}.html"

echo "----------------------------------------------------"
echo "Step 1: Installing PyTorch ${TORCH_VER}+${CUDA}..."
uv pip install "torch==${TORCH_VER}+${CUDA}" \
    --extra-index-url ${TORCH_URL}

echo "----------------------------------------------------"
echo "Step 2: Installing PyG Extensions..."
uv pip install torch-scatter torch-sparse torch-cluster \
    --find-links ${PYG_URL}

echo "----------------------------------------------------"
echo "Step 3: Installing Project..."
# Ensure you added 'torch==2.3.0' to pyproject.toml as discussed!
uv pip install -e '.[all]'

echo "----------------------------------------------------"
echo "Step 4: Installing Pre-commit hooks..."
# Now that we are actually ON Python 3.11, this will succeed
uv run pre-commit install

echo "----------------------------------------------------"
echo "Final Verification:"
uv run python --version  # Should say 3.11.x
uv run python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
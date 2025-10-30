#!/bin/bash -l

# ⚠️  DEPRECATED: This script is no longer maintained
# ⚠️  Please use uv_setup.sh instead for faster, more reliable installation
# 
# To migrate to uv:
# 1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run: bash uv_setup.sh
#
# See UV_MIGRATION.md for more details

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  WARNING: env_setup.sh is DEPRECATED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This script is no longer maintained. Please use uv_setup.sh instead."
echo ""
echo "Benefits of using uv:"
echo "  ⚡ 10-100x faster installation"
echo "  🔒 Deterministic builds with lock file"
echo "  🎯 Automatic CUDA detection"
echo "  🚀 Modern Python package management"
echo ""
echo "To install with uv:"
echo "  1. curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "  2. bash uv_setup.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "Continue with deprecated installation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Please use: bash uv_setup.sh"
    exit 1
fi

pip install --upgrade pip
pip install -e '.[all]'

# Note that not all combinations of torch and CUDA are available
# See https://github.com/pyg-team/pyg-lib to check the configuration that works for you
TORCH="2.3.0"   # available options: 2.0.0, 2.1.0, 2.2.0, 2.3.0, 2.4.0, ...
CUDA="cpu"      # if available, select the CUDA version suitable for your system
                # available options: cpu, cu102, cu113, cu116, cu117, cu118, cu121, ...
pip install torch==${TORCH} --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

#pytest

pre-commit install

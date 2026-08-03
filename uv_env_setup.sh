#!/bin/bash -l

# ==============================================================================
# TopoBench immutable environment setup (Python 3.11)
# ==============================================================================
# usage: source uv_env_setup.sh [cpu|cu121]
# ==============================================================================

if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCKED_PLATFORM="cpu"
else
    LOCKED_PLATFORM="cu121"
fi
PLATFORM="${1:-$LOCKED_PLATFORM}"
if [ "$PLATFORM" != "$LOCKED_PLATFORM" ]; then
    echo "Error: this immutable lock targets '$LOCKED_PLATFORM', not '$PLATFORM'."
    exit 1
fi

echo ""
echo "======================================================="
echo "Initializing TopoBench Environment ($PLATFORM)"
echo "======================================================="
echo "Syncing the committed Python 3.11 lock without resolution..."
if ! uv sync --python 3.11 --all-extras --frozen; then
    echo "Error: frozen uv sync failed."
    return 1 2>/dev/null || exit 1
fi

# ------------------------------------------------------------------------------
# Finalize
# ------------------------------------------------------------------------------
source .venv/bin/activate
python -c "import sys; import torch; print(f'✅ Python Ver    : {sys.version.split()[0]}'); print(f'✅ Torch Version : {torch.__version__}'); print(f'✅ CUDA Available: {torch.cuda.is_available()}'); print(f'✅ CUDA Version  : {torch.version.cuda}')"
echo "======================================================="

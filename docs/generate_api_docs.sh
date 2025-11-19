#!/bin/bash
# Script to automatically generate API documentation using sphinx-apidoc

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
API_DIR="$SCRIPT_DIR/api"

echo "======================================"
echo "Generating API Documentation"
echo "======================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "API output directory: $API_DIR"
echo ""

# Remove old API documentation
if [ -d "$API_DIR" ]; then
    echo "Removing old API documentation..."
    rm -rf "$API_DIR"
fi

# Create API directory
mkdir -p "$API_DIR"

# Generate API documentation using sphinx-apidoc
echo "Running sphinx-apidoc..."
sphinx-apidoc \
    --force \
    --separate \
    --module-first \
    --no-toc \
    --maxdepth 4 \
    -o "$API_DIR" \
    "$PROJECT_DIR/topobench" \
    "$PROJECT_DIR/topobench/__pycache__" \
    "$PROJECT_DIR/topobench/**/__pycache__"

echo ""
echo "sphinx-apidoc completed."
echo ""

# Generate the API index file
echo "Generating API index..."
python "$SCRIPT_DIR/generate_api_index.py"

echo ""
echo "======================================"
echo "API Documentation Generated Successfully!"
echo "======================================"
echo ""
echo "Generated files are in: $API_DIR"
echo ""
echo "Next steps:"
echo "  - Review the generated .rst files in $API_DIR"
echo "  - Run 'make html' to build the documentation"
echo ""

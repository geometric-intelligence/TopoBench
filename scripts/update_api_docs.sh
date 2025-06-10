#!/bin/bash

# Set paths
TOPOBENCH_PATH="topobench"  # Updated path to the cloned repository
DOCS_PATH="_docs/documentation/api"

# Create necessary directories
mkdir -p "$DOCS_PATH"

# Generate API documentation
echo "Generating API documentation..."
python3 scripts/generate_api_docs.py

# Format the documentation
echo "Formatting documentation..."
python3 scripts/format_api_docs.py

echo "API documentation has been updated successfully!" 
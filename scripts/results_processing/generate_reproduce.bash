#!/bin/bash


echo "Checking to see if normalized CSV exists..."

if [ ! -f merged_csv/merged_normalized.csv ]; then
    echo "Not found! Execute preprocess.bash"
fi

echo "Generating reproduction script in the scripts/folder"

python scripts/results_processing/generate_reproduce_script.py
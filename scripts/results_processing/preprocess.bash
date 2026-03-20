#!/bin/bash


echo "Checking to see if merged.csv exists..."

if [ ! -f merged_csv/merged.csv ]; then
    echo "Not found! Execute refetch.bash"
fi

echo "Generating merged_normalized.csv in the merged_csv folder"

python scripts/results_processing/process_json.py
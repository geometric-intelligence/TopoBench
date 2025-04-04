#!/bin/bash


echo "Fetching results from WanDB and merging them..."
python scripts/results_processing/fetch_and_merge.py
echo "Done"

echo "Normalizing columns with JSON data..."
python scripts/results_processing/process_json.py
echo


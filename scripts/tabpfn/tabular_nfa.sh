#!/bin/bash
# This script sets up the logging directory and runs a hyperparameter sweep
# using Bash array iteration and parallel job management.

# ==========================================================
# SETUP: Clean and prepare the logs directory for a fresh run
# ==========================================================
LOG_DIR="./logs/tabular_nfa"
echo "Preparing a clean log directory at: $LOG_DIR"

# If the log directory exists, delete it and everything inside it
if [ -d "$LOG_DIR" ]; then
    rm -r "$LOG_DIR"
fi

# Create a new, empty log directory
mkdir -p "$LOG_DIR"

echo "Folder ready: $LOG_DIR"

# ========================================================================
# Load Utilities and Set Environment
# ========================================================================
# Get the absolute path to the directory where this script is located.
SCRIPT_DIR="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" &> /dev/null && pwd)"


# Add this line to get detailed errors from Hydra
export HYDRA_FULL_ERROR=1
# Source the logging.sh file using the script's directory as an anchor.
# NOTE: Assumes base/logging.sh exists and defines run_and_log

source "$SCRIPT_DIR/base/logging.sh"

# ========================================================================
# Dependency Arrays
# ========================================================================

models=(
    "dt_c"
    "rf_c"
    "hgb_c"
    "lgbm_c"
    "logistic_regression"
    "mlp_c"
)

datasets=(
    "graph/cocitation_cora"
    "graph/cocitation_citeseer"
    "graph/cocitation_pubmed"
    "graph/amazon_ratings"
    "graph/minesweeper"
    "graph/questions"
    "graph/roman_empire"
    "graph/hm-categories"
    "graph/pokec-regions"
    "graph/web-topics"
    "graph/tolokers"
    "graph/tolokers-2"
    "graph/city-reviews"
    "graph/artnet-exp"
    "graph/web-fraud"
    "graph/wiki_cs"
)

DATA_SEEDS=(0 3 5 7 9)

# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

# --- 2. Initialize counters for tracking runs and managing parallel jobs ---
ROOT_LOG_DIR="$LOG_DIR"
run_counter=1
job_counter=0
MAX_PARALLEL=2 # Set the max number of jobs to run at once

# Loop over datasets and batch sizes using array indexing for zipping
num_datasets=${#datasets[@]}

# Loop over models
for model in "${models[@]}"; do
    # Loop over datasets and batch sizes
    for i in $(seq 0 $((num_datasets - 1))); do
        dataset="${datasets[i]}"
        
        for data_seed in "${DATA_SEEDS[@]}"; do      
            # Define a descriptive run name for logging
            run_name="${model##*/}_${dataset##*/}_seed${data_seed}"
            project_name="graph_tabpfn"
            # Construct the command array.
            cmd=(
                "python" "-m" "topobench"
                "model/non_relational/sklearn@model.backbone=${model}"
                "model=non_relational/sklearn_classifier"
                "dataset=${dataset}"
                "dataset.split_params.split_type=stratified"
                "transforms=nfa"
                "train=False"
                "trainer=cpu"
                "evaluator=classification_extended"
                "dataset.split_params.data_seed=${data_seed}"
                "logger.wandb.project=${project_name}"
            )

            echo "============================================================"
            echo "Starting Run #$run_counter: $run_name"
            echo "Command: ${cmd[*]}"
            echo "============================================================"

            run_and_log "${cmd[*]}" "$log_group" "$run_name" "$ROOT_LOG_DIR" &

            # --- 6. Increment counters and manage parallel jobs ---
            # ... (Parallel job management remains the same) ...
            ((run_counter++))
            ((job_counter++))
            if [[ "$job_counter" -ge "$MAX_PARALLEL" ]]; then
                wait -n
                ((job_counter--))
            fi
            
            done
        done
    done
done

# Wait for any remaining jobs to complete before exiting the script
echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."

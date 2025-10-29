#!/bin/bash
# This script sets up the logging directory and runs a hyperparameter sweep
# using Bash array iteration and parallel job management.

# ==========================================================
# SETUP: Clean and prepare the logs directory for a fresh run
# ==========================================================
LOG_DIR="./logs"
echo "Preparing a clean log directory at: $LOG_DIR"

# If the log directory exists, delete it and everything inside it
if [ -d "$LOG_DIR" ]; then
    rm -r "$LOG_DIR"
fi

# Create a new, empty log directory
mkdir -p "$LOG_DIR"

# ========================================================================
# Load Utilities and Set Environment
# ========================================================================
# Get the absolute path to the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Add this line to get detailed errors from Hydra
export HYDRA_FULL_ERROR=1
# Source the logging.sh file using the script's directory as an anchor.
# NOTE: Assumes base/logging.sh exists and defines run_and_log
source "$SCRIPT_DIR/base/logging.sh"

# ========================================================================
# Dependency Arrays
# ========================================================================

models=(
    "hypergraph/allsettransformer"
    "hypergraph/edgnn"
    "hypergraph/unignn2"
)

datasets=(
    "graph/cocitation_cora"
    "graph/cocitation_citeseer"
    "graph/cocitation_pubmed"
    "graph/amazon_ratings"
    "graph/roman_empire"
    "graph/MUTAG"
    "graph/PROTEINS"
    # "graph/ZINC"
)

# NOTE: The batch_sizes array must match the length and order of datasets
batch_sizes=(1 1 1 1 1 256 256 256)

lifting="liftings/graph2hypergraph/knn"


lrs=(0.001 0.01 0.1)
hidden_channels=(32 64 128)
k_values=(3 5 10)
DATA_SEEDS=(0 3 5 7 9)
# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

# --- 2. Initialize counters for tracking runs and managing parallel jobs ---
gpu_id=5  # Specify which GPU to use
ROOT_LOG_DIR="$LOG_DIR"
run_counter=1
job_counter=0
MAX_PARALLEL=4 # Set the max number of jobs to run at once

# Loop over datasets and batch sizes using array indexing for zipping
num_datasets=${#datasets[@]}

# Loop over models
for model in "${models[@]}"; do
    # Loop over datasets and batch sizes
    for i in $(seq 0 $((num_datasets - 1))); do
        dataset="${datasets[i]}"
        batch_size="${batch_sizes[i]}"
            
        # Loop over learning rates
        for lr in "${lrs[@]}"; do
            
            # Loop over hidden channels
            for h in "${hidden_channels[@]}"; do
                
                for k in "${k_values[@]}"; do
                    for data_seed in "${DATA_SEEDS[@]}"; do
                
                        # Define a descriptive run name for logging
                        run_name="${model##*/}_${dataset##*/}_${lifting##*/}_k${k}_lr${lr}_h${h}"
                        log_group="sweep_knn"
                        # Construct the command array.
                        cmd=(
                            "python" "-m" "topobench"
                            "model=${model}"
                            "dataset=${dataset}"
                            "optimizer.parameters.lr=${lr}"
                            "model.feature_encoder.out_channels=${h}"
                            "model.readout.readout_name=PropagateSignalDown"
                            "model.feature_encoder.proj_dropout=0.5"
                            "dataset.dataloader_params.batch_size=${batch_size}"
                            "transforms=[${lifting}]"
                            "transforms.liftings.graph2hypergraph.k_value=${k}" 
                            "dataset.split_params.data_seed=${data_seed}"
                            "trainer.max_epochs=500"
                            "trainer.min_epochs=50"
                            "trainer.check_val_every_n_epoch=5"
                            "trainer.devices=[${gpu_id}]"
                            "callbacks.early_stopping.patience=10"
                            "logger.wandb.project=hypergraph_liftings"
                        )
                        if [[ "${model##*/}" != "edgnn" ]]; then
                            cmd+=("model.backbone.n_layers=2")
                        fi

                        run_and_log "${cmd[*]}" "$log_group" "$run_name" "$ROOT_LOG_DIR" &

                        # --- 6. Increment counters and manage parallel jobs ---
                        # ... (Parallel job management remains the same) ...
                        ((run_counter++))
                        ((job_counter++))
                        if [[ "$job_counter" -ge "$MAX_PARALLEL" ]]; then
                            wait -n
                            ((job_counter--))
                        fi
                    done # END data_seed loop
                done # END k_values loop
            done
        done
    done
done

# Wait for any remaining jobs to complete before exiting the script
echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."
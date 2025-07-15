#!/bin/bash

# Define log files
LOG_FILE="scripts/script_output.log"
ERROR_LOG_FILE="scripts/script_error.log"
FAILED_LOG_FILE="scripts/failed_runs.log"

# Clear previous log files
> $LOG_FILE
> $ERROR_LOG_FILE
> $FAILED_LOG_FILE

# Function to run a command and check for failure
run_command() {
    local cmd="$1"
    
    # Run the command and capture the output and error
    { eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; } 2>> "$ERROR_LOG_FILE"
    
    # Check if the command failed
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Command failed: $cmd" >> "$FAILED_LOG_FILE"
        echo "Check $ERROR_LOG_FILE for details." >> "$FAILED_LOG_FILE"
    fi
}

# List of commands to execute
# transforms.graph2cell_lifting.max_cell_length=10
# model.backbone_wrapper.sampler.n_hops=1,2,3 
commands=(
# Baselines
'python -m topobench model=cell/tabpfn dataset=graph/cocitation_citeseer model.backbone_wrapper.use_embeddings=False model.backbone_wrapper.sampler.sampler_name=GraphHopSampler model.backbone_wrapper.sampler.n_hops=1,2,3 experiment=tabpfn_m dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=tabpfn --multirun'

# Manual PE, use PE embeddigns
'python -m topobench \
model=cell/tabpfn \
dataset=graph/cocitation_citeseer\
model.backbone_wrapper.use_embeddings=True \
experiment=tabpfn_m,tabpfn_g \
model.backbone_wrapper.sampler.sampler_name=GraphHopSampler \
model.backbone_wrapper.sampler.n_hops=1,2,3 \
dataset.split_params.data_seed=0,3,5,7,9 \
logger.wandb.project=tabpfn \
--multirun'


)

# Iterate over the commands and run them
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    run_command "$cmd"
done


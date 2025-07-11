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
'python -m topobench model=cell/tabpfn dataset=graph/cocitation_citeseer,graph/cocitation_cora,graph/cocitation_pubmed,
graph/amazon_ratings,graph/roman_empire,graph/minesweeper,graph/tolokers
model.backbone_wrapper.sampler.k=10,50,100,500,1000 model.backbone_wrapper.sampler.sampler_name=KNNSampler
transforms=tabpfn  dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=tabpfn --multirun'

'python -m topobench model=cell/tabpfn dataset=graph/cocitation_citeseer,graph/cocitation_cora,graph/cocitation_pubmed,
graph/amazon_ratings,graph/roman_empire,graph/minesweeper,graph/tolokers
model.backbone_wrapper.sampler.n_hops=1,2,3,4,5,6 model.backbone_wrapper.sampler.sampler_name=GraphSampler
transforms=tabpfn  dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=tabpfn --multirun'

'python -m topobench model=cell/tabpfn dataset=graph/cocitation_citeseer,graph/cocitation_cora,graph/cocitation_pubmed,
graph/amazon_ratings,graph/roman_empire,graph/minesweeper,graph/tolokers
model.backbone_wrapper.sampler.n_hops=1,2,3,4,5,6 model.backbone_wrapper.sampler.k=10,50,100,500,1000
model.backbone_wrapper.sampler.sampler_name=CompositeSampler
transforms=tabpfn  dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=tabpfn --multirun'
)

# Iterate over the commands and run them
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    run_command "$cmd"
done


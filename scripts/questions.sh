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
commands=(
 'python -m topobench model=graph/gcn dataset=graph/questions optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=64 model.backbone.num_layers=2 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.devices=[0] trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=questions_loris --multirun'
 'python -m topobench model=graph/gin dataset=graph/questions optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=64 model.backbone.num_layers=2 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.devices=[0] trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=questions_loris --multirun'
 'python -m topobench model=graph/gat dataset=graph/questions optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=64 model.backbone.num_layers=2 model.feature_encoder.proj_dropout=0.25 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=1000 trainer.min_epochs=50 trainer.devices=[0] trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50 logger.wandb.project=questions_loris --multirun'
)

# Iterate over the commands and run them
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    run_command "$cmd"
done


run_and_log() {
    local cmd="$1"
    local log_group="$2" # e.g., "xjtu_sy/spectral/lr_0.01"
    local run_name="$3"  # e.g., "stf"
    # Default to "./logs" if the 4th argument (root_dir) is empty or not provided
    local root_dir="${4:-./logs}" 

    # --- Path Definitions ---

    # 1. Define specific log directory for this run's detailed logs
    local specific_log_dir="$root_dir/$log_group"
    mkdir -p "$specific_log_dir"
    
    # 2. Define specific log file paths
    local stdout_log="$specific_log_dir/${run_name}_stdout.log"
    local stderr_log="$specific_log_dir/${run_name}_stderr.log"
    
    # 3. Define global summary log file paths (in the root dir)
    mkdir -p "$root_dir" # Ensure root dir exists
    local success_log="$root_dir/SUCCESSFUL_RUNS.log"
    local failed_log="$root_dir/FAILED_RUNS.log"
    
    # 4. Define temporary log file paths (will be renamed on failure)
    local tmp_stdout="${stdout_log}.tmp"
    local tmp_stderr="${stderr_log}.tmp"

    # --- Execution ---

    echo "--- [START] Running: $run_name (Log Group: $log_group) ---"

    # Execute the command, redirecting output to temporary files
    # We still use 'tee' so you can see the output live in your terminal.
    # We use process substitution >(...) for tee
    eval "$cmd" > >(tee "$tmp_stdout") 2> >(tee "$tmp_stderr" >&2)

    # Check the exit status of the 'eval' command (the first command in the pipe)
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ [SUCCESS] Finished: $run_name"
        # Log success to the global summary file
        echo "$(date): [SUCCESS] ${log_group}/${run_name}" >> "$success_log"
        
        # On success, remove the temporary files (we don't need the full logs)
        rm -f "$tmp_stdout" "$tmp_stderr"
        return 0
    else
        echo "❌ [FAILURE] Finished: $run_name"
        
        # On failure, rename the temp files to permanent logs
        mv "$tmp_stdout" "$stdout_log"
        mv "$tmp_stderr" "$stderr_log"

        echo "----------------- ERROR OUTPUT (Last 15 lines) -----------------"
        # Show a snippet of the error log directly in the console
        tail -n 15 "$stderr_log"
        echo "----------------------------------------------------------------"
        echo "Logs saved to: $specific_log_dir"

        # Log failure details to the global summary file
        echo "=================================" >> "$failed_log"
        echo "FAILURE on $(date): [${log_group}/${run_name}]" >> "$failed_log"
        echo "Command: $cmd" >> "$failed_log"
        echo "See full logs: $stdout_log | $stderr_log" >> "$failed_log"
        echo "=================================" >> "$failed_log"
        return 1
    fi
}


# Example: 

#!/bin/bash

# 1. Source the updated logging utility
# (Adjust path if run_utils.sh is in a different directory)
# source ./run_utils.sh

# # 2. Define a single root directory for all logs
# ROOT_LOG_DIR="scripts/checks/experiment_logs"

# # 3. Clear previous *global summary* log files
# mkdir -p "$ROOT_LOG_DIR"
# > "$ROOT_LOG_DIR/SUCCESSFUL_RUNS.log"
# > "$ROOT_LOG_DIR/FAILED_RUNS.log"


# MODELS=("stf" "patchtst")
# # "lstm"  "stf" "patchtst" "cnn_1d"  "stf" "patchtst"

# # 4. Define command templates in an array
# # Use SINGLE QUOTES (') to prevent ${model} from expanding now
# commands=(
#     'python lmetk/run.py experiment=xjtu_sy/prognostics/spectral/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.01 task_definition.seq_len=16'
#     'python lmetk/run.py experiment=pronostia/prognostics/spectral/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.01 task_definition.seq_len=16'
#     'python lmetk/run.py experiment=pronostia/prognostics/spectral/${model}.yaml trainer.max_epochs=100 trainer.min_epochs=50 optimizer.parameters.lr=0.001 task_definition.seq_len=16'
#     # Add other commands here
# )

# # 5. Define corresponding log groups for each command
# # This array MUST match the 'commands' array element-for-element
# log_groups=(
#     "xjtu_sy/spectral/lr_0.01"
#     "pronostia/spectral/lr_0.01"
#     "pronostia/spectral/lr_0.001"
#     # Add corresponding log group names here
# )

# # 6. Loop using array indices to keep commands and log groups in sync
# for i in "${!commands[@]}"; do
#     cmd_template="${commands[$i]}"
#     log_group="${log_groups[$i]}"

#     echo "============================================================"
#     echo "Starting Experiment Group: $log_group"
#     echo "============================================================"

#     for model in "${MODELS[@]}"; do
        
#         # 7. Use 'eval' to substitute the $model variable into the command
#         eval "final_cmd=\"$cmd_template\""
        
#         # 8. Define a unique run name for this specific job.
#         # Here, we just use the model name.
#         local run_name="$model"
        
#         # 9. Call the new, general-purpose logger
#         # It will handle all logging, directory creation, and error reporting.
#         run_and_log "$final_cmd" "$log_group" "$run_name" "$ROOT_LOG_DIR"
        
#         # The return code (0 for success, 1 for failure) is available
#         # in $? if you need to check it, e.g., to stop the script.
#         # if [ $? -ne 0 ]; then
#         #    echo "!!! Critical error on $model, stopping script."
#         #    exit 1
#         # fi
#     done
# done

# echo "All experiments finished."
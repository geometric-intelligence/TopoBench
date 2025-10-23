# File: run_utils.sh

#!/bin/bash

# This function executes a command, logs its output, and keeps the logs
# ONLY if the command fails.
run_and_log() {
    local cmd="$1"
    local run_name="$2"

    # Define log file paths
    local log_dir="./logs"
    mkdir -p "$log_dir"
    local stdout_log="$log_dir/${run_name}_stdout.log"
    local stderr_log="$log_dir/${run_name}_stderr.log"
    local success_log="$log_dir/SUCCESSFUL_RUNS.log"
    local failed_log="$log_dir/FAILED_RUNS.log"
    
    # Define temporary log file paths
    local tmp_stdout="${stdout_log}.tmp"
    local tmp_stderr="${stderr_log}.tmp"

    echo "--- [START] Running: $run_name ---"

    # Execute the command, redirecting output to temporary files
    # We still use 'tee' so you can see the output live in your terminal.
    eval "$cmd" > >(tee "$tmp_stdout") 2> >(tee "$tmp_stderr" >&2)

    # Check the exit status of the 'eval' command
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ [SUCCESS] Finished: $run_name"
        echo "$(date): ${run_name}" >> "$success_log"
        # On success, remove the temporary files
        rm -f "$tmp_stdout" "$tmp_stderr"
        return 0
    else
        echo "❌ [FAILURE] Finished: $run_name"
        
        # On failure, rename the temp files to permanent logs
        mv "$tmp_stdout" "$stdout_log"
        mv "$tmp_stderr" "$stderr_log"

        echo "----------------- ERROR OUTPUT -----------------"
        tail -n 15 "$stderr_log" # Now operates on the permanent log file
        echo "------------------------------------------------"

        echo "=================================" >> "$failed_log"
        echo "FAILURE on $(date): $run_name" >> "$failed_log"
        echo "Command: $cmd" >> "$failed_log"
        echo "See full logs: $stdout_log | $stderr_log" >> "$failed_log"
        echo "=================================" >> "$failed_log"
        return 1
    fi
}
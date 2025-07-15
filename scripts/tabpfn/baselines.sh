#!/bin/bash

# ======================
# USER CONFIGURATION
# ======================
exp_name="tabpfn_m"   # Set your experiment name here
MAX_PARALLEL=4        # Max parallel jobs

# Default to enabling retry
ENABLE_RETRY=true
RETRY_DELAY_SECONDS=10 # Delay before retry

# Datasets, experiments, seeds
datasets=("graph/cocitation_citeseer" "graph/cocitation_cora")
experiments=("tabpfn_m" "tabpfn_g")
#seeds=(0 3 5 7 9)

# ======================
# ARGUMENT PARSING
# ======================
for arg in "$@"; do
    case $arg in
        --no-retry)
            ENABLE_RETRY=false
            shift # Remove --no-retry from processing
            ;;
        *)
            # Unknown option, potentially other arguments for a future extension
            # For now, just shift it.
            shift
            ;;
    esac
done

# ======================
# INTERNAL SETUP
# ======================
run_counter=0
JOBS=0

# ======================
# RUN FUNCTION WITH LOGGING AND OPTIONAL RETRY
# ======================
run_logged_command() {
    local cmd="$1"
    local dataset_slug="$2"
    local run_id_prefix="$3" # Prefix for run_id, now per dataset/experiment combo
    local max_retries=0       # Default to 0 retries (1 attempt)

    # If retry is enabled, set max_retries to 1 (total of 2 attempts)
    if $ENABLE_RETRY; then
        max_retries=1
    fi

    local attempt_num=0

    # Get directory of this script
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Define log path
    log_dir="$script_dir/logs/$dataset_slug/$exp_name/logs"
    mkdir -p "$log_dir"

    local stdout_log="$log_dir/${run_id_prefix}_stdout.log"
    local stderr_log="$log_dir/${run_id_prefix}_stderr.log"
    local failed_log="$script_dir/logs/$dataset_slug/$exp_name/failed_runs.log"

    while (( attempt_num <= max_retries )); do
        local current_run_tag="[$run_id_prefix] Attempt $((attempt_num + 1))"
        echo "Running $current_run_tag: $cmd" | tee -a "$stdout_log"

        # Execute the command, redirecting stdout to log and teeing to stdout
        # Redirect stderr to a separate log
        { eval "$cmd" 2>&1 | tee -a "$stdout_log"; } 2>> "$stderr_log"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "Run $current_run_tag SUCCEEDED." | tee -a "$stdout_log"
            return 0 # Success, exit the function
        else
            echo "Run $current_run_tag FAILED." | tee -a "$stdout_log"
            if (( attempt_num < max_retries )); then
                echo "Waiting $RETRY_DELAY_SECONDS seconds before retrying..." | tee -a "$stdout_log"
                sleep "$RETRY_DELAY_SECONDS"
                ((attempt_num++))
            else
                echo "Run $current_run_tag FAILED permanently after $max_retries retries: $cmd" >> "$failed_log"
                echo "See logs: $stdout_log | $stderr_log" >> "$failed_log"
                return 1 # Permanent failure
            fi
        fi
    done
}

# ======================
# MAIN EXECUTION LOOP
# ======================
for dataset in "${datasets[@]}"; do
    dataset_slug=$(basename "$dataset")  # e.g., cocitation_citeseer

    for experiment in "${experiments[@]}"; do
        # Increment run_counter for each parallel job (dataset x experiment)
        run_id_for_parallel_job=$(printf "%04d" $run_counter)
        ((run_counter++))

        
        

        cmd="python -m topobench \
            model=cell/tabpfn_classifier \
            dataset=$dataset \
            model.backbone_wrapper.use_embeddings=True \
            experiment=$experiment \
            model.backbone_wrapper.sampler.sampler_name=GraphHopSampler \
            model.backbone_wrapper.sampler.n_hops=1,2,3 \
            dataset.split_params.data_seed=0,3,5,7,9 \
            logger.wandb.project=tabpfn \
            --multirun"

        # The run_id here now represents the specific parallel job (dataset + experiment)
        # We run this in the background
        run_logged_command "$cmd" "$dataset_slug" "${run_id_for_parallel_job}_${experiment}" &

        ((JOBS++))
        if [ "$JOBS" -ge "$MAX_PARALLEL" ]; then
            wait
            JOBS=0
        fi
    done
done

# Wait for any remaining jobs
wait
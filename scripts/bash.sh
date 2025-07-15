#!/bin/bash

# ======================
# COUNTER INITIALIZATION
# ======================
run_counter=0

# ======================
# PARALLELISM CONFIG
# ======================
MAX_PARALLEL=4
JOBS=0

# ======================
# PARAMETER SETUP
# ======================

datasets=("graph/cocitation_citeseer" "graph/cocitation_cora")
experiments=("tabpfn_m" "tabpfn_g")
seeds=(0 3 5 7 9)

# ======================
# RUN FUNCTION WITH LOGGING
# ======================
run_logged_command() {
    local cmd="$1"
    local dataset_slug="$2"
    local run_id="$3"

    # Define log directories and files
    log_dir="scripts/logs/$dataset_slug"
    mkdir -p "$log_dir"

    local stdout_log="$log_dir/${run_id}_stdout.log"
    local stderr_log="$log_dir/${run_id}_stderr.log"
    local failed_log="$log_dir/failed_runs.log"

    echo "Running [$run_id]: $cmd" | tee -a "$stdout_log"
    
    { eval "$cmd" 2>&1 | tee -a "$stdout_log"; } 2>> "$stderr_log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Run [$run_id] FAILED: $cmd" >> "$failed_log"
        echo "See logs: $stdout_log | $stderr_log" >> "$failed_log"
    fi
}

# ======================
# MAIN LOOP
# ======================
for dataset in "${datasets[@]}"; do
    dataset_slug=$(basename "$dataset")  # e.g., cocitation_citeseer

    for experiment in "${experiments[@]}"; do
        for seed in "${seeds[@]}"; do
            run_id=$(printf "%04d" $run_counter)  # 4-digit run counter
            ((run_counter++))

            cmd="python -m topobench \
                model=cell/tabpfn_classifier \
                dataset=$dataset \
                model.backbone_wrapper.use_embeddings=True \
                experiment=$experiment \
                model.backbone_wrapper.sampler.sampler_name=GraphHopSampler \
                model.backbone_wrapper.sampler.n_hops=1,2,3 \
                dataset.split_params.data_seed=$seed \
                logger.wandb.project=tabpfn \
                --multirun"

            run_logged_command "$cmd" "$dataset_slug" "$run_id" &

            ((JOBS++))
            if [ "$JOBS" -ge "$MAX_PARALLEL" ]; then
                wait
                JOBS=0
            fi
        done
    done
done

wait


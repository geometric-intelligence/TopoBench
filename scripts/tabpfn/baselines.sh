#!/bin/bash

# ======================
# USER CONFIGURATION
# ======================
project_name='tabpfn_full_graph'
exp_name="tabpfn"   # Set your experiment name here
MAX_PARALLEL=8        # Max parallel jobs

# Default to enabling retry
ENABLE_RETRY=true
RETRY_DELAY_SECONDS=10 # Delay before retry

# Datasets, experiments, seeds
datasets_classification=("graph/cocitation_citeseer" "graph/cocitation_cora" "graph/minesweeper" "graph/roman_empire" "graph/cocitation_pubmed")
datasets_regression=("graph/US-county-demos")
experiments=("tabpfn_m" "tabpfn_g")
USE_EMBEDDINGS=(True False)
#seeds=(0 3 5 7 9)

# ======================
# ARGUMENT PARSING
# ======================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-retry)
            ENABLE_RETRY=false
            shift
            ;;
        *)
            # preserve or collect unknowns if needed
            shift
            ;;
    esac
done

# ======================
# INTERNAL SETUP
# ======================
run_counter=0
JOBS=8

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
for dataset in "${datasets_regression[@]}"; do
    dataset_slug=$(basename "$dataset")  # e.g., Election

    for experiment in "${experiments[@]}"; do
        for use_embeddings in "${USE_EMBEDDINGS[@]}"; do
        
            # Increment run_counter for each parallel job (dataset x experiment)
            run_id_for_parallel_job=$(printf "%04d" $run_counter)
            ((run_counter++))
            
            
            cmd_all_nodes="python -m topobench \
                    model=cell/tabpfn_regressor \
                    dataset=$dataset \
                    model.backbone_wrapper.use_embeddings=$use_embeddings \
                    experiment=$experiment \
                    dataset.split_params.data_seed=0,3,5,7,9 \
                    logger.wandb.project=$project_name \
                    model.backbone.device=cuda:$JOBS \
                    model.backbone_wrapper.sampler=null \
                    --multirun"

                # All the node features  embedding are in the context
                run_logged_command "$cmd_all_nodes" "$dataset_slug" "${run_id_for_parallel_job}_${experiment}_all_nodes" &

                ((JOBS++))
                if [ "$JOBS" -ge "$MAX_PARALLEL" ]; then
                    wait
                    JOBS=0
                fi
        
                                                # cmd_graph="python -m topobench \
                                                #     model=cell/tabpfn_regressor \
                                                #     dataset=$dataset \
                                                #     model.backbone_wrapper.use_embeddings=True,False \
                                                #     experiment=$experiment \
                                                #     model.backbone_wrapper.sampler.sampler_name=GraphHopSampler \
                                                #     model.backbone_wrapper.sampler.n_hops=1,2,3 \
                                                #     dataset.split_params.data_seed=0,3,5,7,9 \
                                                #     model.backbone.device=cuda:$cuda \
                                                #     logger.wandb.project=tabpfn \
                                                #     --multirun"

                                                # # The run_id here now represents the specific parallel job (dataset + experiment)
                                                # # We run this in the background
                                                # # Using graph sampler
                                                # run_logged_command "$cmd_graph" "$dataset_slug" "${run_id_for_parallel_job}_${experiment}_graph" &

                                                # cmd_knn="python -m topobench \
                                                #     model=cell/tabpfn_regressor \
                                                #     dataset=$dataset \
                                                #     model.backbone_wrapper.use_embeddings=True,False \
                                                #     experiment=$experiment \
                                                #     model.backbone_wrapper.sampler.sampler_name=KNNSampler \
                                                #     model.backbone_wrapper.sampler.k=2,3,4,5,10,20,50,100 \
                                                #     dataset.split_params.data_seed=0,3,5,7,9 \
                                                #     logger.wandb.project=tabpfn \
                                                #     model.backbone.device=cuda:$cuda \
                                                #     --multirun"

                                                # # Using K-nn sampler
                                                # run_logged_command "$cmd_knn" "$dataset_slug" "${run_id_for_parallel_job}_${experiment}_knn" &            
        done
    done
done


for dataset in "${datasets_classification[@]}"; do
    dataset_slug=$(basename "$dataset")  # e.g., cocitation_citeseer

    for experiment in "${experiments[@]}"; do
        for use_embeddings in "${USE_EMBEDDINGS[@]}"; do
            # Increment run_counter for each parallel job (dataset x experiment)
            run_id_for_parallel_job=$(printf "%04d" $run_counter)
            ((run_counter++))
            
            
            cmd_all_nodes="python -m topobench \
            model=cell/tabpfn_classifier \
            dataset=$dataset \
            model.backbone_wrapper.use_embeddings=$use_embeddings \
            experiment=$experiment \
            dataset.split_params.data_seed=0,3,5,7,9 \
            logger.wandb.project=$project_name \
            model.backbone.device=cuda:$JOBS \
            model.backbone_wrapper.sampler=null \
            --multirun"

            # All the node features  embedding are in the context
            run_logged_command "$cmd_all_nodes" "$dataset_slug" "${run_id_for_parallel_job}_${experiment}_all_nodes" &

            ((JOBS++))
            if [ "$JOBS" -ge "$MAX_PARALLEL" ]; then
                wait
                JOBS=0
            fi

        # cmd_graph="python -m topobench \
        #     model=cell/tabpfn_classifier \
        #     dataset=$dataset \
        #     model.backbone_wrapper.use_embeddings=True,False \
        #     experiment=$experiment \
        #     model.backbone_wrapper.sampler.sampler_name=GraphHopSampler \
        #     model.backbone_wrapper.sampler.n_hops=1,2,3 \
        #     dataset.split_params.data_seed=0,3,5,7,9 \
        #     logger.wandb.project=tabpfn \
        #     model.backbone.device=cuda:$cuda \
        #     --multirun"

        # # The run_id here now represents the specific parallel job (dataset + experiment)
        # # We run this in the background
        # # Using graph sampler
        # run_logged_command "$cmd_graph" "$dataset_slug" "${run_id_for_parallel_job}_${experiment}_graph" &

        # cmd_knn="python -m topobench \
        #     model=cell/tabpfn_classifier \
        #     dataset=$dataset \
        #     model.backbone_wrapper.use_embeddings=True,False \
        #     experiment=$experiment \
        #     model.backbone_wrapper.sampler.sampler_name=KNNSampler \
        #     model.backbone_wrapper.sampler.k=2,3,4,5,10,20,50,100 \
        #     dataset.split_params.data_seed=0,3,5,7,9 \
        #     logger.wandb.project=tabpfn \
        #     model.backbone.device=cuda:$cuda \
        #     --multirun"

        # # Using K-nn sampler
        # run_logged_command "$cmd_knn" "$dataset_slug" "${run_id_for_parallel_job}_${experiment}_knn" &

        
        done
    done
done

# Wait for any remaining jobs
wait
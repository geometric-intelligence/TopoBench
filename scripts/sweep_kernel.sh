#!/bin/bash
# This script sets up the logging directory and runs a hyperparameter sweep
# using Bash array iteration and parallel job management.

# ==========================================================
# SETUP: Clean and prepare the logs directory for a fresh run
# ==========================================================
gpu_id=4  # Specify which GPU to use
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
project_name="hypergraph_${script_name}"

LOG_DIR="./logs/kernel"
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
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HYDRA_FULL_ERROR=1
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
)
batch_sizes=(1 1 1 1 1 256 256)

lifting="liftings/graph2hypergraph/kernel"

lrs=(0.001 0.01)
hidden_channels=(32 64 128)
DATA_SEEDS=(0 3 5 7 9)

# Kernel definitions
kernel_feats=(
    "identity"
    "cosine"
    "euclidean"
)
graph_kernels=("heat" "matern" "identity")
combinations=("sum" "first" "second")

# Hyperparameters for kernels
temperatures=(0.1 1.0)
nus=(0.5 1.5 2.5)
kappas=(0.1 0.5)
fractions=(0.01 0.05)
# ========================================================================
# Main Loop with Execution Tracking
# ========================================================================

ROOT_LOG_DIR="$LOG_DIR"
run_counter=1
job_counter=0
MAX_PARALLEL=2

num_datasets=${#datasets[@]}

for model in "${models[@]}"; do
    for i in $(seq 0 $((num_datasets - 1))); do
        dataset="${datasets[i]}"
        batch_size="${batch_sizes[i]}"
            
        for lr in "${lrs[@]}"; do
            for h in "${hidden_channels[@]}"; do
                for data_seed in "${DATA_SEEDS[@]}"; do
                    for fraction in "${fractions[@]}"; do
                        for comb in "${combinations[@]}"; do
                            
                            # 1. Determine Feature Kernels to loop over
                            # If combination is "first", we ignore feature kernels (run once with identity/dummy)
                            if [[ "$comb" == "first" ]]; then
                                current_feat_kernels=("identity")
                            else
                                current_feat_kernels=("${kernel_feats[@]}")
                            fi

                            for k_feat in "${current_feat_kernels[@]}"; do
                                
                                # 2. Determine Graph Kernels to loop over
                                # If combination is "second", we ignore graph kernels (run once with identity/dummy)
                                if [[ "$comb" == "second" ]]; then
                                    current_graph_kernels=("identity")
                                else
                                    current_graph_kernels=("${graph_kernels[@]}")
                                fi

                                for k_graph in "${current_graph_kernels[@]}"; do
                                    
                                    # Prepare to collect hyperparams to loop over
                                    # We construct an array of strings like "t=X nu=Y kappa=Z"
                                    param_configs=()

                                    # 3. Logic for Graph Kernel Hyperparameters
                                    if [[ "$comb" == "second" ]]; then
                                        # If only feature kernel matters, we don't loop over t/nu/kappa
                                        param_configs+=("t=null nu=null kappa=null")
                                    
                                    elif [[ "$k_graph" == "heat" ]]; then
                                        # HEAT: Loop over temperatures
                                        for t in "${temperatures[@]}"; do
                                            param_configs+=("t=${t} nu=null kappa=null")
                                        done
                                    
                                    elif [[ "$k_graph" == "matern" ]]; then
                                        # MATERN: Loop over nus and kappas
                                        for nu in "${nus[@]}"; do
                                            for kap in "${kappas[@]}"; do
                                                param_configs+=("t=null nu=${nu} kappa=${kap}")
                                            done
                                        done
                                    
                                    else 
                                        # Identity or other kernels without params
                                        param_configs+=("t=null nu=null kappa=null")
                                    fi

                                    # 4. Execute Runs
                                    for p_conf in "${param_configs[@]}"; do
                                        # Extract params from the string
                                        # Using simple awk/cut or eval logic
                                        # Here we interpret the string manually
                                        curr_t=$(echo $p_conf | awk '{print $1}' | cut -d= -f2)
                                        curr_nu=$(echo $p_conf | awk '{print $2}' | cut -d= -f2)
                                        curr_kap=$(echo $p_conf | awk '{print $3}' | cut -d= -f2)

                                        # Build Run Name
                                        run_name="${model##*/}_${dataset##*/}_${lifting##*/}_${comb}"
                                        
                                        # Add relevant parts to run_name
                                        if [[ "$comb" != "first" ]]; then
                                            run_name="${run_name}_kf-${k_feat}"
                                        fi
                                        if [[ "$comb" != "second" ]]; then
                                            run_name="${run_name}_kg-${k_graph}"
                                            if [[ "$k_graph" == "heat" ]]; then run_name="${run_name}_t${curr_t}"; fi
                                            if [[ "$k_graph" == "matern" ]]; then run_name="${run_name}_n${curr_nu}_k${curr_kap}"; fi
                                        fi
                                        
                                        run_name="${run_name}_lr${lr}_h${h}_s${data_seed}"
                                        log_group="sweep_kernel"

                                        # Construct Command
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
                                            "dataset.split_params.data_seed=${data_seed}"
                                            "trainer.max_epochs=500"
                                            "trainer.min_epochs=50"
                                            "trainer.check_val_every_n_epoch=5"
                                            "trainer.devices=[${gpu_id}]"
                                            "callbacks.early_stopping.patience=10"
                                            "logger.wandb.project=${project_name}"
                                            
                                            # Dynamic Arguments
                                            "transforms.liftings.graph2hypergraph.C=${comb}"
                                            "transforms.liftings.graph2hypergraph.feat_kernel=${k_feat}"
                                            "transforms.liftings.graph2hypergraph.graph_kernel=${k_graph}"
                                            "transforms.liftings.graph2hypergraph.fraction=${fraction}"
                                        )

                                        if [[ "${model##*/}" != "edgnn" ]]; then
                                            cmd+=("model.backbone.n_layers=2")
                                        fi

                                        # Only append flags if they are not null
                                        if [[ "$curr_t" != "null" ]]; then
                                            cmd+=("transforms.liftings.graph2hypergraph.t=${curr_t}")
                                        fi
                                        if [[ "$curr_nu" != "null" ]]; then
                                            cmd+=("transforms.liftings.graph2hypergraph.nu=${curr_nu}")
                                        fi
                                        if [[ "$curr_kap" != "null" ]]; then
                                            cmd+=("transforms.liftings.graph2hypergraph.kappa=${curr_kap}")
                                        fi

                                        run_and_log "${cmd[*]}" "$log_group" "$run_name" "$ROOT_LOG_DIR" &

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
                    done
                done
            done
        done
    done
done

echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."
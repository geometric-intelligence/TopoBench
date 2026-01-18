#!/bin/bash
# ==============================================================================
# SCRIPT: kernel_baselines_master.sh
# DESCRIPTION:
#   Runs a scalable hyperparameter sweep for Hypergraph Kernels.
#   - Adapted from the old "kernel" script to the new "virtual slot" framework.
#   - Handles complex conditional logic for Graph/Feature kernels.
# ==============================================================================

# ==============================================================================
# SECTION 1: LOGGING & ENVIRONMENT SETUP
# ==============================================================================

# 1.1 Define Project Identifiers
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
project_name="hypergraph_${script_name}"
log_group="sweep_kernel"
LOG_DIR="./logs/${log_group}"

echo "=========================================================="
echo " Preparing log directory: $LOG_DIR"
echo "=========================================================="

# 1.2 Clean up old logs
if [ -d "$LOG_DIR" ]; then rm -r "$LOG_DIR"; fi
mkdir -p "$LOG_DIR"

# 1.3 Robust Dependency Loading
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HYDRA_FULL_ERROR=1

find_logging_script() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/base/logging.sh" ]]; then echo "$dir/base/logging.sh"; return 0; fi
        if [[ -f "$dir/scripts/base/logging.sh" ]]; then echo "$dir/scripts/base/logging.sh"; return 0; fi
        dir="$(dirname "$dir")"
    done
    return 1
}

LOGGING_PATH=$(find_logging_script "$SCRIPT_DIR")
if [[ -n "$LOGGING_PATH" ]]; then
    echo "✔ Found logging utils at: $LOGGING_PATH"
    source "$LOGGING_PATH"
else
    echo "❌ CRITICAL ERROR: Could not locate 'base/logging.sh'."
    exit 1
fi

# ==============================================================================
# SECTION 2: HARDWARE & CONCURRENCY
# ==============================================================================

# 2.1 Configuration
physical_gpus=(0 1 2 3)  # IDs of the GPUs to use
JOBS_PER_GPU=2           # Number of parallel runs allowed per GPU

# 2.2 Create Virtual Slots
gpus=()
for gpu in "${physical_gpus[@]}"; do
    for ((i=1; i<=JOBS_PER_GPU; i++)); do gpus+=("$gpu"); done
done

# 2.3 Initialize Slot Tracking
declare -a slot_pids
for i in "${!gpus[@]}"; do slot_pids[$i]=0; done


# ==============================================================================
# SECTION 3: EXPERIMENT PARAMETERS
# These variables are injected into the Python generator below.
# ==============================================================================

# --- Major Variations ---
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

# --- Hyperparameters ---
lrs=(0.001 0.01)
hidden_channels=(32 64 128)
fractions=(0.01 0.05)
DATA_SEEDS=(0 3 5 7 9)

# --- Kernel Logic Params ---
kernel_feats=("identity" "cosine" "euclidean")
graph_kernels=("heat" "matern" "identity")
combinations=("sum" "first" "second")

# Kernel Specifics
temperatures=(0.1 1.0)
nus=(0.5 1.5 2.5)
kappas=(0.1 0.5)

# --- Fixed Parameters ---
FIXED_ARGS=(
    "transforms=[liftings/graph2hypergraph/kernel]"
    "model.readout.readout_name=PropagateSignalDown"
    "model.feature_encoder.proj_dropout=0.5"
    "trainer.max_epochs=500"
    "trainer.min_epochs=50"
    "trainer.check_val_every_n_epoch=5"
    "callbacks.early_stopping.patience=10"
    "logger.wandb.project=${project_name}"
)


# ==============================================================================
# SECTION 5: PYTHON GENERATOR (CUSTOM LOGIC)
# Explicitly implements the complex Kernel logic from the old script.
# ==============================================================================
generate_combinations() {
# We pass the bash arrays into python by interpolating them into the heredoc string
python3 -c "
import itertools, os

# 1. Load Data from Bash (Interpolated strings)
models = '${models[*]}'.split()
datasets = '${datasets[*]}'.split()
lrs = '${lrs[*]}'.split()
hiddens = '${hidden_channels[*]}'.split()
fractions = '${fractions[*]}'.split()
seeds = '${DATA_SEEDS[*]}'.split()

# Kernel Arrays
combinations = '${combinations[*]}'.split()
kernel_feats = '${kernel_feats[*]}'.split()
graph_kernels = '${graph_kernels[*]}'.split()
temperatures = '${temperatures[*]}'.split()
nus = '${nus[*]}'.split()
kappas = '${kappas[*]}'.split()

run_configs = []

# 2. Build the Logic
# Loop order: Model -> Dataset -> LR -> Hidden -> Fraction -> Combinations -> Kernels -> Seeds
# We put seeds last so they run back-to-back.

base_product = itertools.product(models, datasets, lrs, hiddens, fractions)

for (model, dataset, lr, h, frac) in base_product:
    
    # Logic: Batch Size mapping
    # Old script: MUTAG/PROTEINS -> 256, Others -> 1
    if 'MUTAG' in dataset or 'PROTEINS' in dataset:
        bs = 256
    else:
        bs = 1

    # Logic: EdGNN layers
    # Old script: If not edgnn, set layers=2
    extra_model_args = []
    if 'edgnn' not in model:
        extra_model_args.append('model.backbone.n_layers=2')

    # Logic: Kernels (Nested Conditional Loop)
    for comb in combinations:
        
        # Filter Feature Kernels
        curr_feats = ['identity'] if comb == 'first' else kernel_feats

        for k_feat in curr_feats:
            
            # Filter Graph Kernels
            curr_graphs = ['identity'] if comb == 'second' else graph_kernels

            for k_graph in curr_graphs:
                
                # Determine Hyperparams for Graph Kernel
                param_configs = []
                
                if comb == 'second':
                    # Only feature kernel matters
                    param_configs.append({})
                elif k_graph == 'heat':
                    # Heat: loop temps
                    for t in temperatures:
                        param_configs.append({'t': t})
                elif k_graph == 'matern':
                    # Matern: loop nu and kappa
                    for nu in nus:
                        for kap in kappas:
                            param_configs.append({'nu': nu, 'kappa': kap})
                else:
                    # Identity/Other
                    param_configs.append({})

                # Finally, loop over these kernel configs AND seeds
                for params in param_configs:
                    for seed in seeds:
                        
                        # Build Run Name
                        # ex: unignn2_cora_kernel_sum_kf-cos_kg-heat_t0.1_lr0.01_h64_s0
                        m_short = os.path.basename(model)
                        d_short = os.path.basename(dataset)
                        r_name = f'{m_short}_{d_short}_kernel_{comb}'

                        if comb != 'first':
                            r_name += f'_kf-{k_feat}'
                        
                        if comb != 'second':
                            r_name += f'_kg-{k_graph}'
                            
                            # FIX: Assign to variables first to avoid f-string quote syntax errors
                            if 't' in params: 
                                val_t = params['t']
                                r_name += f'_t{val_t}'
                            if 'nu' in params: 
                                val_nu = params['nu']
                                val_k = params['kappa']
                                r_name += f'_n{val_nu}_k{val_k}'

                        r_name += f'_lr{lr}_h{h}_s{seed}'

                        # Build Arguments List
                        args = [
                            f'model={model}',
                            f'dataset={dataset}',
                            f'optimizer.parameters.lr={lr}',
                            f'model.feature_encoder.out_channels={h}',
                            f'dataset.dataloader_params.batch_size={bs}',
                            f'dataset.split_params.data_seed={seed}',
                            f'transforms.liftings.graph2hypergraph.fraction={frac}',
                            f'transforms.liftings.graph2hypergraph.C={comb}',
                            f'transforms.liftings.graph2hypergraph.feat_kernel={k_feat}',
                            f'transforms.liftings.graph2hypergraph.graph_kernel={k_graph}'
                        ]
                        
                        # Add optional kernel params
                        # FIX: Use variables here too
                        if 't' in params: 
                            val_t = params['t']
                            args.append(f'transforms.liftings.graph2hypergraph.t={val_t}')
                        if 'nu' in params: 
                            val_nu = params['nu']
                            args.append(f'transforms.liftings.graph2hypergraph.nu={val_nu}')
                        if 'kappa' in params: 
                            val_k = params['kappa']
                            args.append(f'transforms.liftings.graph2hypergraph.kappa={val_k}')

                        # Add extra model args
                        args.extend(extra_model_args)

                        run_configs.append((r_name, args))

# 3. Output
print(f'TOTAL;{len(run_configs)}')
for name, args in run_configs:
    print(f'{name};' + ' '.join(args))
"
}


# ==============================================================================
# SECTION 6: MAIN EXECUTION LOOP
# Standard Virtual Slot Loop
# ==============================================================================

echo "----------------------------------------------------------"
echo " Generating experiment combinations..."
echo "----------------------------------------------------------"

total_runs=0
run_counter=0
one_percent_step=1

while IFS=";" read -r col1 col2; do
    
    # 6.1 Handle Header
    if [[ "$col1" == "TOTAL" ]]; then
        total_runs=$col2
        if [ "$total_runs" -gt 0 ]; then one_percent_step=$(( total_runs / 100 )); fi
        if [ "$one_percent_step" -eq 0 ]; then one_percent_step=1; fi
        
        echo "► Total runs planned: $total_runs"
        echo "► Reporting progress every $one_percent_step runs (1%)"
        echo "----------------------------------------------------------"
        continue
    fi

    # 6.2 Parse Run
    run_name="$col1"
    dynamic_args_str="$col2"
    
    ((run_counter++))
    if (( run_counter % one_percent_step == 0 )); then
        if [ "$total_runs" -gt 0 ]; then percent=$(( (run_counter * 100) / total_runs )); else percent=0; fi
        echo "📊 Progress: ${percent}% completed ($run_counter / $total_runs runs launched)"
    fi

    # 6.3 Find Free Slot
    assigned_slot=-1
    while [ "$assigned_slot" -eq -1 ]; do
        for i in "${!gpus[@]}"; do
            pid="${slot_pids[$i]}"
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_slot=$i
                break
            fi
        done
        
        # Wait if all slots are full
        if [ "$assigned_slot" -eq -1 ]; then
            wait -n
        fi
    done

    # 6.4 Prepare Command
    current_gpu=${gpus[$assigned_slot]}
    read -ra DYNAMIC_ARGS_ARRAY <<< "$dynamic_args_str"
    
    cmd=(
        "python" "-m" "topobench"
        "${DYNAMIC_ARGS_ARRAY[@]}"
        "${FIXED_ARGS[@]}"
        "trainer.devices=[${current_gpu}]"
    )

    # 6.5 Execute
    run_and_log "${cmd[*]}" "$log_group" "$run_name" "$LOG_DIR" &
    slot_pids[$assigned_slot]=$!

done < <(generate_combinations)


# ==============================================================================
# SECTION 7: CLEANUP
# ==============================================================================
echo "----------------------------------------------------------"
echo " All jobs launched ($run_counter total)."
echo " Waiting for remaining background jobs to finish..."
echo "----------------------------------------------------------"
wait
echo "✔ All runs complete."
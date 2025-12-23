#!/bin/bash
# ==============================================================================
# SCRIPT: graph_baselines_master.sh
# DESCRIPTION:
#   Runs a scalable hyperparameter sweep for Graph Neural Networks.
#   - ARCHITECTURE: Uses a "Cartesian Product" generation strategy.
#   - CONCURRENCY: Uses "Virtual Slots" to run N jobs per GPU.
#   - ORDERING: Prioritizes running all seeds for a config before moving on.
# ==============================================================================

# ==============================================================================
# SECTION 1: LOGGING & ENVIRONMENT SETUP
# Prepares the logging directory and loads the helper utility script.
# ==============================================================================

# 1.1 Define Project Identifiers
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
project_name="graph_baselines_${script_name}"
log_group="sweep_graphland"
LOG_DIR="./logs/${log_group}"

echo "=========================================================="
echo " Preparing log directory: $LOG_DIR"
echo "=========================================================="

# 1.2 Clean up old logs to ensure a fresh run
if [ -d "$LOG_DIR" ]; then rm -r "$LOG_DIR"; fi
mkdir -p "$LOG_DIR"

# 1.3 Robust Dependency Loading
# This function walks up the directory tree to find 'base/logging.sh'
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
# Defines available GPUs and creates "Virtual Slots".
# ==============================================================================

# 2.1 Configuration
physical_gpus=(0 1 2 3 4 5 6 7)  # IDs of the GPUs to use
JOBS_PER_GPU=1           # Number of parallel runs allowed per GPU

# 2.2 Create Virtual Slots (The 'Scheduling Queue')
# If physical_gpus=(0 1) and JOBS_PER_GPU=2, this creates slots: (0 0 1 1).
# This allows us to treat every "capacity unit" identically in the loop.
gpus=()
for gpu in "${physical_gpus[@]}"; do
    for ((i=1; i<=JOBS_PER_GPU; i++)); do gpus+=("$gpu"); done
done

# 2.3 Initialize Slot Tracking
# We use an array 'slot_pids' to track the Process ID (PID) running in each slot.
# 0 = Empty/Available.
declare -a slot_pids
for i in "${!gpus[@]}"; do slot_pids[$i]=0; done


# ==============================================================================
# SECTION 3: EXPERIMENT PARAMETERS (USER INPUT)
# Define your search space here using standard Bash arrays.
# ==============================================================================

# --- Major Variations ---
models=(
    "graph/gcn"
    "graph/gin"
    "graph/gat"
)

datasets=(
    # # Node classification datasets
    "graph/hm-categories"
    "graph/pokec-regions"
    "graph/tolokers-2"
    "graph/city-reviews"
    "graph/artnet-exp"
    #"graph/web-topics"
    #"graph/web-fraud"

    # # Node regression datasets
    graph/avazu-ctr
    graph/city-roads-L
    graph/city-roads-M
    graph/twitch-views
    graph/artnet-views
    #graph/hm-prices
    #graph/web-traffic

    # # Reruns
    "graph/cocitation_cora"
    "graph/cocitation_citeseer"
    "graph/cocitation_pubmed"
    "graph/amazon_ratings"
    "graph/questions"
    "graph/roman_empire"
)

# --- Hyperparameters ---
batch_sizes=(1)
lrs=(0.001 0.01)
hidden_channels=(16 32 64 128) 
num_layers=(1 2 4 8) 
# The Pivotal Parameter
DATA_SEEDS=(0 3 5 7 9)

# --- Fixed Parameters (Constant for all runs) ---
FIXED_ARGS=(
    "model.feature_encoder.proj_dropout=0.25"
    "model.backbone.dropout=0.25"
    "trainer.max_epochs=500"
    "trainer.min_epochs=50"
    "trainer.check_val_every_n_epoch=5"
    "callbacks.early_stopping.patience=10"
    "logger.wandb.project=${project_name}"
)


# ==============================================================================
# SECTION 4: SWEEP CONFIGURATION MAPPING (CRITICAL ORDERING)
# Format: "ShortTag | HydraKey | ${Array[*]}"
#
# IMPORTANT: The order here defines the execution loop order.
# 1. Top items change SLOWEST (Outer Loop).
# 2. Bottom items change FASTEST (Inner Loop).
#
# To validate results quickly, we put DATA_SEEDS at the very bottom.
# This ensures we run Seed 0, 1, 2... for a specific config BACK-TO-BACK.
# ==============================================================================

SWEEP_CONFIG=(
    # --- LEVEL 1: SLOWEST CHANGING (Outer Loops) ---
    "|model|${models[*]}"
    "|dataset|${datasets[*]}"
    
    # --- LEVEL 2: HYPERPARAMETERS ---
    "L|model.backbone.num_layers|${num_layers[*]}"
    "lr|optimizer.parameters.lr|${lrs[*]}"
    "h|model.feature_encoder.out_channels|${hidden_channels[*]}"
    "bs|dataset.dataloader_params.batch_size|${batch_sizes[*]}"
    
    # --- LEVEL 3: FASTEST CHANGING (Inner Loop) ---
    # We keep seeds last so they run consecutively for every config.
    "seed|dataset.split_params.data_seed|${DATA_SEEDS[*]}"
)


# ==============================================================================
# SECTION 5: PYTHON GENERATOR (INTERNAL UTILITY)
# Runs a temporary Python script to calculate the Cartesian Product.
# It outputs a clean list of runs formatted as: RunName;Args
# ==============================================================================
generate_combinations() {
python3 -c "
import sys, itertools, os

# 1. Parse Input Specs
specs = []
for item in sys.argv[1:]:
    parts = item.split('|')
    tag = parts[0].strip()
    key = parts[1].strip()
    vals = parts[2].split() # Split space-separated string into list
    specs.append({'tag': tag, 'key': key, 'vals': vals})

# 2. Generate Cartesian Product
options = [[(s['tag'], s['key'], val) for val in s['vals']] for s in specs]
combinations = list(itertools.product(*options))

# 3. Print Total Count Header (Using semicolon separator)
print(f'TOTAL;{len(combinations)}')

# 4. Print Each Combination
for combo in combinations:
    # Build Name
    name_parts = []
    for (tag, key, val) in combo:
        clean_val = os.path.basename(val) # 'graph/gcn' -> 'gcn'
        if tag:
            name_parts.append(f'{tag}{clean_val}')
        else:
            name_parts.append(clean_val)
    run_name = '_'.join(name_parts)

    # Build Args
    cmd_args = [f'{key}={val}' for (tag, key, val) in combo]
    
    # Output: RUN_NAME ; ARG1 ARG2 ARG3
    print(f'{run_name};' + ' '.join(cmd_args))
" "${SWEEP_CONFIG[@]}"
}


# ==============================================================================
# SECTION 6: MAIN EXECUTION LOOP
# 1. Reads the Python output line-by-line.
# 2. Finds a free GPU slot.
# 3. Launches the job.
# ==============================================================================

echo "----------------------------------------------------------"
echo " Generating experiment combinations..."
echo "----------------------------------------------------------"

total_runs=0
run_counter=0
one_percent_step=1

# Use process substitution < <(...) to feed the loop
# We use ';' as the IFS delimiter because our python script outputs "Name;Args"
while IFS=";" read -r col1 col2; do
    
    # 6.1 Handle Header (Total Count)
    if [[ "$col1" == "TOTAL" ]]; then
        total_runs=$col2
        # Calculate progress step size
        if [ "$total_runs" -gt 0 ]; then
            one_percent_step=$(( total_runs / 100 ))
        fi
        if [ "$one_percent_step" -eq 0 ]; then one_percent_step=1; fi
        
        echo "► Total runs planned: $total_runs"
        echo "► Reporting progress every $one_percent_step runs (1%)"
        echo "----------------------------------------------------------"
        continue
    fi

    # 6.2 Parse Run Data
    run_name="$col1"
    dynamic_args_str="$col2"
    
    # 6.3 Update Progress
    ((run_counter++))
    if (( run_counter % one_percent_step == 0 )); then
        if [ "$total_runs" -gt 0 ]; then
            percent=$(( (run_counter * 100) / total_runs ))
        else
            percent=0
        fi
        echo "📊 Progress: ${percent}% completed ($run_counter / $total_runs runs launched)"
    fi

    # 6.4 Find a Free GPU Slot (Active Polling)
    # This loop blocks until a slot opens up on ANY GPU.
    assigned_slot=-1
    while [ "$assigned_slot" -eq -1 ]; do
        for i in "${!gpus[@]}"; do
            pid="${slot_pids[$i]}"
            
            # Slot is free if: PID is 0 (start) OR process is dead (kill -0 fails)
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_slot=$i
                break
            fi
        done

        # If all slots are full, wait for the first job to finish to check again
        if [ "$assigned_slot" -eq -1 ]; then
            wait -n
        fi
    done

    # 6.5 Prepare Command
    current_gpu=${gpus[$assigned_slot]}
    read -ra DYNAMIC_ARGS_ARRAY <<< "$dynamic_args_str" # Convert args string to array
    
    cmd=(
        "python" "-m" "topobench"
        "${DYNAMIC_ARGS_ARRAY[@]}"
        "${FIXED_ARGS[@]}"
        "trainer.devices=[${current_gpu}]"
    )

    # 6.6 Execute
    # We launch in background (&) so the loop can continue immediately
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
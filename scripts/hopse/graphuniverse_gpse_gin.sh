#!/bin/bash
# ==============================================================================
# SCRIPT: graphuniverse_gpse_gin.sh
# DESCRIPTION:
#   GraphUniverse GPSE ablation with fixed GIN.
#   - TRANSFORMS vary: no_transform, each PE alone, all PEs combined.
#   - DATASET params vary one sweep-group at a time (not a full cartesian product).
#   - Each unique dataset setting is repeated over GENERATION_SEEDS universe seeds.
#   - Full Hydra cfg (transforms + GraphUniverse generation params) is logged to W&B.
# ==============================================================================

# ==============================================================================
# SECTION 0: USER CONFIGURATION
# ==============================================================================

# Universe generation seeds (dataset.loader...universe_parameters.seed).
# Edit this array to change how many / which seeds are used (e.g. 5 seeds later).
GENERATION_SEEDS=(42 43 44)

wandb_entity="louis-van-langendonck-universitat-polit-cnica-de-catalunya"
wandb_project="graphuniverse_gpse_gin"

# Optional GPU filter (comma-separated indices). Leave empty to use all GPUs.
SELECTED_GPUS="0"

# GraphUniverse generation-parameter sweeps.
# Each JSON object varies ONE parameter; values are tried sequentially (not crossed).
# Baseline values live in configs/dataset/graph/graphuniverse_inductive.yaml:
#   universe: K=20, feature_dim=15, center_variance=0.2, cluster_variance=0.4,
#             edge_propensity_variance=1.0, seed=<GENERATION_SEEDS>
#   family:   n_graphs=1000, n_nodes_range=[50,200], n_communities_range=[3,7],
#             homophily_range=[0.4,0.8], avg_degree_range=[1.0,2.0],
#             degree_separation_range=[0.5,1.0], power_law_exponent_range=[1.5,2.5]
read -r -d '' GU_PARAM_SWEEP_JSON <<'EOF' || true
[
  {"homophily_range": [[0.0, 0.1], [0.45, 0.55], [0.9, 1.0]]},
  {"n_nodes_range": [[50, 100], [100, 200], [200, 300]]},
  {"avg_degree_range": [[1.0, 3.0], [5.0, 7.0], [10.0, 13.0]]},
  {"power_law_exponent_range": [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]}
]
EOF

# --- Transforms (fixed GIN; only encodings change) ---
transform_presets=(
    "notf::no_transform"
    "LapPE::combined_pe@@@transforms.CombinedPSEs.encodings=[LapPE]"
    "RWSE::combined_pe@@@transforms.CombinedPSEs.encodings=[RWSE]"
    "ElectrostaticPE::combined_pe@@@transforms.CombinedPSEs.encodings=[ElectrostaticPE]"
    "HKdiagSE::combined_pe@@@transforms.CombinedPSEs.encodings=[HKdiagSE]"
    "all::combined_pe@@@transforms.CombinedPSEs.encodings=[LapPE,RWSE,ElectrostaticPE,HKdiagSE]"
)

# --- Fixed training / model args (GIN defaults from configs/) ---
FIXED_ARGS=(
    "model=graph/gin"
    "dataset=graph/graphuniverse_inductive"
    "trainer.max_epochs=500"
    "trainer.min_epochs=50"
    "trainer.check_val_every_n_epoch=5"
    "callbacks.early_stopping.patience=5"
)

# ==============================================================================
# SECTION 1: LOGGING & ENVIRONMENT SETUP
# ==============================================================================

script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
project_name="${script_name}"
log_group="graphuniverse_gpse_gin_sweep"
LOG_DIR="./logs/${log_group}"

echo "=========================================================="
echo " Preparing log directory: $LOG_DIR"
echo "=========================================================="

if [ -d "$LOG_DIR" ]; then rm -r "$LOG_DIR"; fi
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HYDRA_FULL_ERROR=1
export GU_PARAM_SWEEP_JSON
export GENERATION_SEEDS_STR="${GENERATION_SEEDS[*]}"
export SELECTED_GPUS

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ==============================================================================
# SECTION 2: HARDWARE & CONCURRENCY
# ==============================================================================

_gpu_info=$(python3 -c "
import subprocess, os

selected_env = os.environ.get('SELECTED_GPUS', '').strip()
allowed_gpus = [x.strip() for x in selected_env.split(',')] if selected_env else None

try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'],
        text=True
    )
    indices, mem_mb = [], []
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        idx = idx.strip()
        if allowed_gpus and idx not in allowed_gpus:
            continue
        indices.append(idx)
        mem_mb.append(int(mem.strip()))

    if not indices:
        print('2 0')
    else:
        min_mem_gb = min(mem_mb) / 1024
        if min_mem_gb >= 80:
            jobs = 4
        elif min_mem_gb <= 30:
            jobs = 2
        else:
            jobs = 3
        print(jobs, ' '.join(indices))
except Exception:
    print('2 0')
")
read -r JOBS_PER_GPU _gpu_ids <<< "$_gpu_info"
read -ra physical_gpus <<< "$_gpu_ids"

echo "✔ Detected ${#physical_gpus[@]} GPU(s): ${physical_gpus[*]}"
echo "✔ Jobs per GPU: $JOBS_PER_GPU"
echo "✔ Generation seeds: ${GENERATION_SEEDS[*]}"

gpus=()
for gpu in "${physical_gpus[@]}"; do
    for ((i=1; i<=JOBS_PER_GPU; i++)); do gpus+=("$gpu"); done
done
echo "✔ Total virtual slots: ${#gpus[@]}"

declare -a slot_pids
for i in "${!gpus[@]}"; do slot_pids[$i]=0; done

# ==============================================================================
# SECTION 3: COMBINATION GENERATOR
# ==============================================================================

generate_combinations() {
python3 -c "
import json, os, sys

# --- inputs from shell ---
sweep_json = os.environ.get('GU_PARAM_SWEEP_JSON', '[]')
generation_seeds = [int(s) for s in os.environ.get('GENERATION_SEEDS_STR', '42 43 44').split()]
transform_presets = sys.argv[1:]

FAMILY_PARAMS = {
    'n_nodes_range', 'n_communities_range', 'homophily_range',
    'avg_degree_range', 'degree_separation_range', 'power_law_exponent_range',
}
UNIVERSE_PARAMS = {
    'K', 'feature_dim', 'center_variance', 'cluster_variance', 'edge_propensity_variance',
}

def hydra_key(param):
    if param in FAMILY_PARAMS:
        return f'dataset.loader.parameters.generation_parameters.family_parameters.{param}'
    if param in UNIVERSE_PARAMS:
        return f'dataset.loader.parameters.generation_parameters.universe_parameters.{param}'
    raise KeyError(f'Unknown GraphUniverse param: {param}')

def hydra_val(value):
    if isinstance(value, list):
        inner = ','.join(hydra_val(v) for v in value)
        return f'[{inner}]'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return format(value, '.15g')
    return str(value)

def alias_val(value):
    if isinstance(value, list):
        return '-'.join(alias_val(v) for v in value)
    if isinstance(value, float):
        s = format(value, '.15g')
        return s.replace('.', 'p')
    return str(value)

def parse_transform_preset(preset):
    if '::' in preset:
        alias, hydra_val_str = preset.split('::', 1)
    else:
        alias, hydra_val_str = preset, preset
    cmd_parts = []
    if '@@@' in hydra_val_str:
        for part in hydra_val_str.split('@@@'):
            part = part.strip()
            if part:
                cmd_parts.append(part)
    else:
        cmd_parts.append(f'transforms={hydra_val_str}')
    return alias, cmd_parts

try:
    sweep_groups = json.loads(sweep_json)
except json.JSONDecodeError as exc:
    print(f'Invalid GU_PARAM_SWEEP_JSON: {exc}', file=sys.stderr)
    sys.exit(1)

if not isinstance(sweep_groups, list):
    print('GU_PARAM_SWEEP_JSON must be a JSON list of objects', file=sys.stderr)
    sys.exit(1)

transforms = [parse_transform_preset(p) for p in transform_presets]
runs = []

for group_idx, group in enumerate(sweep_groups):
    if not isinstance(group, dict) or len(group) != 1:
        print(f'Sweep group {group_idx} must be a single-key dict, got: {group!r}', file=sys.stderr)
        sys.exit(1)
    param_name, values = next(iter(group.items()))
    if not isinstance(values, list):
        values = [values]
    for value in values:
        ds_key = hydra_key(param_name)
        ds_override = f'{ds_key}={hydra_val(value)}'
        ds_alias = f'gu{param_name}_{alias_val(value)}'
        group_name = f'gu_{param_name}'
        for gseed in generation_seeds:
            seed_override = (
                'dataset.loader.parameters.generation_parameters.universe_parameters.seed='
                f'{gseed}'
            )
            for tf_alias, tf_cmd_parts in transforms:
                name_parts = [ds_alias, f'gseed{gseed}', f'tf{tf_alias}']
                run_name = '_'.join(name_parts)
                cmd_args = [ds_override, seed_override] + tf_cmd_parts
                wandb_tags = [
                    'graphuniverse', 'gpse', 'gin', param_name, tf_alias, f'gseed{gseed}',
                ]
                runs.append((run_name, cmd_args, group_name, wandb_tags))

print(f'TOTAL;{len(runs)}')
for run_name, cmd_args, group_name, wandb_tags in runs:
    tags_str = ','.join(wandb_tags)
    print(run_name + ';' + ' '.join(cmd_args) + ';;' + group_name + ';;' + tags_str)
" "${transform_presets[@]}"
}

# ==============================================================================
# SECTION 4: MAIN EXECUTION LOOP
# ==============================================================================

repair_hydra_transforms_arg() {
    local -n _r=$1
    local out=() i
    for ((i = 0; i < ${#_r[@]}; i++)); do
        local t="${_r[i]}"
        if [[ "$t" == transforms=* ]]; then
            out+=("$t")
        elif [[ "$t" == "transforms" && $((i + 1)) -lt ${#_r[@]} ]]; then
            local nxt="${_r[$((i + 1))]}"
            [[ "$nxt" == *"="* ]] && { out+=("$t"); continue; }
            out+=("transforms=$nxt")
            ((i++))
        elif [[ "$t" =~ ^(combined_pe|combined_fe|no_transform)$ ]]; then
            out+=("transforms=$t")
        else
            out+=("$t")
        fi
    done
    _r=("${out[@]}")
}

echo "----------------------------------------------------------"
echo " Generating experiment combinations..."
echo "----------------------------------------------------------"

total_runs=0
run_counter=0
one_percent_step=1

while IFS=";" read -r run_name dynamic_args_str wandb_group wandb_tags_str; do

    if [[ "$run_name" == "TOTAL" ]]; then
        total_runs=$dynamic_args_str
        if [ "$total_runs" -gt 0 ]; then
            one_percent_step=$(( total_runs / 100 ))
        fi
        if [ "$one_percent_step" -eq 0 ]; then one_percent_step=1; fi
        echo "► Total runs planned: $total_runs"
        echo "► Reporting progress every $one_percent_step runs (1%)"
        echo "----------------------------------------------------------"
        continue
    fi

    ((run_counter++))
    if (( run_counter % one_percent_step == 0 )); then
        if [ "$total_runs" -gt 0 ]; then
            percent=$(( (run_counter * 100) / total_runs ))
        else
            percent=0
        fi
        echo "📊 Progress: ${percent}% completed ($run_counter / $total_runs runs launched)"
    fi

    assigned_slot=-1
    while [ "$assigned_slot" -eq -1 ]; do
        for i in "${!gpus[@]}"; do
            pid="${slot_pids[$i]}"
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_slot=$i
                break
            fi
        done
        if [ "$assigned_slot" -eq -1 ]; then
            wait -n
        fi
    done

    current_gpu=${gpus[$assigned_slot]}
    IFS=$' \t\n' read -ra DYNAMIC_ARGS_ARRAY <<< "$dynamic_args_str"
    repair_hydra_transforms_arg DYNAMIC_ARGS_ARRAY

    IFS=',' read -ra WANDB_TAGS_ARRAY <<< "$wandb_tags_str"
    wandb_tags_hydra="[$(printf '%s,' "${WANDB_TAGS_ARRAY[@]}" | sed 's/,$//')]"
    # Hydra list syntax needs quotes for string tags with special chars — use bare tags here.

    cmd=(
        "python" "-m" "topobench"
        "${DYNAMIC_ARGS_ARRAY[@]}"
        "${FIXED_ARGS[@]}"
        "trainer.devices=[${current_gpu}]"
        "+logger.wandb.entity=${wandb_entity}"
        "logger.wandb.project=${wandb_project}"
        "logger.wandb.group=${wandb_group}"
        "logger.wandb.name=${run_name}"
        "tags=${wandb_tags_hydra}"
    )

    cmd_eval=$(printf '%q ' "${cmd[@]}")
    run_and_log "${cmd_eval% }" "$log_group" "$run_name" "$LOG_DIR" &
    slot_pids[$assigned_slot]=$!

done < <(generate_combinations)

echo "----------------------------------------------------------"
echo " All jobs launched ($run_counter total)."
echo " Waiting for remaining background jobs to finish..."
echo "----------------------------------------------------------"
wait
echo "✔ All runs complete."

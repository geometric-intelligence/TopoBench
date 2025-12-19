#!/bin/bash
# Hyperparameter sweep with parallel job management.

LOG_DIR="./logs/tabular_nfa"
echo "Preparing a clean log directory at: $LOG_DIR"

if [ -d "$LOG_DIR" ]; then
    rm -r "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"
echo "Folder ready: $LOG_DIR"

# Project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" &> /dev/null && pwd)"

export HYDRA_FULL_ERROR=1
source "$SCRIPT_DIR/base/logging.sh"

# -------------------------
# Sweep grids
# -------------------------
models=(
    "tabpfn_c"
)

datasets=(
    "graph/roman_empire"
    "graph/cocitation_cora"
    "graph/cocitation_citeseer"
    "graph/cocitation_pubmed"
    "graph/amazon_ratings"
    "graph/minesweeper"
    "graph/questions"
    "graph/tolokers"
    "graph/tolokers-2"
#  "graph/web-topics"
  "graph/hm-categories"
#  "graph/pokec-regions"
#    "graph/city-reviews"
    "graph/artnet-exp"
#    "graph/web-fraud"
    "graph/wiki_cs"
)

# these datasets have more thant 10 classes, which is over the limit for TabPFN
OVR_DATASETS=(
  "graph/roman_empire"
#  "graph/web-topics"
  "graph/hm-categories"
#  "graph/pokec-regions"
)

DATA_SEEDS=(0 3 5 7 9)

# ✅ Loop: use NFA or not
USE_NFA=(true false)

# ✅ Loop: sampler choices (Hydra group names)
# Esempi: cambia questi ai nomi reali nel tuo config (sampler/...)
SAMPLERS=(
# "none"
#  "knn"
#  "graph_hop"
  "composite"
)

HOPS=(1 2 3 4 5)
TEST_POINTS=(1 2 4 8 16 32)
Ks=(5 10 20 50)

# -------------------------
# Execution tracking
# -------------------------
ROOT_LOG_DIR="$LOG_DIR"
run_counter=1
job_counter=0
MAX_PARALLEL=1

num_datasets=${#datasets[@]}

for model in "${models[@]}"; do
  for i in $(seq 0 $((num_datasets - 1))); do
    dataset="${datasets[i]}"

    for data_seed in "${DATA_SEEDS[@]}"; do
      for use_nfa in "${USE_NFA[@]}"; do
        for sampler in "${SAMPLERS[@]}"; do
          for k in "${Ks[@]}"; do
            for hop in "${HOPS[@]}"; do
              for test_points in "${TEST_POINTS[@]}"; do

                project_name="graph_tabpfn"
                log_group="tabular_nfa"
                sampler_tag="sampler${sampler}"

                # NFA on/off: transforms=nfa solo se true
                transforms_args=()
                nfa_tag="noNFA"
                if [[ "$use_nfa" == "true" ]]; then
                  transforms_args=("transforms=nfa")
                  nfa_tag="NFA"
                fi

                # ✅ SEMPRE: overrides di hops/test_points (cambia le chiavi se nel tuo config sono diverse)
                nfa_params_args=(
                  "transforms.nfa.hops=${hop}"
                  "transforms.nfa.test_points=${test_points}"
                )

                # OVR datasets -> change model
                model_tag="tabpfn_c"
                for ovr_ds in "${OVR_DATASETS[@]}"; do
                  if [[ "$dataset" == "$ovr_ds" ]]; then
                    model_tag="tabpfn_ovr"
                    break
                  fi
                done

                run_name="${model##*/}_${dataset##*/}_seed${data_seed}_${model_tag}_${nfa_tag}_hops${hop}k${k}_tp${test_points}_${sampler_tag}"

                cmd=(
                  "python" "-m" "topobench"
                  "model=non_relational/sklearn_classifier"
                  "model/non_relational/sklearn@model.backbone=${model_tag}"
                  "model/non_relational/sklearn/samplers@model.backbone_wrapper.sampler=${sampler}"
                  "model.backbone_wrapper.sampler.n_hops=${hop}"
                  "model.backbone_wrapper.sampler.k=${k}"
                  "model.backbone_wrapper.num_test_nodes=${test_points}"
                  "dataset=${dataset}"
                  "dataset.split_params.split_type=stratified"
                  "train=False"
                  "trainer=gpu"
                  "evaluator=classification_extended"
                  "dataset.split_params.data_seed=${data_seed}"
                  "logger.wandb.project=${project_name}"
                  "${transforms_args[@]}"
                )

                echo "============================================================"
                echo "Starting Run #$run_counter: $run_name"
                echo "Command: ${cmd[*]}"
                echo "============================================================"

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

echo "Waiting for the final batch of jobs to finish..."
wait
echo "All runs complete."
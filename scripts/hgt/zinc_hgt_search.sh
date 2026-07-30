#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-cell-hgt-zinc}"
ACCELERATOR="${TOPOBENCH_ACCELERATOR:-cpu}"
DEVICES="${TOPOBENCH_DEVICES:-1}"

usage() {
    cat >&2 <<'EOF'
Usage:
  zinc_hgt_search.sh depth [seed]
  zinc_hgt_search.sh heads <best_depth> [seed]
  zinc_hgt_search.sh width <best_depth> <best_heads> [seed]
  zinc_hgt_search.sh lr <best_depth> <best_heads> <best_width> [seed]
  zinc_hgt_search.sh final <best_depth> <best_heads> <best_width> <best_lr> [seed]

Environment:
  WANDB_PROJECT          Shared W&B project (default: cell-hgt-zinc)
  WANDB_ENTITY           Optional W&B user or team
  TOPOBENCH_ACCELERATOR  Lightning accelerator (default: cpu)
  TOPOBENCH_DEVICES      Lightning devices value (default: 1)
  DRY_RUN=1              Print commands without starting training
EOF
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 2
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] \
        || fail "${name} must be a positive integer, got '${value}'"
}

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] \
        || fail "${name} must be a non-negative integer, got '${value}'"
}

learning_rate_tag() {
    case "$1" in
        0.0005) printf '5e-4' ;;
        0.001) printf '1e-3' ;;
        0.002) printf '2e-3' ;;
        *) printf '%s' "${1//./p}" ;;
    esac
}

print_command() {
    printf 'DRY RUN: '
    printf '%q ' "$@"
    printf '\n'
}

run_candidate() {
    local phase="$1"
    local depth="$2"
    local heads="$3"
    local width="$4"
    local learning_rate="$5"
    local seed="$6"
    local run_test="$7"

    require_positive_integer "depth" "$depth"
    require_positive_integer "heads" "$heads"
    require_positive_integer "width" "$width"
    require_nonnegative_integer "seed" "$seed"

    if ((10#${width} % 10#${heads} != 0)); then
        fail "width ${width} must be divisible by heads ${heads}"
    fi

    local depth_tag
    local heads_tag
    local width_tag
    local lr_tag
    printf -v depth_tag '%02d' "$depth"
    printf -v heads_tag '%02d' "$heads"
    printf -v width_tag '%03d' "$width"
    lr_tag="$(learning_rate_tag "$learning_rate")"

    local group_name="zinc-hgt-${phase}-s${seed}"
    local job_type="${phase}-screen"
    if [[ "$phase" == "final" ]]; then
        job_type="final-evaluation"
    fi
    local run_name
    run_name="zinc-hgt-${phase}-d${depth_tag}-h${heads_tag}"
    run_name+="-w${width_tag}-lr${lr_tag}-s${seed}"

    local -a wandb_arguments=(
        "logger.wandb.project=${WANDB_PROJECT_NAME}"
        "logger.wandb.group=${group_name}"
        "logger.wandb.job_type=${job_type}"
        "+logger.wandb.name=${run_name}"
        "logger.wandb.tags=[cell,hgt,zinc,hpo,${phase}]"
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        wandb_arguments+=("+logger.wandb.entity=${WANDB_ENTITY}")
    fi

    local -a command=(
        "$PYTHON_BIN"
        -m
        topobench
        "experiment=cell_hgt_zinc"
        "logger=wandb"
        "${wandb_arguments[@]}"
        "seed=${seed}"
        "model.feature_encoder.out_channels=${width}"
        "model.feature_encoder.proj_dropout=0.1"
        "model.backbone.num_layers=${depth}"
        "model.backbone.heads=${heads}"
        "model.backbone.dropout=0.1"
        "dataset.dataloader_params.batch_size=128"
        "optimizer.parameters.lr=${learning_rate}"
        "optimizer.parameters.weight_decay=0.0001"
        "optimizer.scheduler.scheduler_id=StepLR"
        "optimizer.scheduler.scheduler_params.step_size=50"
        "optimizer.scheduler.scheduler_params.gamma=0.5"
        "trainer.accelerator=${ACCELERATOR}"
        "trainer.devices=${DEVICES}"
        "trainer.min_epochs=50"
        "trainer.max_epochs=500"
        "trainer.check_val_every_n_epoch=5"
        "callbacks.early_stopping.patience=10"
        "callbacks.early_stopping.min_delta=0.005"
        "train=true"
        "test=${run_test}"
    )

    if command -v caffeinate >/dev/null 2>&1; then
        command=(caffeinate -i "${command[@]}")
    fi

    printf '\nStarting %s\n' "$run_name"
    printf 'W&B project: %s | group: %s\n' \
        "$WANDB_PROJECT_NAME" "$group_name"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        print_command "${command[@]}"
    else
        "${command[@]}"
    fi
}

[[ -x "$PYTHON_BIN" ]] \
    || fail "Python executable not found at ${PYTHON_BIN}"

phase="${1:-}"
case "$phase" in
    depth)
        [[ "$#" -le 2 ]] || {
            usage
            exit 2
        }
        seed="${2:-0}"
        for depth in 2 4 8; do
            run_candidate "depth" "$depth" 4 64 0.001 "$seed" false
        done
        ;;
    heads)
        [[ "$#" -ge 2 && "$#" -le 3 ]] || {
            usage
            exit 2
        }
        best_depth="$2"
        seed="${3:-0}"
        for heads in 2 8; do
            run_candidate \
                "heads" "$best_depth" "$heads" 64 0.001 "$seed" false
        done
        ;;
    width)
        [[ "$#" -ge 3 && "$#" -le 4 ]] || {
            usage
            exit 2
        }
        best_depth="$2"
        best_heads="$3"
        seed="${4:-0}"
        run_candidate "width" \
            "$best_depth" "$best_heads" 128 0.001 "$seed" false
        ;;
    lr)
        [[ "$#" -ge 4 && "$#" -le 5 ]] || {
            usage
            exit 2
        }
        best_depth="$2"
        best_heads="$3"
        best_width="$4"
        seed="${5:-0}"
        for learning_rate in 0.0005 0.002; do
            run_candidate \
                "lr" \
                "$best_depth" \
                "$best_heads" \
                "$best_width" \
                "$learning_rate" \
                "$seed" \
                false
        done
        ;;
    final)
        [[ "$#" -ge 5 && "$#" -le 6 ]] || {
            usage
            exit 2
        }
        best_depth="$2"
        best_heads="$3"
        best_width="$4"
        best_learning_rate="$5"
        seed="${6:-0}"
        run_candidate \
            "final" \
            "$best_depth" \
            "$best_heads" \
            "$best_width" \
            "$best_learning_rate" \
            "$seed" \
            true
        ;;
    *)
        usage
        exit 2
        ;;
esac

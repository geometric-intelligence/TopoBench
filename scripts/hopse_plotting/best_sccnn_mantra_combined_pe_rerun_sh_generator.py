#!/usr/bin/env python3
"""
Best-val SCCNN + CombinedPE reruns on MANTRA only (optional rebuttal ablation).

Same best-validation selection as ``best_rerun_sh_generator.py`` / the graph HOMP
CombinedPE generator, but restricted to:

- model: ``simplicial/sccnn`` (emitted as ``simplicial/sccnn_custom``)
- datasets: ``simplicial/mantra_{name,orientation,betti_numbers}``

Forces ``transforms=combined_pe`` (PSEs only — mantra is already simplicial, so no
``graph2simplicial`` lifting). W&B project defaults to
``best_runs_sccnn_mantra_combined_pe``.

Usage::

    python scripts/hopse_plotting/best_sccnn_mantra_combined_pe_rerun_sh_generator.py
    python scripts/hopse_plotting/best_sccnn_mantra_combined_pe_rerun_sh_generator.py \\
        -o scripts/sccnn_mantra_combined_pe_reruns_sequential.sh \\
        --output-parallel scripts/sccnn_mantra_combined_pe_reruns_parallel.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import best_rerun_sh_generator as br
from best_homp_combined_pe_rerun_sh_generator import (
    DEFAULT_WANDB_ENTITY,
    _add_shared_cli_args,
    run_generator,
)
from main_loader import DATASETS as LOADER_DATASETS

SCCNN_MODEL_ALLOWLIST = frozenset(
    {
        "simplicial/sccnn",
        "simplicial/sccnn_custom",
    }
)

MANTRA_RERUN_HYDRA_DATASETS: frozenset[str] = frozenset(
    d for d in LOADER_DATASETS if d.startswith("simplicial/mantra")
)

_DEFAULT_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EMIT_SH_SEQUENTIAL = (
    _DEFAULT_SCRIPTS_DIR / "sccnn_mantra_combined_pe_reruns_sequential.sh"
)
DEFAULT_EMIT_SH_PARALLEL = (
    _DEFAULT_SCRIPTS_DIR / "sccnn_mantra_combined_pe_reruns_parallel.sh"
)

DEFAULT_WANDB_PROJECT = "best_runs_sccnn_mantra_combined_pe"


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Emit sequential + parallel bash scripts for best-val SCCNN reruns "
            "with CombinedPE on MANTRA datasets only."
        )
    )
    _add_shared_cli_args(p)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_EMIT_SH_SEQUENTIAL,
        help=f"Sequential .sh path (default: {DEFAULT_EMIT_SH_SEQUENTIAL})",
    )
    p.add_argument(
        "--output-parallel",
        type=Path,
        default=DEFAULT_EMIT_SH_PARALLEL,
        help=f"Parallel .sh path (default: {DEFAULT_EMIT_SH_PARALLEL})",
    )
    p.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project for every command (default: {DEFAULT_WANDB_PROJECT})",
    )
    args = p.parse_args()

    run_generator(
        input_csv=args.input,
        output_sequential=args.output,
        output_parallel=None if args.no_parallel_script else args.output_parallel,
        allowlist=SCCNN_MODEL_ALLOWLIST,
        allowed_hydra_datasets=MANTRA_RERUN_HYDRA_DATASETS,
        wandb_project=str(args.wandb_project),
        interpreter=args.interpreter,
        data_seeds=str(args.data_seeds),
        max_epochs=int(args.max_epochs),
        early_stopping_patience=args.early_stopping_patience,
        fixed_args_profile=str(args.fixed_args_profile),
        append_args=list(args.append_arg),
        keep_row_seed=args.keep_row_seed,
        wandb_entity=str(args.wandb_entity),
        no_wandb_logger=args.no_wandb_logger,
        no_wandb_run_name=args.no_wandb_run_name,
        parallel_gpus=str(args.parallel_gpus),
        parallel_jobs_per_gpu=int(args.parallel_jobs_per_gpu),
        all_matching_datasets=args.all_datasets,
    )


if __name__ == "__main__":
    main()

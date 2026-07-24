#!/usr/bin/env python3
"""
Best-val HOMP reruns with CombinedPE + lifting (rebuttal ablation).

Same best-validation selection as ``best_rerun_sh_generator.py``, but restricted to
real HOMP models:

- ``cell/cwn``
- ``cell/cccn``
- ``simplicial/sccnn`` (emitted as ``simplicial/sccnn_custom``)

and to **graph** datasets from ``main_loader.DATASETS`` (cocitation trio excluded,
same default as the parent generator).

Every command forces:

- cell models  → ``transforms=combined_pe_graph2cell``
- sccnn        → ``transforms=combined_pe_graph2sim``

instead of the CSV / default lifting-only transform. W&B logs to a dedicated project
(``best_runs_homp_combined_pe`` by default).

Usage::

    python scripts/hopse_plotting/best_homp_combined_pe_rerun_sh_generator.py
    python scripts/hopse_plotting/best_homp_combined_pe_rerun_sh_generator.py \\
        -i scripts/hopse_plotting/csvs/hopse_experiments_wandb_export_seed_agg.csv \\
        -o scripts/homp_combined_pe_reruns_sequential.sh \\
        --output-parallel scripts/homp_combined_pe_reruns_parallel.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import best_rerun_sh_generator as br
from main_loader import DATASETS as LOADER_DATASETS
from utils import (
    DEFAULT_AGGREGATED_EXPORT_CSV,
    SEED_COLUMN,
    hydra_dataset_key_from_loader_identity,
    load_wandb_export_csv,
    safe_filename_token,
)

# -----------------------------------------------------------------------------
# HOMP CombinedPE rebuttal settings
# -----------------------------------------------------------------------------
HOMP_MODEL_ALLOWLIST = frozenset(
    {
        "cell/cccn",
        "cell/cwn",
        "simplicial/sccnn",
        "simplicial/sccnn_custom",
    }
)

TRANSFORM_CELL = "combined_pe_graph2cell"
TRANSFORM_SIMPLICIAL = "combined_pe_graph2sim"
# Native simplicial (mantra): CombinedPSEs only — no graph→complex lifting.
TRANSFORM_MANTRA = "combined_pe"

_TRANSDUCTIVE_COCITATION_HYDRA: frozenset[str] = frozenset(
    {
        "graph/cocitation_cora",
        "graph/cocitation_citeseer",
        "graph/cocitation_pubmed",
    }
)
GRAPH_RERUN_HYDRA_DATASETS: frozenset[str] = frozenset(
    d
    for d in LOADER_DATASETS
    if d.startswith("graph/") and d not in _TRANSDUCTIVE_COCITATION_HYDRA
)

_DEFAULT_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EMIT_SH_SEQUENTIAL = _DEFAULT_SCRIPTS_DIR / "homp_combined_pe_reruns_sequential.sh"
DEFAULT_EMIT_SH_PARALLEL = _DEFAULT_SCRIPTS_DIR / "homp_combined_pe_reruns_parallel.sh"

DEFAULT_WANDB_ENTITY = br.DEFAULT_WANDB_ENTITY
DEFAULT_WANDB_PROJECT = "best_runs_homp_combined_pe"

_NESTED_COMBINED_ENCODING_PREFIXES = (
    "transforms.CombinedPSEs.",
    "transforms.CombinedFEs.",
)


def combined_pe_transform_for_homp_model(
    model: str,
    *,
    dataset: str | None = None,
) -> str:
    """Map HOMP model (+ optional dataset) → CombinedPE Hydra transform preset."""
    m = str(model).replace("\r", "").strip().lower()
    ds = hydra_dataset_key_from_loader_identity(
        str(dataset or "").replace("\r", "").strip()
    )
    if ds.startswith("simplicial/mantra"):
        return TRANSFORM_MANTRA
    tail = m.split("/")[-1]
    if m.startswith("cell/") or tail in {"cwn", "cccn"}:
        return TRANSFORM_CELL
    if m.startswith("simplicial/") or tail.startswith("sccnn"):
        return TRANSFORM_SIMPLICIAL
    raise ValueError(f"No CombinedPE transform mapping for model={model!r}")


def force_transforms_override(parts: list[str], transform: str) -> None:
    """Replace or append ``transforms=...``; drop nested CombinedPSE/FE keys from the winner row."""
    target = f"transforms={transform}"
    parts[:] = [
        p
        for p in parts
        if not any(p.startswith(pref) for pref in _NESTED_COMBINED_ENCODING_PREFIXES)
    ]
    for i, p in enumerate(parts):
        if p.startswith("transforms="):
            parts[i] = target
            break
    else:
        parts.append(target)


def _annotate_wandb_name_combined_pe(parts: list[str]) -> None:
    for i, p in enumerate(parts):
        if p.startswith("+logger.wandb.name="):
            name = p.split("=", 1)[1]
            if "__combined_pe" not in name:
                name = safe_filename_token(f"{name}__combined_pe", max_len=120)
                parts[i] = f"+logger.wandb.name={name}"
            break


def install_homp_combined_pe_hooks(*, allowlist: frozenset[str]) -> None:
    """
    Patch ``best_rerun_sh_generator`` so emit helpers use the HOMP allowlist and
    force CombinedPE transforms on every command.
    """
    br.RERUN_MODEL_ALLOWLIST = allowlist
    br.RERUN_HOPSE_M_BRANCHES = None
    _orig_base = br._base_hydra_parts_for_row

    def _base_with_combined_pe(*args, **kwargs):
        model, dataset, parts = _orig_base(*args, **kwargs)
        force_transforms_override(
            parts,
            combined_pe_transform_for_homp_model(model, dataset=dataset),
        )
        _annotate_wandb_name_combined_pe(parts)
        return model, dataset, parts

    br._base_hydra_parts_for_row = _base_with_combined_pe


def _filter_models(df, allowlist: frozenset[str]):
    if "model" not in df.columns:
        raise KeyError("CSV missing 'model' column")

    def keep(m: object) -> bool:
        s = str(m).replace("\r", "").strip()
        return br._csv_model_matches_rerun_allowlist(s, allowlist)

    return df.loc[df["model"].map(keep)].copy()


def run_generator(
    *,
    input_csv: Path,
    output_sequential: Path,
    output_parallel: Path | None,
    allowlist: frozenset[str],
    allowed_hydra_datasets: frozenset[str],
    wandb_project: str,
    interpreter: str = "python",
    data_seeds: str = br.DEFAULT_SWEEP_DATA_SEEDS,
    max_epochs: int = br.DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int | None = None,
    fixed_args_profile: str = "auto",
    append_args: list[str] | None = None,
    keep_row_seed: bool = False,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    no_wandb_logger: bool = False,
    no_wandb_run_name: bool = False,
    parallel_gpus: str = br.DEFAULT_PARALLEL_GPUS,
    parallel_jobs_per_gpu: int = 1,
    all_matching_datasets: bool = False,
) -> None:
    """Shared entry used by the graph HOMP script and the mantra-only SCCNN script."""
    install_homp_combined_pe_hooks(allowlist=allowlist)

    wb_ent: str | None = None
    wb_proj: str | None = None
    if not no_wandb_logger:
        wb_ent = str(wandb_entity).replace("\r", "").strip()
        wb_proj = str(wandb_project).replace("\r", "").strip()

    df = load_wandb_export_csv(input_csv)
    n_in = len(df)
    df = _filter_models(df, allowlist)
    print(f"Model filter: {n_in} -> {len(df)} rows (allowlist={sorted(allowlist)})")

    if not all_matching_datasets:
        n_mid = len(df)
        df = br.dataframe_filter_rerun_datasets(df, allowed_hydra=allowed_hydra_datasets)
        print(
            f"Dataset filter: {n_mid} -> {len(df)} rows "
            f"({len(allowed_hydra_datasets)} allowed Hydra paths)"
        )

    if df.empty:
        raise SystemExit(
            "No rows left after model/dataset filters. Check the seed-aggregated CSV "
            "covers the requested HOMP models and datasets."
        )

    if keep_row_seed and SEED_COLUMN not in df.columns:
        print(
            f"Note: --keep-row-seed but CSV has no {SEED_COLUMN!r} column; "
            "using --data-seeds."
        )

    # No HOPSE-M F/C split needed for these models.
    effective_group = ["model", "dataset"]
    seeds = br._parse_data_seeds(str(data_seeds).replace("\r", ""))

    print(f"Rerun filter: RERUN_MODEL_ALLOWLIST={br.RERUN_MODEL_ALLOWLIST!r}")
    print(f"W&B project: {wb_proj!r}")

    # Sanity: show transform mapping for surviving (model, dataset) pairs.
    pairs = sorted(
        {
            (
                str(r["model"]).replace("\r", "").strip(),
                hydra_dataset_key_from_loader_identity(
                    str(r["dataset"]).replace("\r", "").strip()
                ),
            )
            for _, r in df[["model", "dataset"]].drop_duplicates().iterrows()
        }
    )
    shown: set[tuple[str, str]] = set()
    for m, ds in pairs:
        t = combined_pe_transform_for_homp_model(m, dataset=ds)
        key = (m, t)
        if key in shown:
            continue
        shown.add(key)
        print(f"  transform map: {m} (+ matching datasets) -> {t}")

    common_kw = dict(
        interpreter=interpreter,
        data_seeds=seeds,
        append_args=list(append_args or []),
        keep_row_seed=keep_row_seed,
        group_cols=effective_group,
        max_epochs=int(max_epochs),
        early_stopping_patience=early_stopping_patience,
        fixed_args_profile=str(fixed_args_profile),
        wandb_entity=wb_ent,
        wandb_project=wb_proj,
        wandb_run_name=not no_wandb_run_name,
    )

    n = br.emit_sequential_rerun_script(df, path=output_sequential, **common_kw)
    print(f"Wrote {n} sequential command(s) -> {output_sequential}")

    if output_parallel is not None:
        gpus = br._parse_parallel_gpus(parallel_gpus)
        n2 = br.emit_parallel_rerun_script(
            df,
            path=output_parallel,
            gpu_ids=gpus,
            jobs_per_gpu=int(parallel_jobs_per_gpu),
            **common_kw,
        )
        slots = len(gpus) * max(1, int(parallel_jobs_per_gpu))
        print(
            f"Wrote {n2} parallel command(s) -> {output_parallel} "
            f"(GPUs {gpus}, max {slots} concurrent via slot pool)"
        )


def _add_shared_cli_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_AGGREGATED_EXPORT_CSV,
        help=f"Seed-aggregated CSV (default: {DEFAULT_AGGREGATED_EXPORT_CSV})",
    )
    p.add_argument(
        "--interpreter",
        default="python",
        help="Python executable (default: python)",
    )
    p.add_argument(
        "--data-seeds",
        default=br.DEFAULT_SWEEP_DATA_SEEDS,
        help=(
            "Comma-separated dataset.split_params.data_seed values "
            f"(default: {br.DEFAULT_SWEEP_DATA_SEEDS})"
        ),
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=br.DEFAULT_MAX_EPOCHS,
        help=f"trainer.max_epochs (default: {br.DEFAULT_MAX_EPOCHS})",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        metavar="N",
        help="callbacks.early_stopping.patience; default 10 for HOMP profiles under auto.",
    )
    p.add_argument(
        "--fixed-args-profile",
        choices=("auto", "graph", "hopse", "topotune", "sann", "sccnn", "cwn", "none"),
        default="auto",
        help="Sweep-style extras after row overrides (default: auto).",
    )
    p.add_argument(
        "--append-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override appended last (repeatable).",
    )
    p.add_argument(
        "--keep-row-seed",
        action="store_true",
        help="Prefer CSV data_seed when present (per-run export).",
    )
    p.add_argument(
        "--wandb-entity",
        default=DEFAULT_WANDB_ENTITY,
        help=f"W&B entity (default: {DEFAULT_WANDB_ENTITY})",
    )
    p.add_argument(
        "--no-wandb-logger",
        action="store_true",
        help="Omit logger.wandb entity/project/name overrides.",
    )
    p.add_argument(
        "--no-wandb-run-name",
        action="store_true",
        help="Keep entity/project but omit +logger.wandb.name.",
    )
    p.add_argument(
        "--no-parallel-script",
        action="store_true",
        help="Only write the sequential script.",
    )
    p.add_argument(
        "--parallel-gpus",
        default=br.DEFAULT_PARALLEL_GPUS,
        help=f"Comma-separated GPU indices (default: {br.DEFAULT_PARALLEL_GPUS})",
    )
    p.add_argument(
        "--parallel-jobs-per-gpu",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent jobs per physical GPU in the parallel script (default: 1).",
    )
    p.add_argument(
        "--all-datasets",
        action="store_true",
        help="Skip the allowed-Hydra dataset allowlist (still model-filtered).",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Emit sequential + parallel bash scripts for best-val HOMP reruns "
            "with CombinedPE+lifting transforms on graph datasets."
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
        allowlist=HOMP_MODEL_ALLOWLIST,
        allowed_hydra_datasets=GRAPH_RERUN_HYDRA_DATASETS,
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

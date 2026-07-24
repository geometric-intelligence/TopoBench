#!/usr/bin/env python3
"""
Fetch **best-val rerun** runs from a single W&B project (default ``best_runs_rerun``),
aggregate across ``dataset.split_params.data_seed`` like ``aggregator.py``, write one
seed-aggregated CSV, and emit **collapsed** LaTeX tables (one GNN row per backbone), same
column layout as ``table_generator`` compact mode:

1. **Performance** — test mean ± std from ``summary_test_best_rerun/*`` (same picks as main tables).
2. **Train time per epoch** — from ``AvgTime/train_epoch_mean`` / ``AvgTime/train_epoch_std``.
   TopoBench logs these via ``log_hyperparams`` (W&B **config**, not scalar summary); this
   script copies them into ``summary_*`` for the CSV. For each (model, dataset, submodel),
   among all raw seeds with finite timing, the run with the **lowest within-run epoch std**
   (``summary_AvgTime/train_epoch_std``) is kept as the most stable timing; ± is that
   within-run variability. Two LaTeX tables use the same numbers: one bolds the column
   minimum **per domain** (graph / simplicial / cell), the other **across all models**.
3. **End-to-end wall time** — W&B's ``_runtime`` (seconds) is mapped to ``summary_Runtime`` and
   aggregated across seeds (mean ± std).

Time ``.tex`` tables (runtime + per-epoch): default **bold** = lowest mean in that **dataset
column** within the same **domain band**; **blue** = not significantly different from that
within-domain minimum (two-sided $Z$ at 95\\,\\%, SE $= \\sigma/\\sqrt{n\_\\mathrm{seeds}}$).
Dataset column headers omit performance $\\uparrow$/$\\downarrow$ markers (unlike the main
rerun performance table). The extra ``rerun_time_train_epoch_bold_global.tex`` uses the same
per-epoch cells but bolds the minimum **across all models** in each column.

**Scatter plots** (optional ``--scatter-plots``): three faceted figure families, like
``plot_topology_timing.py`` — **cell** models (``cell/…`` datasets only), **simplicial** models
on **graph** benchmark datasets (e.g. ``graph/MUTAG``), and **simplicial** models on **MANTRA**
datasets. MANTRA Betti uses **one** subplot titled ``$\\beta_1$, $\\beta_2$`` (same run for both
metrics). Panels are arranged in a grid with **at most four columns** (extra datasets wrap to the
next row). One shared x-axis label ``Parameter count (total)``; end-to-end timing y-label notes no
preprocessing. Colors match ``plot_topology_timing.py``.

**Peak GPU memory** (optional ``--memory-plots``): for each raw run, resolve the assigned GPU from
``trainer.devices`` (exactly one index expected — parallel single-GPU jobs on a multi-GPU node).
Read W&B system history (``run.history(stream=\"events\")``; keys like
``system.gpu.{i}.memoryAllocatedBytes``), take **peak** and **mean** allocated bytes on that
device, convert to GB (/1e9), then seed-aggregate like timing. Writes (1) param-count vs
peak/mean-mem scatter plots (same facets as timing) and (2) diagnostic CSVs listing mean±std
per (model, dataset) plus flagged runs (missing/ambiguous ``trainer.devices``, empty system
history, etc.).

Run from repo root (``sys.path`` includes this directory when invoking the script)::

    python scripts/hopse_plotting/process_reruns.py
    python scripts/hopse_plotting/process_reruns.py --keep-incomplete-seeds
    python scripts/hopse_plotting/process_reruns.py --write-raw-csv scripts/hopse_plotting/csvs/best_runs_rerun_raw.csv
    python scripts/hopse_plotting/process_reruns.py --scatter-plots
    python scripts/hopse_plotting/process_reruns.py --memory-plots
    python scripts/hopse_plotting/process_reruns.py --probe-system-columns

Requires ``wandb`` and ``WANDB_API_KEY`` (or ``wandb login``).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from table_generator import (
    TABLES_DIR,
    _finite,
    _latex_cell_body,
    _latex_short_dataset_label,
    _normalize_table_model_id,
    _not_sig_diff_from_best,
    _sem,
    _specs_from_loader_paths,
    _val_mean_for_pick_row,
    build_latex_table,
    cell_submodel_table_rows,
    collapse_gnn_submodel_rows_to_base,
    collect_winner_test_by_submodel,
    dataframe_with_submodel_id,
    expand_mantra_betti_specs,
    hydra_dataset_key_from_loader_identity,
    is_mantra_betti_hydra_dataset,
    optimization_mode_for_metric_tail,
    partition_specs_graph_simplicial,
    simplicial_submodel_table_rows,
    TRANSDUCTIVE_GRAPH_SET,
)
from utils import (
    CSV_DIR,
    MANTRA_BETTI_F1_TAILS,
    MANTRA_BETTI_HYDRA_DATASET,
    MONITOR_METRIC_COLUMN,
    SEED_COLUMN,
    SUMMARY_COLUMN_PREFIX,
    PLOTS_DIR,
    aggregate_wandb_export_by_seed,
    build_seed_bucket_report,
    coalesce_seed_agg_wall_runtime_mean_std,
    dataframe_from_rows,
    filter_aggregated_to_required_n_seeds,
    iter_best_val_group_picks,
    iter_runs,
    list_seed_aggregatable_summary_columns,
    metric_name_tail,
    run_to_row,
    _unwrap_wandb_value,
)

# Match ``best_rerun_sh_generator.py`` defaults.
DEFAULT_WANDB_ENTITY = "gbg141-hopse"
DEFAULT_WANDB_PROJECT = "best_runs_rerun"

DEFAULT_AGG_CSV = CSV_DIR / "best_runs_rerun_seed_agg.csv"

SUMMARY_RUNTIME = f"{SUMMARY_COLUMN_PREFIX}Runtime"
# ``PipelineTimer`` logs these via ``log_hyperparams`` → they live in **run.config**, not summary.
SUMMARY_EPOCH_MEAN = f"{SUMMARY_COLUMN_PREFIX}AvgTime/train_epoch_mean"
SUMMARY_EPOCH_STD = f"{SUMMARY_COLUMN_PREFIX}AvgTime/train_epoch_std"
PARAM_COUNT_COL = "model.params.total"
DEFAULT_SCATTER_DIR = PLOTS_DIR / "rerun_timing_vs_params"
DEFAULT_MEMORY_SCATTER_DIR = PLOTS_DIR / "rerun_memory_vs_params"
DEFAULT_MEMORY_TABLE_CSV = CSV_DIR / "best_runs_rerun_peak_gpu_mem.csv"
DEFAULT_MEMORY_FLAGGED_CSV = CSV_DIR / "best_runs_rerun_peak_gpu_mem_flagged.csv"
# Computed from W&B system history; seed-aggregated like timing summary columns.
SUMMARY_PEAK_GPU_MEM_GB = f"{SUMMARY_COLUMN_PREFIX}peak_gpu_mem_GB"
SUMMARY_MEAN_GPU_MEM_GB = f"{SUMMARY_COLUMN_PREFIX}mean_gpu_mem_GB"
# Single scatter facet for MANTRA Betti (same run / timing for β₁ and β₂ columns).
MANTRA_BETTI_SCATTER_DATASET_H = f"{MANTRA_BETTI_HYDRA_DATASET}__scatter_f1joint"
SCATTER_MAX_COLS = 4

# W&B system stream keys (verified on ``best_runs_rerun``: dots, not slashes).
_SYSTEM_GPU_MEM_BYTES_TMPL = "system.gpu.{i}.memoryAllocatedBytes"
_SYSTEM_GPU_MEM_PCT_TMPL = "system.gpu.{i}.memoryAllocated"
_BYTES_PER_GB = 1e9
DEFAULT_SYSTEM_HISTORY_SAMPLES = 100_000


@dataclass
class PeakGpuMemResult:
    """Per-run GPU memory from W&B system metrics on ``trainer.devices``."""

    peak_mem_gb: float | None = None
    mean_mem_gb: float | None = None
    device_index: int | None = None
    flagged: bool = False
    flag_reason: str = ""
    mean_util_pct: float | None = None
    system_columns_sample: list[str] = field(default_factory=list)


def _coerce_device_index(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
        except ValueError:
            return None
    return None


def parse_trainer_device_indices(config: Mapping[str, Any] | None) -> tuple[list[int] | None, str]:
    """
    Resolve Lightning ``trainer.devices`` to a list of GPU indices.

    Reruns set ``trainer.devices=[k]`` (one physical GPU). A bare integer in Lightning often
    means *device count*, not an index — those are rejected as ambiguous.
    """
    from utils import flatten_config, get_from_flat

    if not config:
        return None, "missing run.config"
    flat = flatten_config(dict(config))
    raw = get_from_flat(flat, "trainer.devices")
    if raw in (None, ""):
        # Nested dict before flatten (some W&B payloads).
        trainer = config.get("trainer") if isinstance(config, Mapping) else None
        if isinstance(trainer, Mapping):
            raw = trainer.get("devices")
    if raw in (None, ""):
        return None, "trainer.devices missing"

    raw = _unwrap_wandb_value(raw)
    if isinstance(raw, (list, tuple)):
        idxs: list[int] = []
        for item in raw:
            ix = _coerce_device_index(_unwrap_wandb_value(item))
            if ix is None:
                return None, f"trainer.devices has non-int entry {item!r}"
            idxs.append(ix)
        return idxs, ""
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return [], ""
            idxs = []
            for part in inner.split(","):
                ix = _coerce_device_index(part.strip())
                if ix is None:
                    return None, f"trainer.devices string not parseable: {raw!r}"
                idxs.append(ix)
            return idxs, ""
        ix = _coerce_device_index(s)
        if ix is None:
            return None, f"trainer.devices string not parseable: {raw!r}"
        return None, (
            f"trainer.devices={raw!r} is a bare int (Lightning device *count*, not index); "
            "expected a one-element list like [0]"
        )
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None, (
            f"trainer.devices={raw!r} is a bare number (Lightning device *count*, not index); "
            "expected a one-element list like [0]"
        )
    return None, f"trainer.devices has unsupported type {type(raw).__name__}: {raw!r}"


def _resolve_system_mem_bytes_column(columns: list[str], device_index: int) -> str | None:
    """Prefer ``memoryAllocatedBytes``; accept slash-separated variants if present."""
    preferred = [
        _SYSTEM_GPU_MEM_BYTES_TMPL.format(i=device_index),
        f"system/gpu.{device_index}.memoryAllocatedBytes",
        f"system.gpu.{device_index}.memory_allocated_bytes",
    ]
    colset = set(columns)
    for c in preferred:
        if c in colset:
            return c
    # Fuzzy: any column mentioning this GPU index and allocated bytes.
    needle = f"gpu.{device_index}."
    for c in columns:
        cl = c.lower().replace("/", ".")
        if needle in cl and "memoryallocatedbytes" in cl.replace("_", ""):
            return c
    return None


def _resolve_system_util_column(columns: list[str], device_index: int) -> str | None:
    preferred = [
        f"system.gpu.{device_index}.gpu",
        f"system/gpu.{device_index}.gpu",
    ]
    colset = set(columns)
    for c in preferred:
        if c in colset:
            return c
    return None


def compute_peak_gpu_mem_for_run(
    run,
    *,
    samples: int = DEFAULT_SYSTEM_HISTORY_SAMPLES,
    print_columns_once: list[bool] | None = None,
) -> PeakGpuMemResult:
    """
    Peak allocated GPU memory (GB) on the single device in ``trainer.devices``.

    Flagged (no peak) when devices ≠ exactly one index, system history is missing, or the
    memory column is absent / all-NaN. Does **not** infer the GPU from utilization (parallel
    single-GPU jobs make multiple GPUs look \"active\" on the node).
    """
    from utils import run_with_wandb_retry

    cfg = dict(run.config or {})
    idxs, err = parse_trainer_device_indices(cfg)
    if err:
        return PeakGpuMemResult(flagged=True, flag_reason=err)
    assert idxs is not None
    if len(idxs) == 0:
        return PeakGpuMemResult(flagged=True, flag_reason="trainer.devices is empty")
    if len(idxs) != 1:
        return PeakGpuMemResult(
            flagged=True,
            flag_reason=f"trainer.devices has {len(idxs)} entries {idxs} (expected exactly 1)",
            device_index=None,
        )
    device_index = int(idxs[0])

    def _fetch():
        # Documented stream name is ``events``; older snippets used ``system`` (alias).
        return run.history(stream="events", samples=int(samples))

    try:
        hist = run_with_wandb_retry(_fetch, label=f"W&B system history {getattr(run, 'id', '?')}")
    except Exception as e:
        return PeakGpuMemResult(
            flagged=True,
            flag_reason=f"history fetch failed: {type(e).__name__}: {e}",
            device_index=device_index,
        )

    if hist is None or getattr(hist, "empty", True):
        return PeakGpuMemResult(
            flagged=True,
            flag_reason="empty system history (stream=events)",
            device_index=device_index,
        )

    columns = [str(c) for c in hist.columns.tolist()]
    if print_columns_once is not None and print_columns_once and not print_columns_once[0]:
        print_columns_once[0] = True
        print("  (probe) system history columns for first run:")
        for c in columns:
            print(f"    {c}")

    mem_col = _resolve_system_mem_bytes_column(columns, device_index)
    if mem_col is None:
        pct_col = _SYSTEM_GPU_MEM_PCT_TMPL.format(i=device_index)
        has_pct = pct_col in columns or f"system/gpu.{device_index}.memoryAllocated" in columns
        return PeakGpuMemResult(
            flagged=True,
            flag_reason=(
                f"no memoryAllocatedBytes column for gpu.{device_index} "
                f"(percent column present={has_pct}); columns sample={columns[:12]!r}…"
            ),
            device_index=device_index,
            system_columns_sample=columns[:40],
        )

    mem_bytes = pd.to_numeric(hist[mem_col], errors="coerce")
    finite = mem_bytes.notna() & np.isfinite(mem_bytes.to_numpy(dtype=float))
    if not bool(finite.any()):
        return PeakGpuMemResult(
            flagged=True,
            flag_reason=f"all-NaN values in {mem_col}",
            device_index=device_index,
            system_columns_sample=columns[:40],
        )
    peak_bytes = float(mem_bytes.loc[finite].max())
    mean_bytes = float(mem_bytes.loc[finite].mean())

    mean_util: float | None = None
    util_col = _resolve_system_util_column(columns, device_index)
    if util_col is not None:
        util = pd.to_numeric(hist[util_col], errors="coerce")
        if util.notna().any():
            mean_util = float(util.mean(skipna=True))

    return PeakGpuMemResult(
        peak_mem_gb=float(peak_bytes) / _BYTES_PER_GB,
        mean_mem_gb=float(mean_bytes) / _BYTES_PER_GB,
        device_index=device_index,
        flagged=False,
        mean_util_pct=mean_util,
        system_columns_sample=columns[:40],
    )


def augment_run_row_peak_gpu_mem(
    row: dict[str, Any],
    run,
    *,
    samples: int,
    flagged_rows: list[dict[str, Any]],
    print_columns_once: list[bool],
) -> dict[str, Any]:
    """Attach peak/mean ``summary_*_gpu_mem_GB`` or append a flagged diagnostic row."""
    from utils import _serialize_cell

    out = dict(row)
    res = compute_peak_gpu_mem_for_run(
        run, samples=samples, print_columns_once=print_columns_once
    )
    meta = {
        "identifiers_run_id": str(out.get("identifiers_run_id", getattr(run, "id", ""))),
        "identifiers_run_name": str(out.get("identifiers_run_name", getattr(run, "name", "") or "")),
        "identifiers_run_url": str(out.get("identifiers_run_url", getattr(run, "url", "") or "")),
        "model": str(out.get("model", "")),
        "dataset": str(out.get("dataset", "")),
        "seed": str(out.get(SEED_COLUMN, "")),
        "trainer_devices_device_index": (
            "" if res.device_index is None else str(res.device_index)
        ),
        "mean_util_pct": (
            "" if res.mean_util_pct is None else f"{res.mean_util_pct:.4g}"
        ),
        "flag_reason": res.flag_reason,
    }
    if res.flagged or res.peak_mem_gb is None or res.mean_mem_gb is None:
        flagged_rows.append(meta)
        return out

    out[SUMMARY_PEAK_GPU_MEM_GB] = _serialize_cell(res.peak_mem_gb)
    out[SUMMARY_MEAN_GPU_MEM_GB] = _serialize_cell(res.mean_mem_gb)
    return out


def augment_run_row_wandb_timing(row: dict[str, Any], run) -> dict[str, Any]:
    """
    Promote timing fields into the same ``summary_*`` columns as scalar metrics so CSV
    export / seed aggregation match ``main_loader``-style tables.

    - **Per-epoch:** ``AvgTime/train_epoch_{mean,std}`` from flattened ``run.config``.
    - **Wall clock:** W&B run duration is usually ``run.summary['_runtime']`` (seconds);
      we normalize to ``summary_Runtime`` for downstream code.
    - **Parameter count:** if ``model.params.total`` is empty in the export row, copy from
      W&B summary ``model/params/total`` (Lightning ``log_hyperparams``).
    """
    from utils import _serialize_cell, flatten_config, get_from_flat

    out = dict(row)
    flat_cfg = flatten_config(dict(run.config or {}))

    for key in ("AvgTime/train_epoch_mean", "AvgTime/train_epoch_std"):
        v = get_from_flat(flat_cfg, key)
        if v is None or v == "":
            continue
        cell = _serialize_cell(v)
        if cell:
            out[f"{SUMMARY_COLUMN_PREFIX}{key}"] = cell

    try:
        summary = dict(run.summary) if run.summary is not None else {}
    except Exception:
        summary = {}

    if not str(out.get(PARAM_COUNT_COL, "")).strip():
        v = summary.get("model/params/total")
        if v is not None and v != "":
            out[PARAM_COUNT_COL] = _serialize_cell(_unwrap_wandb_value(v))

    # If anything logged AvgTime as a scalar metric, prefer filling gaps from summary.
    for key in ("AvgTime/train_epoch_mean", "AvgTime/train_epoch_std"):
        col = f"{SUMMARY_COLUMN_PREFIX}{key}"
        if col in out and str(out[col]).strip():
            continue
        if key in summary:
            out[col] = _serialize_cell(summary[key])

    wall = None
    for wb_key in ("_runtime", "Runtime", "runtime"):
        if wb_key in summary:
            wall = summary[wb_key]
            break
    if wall is None:
        v = get_from_flat(flat_cfg, "_runtime")
        if v not in (None, ""):
            wall = v
    if wall is not None:
        out[SUMMARY_RUNTIME] = _serialize_cell(wall)

    return out


def collect_runs_single_project(
    entity: str,
    project: str,
    *,
    run_state: str | None = "finished",
    verbose: bool = True,
    fetch_peak_gpu_mem: bool = False,
    system_history_samples: int = DEFAULT_SYSTEM_HISTORY_SAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Export finished (or filtered) runs from one project.

    Returns ``(rows, flagged_peak_mem_rows)``. The flagged list is non-empty only when
    ``fetch_peak_gpu_mem`` is True (runs where peak memory could not be attributed safely).
    """
    import wandb

    api = wandb.Api(timeout=120)
    rows: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    print_columns_once = [False]
    _filt = f"state={run_state}" if run_state else "all states"
    if verbose:
        print(f"  (fetch) {entity}/{project} ({_filt})", flush=True)
        if fetch_peak_gpu_mem:
            print(
                f"  (peak GPU mem) system history samples={system_history_samples} "
                f"via trainer.devices -> system.gpu.{{i}}.memoryAllocatedBytes",
                flush=True,
            )
    count = 0
    runs_gen = iter_runs(api, entity, project, state=run_state)
    for run in runs_gen:
        base = run_to_row(entity=entity, project=project, run=run)
        row = augment_run_row_wandb_timing(base, run)
        if fetch_peak_gpu_mem:
            row = augment_run_row_peak_gpu_mem(
                row,
                run,
                samples=system_history_samples,
                flagged_rows=flagged,
                print_columns_once=print_columns_once,
            )
        rows.append(row)
        count += 1
        if verbose and count % 25 == 0:
            extra = f", flagged_mem={len(flagged)}" if fetch_peak_gpu_mem else ""
            print(f"    … {count} run(s) so far{extra}", flush=True)
    if verbose:
        print(f"    -> {count} run(s)", flush=True)
        if fetch_peak_gpu_mem:
            n_ok = sum(1 for r in rows if str(r.get(SUMMARY_PEAK_GPU_MEM_GB, "")).strip())
            print(
                f"    peak GPU mem: {n_ok} ok, {len(flagged)} flagged "
                f"(skipped for aggregation)",
                flush=True,
            )
        if rows:
            peek = pd.DataFrame(rows)
            if "model" in peek.columns:
                models = sorted(peek["model"].astype(str).unique())
                print(f"    Unique models in export ({len(models)}): {models}")
                print(
                    "    (If a model is missing, its reruns may still be non-finished — try "
                    "--run-state all, or confirm runs use this entity/project.)"
                )
    return rows, flagged


def _summary_metric_columns_for_rerun_export(df: pd.DataFrame) -> list[str] | None:
    cols = list_seed_aggregatable_summary_columns(df)
    extra: list[str] = []
    if SUMMARY_RUNTIME in df.columns:
        extra.append(SUMMARY_RUNTIME)
    if SUMMARY_PEAK_GPU_MEM_GB in df.columns:
        extra.append(SUMMARY_PEAK_GPU_MEM_GB)
    if SUMMARY_MEAN_GPU_MEM_GB in df.columns:
        extra.append(SUMMARY_MEAN_GPU_MEM_GB)
    if not cols and not extra:
        return None
    return sorted(set(cols) | set(extra))


def _print_seed_bucket_report(report: pd.DataFrame, *, required_n_seeds: int | None) -> None:
    if report.empty:
        print("Seed-count distribution: (no aggregated hyperparameter groups).")
        return
    if required_n_seeds is not None:
        print(
            f"Seed-count distribution (hyperparameter groups per model+dataset); "
            f"output CSV keeps only n_seeds=={required_n_seeds}."
        )
    else:
        print(
            "Seed-count distribution; output CSV keeps all n_seeds (--keep-incomplete-seeds)."
        )
    for (model, dataset), sub in report.groupby(["model", "dataset"], dropna=False):
        print(f"\n  model={model!r}  dataset={dataset!r}")
        for _, row in sub.sort_values("n_seeds").iterrows():
            k = row["n_seeds"]
            try:
                k_int = int(k) if pd.notna(k) else k
            except (TypeError, ValueError):
                k_int = k
            mark = (
                "  <- rows written to -o"
                if required_n_seeds is not None
                and pd.notna(k)
                and int(k) == int(required_n_seeds)
                else ""
            )
            print(
                f"    n_seeds={k_int}: {int(row['n_groups'])} groups "
                f"({float(row['pct_of_groups']):.2f}% of groups for this pair){mark}"
            )


def _print_silent_failure_report(silent: pd.DataFrame) -> None:
    if silent is None or silent.empty:
        print("\nSilent failures (no summary_test_best_rerun metrics): 0 raw runs dropped.")
        return
    tot = int(pd.to_numeric(silent["n_silent_failures"], errors="coerce").fillna(0).sum())
    print(
        f"\nSilent failures (no finite summary_test_best_rerun/* on raw row): "
        f"{tot} raw run(s) dropped before seed aggregation."
    )


def collect_runtime_stats_from_agg_winners(
    df_agg: pd.DataFrame,
    *,
    mean_col: str,
    std_col: str | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Same validation picks as ``collect_winner_test_by_submodel``, but store
    ``mean_col`` / ``std_col`` in ``test_mean`` / ``test_std`` for reuse of
    ``collapse_gnn_submodel_rows_to_base``.
    """
    work = dataframe_with_submodel_id(df_agg)
    colset = set(work.columns)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    gc = ["model", "dataset", "_sub_id"]
    for keys, pick_idx, monitor_val, tail in iter_best_val_group_picks(
        work, group_cols=gc, monitor_column=MONITOR_METRIC_COLUMN
    ):
        model = _normalize_table_model_id(str(keys[0]))
        dataset_raw = str(keys[1]).strip()
        sub_id = str(keys[2]).strip()
        dataset = hydra_dataset_key_from_loader_identity(dataset_raw)
        row_key = f"{model}|{sub_id}"
        w = work.loc[pick_idx]

        if mean_col not in w.index or mean_col not in colset:
            mu = float("nan")
        else:
            mu = float(pd.to_numeric(w.get(mean_col), errors="coerce"))
            if pd.isna(mu):
                mu = float("nan")

        sd = float("nan")
        if std_col and std_col in colset:
            v = pd.to_numeric(w.get(std_col), errors="coerce")
            if pd.notna(v):
                sd = float(v)

        if is_mantra_betti_hydra_dataset(dataset_raw):
            for fi_tail in MANTRA_BETTI_F1_TAILS:
                col_key = f"{MANTRA_BETTI_HYDRA_DATASET}#{fi_tail}"
                vm = _val_mean_for_pick_row(w, fi_tail, colset)
                out[(row_key, col_key)] = {
                    "test_mean": mu,
                    "test_std": sd,
                    "val_mean": vm,
                    "tail": fi_tail,
                    "mode": "max",
                    "monitor_raw": str(monitor_val).strip(),
                    "n_seeds": int(pd.to_numeric(w.get("n_seeds"), errors="coerce") or 0),
                }
            continue

        mode = optimization_mode_for_metric_tail(tail) if tail else "max"
        vm = _val_mean_for_pick_row(w, tail, colset)
        out[(row_key, dataset)] = {
            "test_mean": mu,
            "test_std": sd,
            "val_mean": vm,
            "tail": tail,
            "mode": mode,
            "monitor_raw": str(monitor_val).strip(),
            "n_seeds": int(pd.to_numeric(w.get("n_seeds"), errors="coerce") or 0),
        }
    return out


def _pick_raw_row_min_epoch_timing_std(
    sub: pd.DataFrame, *, mean_col: str, std_col: str
) -> pd.Series | None:
    """
    Among raw runs in ``sub``, require finite ``mean_col`` (seconds per epoch).
    Prefer the run with the lowest ``std_col`` (within-run epoch timing variability); tie-break
    by lower mean (faster), then lower data seed.
    """
    best: tuple[tuple, pd.Series] | None = None
    for _idx, sr in sub.iterrows():
        mu = pd.to_numeric(sr.get(mean_col), errors="coerce")
        if pd.isna(mu) or not math.isfinite(float(mu)):
            continue
        mu_f = float(mu)
        sig = float("nan")
        if std_col and std_col in sub.columns:
            v = pd.to_numeric(sr.get(std_col), errors="coerce")
            if pd.notna(v) and math.isfinite(float(v)):
                sig = float(v)
        seed_tie = pd.to_numeric(sr.get(SEED_COLUMN), errors="coerce")
        seed_tie = float(seed_tie) if pd.notna(seed_tie) else float("inf")
        if math.isfinite(sig):
            key = (0, sig, mu_f, seed_tie)
        else:
            key = (1, float("inf"), mu_f, seed_tie)
        if best is None or key < best[0]:
            best = (key, sr)
    return None if best is None else best[1]


def collect_epoch_time_stats_min_timing_std(
    df_raw: pd.DataFrame,
    *,
    mean_col: str,
    std_col: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    One raw row per (model, dataset, _sub_id): pick the seed whose logged **epoch-time std**
    (within-run) is smallest among rows with finite per-epoch mean. Values stored as
    ``test_mean`` / ``test_std`` for table + GNN collapse.
    """
    work = dataframe_with_submodel_id(df_raw)
    colset = set(work.columns)
    if mean_col not in work.columns:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}

    for (_model, dataset_raw, sub_id), sub in work.groupby(
        ["model", "dataset", "_sub_id"], dropna=False
    ):
        model = _normalize_table_model_id(str(_model))
        row_key = f"{model}|{sub_id}"
        ds_raw = str(dataset_raw).strip()
        dataset = hydra_dataset_key_from_loader_identity(ds_raw)

        row = _pick_raw_row_min_epoch_timing_std(sub, mean_col=mean_col, std_col=std_col)
        if row is None:
            continue

        mu = float(pd.to_numeric(row.get(mean_col), errors="coerce"))
        if not math.isfinite(mu):
            mu = float("nan")
        sd = float("nan")
        if std_col and std_col in work.columns:
            v = pd.to_numeric(row.get(std_col), errors="coerce")
            if pd.notna(v):
                sd = float(v)

        if is_mantra_betti_hydra_dataset(ds_raw):
            for fi_tail in MANTRA_BETTI_F1_TAILS:
                col_key = f"{MANTRA_BETTI_HYDRA_DATASET}#{fi_tail}"
                vm = _val_mean_for_pick_row(row, fi_tail, colset)
                out[(row_key, col_key)] = {
                    "test_mean": mu,
                    "test_std": sd,
                    "val_mean": vm,
                    "tail": fi_tail,
                    "mode": "max",
                    "monitor_raw": str(row.get(MONITOR_METRIC_COLUMN, "") or "").strip(),
                    "n_seeds": 1,
                }
            continue

        mon = str(row.get(MONITOR_METRIC_COLUMN, "") or "").strip()
        tail = metric_name_tail(mon)
        mode = optimization_mode_for_metric_tail(tail) if tail else "max"
        vm = _val_mean_for_pick_row(row, tail, colset)
        out[(row_key, dataset)] = {
            "test_mean": mu,
            "test_std": sd,
            "val_mean": vm,
            "tail": tail,
            "mode": mode,
            "monitor_raw": mon,
            "n_seeds": 1,
        }
    return out


def _domain_row_keys_for_time_table(
    row_key: str,
    *,
    graph_rows: list[tuple[str, str]],
    simplicial_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
) -> list[str]:
    """Stats row keys in the same rotated band as ``row_key`` (graph / simplicial / cell)."""
    if str(row_key).startswith("graph/"):
        return [rk for rk, _ in graph_rows]
    if str(row_key).startswith("simplicial/"):
        return [rk for rk, _ in simplicial_rows]
    if str(row_key).startswith("cell/"):
        return [rk for rk, _ in cell_rows]
    return [rk for rk, _ in graph_rows + simplicial_rows + cell_rows]


def _all_row_keys_for_time_table(
    *,
    graph_rows: list[tuple[str, str]],
    simplicial_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
) -> list[str]:
    """Every model row key (graph + simplicial + cell) for column-global minimum / Z comparisons."""
    return [rk for rk, _ in graph_rows + simplicial_rows + cell_rows]


def _dataset_header_timing(path: str) -> str:
    """LaTeX column title for timing tables: no performance ↑/↓ (seconds are not higher/lower-is-better)."""
    if "#" in path:
        _base, _, suf = path.partition("#")
        beta_only = {
            "f1-1": r"$\beta_1$",
            "f1-2": r"$\beta_2$",
            "f1-0": r"$\beta_0$",
        }.get(suf)
        if beta_only is not None:
            return beta_only
    return _latex_short_dataset_label(path)


def _column_groups_timing_headers(
    column_groups: list[tuple[str, list[tuple[str, str]]]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Same hydra paths as ``column_groups``; headers match performance short names but omit arrows."""
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for title, block in column_groups:
        out.append((title, [(p, _dataset_header_timing(p)) for p, _ in block]))
    return out


def build_rerun_metric_table_plain(
    stats: dict[tuple[str, str], dict[str, Any]],
    *,
    column_groups: list[tuple[str, list[tuple[str, str]]]],
    graph_rows: list[tuple[str, str]],
    simplicial_rows: list[tuple[str, str]],
    cell_rows: list[tuple[str, str]],
    caption: str,
    label: str,
    decimals: int = 2,
    comparison_scope: str = "domain",
) -> str:
    """
    Same geometry as ``build_latex_table``. **Lower mean time is better** per dataset column.

    ``comparison_scope``:

    - ``"domain"`` (default): bold / blue use the minimum **within the same band** (graph,
      simplicial, or cell) as ``row_key``.
    - ``"global"``: bold / blue use the minimum **across all model rows** in that column.
    """
    dataset_specs: list[tuple[str, str]] = []
    group_ranges: list[tuple[str, int, int]] = []
    for title, block in column_groups:
        if not block:
            continue
        i0 = len(dataset_specs)
        dataset_specs.extend(block)
        group_ranges.append((title, i0, len(dataset_specs) - 1))

    n_d = len(dataset_specs)
    colspec = "@{}ll" + "c" * n_d + "@{}"

    def _tol_eq(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-9 * (1.0 + abs(b))

    def cell_time_colored(row_key: str, ds_path: str) -> str:
        ds_key = hydra_dataset_key_from_loader_identity(ds_path)
        st = stats.get((row_key, ds_key))
        if not st or not _finite(st.get("test_mean", float("nan"))):
            return "-"
        mu = float(st["test_mean"])
        sd = float(st["test_std"]) if _finite(st.get("test_std")) else float("nan")
        n_raw = st.get("n_seeds", 0)
        n_seeds = int(pd.to_numeric(n_raw, errors="coerce")) if _finite(n_raw) else 0
        se = _sem(sd, max(n_seeds, 0))

        if comparison_scope == "global":
            band_keys = _all_row_keys_for_time_table(
                graph_rows=graph_rows,
                simplicial_rows=simplicial_rows,
                cell_rows=cell_rows,
            )
        else:
            band_keys = _domain_row_keys_for_time_table(
                row_key,
                graph_rows=graph_rows,
                simplicial_rows=simplicial_rows,
                cell_rows=cell_rows,
            )
        mus: list[float] = []
        for rk in band_keys:
            t = stats.get((rk, ds_key))
            if t and _finite(t.get("test_mean")):
                mus.append(float(t["test_mean"]))
        if not mus:
            return _latex_cell_body(
                mu, sd, se, is_best=False, blue_tie=False, decimals=decimals, scale=1.0
            )

        best_val = min(mus)
        is_best = _tol_eq(mu, best_val)

        ref_mu, ref_se = best_val, 0.0
        for rk in band_keys:
            t = stats.get((rk, ds_key))
            if not t or not _finite(t.get("test_mean")):
                continue
            if not _tol_eq(float(t["test_mean"]), best_val):
                continue
            ref_mu = float(t["test_mean"])
            ns_ref = pd.to_numeric(t.get("n_seeds", 0), errors="coerce")
            n_ref = int(ns_ref) if pd.notna(ns_ref) else 0
            ref_se = _sem(
                float(t["test_std"]) if _finite(t.get("test_std")) else 0.0,
                n_ref,
            )
            break

        blue = not is_best and _not_sig_diff_from_best(mu, se, ref_mu, ref_se)
        return _latex_cell_body(
            mu, sd, se, is_best=is_best, blue_tie=blue, decimals=decimals, scale=1.0
        )

    lines: list[str] = []
    lines.append(
        "% --- Requires: \\usepackage{booktabs,multirow,adjustbox,graphicx,xcolor,colortbl}"
    )
    lines.append(
        "\\definecolor{stdblue}{HTML}{C9DAF8}% same swatch as table_generator (non-significant vs column best)"
    )
    lines.append("\\definecolor{bestgray}{HTML}{D9D9D9}")
    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{width=1.\\textwidth}")
    lines.append("\\renewcommand{\\arraystretch}{1.4}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    if n_d > 0 and group_ranges:
        multicols = []
        cmid_parts = []
        for title, i0, i1 in group_ranges:
            span = i1 - i0 + 1
            multicols.append(f"\\multicolumn{{{span}}}{{c}}{{\\mbox{{{title}}}}}")
            cmid_parts.append(f"\\cmidrule(lr){{{3 + i0}-{3 + i1}}}")
        lines.append("  &  & " + " & ".join(multicols) + " \\\\")
        lines.append(" ".join(cmid_parts))

    hdr = " & \\textbf{Model}"
    for _p, h in dataset_specs:
        hdr += f" & \\scriptsize {h}"
    hdr += " \\\\"
    lines.append(hdr)
    lines.append("\\midrule")

    def emit_block(rotate: str, rows: list[tuple[str, str]]) -> None:
        n_r = len(rows)
        rk0, lab0 = rows[0]
        row = (
            f"\\multirow{{{n_r}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{rotate}}}}}}} "
            f"& {lab0}"
        )
        for ds_path, _h in dataset_specs:
            row += " & " + cell_time_colored(rk0, ds_path)
        lines.append(row + " \\\\")
        for rk, lab in rows[1:]:
            row = f"& {lab}"
            for ds_path, _h in dataset_specs:
                row += " & " + cell_time_colored(rk, ds_path)
            lines.append(row + " \\\\")

    emit_block("Graph", graph_rows)
    lines.append("\\midrule")
    emit_block("Simplicial", simplicial_rows)
    lines.append("\\midrule")
    emit_block("Cell", cell_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


_COAL_WALL_MEAN = "_coalesced_wall_runtime_mean"
_COAL_WALL_STD = "_coalesced_wall_runtime_std"


def build_rerun_timing_vs_params_frame(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Best-val rerun row per ``(model, dataset, _sub_id)`` (same picks as performance tables),
    with parameter count, wall runtime mean±std over seeds, train-epoch mean mean±std
    over seeds, and (when present) peak GPU memory GB mean±std over seeds.
    """
    from plot_topology_timing import _domain_from_model, _pretty_legend_label, enrich_submodel_columns

    if agg.empty or "model" not in agg.columns:
        return pd.DataFrame()

    work = enrich_submodel_columns(agg)
    wm_ser, ws_ser = coalesce_seed_agg_wall_runtime_mean_std(work)
    work = work.copy()
    work[_COAL_WALL_MEAN] = wm_ser
    work[_COAL_WALL_STD] = ws_ser
    ep_m = f"{SUMMARY_EPOCH_MEAN}__mean"
    ep_s = f"{SUMMARY_EPOCH_MEAN}__std"
    has_ep = ep_m in work.columns
    mem_m = f"{SUMMARY_PEAK_GPU_MEM_GB}__mean"
    mem_s = f"{SUMMARY_PEAK_GPU_MEM_GB}__std"
    has_mem = mem_m in work.columns
    mean_mem_m = f"{SUMMARY_MEAN_GPU_MEM_GB}__mean"
    mean_mem_s = f"{SUMMARY_MEAN_GPU_MEM_GB}__std"
    has_mean_mem = mean_mem_m in work.columns

    rows: list[dict[str, Any]] = []
    gc = ["model", "dataset", "_sub_id"]
    for _keys, pick_idx, _mon, _tail in iter_best_val_group_picks(
        work, group_cols=gc, monitor_column=MONITOR_METRIC_COLUMN
    ):
        w = work.loc[pick_idx]
        ds_raw = str(w.get("dataset", "")).strip()
        mk = str(w.get("model_row_key", "")).strip()
        bb = str(w.get("model_backbone", "")).strip()
        if not mk:
            continue

        nparams = pd.to_numeric(w.get(PARAM_COUNT_COL), errors="coerce")
        pcount = float(nparams) if pd.notna(nparams) else float("nan")

        wm = float(pd.to_numeric(w.get(_COAL_WALL_MEAN), errors="coerce"))
        wsd = float(pd.to_numeric(w.get(_COAL_WALL_STD), errors="coerce"))
        if pd.isna(wm):
            wm = float("nan")
        if pd.isna(wsd):
            wsd = float("nan")

        em = float("nan")
        esd = float("nan")
        if has_ep:
            em = float(pd.to_numeric(w.get(ep_m), errors="coerce"))
            if ep_s in w.index:
                esd = float(pd.to_numeric(w.get(ep_s), errors="coerce"))

        pmem = float("nan")
        pmem_sd = float("nan")
        if has_mem:
            pmem = float(pd.to_numeric(w.get(mem_m), errors="coerce"))
            if mem_s in w.index:
                pmem_sd = float(pd.to_numeric(w.get(mem_s), errors="coerce"))

        mmem = float("nan")
        mmem_sd = float("nan")
        if has_mean_mem:
            mmem = float(pd.to_numeric(w.get(mean_mem_m), errors="coerce"))
            if mean_mem_s in w.index:
                mmem_sd = float(pd.to_numeric(w.get(mean_mem_s), errors="coerce"))

        legend = _pretty_legend_label(mk, bb)
        model = str(w.get("model", "")).strip()
        plot_dom = _domain_from_model(model) or ""
        n_seeds = pd.to_numeric(w.get("n_seeds"), errors="coerce")
        n_seeds_i = int(n_seeds) if pd.notna(n_seeds) else 0

        def one_row(dataset_h: str) -> None:
            rows.append(
                {
                    "dataset_h": dataset_h,
                    "model_row_key": mk,
                    "model_backbone": bb,
                    "legend_label": legend,
                    "model": model,
                    "plot_domain": plot_dom,
                    "param_count": pcount,
                    "wall_mean": wm,
                    "wall_std": wsd,
                    "epoch_mean": em,
                    "epoch_std": esd,
                    "peak_mem_mean": pmem,
                    "peak_mem_std": pmem_sd,
                    "mean_mem_mean": mmem,
                    "mean_mem_std": mmem_sd,
                    "n_seeds": n_seeds_i,
                }
            )

        if is_mantra_betti_hydra_dataset(ds_raw):
            one_row(MANTRA_BETTI_SCATTER_DATASET_H)
        else:
            one_row(hydra_dataset_key_from_loader_identity(ds_raw))

    return pd.DataFrame(rows)


def build_peak_mem_diagnostic_table(
    df_raw: pd.DataFrame,
    flagged: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Per ``(model, dataset)``: seed-mean/std of per-run peak and within-run-mean GPU GB,
    plus ``n_seeds`` / ``n_flagged``.

    Uses raw per-run memory columns (not best-val collapse) so the diagnostic matches the
    requested grouping.
    """
    out_cols = [
        "model",
        "dataset",
        "mean_peak_mem_GB",
        "std_peak_mem_GB",
        "mean_mean_mem_GB",
        "std_mean_mem_GB",
        "n_seeds",
        "n_flagged",
    ]
    flagged_df = pd.DataFrame(flagged) if flagged else pd.DataFrame()
    if not flagged_df.empty and {"model", "dataset"}.issubset(flagged_df.columns):
        n_flag = (
            flagged_df.groupby(["model", "dataset"], dropna=False)
            .size()
            .rename("n_flagged")
            .reset_index()
        )
    else:
        n_flag = pd.DataFrame(columns=["model", "dataset", "n_flagged"])

    has_peak = SUMMARY_PEAK_GPU_MEM_GB in df_raw.columns
    has_mean = SUMMARY_MEAN_GPU_MEM_GB in df_raw.columns
    if df_raw.empty or not (has_peak or has_mean):
        if n_flag.empty:
            return pd.DataFrame(columns=out_cols)
        n_flag = n_flag.copy()
        for c in (
            "mean_peak_mem_GB",
            "std_peak_mem_GB",
            "mean_mean_mem_GB",
            "std_mean_mem_GB",
        ):
            n_flag[c] = float("nan")
        n_flag["n_seeds"] = 0
        return n_flag[out_cols]

    work = df_raw.copy()
    if has_peak:
        work["_peak"] = pd.to_numeric(work[SUMMARY_PEAK_GPU_MEM_GB], errors="coerce")
    else:
        work["_peak"] = np.nan
    if has_mean:
        work["_mean"] = pd.to_numeric(work[SUMMARY_MEAN_GPU_MEM_GB], errors="coerce")
    else:
        work["_mean"] = np.nan

    valid = work.loc[
        (work["_peak"].notna() & np.isfinite(work["_peak"].to_numpy(dtype=float)))
        | (work["_mean"].notna() & np.isfinite(work["_mean"].to_numpy(dtype=float)))
    ]
    if valid.empty:
        stats = pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "mean_peak_mem_GB",
                "std_peak_mem_GB",
                "mean_mean_mem_GB",
                "std_mean_mem_GB",
                "n_seeds",
            ]
        )
    else:

        def _std0(s: pd.Series) -> float:
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                return float("nan")
            return float(s.std(ddof=0))

        g = valid.groupby(["model", "dataset"], dropna=False)
        stats = g.agg(
            mean_peak_mem_GB=("_peak", "mean"),
            std_peak_mem_GB=("_peak", _std0),
            mean_mean_mem_GB=("_mean", "mean"),
            std_mean_mem_GB=("_mean", _std0),
            n_seeds=("_peak", "size"),
        ).reset_index()

    out = stats.merge(n_flag, on=["model", "dataset"], how="outer")
    if "n_flagged" not in out.columns:
        out["n_flagged"] = 0
    out["n_flagged"] = pd.to_numeric(out["n_flagged"], errors="coerce").fillna(0).astype(int)
    if "n_seeds" not in out.columns:
        out["n_seeds"] = 0
    out["n_seeds"] = pd.to_numeric(out["n_seeds"], errors="coerce").fillna(0).astype(int)
    for c in (
        "mean_peak_mem_GB",
        "std_peak_mem_GB",
        "mean_mean_mem_GB",
        "std_mean_mem_GB",
    ):
        if c not in out.columns:
            out[c] = float("nan")
    out = out.sort_values(["model", "dataset"], kind="stable").reset_index(drop=True)
    return out[out_cols]


def probe_one_run_system_columns(entity: str, project: str, *, run_state: str | None) -> None:
    """Print system-history columns for one finished run (column-name verification)."""
    import wandb

    api = wandb.Api(timeout=120)
    runs_gen = iter_runs(api, entity, project, state=run_state)
    run = next(iter(runs_gen), None)
    if run is None:
        print(f"No runs in {entity}/{project} (state={run_state!r}).")
        return
    print(f"Probe run id={run.id} name={run.name!r}")
    idxs, err = parse_trainer_device_indices(dict(run.config or {}))
    print(f"  trainer.devices -> {idxs!r}" + (f" ({err})" if err else ""))
    hist = run.history(stream="events", samples=min(5000, DEFAULT_SYSTEM_HISTORY_SAMPLES))
    cols = [] if hist is None else [str(c) for c in hist.columns.tolist()]
    print(f"  history rows={0 if hist is None else len(hist)}; columns ({len(cols)}):")
    for c in cols:
        print(f"    {c}")
    if idxs and len(idxs) == 1:
        res = compute_peak_gpu_mem_for_run(run, samples=DEFAULT_SYSTEM_HISTORY_SAMPLES)
        print(
            f"  peak_mem_gb={res.peak_mem_gb} mean_mem_gb={res.mean_mem_gb} "
            f"device={res.device_index} flagged={res.flagged} reason={res.flag_reason!r} "
            f"mean_util={res.mean_util_pct}"
        )


def _expanded_hydra_paths_in_table_order() -> list[str]:
    return [p for p, _ in expand_mantra_betti_specs(_specs_from_loader_paths())]


def _dataset_order_cell() -> list[str]:
    return [p for p in _expanded_hydra_paths_in_table_order() if p.startswith("cell/")]


def _dataset_order_simplicial_graph_benchmarks() -> list[str]:
    return [
        p
        for p in _expanded_hydra_paths_in_table_order()
        if p.startswith("graph/") and p not in TRANSDUCTIVE_GRAPH_SET
    ]


def _dataset_order_simplicial_mantra() -> list[str]:
    """MANTRA paths in table order; Betti ``#f1-1`` / ``#f1-2`` replaced by one combined facet."""
    from plot_topology_timing import _is_mantra_simplicial_dataset

    raw = [p for p in _expanded_hydra_paths_in_table_order() if _is_mantra_simplicial_dataset(p)]
    betti_cols = {f"{MANTRA_BETTI_HYDRA_DATASET}#{t}" for t in MANTRA_BETTI_F1_TAILS}
    out: list[str] = []
    inserted_joint = False
    for p in raw:
        if p in betti_cols:
            if not inserted_joint:
                out.append(MANTRA_BETTI_SCATTER_DATASET_H)
                inserted_joint = True
            continue
        out.append(p)
    return out


def _scatter_panel_title(dataset_h: str) -> str:
    if dataset_h == MANTRA_BETTI_SCATTER_DATASET_H:
        return r"$\beta_1$, $\beta_2$"
    return _latex_short_dataset_label(dataset_h)


def _filter_scatter_cell(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["model"].astype(str).str.startswith("cell/")].copy()


def _filter_scatter_simplicial_graph_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    from plot_topology_timing import _is_mantra_simplicial_dataset

    m = df["model"].astype(str).str.startswith("simplicial/")
    not_mantra_ds = ~df["dataset_h"].astype(str).map(_is_mantra_simplicial_dataset)
    return df.loc[m & not_mantra_ds].copy()


def _filter_scatter_simplicial_mantra(df: pd.DataFrame) -> pd.DataFrame:
    from plot_topology_timing import _is_mantra_simplicial_dataset

    m = df["model"].astype(str).str.startswith("simplicial/")
    mantra_ds = df["dataset_h"].astype(str).map(_is_mantra_simplicial_dataset)
    return df.loc[m & mantra_ds].copy()


def emit_rerun_scatter_timing_vs_params(
    plot_df: pd.DataFrame,
    *,
    y_mean_col: str,
    y_std_col: str | None,
    stem: str,
    suptitle: str,
    y_label: str,
    x_label: str,
    dataset_order: list[str],
    out_dir: Path,
    dpi: int,
) -> None:
    """Faceted scatter: x = parameter count, y = timing, error bars on y from seed std."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    from plot_topology_timing import (
        FACET_FS_LEGEND,
        FACET_FS_TICK,
        FACET_FS_TITLE,
        FACET_FS_XY,
        FACET_LEGEND_BORDERPAD,
        FACET_LEGEND_HANDLEHEIGHT,
        FACET_LEGEND_HANDLELENGTH,
        FACET_LEGEND_LABELSPACING,
        FACET_SUPTITLE_Y,
        _infer_plot_domain,
        _ordered_model_row_keys,
        build_row_key_color_map,
    )

    sub_mean = pd.to_numeric(plot_df[y_mean_col], errors="coerce")
    plot_df = plot_df.loc[sub_mean.notna() & np.isfinite(sub_mean.to_numpy())].copy()
    if plot_df.empty:
        print(f"  (skip empty scatter) {stem}.png")
        return

    present = set(plot_df["dataset_h"].astype(str).unique())
    dss = [d for d in dataset_order if d in present]
    dss.extend(sorted(present - set(dss)))

    dom = _infer_plot_domain(plot_df)
    row_keys = _ordered_model_row_keys(plot_df, sorted(plot_df["model_row_key"].astype(str).unique()), dom)
    color_of = build_row_key_color_map(plot_df, row_keys, domain=dom)

    n_d = len(dss)
    if n_d == 0:
        print(f"  (skip empty scatter) {stem}.png")
        return

    n_cols = min(SCATTER_MAX_COLS, n_d)
    n_rows = (n_d + n_cols - 1) // n_cols
    n_pad = n_rows * n_cols
    # Wider panels: width scales with number of columns in the grid (max 4).
    fig_w = max(9.0, min(42.0, 3.55 * n_cols + 2.35))
    fig_h = max(4.6, 5.05 * n_rows + 1.95)
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        sharey=False,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.05, wspace=None, hspace=None)
    ax_arr = np.atleast_2d(axs)
    axes_flat: list[Any] = list(ax_arr.ravel())

    # Larger than topology faceting defaults for rerun scatter readability.
    fs_suptitle = FACET_FS_TITLE + 2.5
    fs_axis_label = FACET_FS_XY + 2.0
    fs_panel_title = FACET_FS_TICK + 2.2
    fs_tick = FACET_FS_TICK + 1.8
    fs_legend = FACET_FS_LEGEND + 2.8
    fs_ann = FACET_FS_TICK + 3.5
    scatter_s = 92
    scatter_lw = 0.85
    err_lw = 1.2
    err_capsize = 4.2

    for i, ds in enumerate(dss):
        ax = axes_flat[i]
        sub = plot_df.loc[plot_df["dataset_h"].astype(str) == ds].copy()

        for mk in row_keys:
            m = sub.loc[sub["model_row_key"].astype(str) == mk]
            if m.empty:
                continue
            xi = pd.to_numeric(m["param_count"], errors="coerce").to_numpy(dtype=float)
            yi = pd.to_numeric(m[y_mean_col], errors="coerce").to_numpy(dtype=float)
            if y_std_col and y_std_col in m.columns:
                ei = pd.to_numeric(m[y_std_col], errors="coerce").to_numpy(dtype=float)
                ei = np.where(np.isfinite(ei) & (ei >= 0), ei, 0.0)
            else:
                ei = np.zeros_like(yi, dtype=float)
            c = color_of.get(mk, (0.35, 0.35, 0.35, 1.0))
            ax.errorbar(
                xi,
                yi,
                yerr=ei,
                fmt="none",
                ecolor="0.32",
                elinewidth=err_lw,
                capsize=err_capsize,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                xi,
                yi,
                color=c,
                s=scatter_s,
                edgecolors="0.22",
                linewidths=scatter_lw,
                zorder=3,
                label=None,
            )
            for _, rr in m.iterrows():
                lab = str(rr.get("legend_label", mk)).strip()
                if not lab:
                    continue
                ax.annotate(
                    lab,
                    (float(rr["param_count"]), float(rr[y_mean_col])),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=fs_ann,
                    alpha=0.9,
                    clip_on=False,
                    color="0.15",
                )

        ax.set_xlabel("")
        if i % n_cols == 0:
            ax.set_ylabel(y_label, fontsize=fs_axis_label)
        ax.set_title(_scatter_panel_title(ds), fontsize=fs_panel_title, fontweight="bold", pad=8)
        ax.tick_params(axis="both", which="major", labelsize=fs_tick, width=1.05, length=5)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.set_axisbelow(True)

    for j in range(n_d, n_pad):
        axes_flat[j].set_visible(False)

    model_tick_labels: list[str] = []
    for mk in row_keys:
        subm = plot_df.loc[plot_df["model_row_key"].astype(str) == mk]
        model_tick_labels.append(str(subm["legend_label"].iloc[0]) if len(subm) else mk)

    legend_handles = [
        Patch(
            facecolor=color_of[mk],
            edgecolor="0.25",
            linewidth=1.2,
            label=lab,
            alpha=0.88,
        )
        for mk, lab in zip(row_keys, model_tick_labels, strict=True)
    ]
    n_leg = max(1, len(model_tick_labels))
    fig.legend(
        legend_handles,
        model_tick_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(n_leg, 6),
        fontsize=fs_legend,
        framealpha=0.92,
        handlelength=FACET_LEGEND_HANDLELENGTH + 0.35,
        handleheight=FACET_LEGEND_HANDLEHEIGHT + 0.12,
        labelspacing=FACET_LEGEND_LABELSPACING + 0.12,
        borderpad=FACET_LEGEND_BORDERPAD + 0.08,
        columnspacing=1.45,
    )

    fig.suptitle(suptitle, fontsize=fs_suptitle, y=FACET_SUPTITLE_Y)
    fig.supxlabel(x_label, fontsize=fs_axis_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="W&B best_runs_rerun: export, seed-aggregate, emit LaTeX tables (perf + times)."
    )
    p.add_argument("--entity", default=DEFAULT_WANDB_ENTITY, help="W&B entity")
    p.add_argument("--project", default=DEFAULT_WANDB_PROJECT, help="Single W&B project for all reruns")
    p.add_argument(
        "--run-state",
        default="finished",
        metavar="STATE",
        help='W&B run filter: "finished" (default), "running", "all", …',
    )
    p.add_argument("--quiet", action="store_true", help="Less console output")
    p.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=DEFAULT_AGG_CSV,
        help=f"Seed-aggregated CSV (default: {DEFAULT_AGG_CSV})",
    )
    p.add_argument(
        "--write-raw-csv",
        type=Path,
        default=None,
        help="Optional path to write per-run export before aggregation.",
    )
    p.add_argument(
        "--required-seeds",
        type=int,
        default=5,
        metavar="N",
        help="Keep only hyperparameter groups with exactly N raw runs (default: 5). Ignored with --keep-incomplete-seeds.",
    )
    p.add_argument(
        "--keep-incomplete-seeds",
        action="store_true",
        help="Do not filter on n_seeds.",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        help=f"Directory for emitted .tex files (default: {TABLES_DIR})",
    )
    p.add_argument("--decimals", type=int, default=2, help="Decimal places in time tables (default: 2)")
    p.add_argument(
        "--no-scale-fractions",
        action="store_true",
        help="Performance table: do not scale accuracy-like metrics by 100.",
    )
    p.add_argument(
        "--scatter-plots",
        action="store_true",
        help=(
            "Write PNG scatter plots: parameter count vs rerun timing (best-val pick per "
            "model/dataset/submodel; y error bars = std over seeds)."
        ),
    )
    p.add_argument(
        "--scatter-dir",
        type=Path,
        default=DEFAULT_SCATTER_DIR,
        help=f"Directory for --scatter-plots (default: {DEFAULT_SCATTER_DIR})",
    )
    p.add_argument(
        "--scatter-dpi",
        type=int,
        default=150,
        metavar="DPI",
        help="DPI for --scatter-plots / --memory-plots PNGs (default: 150)",
    )
    p.add_argument(
        "--memory-plots",
        action="store_true",
        help=(
            "Fetch W&B system history peak GPU memory (assigned GPU from trainer.devices), "
            "write diagnostic CSVs and param-count vs peak-mem scatter plots."
        ),
    )
    p.add_argument(
        "--memory-dir",
        type=Path,
        default=DEFAULT_MEMORY_SCATTER_DIR,
        help=f"Directory for --memory-plots PNGs (default: {DEFAULT_MEMORY_SCATTER_DIR})",
    )
    p.add_argument(
        "--memory-table-csv",
        type=Path,
        default=DEFAULT_MEMORY_TABLE_CSV,
        help=f"Per (model, dataset) peak-mem summary CSV (default: {DEFAULT_MEMORY_TABLE_CSV})",
    )
    p.add_argument(
        "--memory-flagged-csv",
        type=Path,
        default=DEFAULT_MEMORY_FLAGGED_CSV,
        help=f"Flagged runs CSV for --memory-plots (default: {DEFAULT_MEMORY_FLAGGED_CSV})",
    )
    p.add_argument(
        "--system-history-samples",
        type=int,
        default=DEFAULT_SYSTEM_HISTORY_SAMPLES,
        metavar="N",
        help=(
            "W&B run.history(stream='events', samples=N) for peak memory "
            f"(default: {DEFAULT_SYSTEM_HISTORY_SAMPLES}). Increase for very long runs."
        ),
    )
    p.add_argument(
        "--probe-system-columns",
        action="store_true",
        help="Print system-history columns for one run and exit (verify GPU metric key names).",
    )
    args = p.parse_args()

    run_state: str | None
    if str(args.run_state).lower() == "all":
        run_state = None
    else:
        run_state = str(args.run_state)

    if args.probe_system_columns:
        probe_one_run_system_columns(args.entity, args.project, run_state=run_state)
        return

    if not args.keep_incomplete_seeds and int(args.required_seeds) < 1:
        p.error("--required-seeds must be >= 1 unless --keep-incomplete-seeds is set.")

    verbose = not args.quiet
    print(f"Entity: {args.entity}  project: {args.project!r}")
    rows, flagged_mem = collect_runs_single_project(
        args.entity,
        args.project,
        run_state=run_state,
        verbose=verbose,
        fetch_peak_gpu_mem=bool(args.memory_plots),
        system_history_samples=int(args.system_history_samples),
    )
    df_raw = dataframe_from_rows(rows)
    if args.write_raw_csv is not None:
        args.write_raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(args.write_raw_csv, index=False)
        print(f"Wrote raw export: {args.write_raw_csv} ({len(df_raw)} rows)")

    sm_cols = _summary_metric_columns_for_rerun_export(df_raw)
    agg, silent = aggregate_wandb_export_by_seed(df_raw, summary_metric_columns=sm_cols)
    report = build_seed_bucket_report(agg)
    req = None if args.keep_incomplete_seeds else int(args.required_seeds)
    if req is not None:
        agg = filter_aggregated_to_required_n_seeds(agg, req)
    agg = agg.fillna("")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output_csv, index=False)
    print(f"Wrote seed-aggregated CSV: {args.output_csv} ({len(agg)} rows x {len(agg.columns)} cols)")

    _print_silent_failure_report(silent)
    _print_seed_bucket_report(report, required_n_seeds=req)

    # --- LaTeX tables (same dataset / row layout as table_generator) ---
    base_specs = expand_mantra_betti_specs(_specs_from_loader_paths())
    groups = partition_specs_graph_simplicial(base_specs)
    groups_timing = _column_groups_timing_headers(groups)
    stats_perf = collect_winner_test_by_submodel(agg)
    simplicial_rows_sub = simplicial_submodel_table_rows()
    cell_rows_sub = cell_submodel_table_rows()
    graph_rows_compact = [
        ("graph/gcn", "GCN"),
        ("graph/gat", "GAT"),
        ("graph/gin", "GIN"),
    ]
    stats_perf_compact = collapse_gnn_submodel_rows_to_base(stats_perf)
    tex_perf_compact = build_latex_table(
        stats_perf_compact,
        column_groups=groups,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        decimals=args.decimals,
        scale_fraction_metrics=not args.no_scale_fractions,
        label="tbl:best_rerun_perf",
        caption="Rerun test, mean $\\pm$ std over seeds (bold $=$ best; blue $=$ not worse, 95\\,\\%).",
    )

    agg_rt = agg.copy()
    _wm, _ws = coalesce_seed_agg_wall_runtime_mean_std(agg_rt)
    agg_rt[_COAL_WALL_MEAN] = _wm
    agg_rt[_COAL_WALL_STD] = _ws
    stats_rt_sub = collect_runtime_stats_from_agg_winners(
        agg_rt, mean_col=_COAL_WALL_MEAN, std_col=_COAL_WALL_STD
    )
    stats_rt_compact = collapse_gnn_submodel_rows_to_base(stats_rt_sub) if stats_rt_sub else {}

    cap_rt = (
        "End-to-end time (without preprocessing) in seconds (mean $\\pm$ std over seeds). "
        "\\textbf{Bold}: lowest mean per dataset (column) and domain type (graph, simplicial or cell). "
        "\\protect\\colorbox{stdblue}{blue}: not significantly slower than that minimum (95\\,\\%, two-sided $Z$)."
    )
    tex_rt_compact = build_rerun_metric_table_plain(
        stats_rt_compact,
        column_groups=groups_timing,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        caption=cap_rt,
        label="tbl:best_rerun_runtime",
        decimals=args.decimals,
    )

    stats_ep_sub: dict[tuple[str, str], dict[str, Any]] = {}
    if SUMMARY_EPOCH_MEAN in df_raw.columns:
        stats_ep_sub = collect_epoch_time_stats_min_timing_std(
            df_raw,
            mean_col=SUMMARY_EPOCH_MEAN,
            std_col=SUMMARY_EPOCH_STD if SUMMARY_EPOCH_STD in df_raw.columns else "",
        )
        # Match compact GNN collapse to the performance table (val from seed-aggregated picks).
        for k in list(stats_ep_sub.keys()):
            if k in stats_perf:
                stats_ep_sub[k]["val_mean"] = float(stats_perf[k]["val_mean"])
    stats_ep_compact = collapse_gnn_submodel_rows_to_base(stats_ep_sub) if stats_ep_sub else {}

    cap_ep = (
        "Train seconds per epoch (mean $\\pm$ std) "
        "\\textbf{Bold}: lowest mean per dataset (column) within domain (graph, simplicial or cell). "
        "\\protect\\colorbox{stdblue}{blue}: not significantly slower than that minimum (95\\,\\%, two-sided $Z$)."
    )
    tex_ep_compact = build_rerun_metric_table_plain(
        stats_ep_compact,
        column_groups=groups_timing,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        caption=cap_ep,
        label="tbl:best_rerun_epoch_time",
        decimals=args.decimals,
        comparison_scope="domain",
    )
    cap_ep_global = (
        "Same per-epoch times as the domain-banded table. "
        "\\textbf{Bold}: lowest mean per dataset (column) across \\textbf{all} models (not per domain). "
        "\\protect\\colorbox{stdblue}{blue}: not significantly slower than that column minimum "
        "(95\\,\\%, two-sided $Z$)."
    )
    tex_ep_global = build_rerun_metric_table_plain(
        stats_ep_compact,
        column_groups=groups_timing,
        graph_rows=graph_rows_compact,
        simplicial_rows=simplicial_rows_sub,
        cell_rows=cell_rows_sub,
        caption=cap_ep_global,
        label="tbl:best_rerun_epoch_time_global_min",
        decimals=args.decimals,
        comparison_scope="global",
    )

    out_dir = Path(args.tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rerun_main_table.tex": tex_perf_compact,
        "rerun_time_runtime.tex": tex_rt_compact,
        "rerun_time_train_epoch.tex": tex_ep_compact,
        "rerun_time_train_epoch_bold_global.tex": tex_ep_global,
    }
    for name, body in paths.items():
        path = out_dir / name
        path.write_text(body, encoding="utf-8")
        print(f"Wrote {path}")

    if not stats_rt_sub:
        print(
            "Note: no wall-clock timing in aggregated CSV — expected ``summary_Runtime`` "
            "from W&B ``_runtime`` (or ``Runtime``) in run summary / ``_runtime`` in config."
        )
    if not stats_ep_sub:
        print(
            f"Note: no column {SUMMARY_EPOCH_MEAN!r} in raw export — "
            "``PipelineTimer`` stores AvgTime in run **config** via ``log_hyperparams``; "
            "if still empty, training may have ended before enough epochs to log averages."
        )

    if args.scatter_plots:
        scatter_df = build_rerun_timing_vs_params_frame(agg)
        if scatter_df.empty:
            print("Scatter plots: no data (empty aggregate or could not build best-val picks).")
        else:
            _pc = pd.to_numeric(scatter_df["param_count"], errors="coerce")
            scatter_df = scatter_df.loc[_pc.notna() & np.isfinite(_pc.to_numpy(dtype=float))].copy()
            if scatter_df.empty:
                print("Scatter plots: no rows with finite parameter count.")
            else:
                wall_m, wall_s = "wall_mean", "wall_std"
                ep_m, ep_s = "epoch_mean", "epoch_std"
                scatter_bands: tuple[
                    tuple[str, list[str], Callable[[pd.DataFrame], pd.DataFrame], str], ...
                ] = (
                    ("cell", _dataset_order_cell(), _filter_scatter_cell, "Cell"),
                    (
                        "simplicial_graph",
                        _dataset_order_simplicial_graph_benchmarks(),
                        _filter_scatter_simplicial_graph_benchmarks,
                        "Simplicial (graph)",
                    ),
                    (
                        "simplicial_mantra",
                        _dataset_order_simplicial_mantra(),
                        _filter_scatter_simplicial_mantra,
                        "Simplicial (MANTRA)",
                    ),
                )
                for stem_suffix, ds_order, filt_fn, title_suffix in scatter_bands:
                    sub = filt_fn(scatter_df)
                    if sub.empty:
                        print(f"Scatter plots: skip band {stem_suffix!r} (no rows after filter).")
                        continue
                    if wall_m in sub.columns and sub[wall_m].notna().any() and np.isfinite(
                        sub[wall_m].to_numpy(dtype=float)
                    ).any():
                        emit_rerun_scatter_timing_vs_params(
                            sub,
                            y_mean_col=wall_m,
                            y_std_col=wall_s,
                            stem=f"rerun_wall_runtime_vs_params_{stem_suffix}",
                            suptitle=f"Best model runtime vs. parameter count: {title_suffix}",
                            y_label="End-to-end training time (s) (no preprocessing)",
                            x_label="Parameter count (total)",
                            dataset_order=ds_order,
                            out_dir=Path(args.scatter_dir),
                            dpi=int(args.scatter_dpi),
                        )
                    else:
                        print(
                            f"Scatter plots: skip wall-clock figure for band {stem_suffix!r} "
                            "(no finite wall_mean)."
                        )
                    if ep_m in sub.columns and sub[ep_m].notna().any() and np.isfinite(
                        sub[ep_m].to_numpy(dtype=float)
                    ).any():
                        emit_rerun_scatter_timing_vs_params(
                            sub,
                            y_mean_col=ep_m,
                            y_std_col=ep_s,
                            stem=f"rerun_train_epoch_vs_params_{stem_suffix}",
                            suptitle=f"Best model train epoch time vs. parameter count: {title_suffix}",
                            y_label="Train time per epoch (s)",
                            x_label="Parameter count (total)",
                            dataset_order=ds_order,
                            out_dir=Path(args.scatter_dir),
                            dpi=int(args.scatter_dpi),
                        )
                    else:
                        print(
                            f"Scatter plots: skip per-epoch figure for band {stem_suffix!r} "
                            "(no finite epoch_mean)."
                        )

    if args.memory_plots:
        mem_table = build_peak_mem_diagnostic_table(df_raw, flagged_mem)
        args.memory_table_csv.parent.mkdir(parents=True, exist_ok=True)
        mem_table.to_csv(args.memory_table_csv, index=False)
        print(
            f"Wrote peak-mem diagnostic table: {args.memory_table_csv} "
            f"({len(mem_table)} rows)"
        )
        if not mem_table.empty and verbose:
            n_flag_tot = int(mem_table["n_flagged"].sum())
            n_ok_tot = int(mem_table["n_seeds"].sum())
            print(
                f"  peak-mem seeds ok={n_ok_tot}, flagged runs={n_flag_tot} "
                f"({100.0 * n_flag_tot / max(n_ok_tot + n_flag_tot, 1):.1f}% of raw attempts)"
            )
            # Compact console preview of groups with any flags.
            flagged_groups = mem_table.loc[mem_table["n_flagged"] > 0]
            if not flagged_groups.empty:
                print("  groups with n_flagged > 0:")
                for _, rr in flagged_groups.iterrows():
                    print(
                        f"    {rr['model']} | {rr['dataset']}: "
                        f"n_seeds={int(rr['n_seeds'])} n_flagged={int(rr['n_flagged'])}"
                    )

        flagged_df = pd.DataFrame(flagged_mem) if flagged_mem else pd.DataFrame(
            columns=[
                "identifiers_run_id",
                "identifiers_run_name",
                "identifiers_run_url",
                "model",
                "dataset",
                "seed",
                "trainer_devices_device_index",
                "mean_util_pct",
                "flag_reason",
            ]
        )
        args.memory_flagged_csv.parent.mkdir(parents=True, exist_ok=True)
        flagged_df.to_csv(args.memory_flagged_csv, index=False)
        print(f"Wrote flagged peak-mem runs: {args.memory_flagged_csv} ({len(flagged_df)} rows)")

        scatter_df = build_rerun_timing_vs_params_frame(agg)
        if scatter_df.empty:
            print("Memory plots: no data (empty aggregate or could not build best-val picks).")
        else:
            _pc = pd.to_numeric(scatter_df["param_count"], errors="coerce")
            scatter_df = scatter_df.loc[_pc.notna() & np.isfinite(_pc.to_numpy(dtype=float))].copy()
            if scatter_df.empty:
                print("Memory plots: no rows with finite parameter count.")
            else:
                has_peak_y = (
                    "peak_mem_mean" in scatter_df.columns
                    and scatter_df["peak_mem_mean"].notna().any()
                    and np.isfinite(scatter_df["peak_mem_mean"].to_numpy(dtype=float)).any()
                )
                has_mean_y = (
                    "mean_mem_mean" in scatter_df.columns
                    and scatter_df["mean_mem_mean"].notna().any()
                    and np.isfinite(scatter_df["mean_mem_mean"].to_numpy(dtype=float)).any()
                )
                if not has_peak_y and not has_mean_y:
                    print(
                        "Memory plots: no finite peak/mean mem in best-val picks "
                        "(check flagged CSV / system metrics)."
                    )
                else:
                    scatter_bands: tuple[
                        tuple[str, list[str], Callable[[pd.DataFrame], pd.DataFrame], str], ...
                    ] = (
                        ("cell", _dataset_order_cell(), _filter_scatter_cell, "Cell"),
                        (
                            "simplicial_graph",
                            _dataset_order_simplicial_graph_benchmarks(),
                            _filter_scatter_simplicial_graph_benchmarks,
                            "Simplicial (graph)",
                        ),
                        (
                            "simplicial_mantra",
                            _dataset_order_simplicial_mantra(),
                            _filter_scatter_simplicial_mantra,
                            "Simplicial (MANTRA)",
                        ),
                    )
                    for stem_suffix, ds_order, filt_fn, title_suffix in scatter_bands:
                        sub = filt_fn(scatter_df)
                        if sub.empty:
                            print(
                                f"Memory plots: skip band {stem_suffix!r} (no rows after filter)."
                            )
                            continue
                        if (
                            "peak_mem_mean" in sub.columns
                            and sub["peak_mem_mean"].notna().any()
                            and np.isfinite(sub["peak_mem_mean"].to_numpy(dtype=float)).any()
                        ):
                            emit_rerun_scatter_timing_vs_params(
                                sub,
                                y_mean_col="peak_mem_mean",
                                y_std_col="peak_mem_std",
                                stem=f"rerun_peak_gpu_mem_vs_params_{stem_suffix}",
                                suptitle=(
                                    "Best model peak GPU memory vs. parameter count: "
                                    f"{title_suffix}"
                                ),
                                y_label="Peak GPU memory (GB)",
                                x_label="Parameter count (total)",
                                dataset_order=ds_order,
                                out_dir=Path(args.memory_dir),
                                dpi=int(args.scatter_dpi),
                            )
                        else:
                            print(
                                f"Memory plots: skip peak figure for band {stem_suffix!r} "
                                "(no finite peak_mem_mean)."
                            )
                        if (
                            "mean_mem_mean" in sub.columns
                            and sub["mean_mem_mean"].notna().any()
                            and np.isfinite(sub["mean_mem_mean"].to_numpy(dtype=float)).any()
                        ):
                            emit_rerun_scatter_timing_vs_params(
                                sub,
                                y_mean_col="mean_mem_mean",
                                y_std_col="mean_mem_std",
                                stem=f"rerun_mean_gpu_mem_vs_params_{stem_suffix}",
                                suptitle=(
                                    "Best model mean GPU memory vs. parameter count: "
                                    f"{title_suffix}"
                                ),
                                y_label="Mean GPU memory allocated (GB)",
                                x_label="Parameter count (total)",
                                dataset_order=ds_order,
                                out_dir=Path(args.memory_dir),
                                dpi=int(args.scatter_dpi),
                            )
                        else:
                            print(
                                f"Memory plots: skip mean figure for band {stem_suffix!r} "
                                "(no finite mean_mem_mean)."
                            )


if __name__ == "__main__":
    main()

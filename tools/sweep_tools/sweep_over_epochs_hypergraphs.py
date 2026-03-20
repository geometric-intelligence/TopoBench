import os
import os.path as osp
import shutil
from collections import defaultdict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from topobench.data.preprocessor import PreProcessor
from topobench.data.utils import build_cluster_transform
from topobench.dataloader.dataload_cluster import (
    BlockCSRBatchCollator,
    _HandleAdapter,
    _PartIdListDataset,
)


# 1. BUILD GOLDEN & LOADER (for multi-epoch runs)
def build_golden_and_loader_from_cfg(
    cfg: DictConfig, num_parts: int = 10, batch_size: int = 1
):
    """Build golden graph and candidate loader for multi-epoch runs.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for dataset and transforms.
    num_parts : int, optional
        Number of graph partitions, by default 10.
    batch_size : int, optional
        Batch size for the candidate DataLoader, by default 1.

    Returns
    -------
    golden : torch_geometric.data.Data
        Full reference (golden) graph.
    loader : torch.utils.data.DataLoader
        DataLoader over partition IDs producing candidate batches.
    handle : dict
        Partition handle with metadata and paths.
    """
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    transforms_config = cfg.get("transforms", None)

    golden_pre = PreProcessor(dataset, dataset_dir, transforms_config)
    assert len(golden_pre.data_list) == 1, "Expected a single graph for transductive setting."
    golden = golden_pre.data_list[0]
    print("Golden keys:", sorted(golden.keys()))

    raw_pre = PreProcessor(dataset, dataset_dir, transforms_config=None)

    cluster_params = cfg.dataset.loader.parameters.get("cluster", {})
    if hasattr(cluster_params, "to_container"):
        cluster_params = cluster_params.to_container(resolve=True)
    cluster_params = dict(cluster_params)
    cluster_params["num_parts"] = int(num_parts)

    handle = raw_pre.pack_global_partition(
        split_params=cfg.dataset.get("split_params", {}),
        cluster_params=cluster_params,
        stream_params=cfg.dataset.loader.parameters.get("stream", {}),
        dtype_policy=cfg.dataset.loader.parameters.get("dtype_policy", "preserve"),
        pack_db=True,
        pack_memmaps=True,
    )

    post_batch_transform = build_cluster_transform(transforms_config)
    print("Post-batch transform:", post_batch_transform)

    adapter = _HandleAdapter(handle)
    part_ids = np.arange(adapter.num_parts, dtype=np.int64)
    part_ds = _PartIdListDataset(part_ids)

    active_split = cfg.get("cluster_eval", {}).get("active_split", "train")

    collate = BlockCSRBatchCollator(
        adapter,
        device=None,
        with_edge_attr=handle.get("has_edge_attr", False),
        active_split=active_split,
        post_batch_transform=post_batch_transform,
    )

    loader = DataLoader(
        part_ds,
        batch_size=batch_size,
        shuffle=True,   # important: new cluster combinations per epoch
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )

    return golden, loader, handle


# 2. PERMUTATION HANDLING
def load_perm_to_global(handle):
    """Load mapping from permuted node IDs to global node IDs.

    Parameters
    ----------
    handle : dict
        Partition handle containing ``processed_dir``.

    Returns
    -------
    torch.Tensor or None
        Long tensor of shape (num_nodes,) mapping permuted IDs to global IDs,
        or ``None`` if the mapping file does not exist.
    """
    base_dir = handle["processed_dir"]
    perm_path = osp.join(base_dir, "perm_memmap", "perm_to_global.npy")

    if not osp.exists(perm_path):
        print("No perm_to_global.npy found; assuming global_nid are already true IDs.")
        return None

    arr = np.load(perm_path, mmap_mode="r")
    perm_to_global = torch.from_numpy(arr).long()
    print("Loaded perm_to_global of shape", perm_to_global.shape)
    return perm_to_global


def resolve_true_global_ids(golden, batch, perm_to_global: torch.Tensor | None):
    """Resolve true global node IDs for a batch.

    Parameters
    ----------
    golden : torch_geometric.data.Data
        Golden reference graph with baseline features.
    batch : torch_geometric.data.Data
        Candidate batch with optional ``global_nid``.
    perm_to_global : torch.Tensor or None
        Mapping from permuted IDs to global IDs.

    Returns
    -------
    torch.Tensor
        Long tensor of global node IDs on the batch device.
    """
    n = batch.x.size(0)
    device = batch.x.device

    if hasattr(batch, "global_nid"):
        gids = batch.global_nid.clone().detach().long()

        if perm_to_global is not None:
            true_gids = perm_to_global[gids]
        else:
            true_gids = gids

        if hasattr(golden, "x_0") and hasattr(batch, "x_0"):
            k = min(50, n)
            idx = torch.randint(0, n, (k,))
            try:
                diff = (batch.x_0[idx].cpu() - golden.x_0[true_gids[idx]].cpu()).abs().max()
                print("  sanity max|x_0_batch - x_0_golden[true_gids]| =", float(diff))
            except Exception as e:
                print("  sanity check failed:", e)

        return true_gids.to(device)

    return torch.arange(n, device=device)


# 3. HYPERGRAPH STRUCTURE
def hyperedges_from_incidence_hyperedges(data, true_global_ids: torch.Tensor):
    """Extract hyperedges from an incidence matrix.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data with ``incidence_hyperedges`` attribute.
    true_global_ids : torch.Tensor
        Global node IDs for the incidence row index.

    Returns
    -------
    set of tuple of int
        Set of hyperedges as sorted tuples of global node IDs.
    """
    if not hasattr(data, "incidence_hyperedges"):
        return set()

    H = data.incidence_hyperedges
    if H is None or H.numel() == 0:
        return set()

    H = H.cpu()
    gids = true_global_ids.cpu()

    he_to_nodes = defaultdict(set)

    if getattr(H, "is_sparse", False):
        H = H.coalesce()
        rows, cols = H.indices()
        for i, j in zip(rows.tolist(), cols.tolist(), strict=False):
            he_to_nodes[int(j)].add(int(i))
    else:
        rows, cols = (H != 0).nonzero(as_tuple=True)
        for i, j in zip(rows.tolist(), cols.tolist(), strict=False):
            he_to_nodes[int(j)].add(int(i))

    hyperedges = set()
    for nodes in he_to_nodes.values():
        if len(nodes) <= 1:
            continue
        vs = sorted({int(gids[i]) for i in nodes})
        if len(vs) > 1:
            hyperedges.add(tuple(vs))

    return hyperedges


# 4. METRICS
def compute_hyperedge_coverage_metrics(gold_hyperedges, cand_hyperedges):
    """Compute strict and partial hyperedge coverage metrics.

    Parameters
    ----------
    gold_hyperedges : iterable of tuple of int
        Ground-truth hyperedges.
    cand_hyperedges : iterable of tuple of int
        Candidate hyperedges.

    Returns
    -------
    dict
        Dictionary with keys
        ``gold_n``, ``cand_n``, ``strict_match``, ``partial_covered``,
        ``strict_recall``, ``partial_recall``.
    """
    gold_fs = {frozenset(h) for h in gold_hyperedges}
    cand_fs = {frozenset(h) for h in cand_hyperedges}

    G = len(gold_fs)
    if G == 0:
        return {
            "gold_n": 0,
            "cand_n": len(cand_fs),
            "strict_match": 0,
            "partial_covered": 0,
            "strict_recall": 0.0,
            "partial_recall": 0.0,
        }

    common = gold_fs & cand_fs
    strict_match = len(common)

    missing = list(gold_fs - cand_fs)
    new = list(cand_fs - gold_fs)

    missing_sets = [set(h) for h in missing]

    partially_covered_gold = set(common)

    for mh_fs, mh_set in zip(missing, missing_sets, strict=False):
        for c in new:
            cs = set(c)
            if len(cs) > 1 and cs < mh_set:
                partially_covered_gold.add(mh_fs)
                break

    partial_covered = len(partially_covered_gold)
    strict_recall = strict_match / G
    partial_recall = partial_covered / G

    return {
        "gold_n": G,
        "cand_n": len(cand_fs),
        "strict_match": strict_match,
        "partial_covered": partial_covered,
        "strict_recall": strict_recall,
        "partial_recall": partial_recall,
    }


def summarize_hyperedge_differences(gold_hyperedges, cand_hyperedges) -> str:
    """Summarize overlap and differences between golden and candidate hyperedges.

    Parameters
    ----------
    gold_hyperedges : iterable of tuple of int
        Ground-truth hyperedges.
    cand_hyperedges : iterable of tuple of int
        Candidate hyperedges.

    Returns
    -------
    str
        Human-readable summary of exact matches, splits, and new or lost edges.
    """
    gold_fs = {frozenset(h) for h in gold_hyperedges}
    cand_fs = {frozenset(h) for h in cand_hyperedges}

    common = gold_fs & cand_fs
    missing = list(gold_fs - cand_fs)
    new = list(cand_fs - gold_fs)

    missing_sets = [set(h) for h in missing]

    explained_missing = set()
    split_fragments = 0
    unexplained_new = 0

    for new_h in new:
        hs = set(new_h)
        found_parent = False
        for mh_fs, mh_set in zip(missing, missing_sets, strict=False):
            if hs < mh_set:
                split_fragments += 1
                explained_missing.add(mh_fs)
                found_parent = True
                break
        if not found_parent:
            unexplained_new += 1

    completely_lost = len(missing) - len(explained_missing)

    return (
        f"Hyperedges: {len(common)} match exactly, {len(missing)} are missing in the "
        f"clustered pipeline ({split_fragments} explained as partition-induced splits of "
        f"larger golden hyperedges and {completely_lost} completely lost), and {len(new)} "
        f"appear only in the clustered pipeline ({unexplained_new} not explainable as "
        f"subsets of golden hyperedges)."
    )


# 5. MULTI-EPOCH FOR A SINGLE (bs, num_parts)
def run_single_experiment_multi_epoch(
    cfg: DictConfig,
    num_parts: int = 10,
    bs: int = 1,
    num_epochs: int = 5,
    accumulate: bool = True,
):
    """Run a multi-epoch hyperedge coverage experiment for one configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for dataset and model.
    num_parts : int, optional
        Number of graph partitions, by default 10.
    bs : int, optional
        Batch size for candidate loader, by default 1.
    num_epochs : int, optional
        Number of epochs (passes over the loader), by default 5.
    accumulate : bool, optional
        If True, union hyperedges across epochs; otherwise evaluate per epoch,
        by default True.

    Returns
    -------
    list of dict
        List of metrics dicts (one per epoch) from
        :func:`compute_hyperedge_coverage_metrics`.
    """
    golden, loader, handle = build_golden_and_loader_from_cfg(
        cfg, num_parts=num_parts, batch_size=bs
    )

    requested_num_parts = int(num_parts)
    actual_num_parts = int(handle.get("num_parts", -1))
    if actual_num_parts != requested_num_parts:
        raise RuntimeError(
            f"Partitioner produced {actual_num_parts} parts but requested {requested_num_parts}"
        )

    perm_to_global = load_perm_to_global(handle)

    N = golden.x.size(0)
    base_ids = torch.arange(N)
    gold_hyperedges = hyperedges_from_incidence_hyperedges(golden, base_ids)

    if len(gold_hyperedges) == 0:
        print("Golden has no hyperedges; aborting.")
        return []

    metrics_per_epoch = []
    cumulative_cand_hyperedges = set()

    for epoch in range(1, num_epochs + 1):
        epoch_hyperedges = set()

        for batch in loader:  # one pass over all parts, in random order
            true_global_ids = resolve_true_global_ids(golden, batch, perm_to_global)
            epoch_hyperedges |= hyperedges_from_incidence_hyperedges(batch, true_global_ids)

        if accumulate:
            cumulative_cand_hyperedges |= epoch_hyperedges
            cand_hyperedges = cumulative_cand_hyperedges
        else:
            cand_hyperedges = epoch_hyperedges

        m = compute_hyperedge_coverage_metrics(gold_hyperedges, cand_hyperedges)
        summary = summarize_hyperedge_differences(gold_hyperedges, cand_hyperedges)

        print(
            f"[epoch={epoch}, bs={bs}, num_parts={num_parts}] "
            f"strict_recall={m['strict_recall']:.4f}, partial_recall={m['partial_recall']:.4f}"
        )
        print(summary)

        metrics_per_epoch.append(m)

    return metrics_per_epoch


# 6. SWEEP OVER (bs, num_parts) FOR MULTIPLE EPOCHS
def run_sweep_multi_epoch(num_epochs: int = 4, accumulate: bool = True):
    """Sweep over (batch_size, num_parts) for multi-epoch experiments.

    For each configuration, run :func:`run_single_experiment_multi_epoch` and
    store per-epoch metrics.

    Parameters
    ----------
    num_epochs : int, optional
        Number of epochs per configuration, by default 4.
    accumulate : bool, optional
        If True, union hyperedges across epochs; otherwise per-epoch only,
        by default True.

    Returns
    -------
    list of int
        Tested batch sizes.
    list of int
        Tested numbers of partitions.
    dict
        Mapping ``(bs, num_parts) -> [metrics_epoch1, ...]``.
    """
    dataset_name = "graph/cocitation_cora_cluster"
    batch_sizes = [1, 2, 4, 8, 16, 32]
    num_parts_list = [2, 4, 8, 16, 32]

    results_per_pair = {}

    with hydra.initialize(config_path="../../configs", job_name="topo_debug_sweep_epochs"):
        for num_parts in num_parts_list:
            for bs in batch_sizes:
                splits_dir = osp.join("datasets", "data_splits", "PubMed")
                if osp.exists(splits_dir):
                    shutil.rmtree(splits_dir)

                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        f"dataset={dataset_name}",
                        "model=hypergraph/edgnn",
                        "trainer.max_epochs=1",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "trainer.accelerator=cpu",
                        "trainer.devices=1",
                        "seed=42",
                        "paths=test",
                    ],
                    return_hydra_config=True,
                )

                if bs <= num_parts:
                    metrics_per_epoch = run_single_experiment_multi_epoch(
                        cfg,
                        num_parts=num_parts,
                        bs=bs,
                        num_epochs=num_epochs,
                        accumulate=accumulate,
                    )
                    results_per_pair[(bs, num_parts)] = metrics_per_epoch
                else:
                    print(f"Skipping bs={bs}, num_parts={num_parts} (bs>num_parts).")

    return batch_sizes, num_parts_list, results_per_pair


# 7. MATRIX HELPER + SUMMARY PLOT
def build_metric_matrix_for_epoch(batch_sizes, num_parts_list, results_per_pair, epoch, metric_key):
    """Build a metric matrix for a given epoch.

    Parameters
    ----------
    batch_sizes : sequence of int
        Batch sizes (row order).
    num_parts_list : sequence of int
        Numbers of partitions (column order).
    results_per_pair : dict
        Mapping ``(bs, num_parts) -> [metrics_epoch1, ...]``.
    epoch : int
        1-based epoch index to extract.
    metric_key : str
        Metric key to extract and convert to percent.

    Returns
    -------
    ndarray
        Matrix of metric values in percent with NaNs for missing entries.
    """
    """
    Build bs × num_parts matrix for a given epoch index (1-based).

    results_per_pair[(bs, num_parts)] = [metrics_epoch1, metrics_epoch2, ...]
    """
    B = len(batch_sizes)
    P = len(num_parts_list)
    M = np.full((B, P), np.nan, dtype=float)

    for i, bs in enumerate(batch_sizes):
        for j, nparts in enumerate(num_parts_list):
            metrics_list = results_per_pair.get((bs, nparts))
            if metrics_list is None or len(metrics_list) < epoch:
                continue
            m = metrics_list[epoch - 1]
            val = m.get(metric_key)
            if val is None:
                continue
            M[i, j] = 100.0 * float(val)  # percent

    return M


def plot_summary_4panel(
    batch_sizes,
    num_parts_list,
    results_per_pair,
    num_epochs: int,
    metric_key: str = "strict_recall",
    target_bs: int = 8,
    target_num_parts: int = 32,
    save_prefix: str = "cora_hypergraph_summary",
):
    """Plot a 4-panel summary for recall across epochs.

    Panels:
      (0, 0) First epoch heatmap.
      (0, 1) Last epoch heatmap.
      (1, 0) Ratio of last to first recall.
      (1, 1) Recall vs. epochs for a chosen (bs, num_parts).

    Parameters
    ----------
    batch_sizes : sequence of int
        Batch sizes (row labels).
    num_parts_list : sequence of int
        Numbers of partitions (column labels).
    results_per_pair : dict
        Mapping ``(bs, num_parts) -> [metrics_epoch1, ...]``.
    num_epochs : int
        Number of epochs to summarize.
    metric_key : str, optional
        Metric key to visualize, by default "strict_recall".
    target_bs : int, optional
        Batch size for the trajectory panel, by default 8.
    target_num_parts : int, optional
        Number of partitions for the trajectory panel, by default 32.
    save_prefix : str, optional
        Filename prefix for saved figures, by default "cora_hypergraph_summary".
    """
    """
    Make a 2x2 figure:
      (0,0) first epoch heatmap
      (0,1) last epoch heatmap
      (1,0) ratio (last / first)
      (1,1) recall vs epochs for (target_bs, target_num_parts)
    """
    # --- matrices for first and last epoch ---
    first_M = build_metric_matrix_for_epoch(
        batch_sizes, num_parts_list, results_per_pair, epoch=1, metric_key=metric_key
    )
    last_M = build_metric_matrix_for_epoch(
        batch_sizes, num_parts_list, results_per_pair, epoch=num_epochs, metric_key=metric_key
    )

    # ratio (last / first)
    ratio_M = np.full_like(first_M, np.nan, dtype=float)
    for (i, j), first_val in np.ndenumerate(first_M):
        last_val = last_M[i, j]
        if np.isnan(first_val) or np.isnan(last_val) or first_val <= 0:
            continue
        ratio_M[i, j] = last_val / first_val  # dimensionless factor

    # --- set up figure and axes ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax_first, ax_last, ax_ratio, ax_curve = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    vmin, vmax = 0, 100
    xticks = np.arange(len(num_parts_list))
    yticks = np.arange(len(batch_sizes))

    # ---------- top-left: first epoch ----------
    im_first = ax_first.imshow(first_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax_first.set_xticks(xticks)
    ax_first.set_yticks(yticks)
    ax_first.set_xticklabels(num_parts_list)
    ax_first.set_yticklabels(batch_sizes)
    ax_first.set_xlabel("num_parts")
    ax_first.set_ylabel("batch_size")
    ax_first.set_title("Recall % – epoch 1")

    for (i, j), val in np.ndenumerate(first_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax_first.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)

    # ---------- top-right: last epoch ----------
    ax_last.imshow(last_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax_last.set_xticks(xticks)
    ax_last.set_yticks(yticks)
    ax_last.set_xticklabels(num_parts_list)
    ax_last.set_yticklabels([])  # share y labels with left
    ax_last.set_xlabel("num_parts")
    ax_last.set_title(f"Recall % – epoch {num_epochs}")

    for (i, j), val in np.ndenumerate(last_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax_last.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)

    # shared colorbar for the two recall heatmaps
    cbar_top = fig.colorbar(im_first, ax=[ax_first, ax_last], fraction=0.035, pad=0.02)
    label = "Strict recall (%)" if metric_key == "strict_recall" else "Partial recall (%)"
    cbar_top.set_label(label)

    # ---------- bottom-left: ratio ----------
    if np.all(np.isnan(ratio_M)):
        pass
    else:
        max(1.0, float(np.nanmax(ratio_M)))
    im_ratio = ax_ratio.imshow(ratio_M, origin="lower", aspect="auto", cmap="Reds")

    ax_ratio.set_xticks(xticks)
    ax_ratio.set_yticks(yticks)
    ax_ratio.set_xticklabels(num_parts_list)
    ax_ratio.set_yticklabels(batch_sizes)
    ax_ratio.set_xlabel("num_parts")
    ax_ratio.set_ylabel("batch_size")
    ax_ratio.set_title("Final / initial recall (×)")

    for (i, j), val in np.ndenumerate(ratio_M):
        if np.isnan(val):
            continue
        color = "white" if val < 1.0 else "black"
        ax_ratio.text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=7, color=color)

    cbar_ratio = fig.colorbar(im_ratio, ax=ax_ratio, fraction=0.035, pad=0.02)
    cbar_ratio.set_label("Final / initial")

    # ---------- bottom-right: trajectory for specific (bs, num_parts) ----------
    metrics_list = results_per_pair.get((target_bs, target_num_parts))
    if metrics_list is None or len(metrics_list) == 0:
        ax_curve.text(
            0.5,
            0.5,
            f"No data for bs={target_bs}, num_parts={target_num_parts}",
            ha="center",
            va="center",
        )
        ax_curve.axis("off")
    else:
        epochs = np.arange(1, len(metrics_list) + 1)
        vals = [m[metric_key] * 100.0 for m in metrics_list]

        ax_curve.plot(epochs, vals, marker="o")
        ax_curve.set_xlabel("epoch")
        ax_curve.set_ylabel("recall (%)")
        ax_curve.set_title(f"Recall vs epochs (bs={target_bs}, num_parts={target_num_parts})")
        ax_curve.grid(alpha=0.3)
        ax_curve.set_ylim(0, 105)

    out_path = f"sweep_tools/outputs/{save_prefix}_{metric_key}.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved 4-panel summary: {out_path}")
    # plt.show()



# 8. MAIN
if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    os.makedirs("sweep_tools/outputs", exist_ok=True)

    NUM_EPOCHS = 10  # how many "epochs thus far" panels you want

    batch_sizes, num_parts_list, results_per_pair = run_sweep_multi_epoch(
        num_epochs=NUM_EPOCHS,
        accumulate=True,   # union of hyperedges across epochs
    )

    plot_summary_4panel(
        batch_sizes,
        num_parts_list,
        results_per_pair,
        num_epochs=NUM_EPOCHS,
        metric_key="strict_recall",
        target_bs=8,
        target_num_parts=32,
        save_prefix="cora_hypergraph_4panel",
    )

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


# 1. BUILD GOLDEN & CANDIDATE
def build_golden_and_candidate_from_cfg(cfg: DictConfig, num_parts: int = 10, batch_size: int = 1):
    """Build golden graph and candidate cluster batches from config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for dataset and transforms.
    num_parts : int, optional
        Number of graph partitions, by default 10.
    batch_size : int, optional
        DataLoader batch size, by default 1.

    Returns
    -------
    golden : torch_geometric.data.Data
        Full reference (golden) graph.
    candidate_batches : list of torch_geometric.data.Data
        List of cluster-wise candidate batches.
    handle : dict
        Partition handle with metadata and paths.
    """
    # Load base dataset
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    transforms_config = cfg.get("transforms", None)

    # Golden: global lifting (Option A)
    golden_pre = PreProcessor(dataset, dataset_dir, transforms_config)
    assert len(golden_pre.data_list) == 1, "Expected a single graph for transductive setting."
    golden = golden_pre.data_list[0]
    print("Golden keys:", sorted(golden.keys()))

    # Candidate: partition raw graph, then apply same lifting per cluster (Option B)
    raw_pre = PreProcessor(dataset, dataset_dir, transforms_config=None)

    # Ensure cluster_params is a plain dict and set num_parts explicitly
    cluster_params = cfg.dataset.loader.parameters.get("cluster", {})
    # If it's an OmegaConf DictConfig, convert to a dict to avoid weird mutation
    if hasattr(cluster_params, "to_container"):
        cluster_params = cluster_params.to_container(resolve=True)
    cluster_params = dict(cluster_params)  # shallow copy
    cluster_params["num_parts"] = int(num_parts)

    handle = raw_pre.pack_global_partition(
        split_params=cfg.dataset.get("split_params", {}),
        cluster_params=cluster_params,
        stream_params=cfg.dataset.loader.parameters.get("stream", {}),
        dtype_policy=cfg.dataset.loader.parameters.get("dtype_policy", "preserve"),
        pack_db=True,
        pack_memmaps=True,
    )

    # Same lifting used as post-batch transform
    post_batch_transform = build_cluster_transform(transforms_config)
    print("Post-batch transform:", post_batch_transform)

    adapter = _HandleAdapter(handle)
    part_ids = np.arange(adapter.num_parts, dtype=np.int64)
    part_ds = _PartIdListDataset(part_ids)

    # Batch size & split for eval come from cfg so we can sweep
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
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )

    candidate_batches = [batch for batch in loader]

    print("Num candidate batches:", len(candidate_batches))
    if candidate_batches:
        b0 = candidate_batches[0]
        print("First candidate keys:", sorted(b0.keys()))
        if hasattr(b0, "global_nid"):
            print("First candidate global_nid shape:", b0.global_nid.shape)

    return golden, candidate_batches, handle


# 2. PERMUTATION HANDLING
def load_perm_to_global(handle):
    """Load mapping from permuted node IDs to global IDs.

    Parameters
    ----------
    handle : dict
        Partition handle containing processed paths.

    Returns
    -------
    torch.Tensor or None
        Long tensor of shape (num_nodes,) mapping permuted IDs to global IDs,
        or ``None`` if no mapping is found.
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
        Golden reference graph.
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

        # Optional sanity check on x_0 if available
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
        Global node IDs for rows in the incidence matrix.

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


# 4. METRICS (STRICT + PARTIAL)
def compute_hyperedge_coverage_metrics(gold_hyperedges, cand_hyperedges):
    """Compute strict and partial recall for hyperedge coverage.

    Parameters
    ----------
    gold_hyperedges : iterable of tuple of int
        Ground-truth hyperedges.
    cand_hyperedges : iterable of tuple of int
        Candidate hyperedges.

    Returns
    -------
    dict
        Dictionary with keys:
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
            if len(cs) > 1 and cs < mh_set:  # strict subset fragment
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
        Human-readable summary of matches, splits, and lost or new hyperedges.
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


# 5. SINGLE EXPERIMENT (ONE bs, num_parts)
def run_single_experiment(cfg: DictConfig, num_parts: int = 10, bs = 1):
    """Run a single hyperedge coverage experiment for given batch size and partitions.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for dataset and model.
    num_parts : int, optional
        Number of graph partitions, by default 10.
    bs : int, optional
        Batch size, by default 1.

    Returns
    -------
    dict or None
        Coverage metrics from :func:`compute_hyperedge_coverage_metrics`,
        or ``None`` if no candidate batches exist.
    """
    golden, candidate_batches, handle = build_golden_and_candidate_from_cfg(cfg, num_parts=num_parts, batch_size = bs)

    # loud runtime check
    requested_num_parts = int(num_parts)
    actual_num_parts = int(handle.get("num_parts", -1))
    if actual_num_parts != requested_num_parts:
        raise RuntimeError(f"Partitioner produced {actual_num_parts} parts but requested {requested_num_parts}")

    if not candidate_batches:
        print("No candidate batches; abort.")
        return None

    perm_to_global = load_perm_to_global(handle)

    N = golden.x.size(0)
    base_ids = torch.arange(N)

    gold_hyperedges = hyperedges_from_incidence_hyperedges(golden, base_ids)

    cand_hyperedges = set()
    for batch in candidate_batches:
        true_global_ids = resolve_true_global_ids(golden, batch, perm_to_global)
        cand_hyperedges |= hyperedges_from_incidence_hyperedges(batch, true_global_ids)
    
    m = compute_hyperedge_coverage_metrics(gold_hyperedges, cand_hyperedges)
    summary = summarize_hyperedge_differences(gold_hyperedges, cand_hyperedges)

    # Read batch size safely
    print(
        f"[bs={bs}, num_parts={num_parts}] "
        f"strict_recall={m['strict_recall']:.4f}, partial_recall={m['partial_recall']:.4f}"
    )
    print(summary)

    return m


# 6. SWEEP OVER (batch_size, num_parts)
def run_sweep():
    """Sweep over batch size and number of partitions and collect metrics.

    Returns
    -------
    list of int
        Tested batch sizes.
    list of int
        Tested numbers of partitions.
    dict
        Mapping ``(batch_size, num_parts) -> metrics dict`` from
        :func:`compute_hyperedge_coverage_metrics`.
    """
    dataset_name = "graph/cocitation_cora_cluster"  # adapt as needed (string Hydra expects)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_parts_list = [2, 4, 8, 16, 32, 64, 128, 256]

    results = {}

    with hydra.initialize(config_path="../../configs", job_name="topo_debug_sweep"):
        for num_parts in num_parts_list:
            for bs in batch_sizes:
                # clear cached splits each run
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
                    m = run_single_experiment(cfg, num_parts=num_parts, bs = bs)
                    results[(bs, num_parts)] = m
                else:
                    print(f"Run failed for bs={bs}, num_parts={num_parts}")
    return batch_sizes, num_parts_list, results

def build_metric_matrix(batch_sizes, num_parts_list, results, metric_key):
    """Build a dense metric matrix in percent.

    Parameters
    ----------
    batch_sizes : sequence of int
        Batch sizes (row order).
    num_parts_list : sequence of int
        Numbers of partitions (column order).
    results : dict
        Mapping ``(batch_size, num_parts) -> metrics dict``.
    metric_key : {"strict_recall", "partial_recall"}
        Metric key to extract and scale to percent.

    Returns
    -------
    ndarray
        Matrix of metric values (%) with NaNs for missing entries.
    """
    B = len(batch_sizes)
    P = len(num_parts_list)
    M = np.full((B, P), np.nan, dtype=float)

    for i, bs in enumerate(batch_sizes):
        for j, nparts in enumerate(num_parts_list):
            m = results.get((bs, nparts))
            if m is None:
                continue
            val = m.get(metric_key)
            if val is None:
                continue
            M[i, j] = 100.0 * float(val)
    return M


def plot_two_heatmaps(batch_sizes, num_parts_list, strict_M, partial_M, save_prefix="coverage"):
    """Plot side-by-side heatmaps for strict and partial recall.

    Both heatmaps share axes and a [0, 100] color scale.

    Parameters
    ----------
    batch_sizes : sequence of int
        Batch sizes (row labels).
    num_parts_list : sequence of int
        Numbers of partitions (column labels).
    strict_M : array-like, shape (B, P)
        Strict recall values in percent.
    partial_M : array-like, shape (B, P)
        Partial recall values in percent.
    save_prefix : str, optional
        Filename prefix for the saved figure, by default "coverage".
    """
    assert strict_M.shape == partial_M.shape
    B, P = strict_M.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    vmin, vmax = 0, 100

    xticks = np.arange(P)
    yticks = np.arange(B)

    # Left: strict
    ax = axes[0]
    im0 = ax.imshow(strict_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(num_parts_list)
    ax.set_yticks(yticks)
    ax.set_yticklabels(batch_sizes)
    ax.set_xlabel("num_parts")
    ax.set_ylabel("batch_size")
    ax.set_title("Strict recall (exact match) %")

    # Annotate (skip NaNs)
    for (i, j), val in np.ndenumerate(strict_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)

    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.02)
    cbar0.set_label("Strict recall (%)")

    # Right: partial
    ax = axes[1]
    im1 = ax.imshow(partial_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(num_parts_list)
    ax.set_yticks(yticks)
    # only show y tick labels on left plot to avoid duplication, but it's fine either way
    ax.set_yticklabels(batch_sizes)
    ax.set_xlabel("num_parts")
    ax.set_title("Partial recall (exact or fragment) %")

    for (i, j), val in np.ndenumerate(partial_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)

    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.02)
    cbar1.set_label("Partial recall (%)")

    # Save and show
    out_combined = f"sweep_tools/outputs/{save_prefix}_strict_vs_partial.png"
    fig.savefig(out_combined, dpi=200)
    print(f"Saved combined heatmap: {out_combined}")
    # plt.show()



def plot_one_heatmap(batch_sizes, num_parts_list, strict_M, save_prefix="coverage"):
    """Plot a single heatmap for strict recall.

    Parameters
    ----------
    batch_sizes : sequence of int
        Batch sizes (row labels).
    num_parts_list : sequence of int
        Numbers of partitions (column labels).
    strict_M : array-like, shape (B, P)
        Strict recall values in percent. NaNs are not annotated.
    save_prefix : str, optional
        Filename prefix for the saved figure, by default "coverage".

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the heatmap.
    matplotlib.axes.Axes
        Axes object for further customization.
    """
    strict_M = np.asarray(strict_M)
    B, P = strict_M.shape

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    vmin, vmax = 0, 100
    xticks = np.arange(P)
    yticks = np.arange(B)

    im = ax.imshow(strict_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(num_parts_list)
    ax.set_yticks(yticks)
    ax.set_yticklabels(batch_sizes)
    ax.set_xlabel("num_parts")
    ax.set_ylabel("batch_size")
    ax.set_title("Hyperedges: Recall %")

    # Annotate (skip NaNs)
    for (i, j), val in np.ndenumerate(strict_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Recall (%)")

    out_path = f"sweep_tools/outputs/{save_prefix}_strict.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved strict heatmap: {out_path}")
    # plt.show()


    return fig, ax

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    os.makedirs("sweep_tools/outputs", exist_ok=True)

    batch_sizes, num_parts_list, results = run_sweep()
    
    print(results)

    strict_M = build_metric_matrix(batch_sizes, num_parts_list, results, "strict_recall")
    partial_M = build_metric_matrix(batch_sizes, num_parts_list, results, "partial_recall")

    plot_two_heatmaps(batch_sizes, num_parts_list, strict_M, partial_M, save_prefix="cora_hypergraph_final")
    plot_one_heatmap(batch_sizes, num_parts_list, strict_M, save_prefix="cora_hypergraph")

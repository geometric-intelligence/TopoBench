import os
import os.path as osp
import shutil
from collections import defaultdict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.utils import coalesce, remove_self_loops

from topobench.data.preprocessor import PreProcessor
from topobench.data.utils import build_cluster_transform
from topobench.dataloader.dataload_cluster import (
    BlockCSRBatchCollator,
    _HandleAdapter,
    _PartIdListDataset,
)

# 1. BATCH PROCESSOR (REMAP + LIFT)

def process_raw_batch_to_lifted(
    raw_batch,
    perm_to_global,
    post_batch_transform,
    target_n0: int,
    pad_to_global_fn,
):
    """Remap a raw cluster batch to true IDs, pad, clean, and lift.

    Parameters
    ----------
    raw_batch : torch_geometric.data.Data
        Raw cluster-level batch.
    perm_to_global : torch.Tensor or None
        Mapping from permuted IDs to global IDs.
    post_batch_transform : callable or None
        Lifting transform applied after remapping.
    target_n0 : int
        Total number of 0-cells in the global complex.
    pad_to_global_fn : callable
        Function that pads node-aligned tensors to size ``target_n0``.

    Returns
    -------
    torch_geometric.data.Data
        Remapped and (optionally) lifted batch aligned to global nodes.
    """
    # 1) Map local node IDs to true global IDs
    if hasattr(raw_batch, "global_nid"):
        gids_local = raw_batch.global_nid.long()
    else:
        gids_local = torch.arange(raw_batch.num_nodes, device=raw_batch.edge_index.device)

    if perm_to_global is not None:
        true_gids = perm_to_global[gids_local]
    else:
        true_gids = gids_local

    # 2) Remap edges to true global ID space
    remapped_edge_index = true_gids[raw_batch.edge_index]

    # 3) Carry over features and attributes
    batch_dict = raw_batch.to_dict()
    batch_dict["edge_index"] = remapped_edge_index

    # Canonicalize node order for node-aligned tensors
    sorted_gids, sorted_idx = torch.sort(true_gids)

    # Reindex then pad to global size for all node-aligned tensors
    for key, value in list(batch_dict.items()):
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == raw_batch.num_nodes:
            val_sorted = value[sorted_idx]
            batch_dict[key] = pad_to_global_fn(val_sorted, sorted_gids)

    device = remapped_edge_index.device
    full_ids = torch.arange(target_n0, device=device, dtype=torch.long)
    batch_dict["true_global_nid"] = full_ids
    batch_dict["global_nid"] = full_ids
    present_mask_0 = torch.zeros(target_n0, dtype=torch.bool, device=device)
    present_mask_0[sorted_gids] = True
    batch_dict["present_mask_0"] = present_mask_0
    batch_dict["is_true_global_remapped"] = True
    batch_dict["num_nodes"] = target_n0

    remapped_graph = torch_geometric.data.Data.from_dict(batch_dict)

    # Clean edges
    if hasattr(remapped_graph, "edge_index"):
        cleaned_ei, _ = remove_self_loops(remapped_graph.edge_index)
        remapped_graph.edge_index = coalesce(cleaned_ei, num_nodes=remapped_graph.num_nodes)

    # Lifting
    if post_batch_transform is not None:
        lifted_graph = post_batch_transform(remapped_graph)
        lifted_graph.true_global_nid = full_ids
        lifted_graph.global_nid = full_ids
        lifted_graph.present_mask_0 = present_mask_0
        lifted_graph.is_true_global_remapped = True
        return lifted_graph

    return remapped_graph


def build_golden_and_loader_from_cfg_cells(
    cfg: DictConfig,
    num_parts: int = 10,
    batch_size: int = 1,
):
    """Build golden complex and DataLoader for multi-epoch cell experiments.

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
        Golden (full) lifted cell complex.
    loader : torch.utils.data.DataLoader
        Loader over partition IDs producing raw batches.
    handle : dict
        Partition handle with metadata and paths.
    perm_to_global : torch.Tensor or None
        Mapping from permuted IDs to global IDs.
    post_batch_transform : callable or None
        Lifting transform applied after remapping.
    target_n0 : int
        Total number of 0-cells in the global complex.
    pad_to_global_fn : callable
        Function for padding node-aligned tensors to size ``target_n0``.
    """
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    transforms_config = cfg.get("transforms", None)

    golden_pre = PreProcessor(dataset, dataset_dir, transforms_config)
    assert len(golden_pre.data_list) == 1, "Expected a single graph for transductive setting."
    golden = golden_pre.data_list[0]
    print("Golden keys:", sorted(golden.keys()))
    if not hasattr(golden, "x_0"):
        raise RuntimeError("Golden data has no x_0; this script is for cell complexes.")

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
        post_batch_transform=None,  # transform applied after remapping
    )

    loader = DataLoader(
        part_ds,
        batch_size=batch_size,
        shuffle=True,   # shuffle so epochs differ
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )

    perm_to_global = load_perm_to_global(handle)
    if perm_to_global is None:
        print("No perm_to_global.npy found; assuming global_nid are already true IDs.")

    target_n0 = int(golden.num_nodes)

    # Canonicalize golden edges
    if hasattr(golden, "edge_index"):
        cleaned_ei, _ = remove_self_loops(golden.edge_index)
        golden.edge_index = coalesce(cleaned_ei, num_nodes=golden.num_nodes)

    def _pad_to_global(val: torch.Tensor, sorted_gids: torch.Tensor):
        """Pad a node-aligned tensor from a batch to global size."""
        if val.size(0) == target_n0:
            return val
        shape = (target_n0,) + tuple(val.shape[1:])
        if val.dtype == torch.bool:
            out = torch.zeros(shape, dtype=torch.bool, device=val.device)
        else:
            out = torch.zeros(shape, dtype=val.dtype, device=val.device)
        out[sorted_gids] = val
        return out

    return golden, loader, handle, perm_to_global, post_batch_transform, target_n0, _pad_to_global


# 2. PERMUTATION HANDLING

def load_perm_to_global(handle):
    """Load mapping from permuted node IDs to global IDs.

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
    print("Loaded perm_to_global of shape", tuple(perm_to_global.shape))
    return perm_to_global


def resolve_true_global_ids(golden, batch, perm_to_global):
    """Map local 0-cell IDs in a batch to true global 0-cell IDs.

    Parameters
    ----------
    golden : torch_geometric.data.Data
        Golden cell complex with reference 0-cell features.
    batch : torch_geometric.data.Data
        Batch with 0-cell features and optional ID metadata.
    perm_to_global : torch.Tensor or None
        Mapping from permuted IDs to global IDs.

    Returns
    -------
    torch.Tensor
        Long tensor of true global 0-cell IDs for the batch.
    """
    if hasattr(batch, "x_0"):
        n0 = batch.x_0.size(0)
        device = batch.x_0.device
    elif hasattr(batch, "x"):
        n0 = batch.x.size(0)
        device = batch.x.device
    else:
        raise RuntimeError("Batch has no x_0 or x; cannot resolve 0-cell IDs.")

    if getattr(batch, "is_true_global_remapped", False):
        if hasattr(batch, "true_global_nid") and getattr(batch.true_global_nid, "numel", lambda: 0)() == n0:
            return batch.true_global_nid.to(device)
        return torch.arange(n0, device=device, dtype=torch.long)

    if not hasattr(batch, "global_nid"):
        print("Batch has no global_nid; using local indices as global IDs.")
        return torch.arange(n0, device=device)

    gids = batch.global_nid.clone().detach().long()
    true_gids = perm_to_global[gids] if perm_to_global is not None else gids

    if hasattr(golden, "x_0") and hasattr(batch, "x_0") and n0 > 0:
        k = min(50, n0)
        idx = torch.randint(0, n0, (k,))
        try:
            diff = (batch.x_0[idx].cpu() - golden.x_0[true_gids[idx]].cpu()).abs().max()
            print("  sanity max|x_0_batch - x_0_golden[true_gids]| =", float(diff))
        except Exception as e:
            print("  sanity check failed:", e)

    return true_gids.to(device)


# 3. HELPERS FOR INCIDENCE HANDLING

def _build_col_to_rows_mapping(M, n_rows):
    """Build a column-to-row index mapping from an incidence-like matrix.

    Parameters
    ----------
    M : torch.Tensor or torch.sparse or COO-like
        Incidence-like matrix where rows index lower-dimensional cells.
    n_rows : int
        Expected number of rows (lower-dimensional cells).

    Returns
    -------
    dict or None
        Mapping ``col_index -> set(row_indices)`` or ``None`` if shapes mismatch
        or the format is unsupported.
    """
    if M is None:
        return None

    if isinstance(M, torch.Tensor) and M.is_sparse:
        M = M.coalesce()
        if M.ndim != 2 or M.size(0) != n_rows:
            return None
        rows, cols = M.indices()
        mapping = defaultdict(set)
        for i, j in zip(rows.tolist(), cols.tolist(), strict=False):
            mapping[int(j)].add(int(i))
        return mapping

    if hasattr(M, "coo"):
        row, col, _ = M.coo()
        if hasattr(M, "sparse_sizes"):
            sizes = M.sparse_sizes()
            if sizes[0] != n_rows:
                return None
        mapping = defaultdict(set)
        for i, j in zip(row.tolist(), col.tolist(), strict=False):
            mapping[int(j)].add(int(i))
        return mapping

    if isinstance(M, torch.Tensor):
        if M.ndim != 2 or M.size(0) != n_rows:
            return None
        rows, cols = (M != 0).nonzero(as_tuple=True)
        mapping = defaultdict(set)
        for i, j in zip(rows.tolist(), cols.tolist(), strict=False):
            mapping[int(j)].add(int(i))
        return mapping

    return None


# 4. 1-CELLS & 2-CELLS AS SETS OF 0-CELLS

def extract_1_cells(data, true_global_ids: torch.Tensor):
    """Extract 1-cells as sets of global 0-cell IDs.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Cell complex data with ``x_0`` and ``incidence_1``.
    true_global_ids : torch.Tensor
        Global 0-cell IDs for the 0-cell axis.

    Returns
    -------
    list of frozenset
        Per-1-cell list of 0-cell sets (may contain empty sets).
    set of frozenset
        Set of non-empty 1-cells as unique 0-cell sets.
    """
    if true_global_ids is None:
        return [], set()

    gids = true_global_ids.cpu()

    if not hasattr(data, "x_0"):
        return [], set()
    n0 = data.x_0.size(0)

    M = getattr(data, "incidence_1", None)
    if M is None:
        return [], set()

    mapping = _build_col_to_rows_mapping(M, n0)
    if mapping is None:
        return [], set()

    n1 = max(mapping.keys()) + 1 if mapping else 0
    cells_1_list = []
    cells_1_set = set()

    for j in range(n1):
        rows = mapping.get(j, set())
        if not rows:
            cells_1_list.append(frozenset())
            continue
        vs = frozenset(int(gids[i]) for i in rows)
        cells_1_list.append(vs)
        if vs:
            cells_1_set.add(vs)

    return cells_1_list, cells_1_set


def extract_2_cells(data,
                    true_global_ids: torch.Tensor,
                    cells_1_list):
    """Extract 2-cells as sets of global 0-cell IDs.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Cell complex data with ``x_0`` and ``incidence_2``.
    true_global_ids : torch.Tensor
        Global 0-cell IDs for the 0-cell axis.
    cells_1_list : list of frozenset
        Per-1-cell list of 0-cell sets, used when 2-cells are defined over 1-cells.

    Returns
    -------
    set of frozenset
        Set of 2-cells, each as a 0-cell frozenset.
    """
    if true_global_ids is None:
        return set()

    gids = true_global_ids.cpu()

    if not hasattr(data, "x_0"):
        return set()
    n0 = data.x_0.size(0)

    M = getattr(data, "incidence_2", None)
    if M is None:
        return set()

    # Case 1: 2-cells defined directly over 0-cells
    mapping_0_2 = _build_col_to_rows_mapping(M, n0)
    if mapping_0_2 is not None:
        cells_2 = set()
        for rows in mapping_0_2.values():
            if not rows:
                continue
            vs = frozenset(int(gids[i]) for i in rows)
            if vs:
                cells_2.add(vs)
        return cells_2

    # Case 2: 2-cells defined over 1-cells, then expanded to 0-cells
    n1 = len(cells_1_list)
    if n1 > 0:
        mapping_1_2 = _build_col_to_rows_mapping(M, n1)
        if mapping_1_2 is not None:
            cells_2 = set()
            for one_cells in mapping_1_2.values():
                vs = set()
                for oc in one_cells:
                    if 0 <= oc < n1:
                        vs |= set(cells_1_list[oc])
                if vs:
                    cells_2.add(frozenset(vs))
            return cells_2

    return set()


# 5. METRICS (STRICT + PARTIAL)

def compute_structure_coverage_metrics(
    gold_structs,
    cand_structs,
    *,
    mode: str = "subset",
    min_subset_size: int = 1,
    union_frac: float | None = None,
    jaccard_thresh: float = 0.5,
):
    """Compute strict and partial coverage metrics for cell-like structures.

    Parameters
    ----------
    gold_structs : iterable of hashable
        Ground-truth structures, typically frozensets of node IDs.
    cand_structs : iterable of hashable
        Candidate structures from the clustered pipeline.
    mode : {"subset"}, optional
        Partial coverage mode, currently only "subset", by default "subset".
    min_subset_size : int, optional
        Minimum size for candidate subsets when ``mode="subset"``, by default 1.
    union_frac : float or None, optional
        Minimum fraction of a golden structure covered by the union of subsets.
        If ``None``, any subset presence counts as partial, by default None.
    jaccard_thresh : float, optional
        Unused placeholder for future Jaccard-based modes, by default 0.5.

    Returns
    -------
    dict
        Dictionary with keys:
        ``gold_n``, ``cand_n``, ``strict_match``, ``partial_covered``,
        ``strict_recall``, ``partial_recall``.
    """
    gold_fs = set(gold_structs)
    cand_fs = set(cand_structs)

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

    partially_covered_gold = set(common)

    if mode == "subset":
        for g in missing:
            g_set = set(g)
            subs = [set(c) for c in new if len(c) >= min_subset_size and set(c) < g_set]
            if not subs:
                continue
            if union_frac is None:
                partially_covered_gold.add(g)
            else:
                union_cov = set().union(*subs)
                if len(union_cov) / max(1, len(g_set)) >= union_frac:
                    partially_covered_gold.add(g)

    partial_covered = len(partially_covered_gold)

    return {
        "gold_n": G,
        "cand_n": len(cand_fs),
        "strict_match": strict_match,
        "partial_covered": partial_covered,
        "strict_recall": strict_match / G,
        "partial_recall": partial_covered / G,
    }


# 6. MULTI-EPOCH EXPERIMENT FOR CELLS

def run_single_experiment_multi_epoch_cells(
    cfg: DictConfig,
    num_parts: int = 10,
    bs: int = 1,
    num_epochs: int = 5,
    accumulate: bool = True,
):
    """Run a multi-epoch coverage experiment for 1- and 2-cells.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for dataset and model.
    num_parts : int, optional
        Number of graph partitions, by default 10.
    bs : int, optional
        Batch size, by default 1.
    num_epochs : int, optional
        Number of passes over the partition loader, by default 5.
    accumulate : bool, optional
        If True, coverage accumulates across epochs; otherwise each epoch
        is evaluated independently, by default True.

    Returns
    -------
    dict
        Mapping ``{1: [metrics_epoch1, ...], 2: [metrics_epoch1, ...]}``.
        The list for dimension 2 may be empty if no 2-cells exist.
    """
    (
        golden,
        loader,
        handle,
        perm_to_global,
        post_batch_transform,
        target_n0,
        pad_to_global_fn,
    ) = build_golden_and_loader_from_cfg_cells(cfg, num_parts=num_parts, batch_size=bs)

    requested_num_parts = int(num_parts)
    actual_num_parts = int(handle.get("num_parts", -1))
    if actual_num_parts != requested_num_parts:
        raise RuntimeError(
            f"Partitioner produced {actual_num_parts} parts but requested {requested_num_parts}"
        )

    # Golden cells
    N0 = golden.x_0.size(0)
    base_ids = torch.arange(N0)
    gold_1_list, gold_1_set = extract_1_cells(golden, base_ids)
    gold_2_set = extract_2_cells(golden, base_ids, gold_1_list)
    print(f"Golden: 1-cells={len(gold_1_set)}, 2-cells={len(gold_2_set)}")

    metrics_dim1 = []
    metrics_dim2 = []

    cumulative_1 = set()
    cumulative_2 = set()

    for epoch in range(1, num_epochs + 1):
        cand_1_epoch = set()
        cand_2_epoch = set()

        for raw_batch in loader:
            lifted_graph = process_raw_batch_to_lifted(
                raw_batch,
                perm_to_global=perm_to_global,
                post_batch_transform=post_batch_transform,
                target_n0=target_n0,
                pad_to_global_fn=pad_to_global_fn,
            )
            true_ids = resolve_true_global_ids(golden, lifted_graph, perm_to_global=None)

            b1_list, b1_set = extract_1_cells(lifted_graph, true_ids)
            b2_set = extract_2_cells(lifted_graph, true_ids, b1_list)

            cand_1_epoch |= b1_set
            cand_2_epoch |= b2_set

        if accumulate:
            cumulative_1 |= cand_1_epoch
            cumulative_2 |= cand_2_epoch
            cand_1 = cumulative_1
            cand_2 = cumulative_2
        else:
            cand_1 = cand_1_epoch
            cand_2 = cand_2_epoch

        m1 = compute_structure_coverage_metrics(gold_1_set, cand_1)
        metrics_dim1.append(m1)
        print(
            f"[dim=1, epoch={epoch}, bs={bs}, num_parts={num_parts}] "
            f"strict={m1['strict_recall']:.4f}, partial={m1['partial_recall']:.4f}"
        )

        if len(gold_2_set) > 0:
            m2 = compute_structure_coverage_metrics(gold_2_set, cand_2)
            metrics_dim2.append(m2)
            print(
                f"[dim=2, epoch={epoch}, bs={bs}, num_parts={num_parts}] "
                f"strict={m2['strict_recall']:.4f}, partial={m2['partial_recall']:.4f}"
            )

    results_by_dim_epochs = {1: metrics_dim1}
    if len(metrics_dim2) > 0:
        results_by_dim_epochs[2] = metrics_dim2
    else:
        results_by_dim_epochs[2] = []

    return results_by_dim_epochs


# 7. SWEEP (MULTI-EPOCH CELLS)

def run_sweep_cells_multi_epoch(
    num_epochs: int = 10,
    accumulate: bool = True,
):
    """Sweep over (batch_size, num_parts) for multi-epoch cell coverage.

    Parameters
    ----------
    num_epochs : int, optional
        Number of epochs per configuration, by default 10.
    accumulate : bool, optional
        If True, coverage accumulates across epochs; otherwise per-epoch
        coverage is independent, by default True.

    Returns
    -------
    list of int
        Tested batch sizes.
    list of int
        Tested numbers of partitions.
    dict
        Nested results mapping
        ``results_by_dim_epochs[dim][(bs, num_parts)] = [metrics_epoch1, ...]``.
    """
    dataset_name = "graph/cocitation_cora_cluster"
    splits_dir = osp.join("datasets", "data_splits", "Cora")

    batch_sizes = [1, 2, 4, 8, 16, 32]
    num_parts_list = [2, 4, 8, 16, 32]

    results_by_dim_epochs = {1: {}, 2: {}}

    with hydra.initialize(config_path="../../configs", job_name="topo_cells_epochs"):
        for num_parts in num_parts_list:
            for bs in batch_sizes:
                if bs > num_parts:
                    print(f"Skip bs={bs}, num_parts={num_parts} (bs > num_parts)")
                    continue

                if osp.exists(splits_dir):
                    shutil.rmtree(splits_dir)

                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        f"dataset={dataset_name}",
                        "model=cell/topotune",
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

                print(f"\n=== EPOCH SWEEP bs={bs}, num_parts={num_parts} ===")
                m_by_dim_epochs = run_single_experiment_multi_epoch_cells(
                    cfg,
                    num_parts=num_parts,
                    bs=bs,
                    num_epochs=num_epochs,
                    accumulate=accumulate,
                )

                for dim in [1, 2]:
                    if dim not in results_by_dim_epochs:
                        results_by_dim_epochs[dim] = {}
                    results_by_dim_epochs[dim][(bs, num_parts)] = m_by_dim_epochs.get(dim, [])

    return batch_sizes, num_parts_list, results_by_dim_epochs


# 8. MATRIX HELPERS + 4-PANEL PLOTTING PER DIM

def build_metric_matrix_for_epoch(batch_sizes, num_parts_list, results_per_pair, epoch, metric_key):
    """Build a metric matrix for a specific epoch across (bs, num_parts) pairs.

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
        Metric key (e.g., ``"strict_recall"`` or ``"partial_recall"``).

    Returns
    -------
    ndarray
        Matrix of metric values in percent, with NaNs for missing entries.
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
            M[i, j] = 100.0 * float(val)
    return M


def plot_summary_4panel_dim(
    batch_sizes,
    num_parts_list,
    results_per_pair_dim,
    num_epochs: int,
    metric_key: str,
    target_bs: int,
    target_num_parts: int,
    save_prefix: str,
    dim_label: str,
):
    """Plot a 4-panel summary for one cell dimension.

    The panels show:
      (0, 0) Epoch 1 heatmap.
      (0, 1) Last epoch heatmap.
      (1, 0) Ratio last/first.
      (1, 1) Recall vs. epochs for a chosen (bs, num_parts).

    Parameters
    ----------
    batch_sizes : sequence of int
        Batch sizes (row labels).
    num_parts_list : sequence of int
        Numbers of partitions (column labels).
    results_per_pair_dim : dict
        Mapping ``(bs, num_parts) -> [metrics_epoch1, ...]`` for this dimension.
    num_epochs : int
        Number of epochs to visualize.
    metric_key : str
        Metric key to plot (e.g., ``"strict_recall"``).
    target_bs : int
        Batch size for the recall-vs-epochs curve.
    target_num_parts : int
        Number of partitions for the recall-vs-epochs curve.
    save_prefix : str
        Filename prefix for the saved figure.
    dim_label : str
        Human-readable label for the dimension (e.g., "Dim 1 (1-cells)").
    """
    first_M = build_metric_matrix_for_epoch(
        batch_sizes, num_parts_list, results_per_pair_dim, epoch=1, metric_key=metric_key
    )
    last_M = build_metric_matrix_for_epoch(
        batch_sizes, num_parts_list, results_per_pair_dim, epoch=num_epochs, metric_key=metric_key
    )

    ratio_M = np.full_like(first_M, np.nan, dtype=float)
    for (i, j), first_val in np.ndenumerate(first_M):
        last_val = last_M[i, j]
        if np.isnan(first_val) or np.isnan(last_val) or first_val <= 0:
            continue
        ratio_M[i, j] = last_val / first_val

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax_first, ax_last, ax_ratio, ax_curve = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    vmin, vmax = 0, 100
    xticks = np.arange(len(num_parts_list))
    yticks = np.arange(len(batch_sizes))

    # Epoch 1 heatmap
    im_first = ax_first.imshow(first_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax_first.set_xticks(xticks)
    ax_first.set_yticks(yticks)
    ax_first.set_xticklabels(num_parts_list)
    ax_first.set_yticklabels(batch_sizes)
    ax_first.set_xlabel("num_parts")
    ax_first.set_ylabel("batch_size")
    ax_first.set_title(f"{dim_label} – Recall % (epoch 1)")

    for (i, j), val in np.ndenumerate(first_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax_first.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)

    # Last epoch heatmap
    ax_last.imshow(last_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax_last.set_xticks(xticks)
    ax_last.set_yticks(yticks)
    ax_last.set_xticklabels(num_parts_list)
    ax_last.set_yticklabels([])
    ax_last.set_xlabel("num_parts")
    ax_last.set_title(f"{dim_label} – Recall % (epoch {num_epochs})")

    for (i, j), val in np.ndenumerate(last_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax_last.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)

    cbar_top = fig.colorbar(im_first, ax=[ax_first, ax_last], fraction=0.035, pad=0.02)
    label = "Strict recall (%)" if metric_key == "strict_recall" else "Partial recall (%)"
    cbar_top.set_label(label)

    # Ratio heatmap
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
    ax_ratio.set_title(f"{dim_label} – Final / initial recall (×)")

    for (i, j), val in np.ndenumerate(ratio_M):
        if np.isnan(val):
            continue
        color = "white" if val < 1.0 else "black"
        ax_ratio.text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=7, color=color)

    cbar_ratio = fig.colorbar(im_ratio, ax=ax_ratio, fraction=0.035, pad=0.02)
    cbar_ratio.set_label("Final / initial")

    # Curve for specific (bs, num_parts)
    metrics_list = results_per_pair_dim.get((target_bs, target_num_parts))
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
        ax_curve.set_title(
            f"{dim_label} – Recall vs epochs\n(bs={target_bs}, num_parts={target_num_parts})"
        )
        ax_curve.grid(alpha=0.3)
        ax_curve.set_ylim(0, 105)

    out_path = f"sweep_tools/outputs/{save_prefix}_{metric_key}.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved 4-panel summary for {dim_label}: {out_path}")
    # plt.show()



# 9. MAIN

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    
    os.makedirs("sweep_tools/outputs", exist_ok=True)
    
    NUM_EPOCHS = 10
    TARGET_BS = 8
    TARGET_NUM_PARTS = 32

    batch_sizes, num_parts_list, results_by_dim_epochs = run_sweep_cells_multi_epoch(
        num_epochs=NUM_EPOCHS,
        accumulate=True,
    )

    # 4-panel for 1-cells
    results_dim1 = results_by_dim_epochs[1]
    plot_summary_4panel_dim(
        batch_sizes,
        num_parts_list,
        results_dim1,
        num_epochs=NUM_EPOCHS,
        metric_key="strict_recall",
        target_bs=TARGET_BS,
        target_num_parts=TARGET_NUM_PARTS,
        save_prefix="cells_dim1_summary",
        dim_label="Dim 1 (1-cells)",
    )

    # 4-panel for 2-cells, if any 2-cells exist
    if any(len(v) > 0 for v in results_by_dim_epochs[2].values()):
        results_dim2 = results_by_dim_epochs[2]
        plot_summary_4panel_dim(
            batch_sizes,
            num_parts_list,
            results_dim2,
            num_epochs=NUM_EPOCHS,
            metric_key="strict_recall",
            target_bs=TARGET_BS,
            target_num_parts=TARGET_NUM_PARTS,
            save_prefix="cells_dim2_summary",
            dim_label="Dim 2 (2-cells)",
        )

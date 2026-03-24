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


# 1. BUILD GOLDEN & CANDIDATE (CELLS)
def build_golden_and_candidate_from_cfg(cfg: DictConfig,
                                        num_parts: int = 10,
                                        batch_size: int = 1):
    """Build golden and candidate cell-complex graphs from config.

    The golden graph is obtained by global lifting on the full dataset.
    Candidate graphs are built by partitioning, remapping to true node
    IDs, padding node-aligned tensors to global size, and applying the
    same lifting per cluster.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with dataset and transform settings.
    num_parts : int, optional
        Number of graph partitions for the candidate setup.
    batch_size : int, optional
        Number of clusters per candidate mini-batch.

    Returns
    -------
    tuple
        ``(golden, candidate_batches, handle)``, where ``golden`` is a
        single lifted graph, ``candidate_batches`` is a list of lifted
        candidate graphs, and ``handle`` is the packed partition handle.
    """
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    transforms_config = cfg.get("transforms", None)

    # Golden: global lifting
    golden_pre = PreProcessor(dataset, dataset_dir, transforms_config)
    assert len(golden_pre.data_list) == 1, "Expected a single graph for transductive setting."
    golden = golden_pre.data_list[0]
    print("Golden keys:", sorted(golden.keys()))

    if not hasattr(golden, "x_0"):
        raise RuntimeError("Golden data has no x_0; this script is for cell complexes.")

    # Candidate: raw, then partition + remap + cluster lifting
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

    # IMPORTANT: do NOT apply the transform in the collator; we must remap first.
    collate = BlockCSRBatchCollator(
        adapter,
        device=None,
        with_edge_attr=handle.get("has_edge_attr", False),
        active_split=active_split,
        post_batch_transform=None,
    )

    loader = DataLoader(
        part_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )

    # Load permutation to global for remapping candidate batches
    perm_to_global = load_perm_to_global(handle)
    if perm_to_global is None:
        print("No perm_to_global.npy found; assuming global_nid are already true IDs.")

    target_n0 = int(golden.num_nodes)

    def _pad_to_global(val: torch.Tensor, sorted_gids: torch.Tensor):
        """Pad node-aligned tensor to global length.

        Parameters
        ----------
        val : torch.Tensor
            Node-aligned tensor of shape ``(local_n, ...)``.
        sorted_gids : torch.Tensor
            Sorted global IDs for the local nodes.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(target_n0, ...)`` with values placed at
            global positions and zeros elsewhere.
        """
        if val.size(0) == target_n0:
            return val  # already global-sized
        shape = (target_n0,) + tuple(val.shape[1:])
        if val.dtype == torch.bool:
            out = torch.zeros(shape, dtype=torch.bool, device=val.device)
        else:
            out = torch.zeros(shape, dtype=val.dtype, device=val.device)
        out[sorted_gids] = val
        return out

    candidate_batches = []
    print("\n--- Loading, remapping, and transforming candidate batches ---")
    for _i, raw_batch in enumerate(loader):
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

        # Reindex then PAD to global size for all node-aligned tensors
        for key, value in list(batch_dict.items()):
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == raw_batch.num_nodes:
                val_sorted = value[sorted_idx]
                batch_dict[key] = _pad_to_global(val_sorted, sorted_gids)

        # Provide identity global mapping over the full domain (x_0 is now global-sized)
        device = remapped_edge_index.device
        full_ids = torch.arange(target_n0, device=device, dtype=torch.long)
        batch_dict["true_global_nid"] = full_ids
        batch_dict["global_nid"] = full_ids
        # Optional: remember which global nodes this part actually touched
        present_mask_0 = torch.zeros(target_n0, dtype=torch.bool, device=device)
        present_mask_0[sorted_gids] = True
        batch_dict["present_mask_0"] = present_mask_0
        batch_dict["is_true_global_remapped"] = True

        # Keep PyG happy: set num_nodes to the global size
        batch_dict["num_nodes"] = target_n0

        remapped_graph = torch_geometric.data.Data.from_dict(batch_dict)

        # 5) Clean edges (self-loops) and coalesce
        if hasattr(remapped_graph, "edge_index"):
            cleaned_ei, _ = remove_self_loops(remapped_graph.edge_index)
            remapped_graph.edge_index = coalesce(cleaned_ei, num_nodes=remapped_graph.num_nodes)

        # 6) Finally apply the (cluster) lifting after remapping
        if post_batch_transform is not None:
            lifted_graph = post_batch_transform(remapped_graph)
            # Preserve helper attributes on lifted graph
            lifted_graph.true_global_nid = full_ids
            lifted_graph.global_nid = full_ids
            lifted_graph.present_mask_0 = present_mask_0
            lifted_graph.is_true_global_remapped = True
            candidate_batches.append(lifted_graph)
        else:
            candidate_batches.append(remapped_graph)

    print("--- Remapping and transformation complete ---\n")

    print("Num candidate batches:", len(candidate_batches))
    if candidate_batches:
        b0 = candidate_batches[0]
        print("First candidate keys:", sorted(b0.keys()))
        if hasattr(b0, "global_nid"):
            print("First candidate global_nid shape:", tuple(b0.global_nid.shape))

    # Canonicalize golden edges too (mirror debug hygiene)
    if hasattr(golden, "edge_index"):
        cleaned_ei, _ = remove_self_loops(golden.edge_index)
        golden.edge_index = coalesce(cleaned_ei, num_nodes=golden.num_nodes)

    return golden, candidate_batches, handle


# 2. PERMUTATION HANDLING
def load_perm_to_global(handle):
    """Load permuted-to-global node ID mapping from disk.

    Parameters
    ----------
    handle : dict
        Partition handle containing ``processed_dir``.

    Returns
    -------
    torch.Tensor or None
        Tensor of shape ``(N,)`` mapping permuted IDs to global IDs, or
        ``None`` if the mapping file is missing.
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
    """Resolve 0-cell global IDs for a candidate batch.

    Uses cached identity mappings when present, otherwise maps from
    permuted IDs via ``perm_to_global`` or falls back to local indices.

    Parameters
    ----------
    golden : Data
        Golden reference graph with global 0-cells.
    batch : Data
        Candidate batch graph.
    perm_to_global : torch.Tensor or None
        Mapping from permuted to global IDs, or ``None``.

    Returns
    -------
    torch.Tensor
        Tensor of true global 0-cell IDs for the batch.
    """
    if hasattr(batch, "x_0"):
        n0 = batch.x_0.size(0)
        device = batch.x_0.device
    elif hasattr(batch, "x"):
        n0 = batch.x.size(0)
        device = batch.x.device
    else:
        raise RuntimeError("Batch has no x_0 or x; cannot resolve 0-cell IDs.")

    # If we've padded to global size, use identity mapping
    if getattr(batch, "is_true_global_remapped", False):
        # true_global_nid may be full identity already; if not, identity works.
        if hasattr(batch, "true_global_nid") and getattr(batch.true_global_nid, "numel", lambda: 0)() == n0:
            return batch.true_global_nid.to(device)
        return torch.arange(n0, device=device, dtype=torch.long)

    # Legacy path (not padded)
    if not hasattr(batch, "global_nid"):
        print("Batch has no global_nid; using local indices as global IDs.")
        return torch.arange(n0, device=device)

    gids = batch.global_nid.clone().detach().long()
    true_gids = perm_to_global[gids] if perm_to_global is not None else gids

    # Optional sanity check on x_0
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
    """Build column-to-row index mapping from an incidence-like matrix.

    Supports dense tensors, sparse COO tensors, and objects exposing a
    ``.coo()`` method.

    Parameters
    ----------
    M : Any
        Incidence-like matrix or sparse structure.
    n_rows : int
        Expected number of rows.

    Returns
    -------
    dict or None
        Mapping ``col_index -> set(row_indices)`` or ``None`` if the
        shape does not match or the type is unsupported.
    """
    if M is None:
        return None

    # torch.sparse_coo_tensor
    if isinstance(M, torch.Tensor) and M.is_sparse:
        M = M.coalesce()
        if M.ndim != 2 or M.size(0) != n_rows:
            return None
        rows, cols = M.indices()
        mapping = defaultdict(set)
        for i, j in zip(rows.tolist(), cols.tolist(), strict=False):
            mapping[int(j)].add(int(i))
        return mapping

    # torch_sparse.SparseTensor
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

    # Dense tensor
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
    """Extract 1-cells as sets of 0-cell global IDs.

    Assumes ``incidence_1`` represents 0→1 incidence with shape
    ``(n0, n1)``, where each column is a 1-cell.

    Parameters
    ----------
    data : Data
        Cell-complex graph with ``x_0`` and ``incidence_1``.
    true_global_ids : torch.Tensor
        Global 0-cell IDs of shape ``(n0,)``.

    Returns
    -------
    tuple
        ``(cells_1_list, cells_1_set)``, where ``cells_1_list`` is a
        list of frozensets for each 1-cell (possibly empty) and
        ``cells_1_set`` is the set of non-empty 1-cells.
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
    """Extract 2-cells as sets of 0-cell global IDs.

    Two interpretations of ``incidence_2`` are supported:

    * Case A: shape ``(n0, n2)`` (0→2 incidence).
    * Case B: shape ``(n1, n2)`` (1→2 incidence), where each 2-cell is
      the union of its incident 1-cells.

    Parameters
    ----------
    data : Data
        Cell-complex graph with ``x_0`` and optionally ``incidence_2``.
    true_global_ids : torch.Tensor
        Global 0-cell IDs of shape ``(n0,)``.
    cells_1_list : list of frozenset
        List of 1-cells as frozensets of 0-cell IDs.

    Returns
    -------
    set of frozenset
        Set of 2-cells represented as frozensets of 0-cell IDs.
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

    # Try case (A): 0 -> 2
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

    # Try case (B): 1 -> 2 (use 1-cells' vertex sets)
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

    # If neither interpretation worked, return empty
    return set()


# 5. METRICS (STRICT + PARTIAL)
def compute_structure_coverage_metrics(
    gold_structs,
    cand_structs,
    *,
    # Partial coverage options:
    mode: str = "subset",          # "subset" or "jaccard"
    min_subset_size: int = 1,      # ignore tiny fragments
    union_frac: float | None = None,  # if set (e.g. 0.5), require union of candidate subsets
    jaccard_thresh: float = 0.5    # used when mode="jaccard"
):
    """Compute strict and partial coverage metrics between structures.

    Structures are sets of 0-cell IDs represented as frozensets. Strict
    matches require exact equality. Partial matches depend on the
    chosen mode:

    * ``"subset"``: candidate strict subsets above a size threshold,
      optionally requiring their union to cover a fraction of gold.
    * ``"jaccard"``: maximum Jaccard similarity above a threshold.

    Parameters
    ----------
    gold_structs : iterable of frozenset
        Ground truth structures.
    cand_structs : iterable of frozenset
        Candidate structures.
    mode : {"subset", "jaccard"}, optional
        Partial coverage criterion. Default is "subset".
    min_subset_size : int, optional
        Minimum size for subset fragments in subset mode.
    union_frac : float or None, optional
        Required coverage fraction of the union of fragments in subset
        mode. If ``None``, any qualifying fragment suffices.
    jaccard_thresh : float, optional
        Minimum Jaccard similarity in jaccard mode.

    Returns
    -------
    dict
        Dictionary with keys ``"gold_n"``, ``"cand_n"``,
        ``"strict_match"``, ``"partial_covered"``, ``"strict_recall"``,
        and ``"partial_recall"``.
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
            # all candidate fragments that are strict subsets of g and large enough
            subs = [set(c) for c in new if len(c) >= min_subset_size and set(c) < g_set]
            if not subs:
                continue
            if union_frac is None:
                # any single subset fragment is enough
                partially_covered_gold.add(g)
            else:
                # require that the union covers a fraction of g
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

# 6. SINGLE EXPERIMENT (1-CELLS & 2-CELLS)
def run_single_experiment(cfg: DictConfig,
                          num_parts: int = 10,
                          bs: int = 1):
    """Run a single cell-coverage experiment for given partition settings.

    Builds golden and candidate cell complexes, aggregates candidate
    1-cells and 2-cells, and computes structural coverage metrics per
    dimension.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    num_parts : int, optional
        Number of graph partitions. Default is 10.
    bs : int, optional
        Batch size (number of clusters per step). Default is 1.

    Returns
    -------
    dict
        Mapping from cell dimension (1 or 2) to metric dictionaries.
    """
    golden, candidate_batches, handle = build_golden_and_candidate_from_cfg(
        cfg, num_parts=num_parts, batch_size=bs
    )

    requested_num_parts = int(num_parts)
    actual_num_parts = int(handle.get("num_parts", -1))
    if actual_num_parts != requested_num_parts:
        raise RuntimeError(
            f"Partitioner produced {actual_num_parts} parts but requested {requested_num_parts}"
        )

    if not candidate_batches:
        print("No candidate batches; abort.")
        return None

    perm_to_global = load_perm_to_global(handle)

    # Golden 0-cells
    N0 = golden.x_0.size(0)
    base_ids = torch.arange(N0)

    # Golden 1-cells & 2-cells
    gold_1_list, gold_1_set = extract_1_cells(golden, base_ids)
    gold_2_set = extract_2_cells(golden, base_ids, gold_1_list)

    print(f"Golden: 1-cells={len(gold_1_set)}, 2-cells={len(gold_2_set)}")

    # Candidate aggregated
    cand_1_set = set()
    cand_2_set = set()

    for bi, batch in enumerate(candidate_batches):
        print(f"Processing candidate batch {bi}...")
        true_ids = resolve_true_global_ids(golden, batch, perm_to_global)
        b1_list, b1_set = extract_1_cells(batch, true_ids)
        b2_set = extract_2_cells(batch, true_ids, b1_list)

        cand_1_set |= b1_set
        cand_2_set |= b2_set

    results_by_dim = {}

    # 1-cells metrics
    m1 = compute_structure_coverage_metrics(gold_1_set, cand_1_set)
    print(
        f"[dim=1, bs={bs}, num_parts={num_parts}] "
        f"strict={m1['strict_recall']:.4f}, partial={m1['partial_recall']:.4f}, "
        f"gold_n={m1['gold_n']}, cand_n={m1['cand_n']}"
    )
    results_by_dim[1] = m1

    # 2-cells metrics (only if there are golden 2-cells)
    if len(gold_2_set) > 0:
        m2 = compute_structure_coverage_metrics(gold_2_set, cand_2_set)
        print(
            f"[dim=2, bs={bs}, num_parts={num_parts}] "
            f"strict={m2['strict_recall']:.4f}, partial={m2['partial_recall']:.4f}, "
            f"gold_n={m2['gold_n']}, cand_n={m2['cand_n']}"
        )
        results_by_dim[2] = m2

    return results_by_dim


# 7. SWEEP (STORE PER DIM)
def run_sweep():
    """Run a sweep over batch sizes and num_parts and collect metrics.

    Uses Hydra to instantiate configurations for each combination of
    batch size and number of parts, then runs a single experiment per
    combination and stores results per cell dimension.

    Returns
    -------
    tuple
        ``(batch_sizes, num_parts_list, results_by_dim)`` where
        ``results_by_dim[dim][(bs, num_parts)]`` holds metric dicts.
    """
    # Adjust dataset_name / splits_dir if needed
    dataset_name = "graph/cocitation_cora_cluster"
    splits_dir = osp.join("datasets", "data_splits", "Cora")

    batch_sizes = [1, 2, 4, 8, 16, 32]
    num_parts_list = [2, 4, 8, 16, 32]

    # results_by_dim[dim][(bs, num_parts)] = metrics
    results_by_dim = {1: {}, 2: {}}

    with hydra.initialize(config_path="../../configs", job_name="topo_cells_1_2_sweep"):
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

                print(f"\n=== RUN bs={bs}, num_parts={num_parts} ===")
                m_by_dim = run_single_experiment(cfg, num_parts=num_parts, bs=bs)

                for dim, m in m_by_dim.items():
                    if dim in results_by_dim:
                        results_by_dim[dim][(bs, num_parts)] = m

    return batch_sizes, num_parts_list, results_by_dim


# 8. MATRIX BUILDERS + PLOTTING PER DIM
def build_metric_matrix(batch_sizes, num_parts_list, results_dim, metric_key):
    """Build a 2D matrix of a metric over batch size and num_parts grid.

    Parameters
    ----------
    batch_sizes : list of int
        Batch sizes in the sweep.
    num_parts_list : list of int
        Numbers of partitions in the sweep.
    results_dim : dict
        Mapping ``(bs, num_parts) -> metrics`` for a given dimension.
    metric_key : str
        Key of the metric to extract (e.g. ``"strict_recall"``).

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(len(batch_sizes), len(num_parts_list))`` with
        metric values in percent, or NaN where missing.
    """
    B = len(batch_sizes)
    P = len(num_parts_list)
    M = np.full((B, P), np.nan, dtype=float)

    for i, bs in enumerate(batch_sizes):
        for j, nparts in enumerate(num_parts_list):
            m = results_dim.get((bs, nparts))
            if not m:
                continue
            val = m.get(metric_key)
            if val is None:
                continue
            M[i, j] = 100.0 * float(val)
    return M


def plot_two_heatmaps(batch_sizes, num_parts_list,
                      strict_M, partial_M,
                      save_prefix, title_prefix):
    """Plot side-by-side strict and partial recall heatmaps.

    Parameters
    ----------
    batch_sizes : list of int
        Batch sizes used on the y-axis.
    num_parts_list : list of int
        Numbers of partitions used on the x-axis.
    strict_M : numpy.ndarray
        Strict recall matrix in percent.
    partial_M : numpy.ndarray
        Partial recall matrix in percent.
    save_prefix : str
        Prefix for the saved figure filename.
    title_prefix : str
        Title prefix indicating the cell dimension.
    """
    assert strict_M.shape == partial_M.shape
    B, P = strict_M.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    vmin, vmax = 0, 100
    xticks = np.arange(P)
    yticks = np.arange(B)

    # Strict recall
    ax = axes[0]
    im0 = ax.imshow(strict_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(num_parts_list)
    ax.set_yticks(yticks)
    ax.set_yticklabels(batch_sizes)
    ax.set_xlabel("num_parts")
    ax.set_ylabel("batch_size")
    ax.set_title(f"{title_prefix}: Strict recall (%)")
    for (i, j), val in np.ndenumerate(strict_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)
    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.02)
    cbar0.set_label("Strict recall (%)")

    # Partial recall
    ax = axes[1]
    im1 = ax.imshow(partial_M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(num_parts_list)
    ax.set_yticks(yticks)
    ax.set_yticklabels(batch_sizes)
    ax.set_xlabel("num_parts")
    ax.set_title(f"{title_prefix}: Partial recall (%)")
    for (i, j), val in np.ndenumerate(partial_M):
        if np.isnan(val):
            continue
        color = "white" if val < 50 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.02)
    cbar1.set_label("Partial recall (%)")

    out_path = f"sweep_tools/outputs/{save_prefix}_strict_vs_partial.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {title_prefix} heatmaps to: {out_path}")
    # plt.show()


def plot_recall_by_dim(batch_sizes, num_parts_list,
                       M_dim1=None, M_dim2=None,
                       save_prefix="cells_recall",
                       title_prefix="Recall"):
    """Plot strict recall heatmaps for 1-cells and 2-cells.

    If a matrix is ``None`` for a given dimension, that subplot is
    omitted.

    Parameters
    ----------
    batch_sizes : list of int
        Batch sizes used on the y-axis.
    num_parts_list : list of int
        Numbers of partitions used on the x-axis.
    M_dim1 : numpy.ndarray or None, optional
        Strict recall matrix for 1-cells.
    M_dim2 : numpy.ndarray or None, optional
        Strict recall matrix for 2-cells.
    save_prefix : str, optional
        Prefix for the saved figure filename. Default is "cells_recall".
    title_prefix : str, optional
        Overall title prefix. Default is "Recall".
    """
    mats = []

    if M_dim1 is not None:
        mats.append(("Dim 1 (1-cells)", M_dim1))
    if M_dim2 is not None:
        mats.append(("Dim 2 (2-cells)", M_dim2))

    if not mats:
        print("Nothing to plot (no recall matrices provided).")
        return

    ncols = len(mats)
    B, P = mats[0][1].shape  # assume same grid
    vmin, vmax = 0, 100
    xticks = np.arange(P)
    yticks = np.arange(B)

    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 6), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, (subtitle, M) in zip(axes, mats, strict=False):
        im = ax.imshow(M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(xticks)
        ax.set_xticklabels(num_parts_list)
        ax.set_yticks(yticks)
        ax.set_yticklabels(batch_sizes)
        ax.set_xlabel("num_parts")
        ax.set_ylabel("batch_size")
        ax.set_title(f"{subtitle}: Recall (%)")

        for (i, j), val in np.ndenumerate(M):
            if np.isnan(val):
                continue
            color = "white" if val < 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Recall (%)")

    out_path = f"sweep_tools/outputs/{save_prefix}.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {title_prefix} to: {out_path}")
    # plt.show()


if __name__ == "__main__":
    os.makedirs("sweep_tools/outputs", exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    batch_sizes, num_parts_list, results_by_dim = run_sweep()
    print("Results by dim:", results_by_dim[2])

    # 1-cells
    if results_by_dim[1]:
        strict_M1 = build_metric_matrix(batch_sizes, num_parts_list,
                                        results_by_dim[1], "strict_recall")
        partial_M1 = build_metric_matrix(batch_sizes, num_parts_list,
                                          results_by_dim[1], "partial_recall")
        plot_two_heatmaps(batch_sizes, num_parts_list,
                          strict_M1, partial_M1,
                          save_prefix="cells_dim1",
                          title_prefix="Dim 1 (1-cells)")

    # 2-cells (only if present anywhere)
    if results_by_dim[2]:
        strict_M2 = build_metric_matrix(batch_sizes, num_parts_list,
                                        results_by_dim[2], "strict_recall")
        partial_M2 = build_metric_matrix(batch_sizes, num_parts_list,
                                         results_by_dim[2], "partial_recall")
        plot_two_heatmaps(batch_sizes, num_parts_list,
                          strict_M2, partial_M2,
                          save_prefix="cells_dim2",
                          title_prefix="Dim 2 (2-cells)")
        
    plot_recall_by_dim(batch_sizes, num_parts_list,
                   M_dim1=strict_M1,
                   M_dim2=strict_M2,
                   save_prefix="cells_recall",
                   title_prefix="Recall")

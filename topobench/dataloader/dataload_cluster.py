# topobench/dataloader/dataload_cluster.py

import os.path as osp
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from .utils import (
    _HandleAdapter,
    _PartIdListDataset,
)


# Collator: stream CSR blocks + masks
class BlockCSRBatchCollator:
    """Collate Cluster-GCN mini-batches from CSR memmaps.

    Streams cluster blocks from disk (CSR structure, optional features,
    labels and edge attributes) and builds a :class:`Data` batch with
    node-level supervision masks and global node IDs.

    Parameters
    ----------
    ds_like : _HandleAdapter
        Adapter providing access to cluster metadata and memmaps.
    device : torch.device or None, optional
        Device to move the batch to. If ``None``, stays on CPU.
    with_edge_attr : bool, optional
        If True, reads and includes edge attributes. Default is False.
    active_split : {"train", "val", "test"}, optional
        Active split whose supervision mask is used. Default is "train".
    post_batch_transform : callable or None, optional
        Optional transform applied to the assembled batch.
    """
    def __init__(
        self,
        ds_like: _HandleAdapter,
        *,
        device: torch.device | None = None,
        with_edge_attr: bool = False,
        active_split: str = "train",  # "train" | "val" | "test"
        post_batch_transform: Callable[..., Any] | None = None,
    ) -> None:
        self.ds = ds_like
        self.device = device
        self.with_edge_attr = with_edge_attr
        self.active_split = str(active_split).lower()
        assert self.active_split in ("train", "val", "test")
        self.post_batch_transform = post_batch_transform

        mm_dir = osp.join(self.ds.processed_dir, "perm_memmap")
        # Structural memmaps:
        self.partptr = np.load(osp.join(mm_dir, "partptr.npy"), mmap_mode="r")
        self.indptr  = np.load(osp.join(mm_dir, "indptr.npy"),  mmap_mode="r")
        self.indices = np.load(osp.join(mm_dir, "indices.npy"), mmap_mode="r")

        # Optional arrays:
        self.X = None
        self.Y = None
        self.EA = None
        x_path = osp.join(mm_dir, "X_perm.npy")
        y_path = osp.join(mm_dir, "y_perm.npy")
        ea_path = osp.join(mm_dir, "edge_attr_perm.npy")
        if osp.exists(x_path):
            self.X  = np.load(x_path, mmap_mode="r")
        if osp.exists(y_path):
            self.Y  = np.load(y_path, mmap_mode="r")
        if with_edge_attr and osp.exists(ea_path):
            self.EA = np.load(ea_path, mmap_mode="r")

        # Split masks (permuted)
        m_train = osp.join(mm_dir, "train_mask_perm.npy")
        m_val   = osp.join(mm_dir, "val_mask_perm.npy")
        m_test  = osp.join(mm_dir, "test_mask_perm.npy")
        if not (osp.exists(m_train) and osp.exists(m_val) and osp.exists(m_test)):
            raise FileNotFoundError("Permuted split masks not found in memmap dir.")
        self.train_mask_perm = np.load(m_train, mmap_mode="r")
        self.val_mask_perm   = np.load(m_val,   mmap_mode="r")
        self.test_mask_perm  = np.load(m_test,  mmap_mode="r")

        assert self.ds.sparse_format == "csr", f"Expected CSR, got {self.ds.sparse_format}"

    # small helper to choose active split mask array
    def _active_mask_array(self) -> np.ndarray:
        if self.active_split == "train":
            return self.train_mask_perm
        if self.active_split == "val":
            return self.val_mask_perm
        return self.test_mask_perm

    def __call__(self, parts: list[int]) -> Data:
        """Build a union batch from a list of cluster IDs.

        For the given cluster IDs, collects their CSR rows, node
        features, labels and (optionally) edge attributes, then returns
        a single :class:`Data` object.

        Parameters
        ----------
        parts : list of int
            Cluster IDs to merge into a mini-batch (length == ``q``).

        Returns
        -------
        Data
            Batched graph with fields such as ``edge_index``, ``x``,
            ``y``, ``edge_attr``, ``supervised_mask`` and ``global_nid``.
        """
        # ranges for selected clusters (sorted for monotonic slices)
        parts = np.asarray(parts, dtype=np.int64)
        starts = self.partptr[parts]
        ends   = self.partptr[parts + 1]
        order = np.argsort(starts)
        starts, ends = starts[order], ends[order]

        # gather node features/labels and build global_nid list
        offsets = np.zeros(len(starts), dtype=np.int64)
        total_nodes = 0
        xs, ys = [], []
        global_ids = []
        for i, (s, e) in enumerate(zip(starts, ends, strict=False)):
            offsets[i] = total_nodes
            # append features
            if self.X is not None:
                xs.append(torch.from_numpy(self.X[s:e]))
            # append labels
            if self.Y is not None:
                ys.append(torch.from_numpy(self.Y[s:e]))
            # append global permuted ids for these rows
            if e > s:
                global_ids.append(np.arange(s, e, dtype=np.int64))
            total_nodes += (e - s)

        x = torch.cat(xs, dim=0) if xs else None
        y = torch.cat(ys, dim=0) if ys else None
        global_ids = np.concatenate(global_ids, axis=0) if len(global_ids) else np.empty((0,), dtype=np.int64)

        # 3) stream CSR rows for each [s:e) -> make row/col (global ids)
        row_chunks, col_chunks = [], []
        ea_chunks = [] if self.EA is not None else None

        for s, e, off in zip(starts, ends, offsets, strict=False):
            rowptr = self.indptr[s:e+1]                 # shape (e-s+1,)
            deg    = rowptr[1:] - rowptr[:-1]           # per-row degrees
            beg, fin = int(rowptr[0]), int(rowptr[-1])  # contiguous span in indices
            cols = torch.from_numpy(self.indices[beg:fin].astype(np.int64, copy=False))
            rows = torch.arange(e - s, dtype=torch.int64).repeat_interleave(
                torch.from_numpy(deg.astype(np.int64))
            ) + int(off)
            row_chunks.append(rows)
            col_chunks.append(cols)

            if ea_chunks is not None:
                ea_chunks.append(torch.from_numpy(self.EA[beg:fin]))

        row = torch.cat(row_chunks, dim=0) if row_chunks else torch.empty(0, dtype=torch.int64)
        col = torch.cat(col_chunks, dim=0) if col_chunks else torch.empty(0, dtype=torch.int64)
        edge_attr = torch.cat(ea_chunks, dim=0) if ea_chunks else None

        # keep only edges whose dst is inside the union of selected ranges
        starts_t = torch.from_numpy(starts)
        ends_t   = torch.from_numpy(ends)
        offsets_t= torch.from_numpy(offsets)

        idx = torch.bucketize(col, starts_t, right=True) - 1
        valid = (idx >= 0) & (col < ends_t.gather(0, idx.clamp_min(0)))

        row = row[valid]
        col = col[valid]
        idx = idx[valid]
        if edge_attr is not None:
            edge_attr = edge_attr[valid]

        # global->local column ids: col_local = col - starts[idx] + offsets[idx]
        col_local = col - starts_t.gather(0, idx) + offsets_t.gather(0, idx)
        edge_index = torch.stack([row, col_local], dim=0)

        data = Data(edge_index=edge_index)
        if x is not None:
            data.x = x
        if y is not None:
            data.y = y
        if edge_attr is not None:
            data.edge_attr = edge_attr
        data.num_nodes = int(total_nodes)

        # ---- split-specific masks & ids ----
        active_mask = self._active_mask_array()
        supervised_mask = torch.from_numpy(active_mask[global_ids]).to(torch.bool)
        global_nid = torch.from_numpy(global_ids.astype(np.int64, copy=False))
        
        # Apply transforms on the full batch
        if self.post_batch_transform is not None:
            data = self.post_batch_transform(data)

        data.supervised_mask = supervised_mask
        data.global_nid = global_nid

        # Backwards-compatible attributes expected by existing code:
        if self.active_split == "train":
            data.train_mask = supervised_mask
        elif self.active_split == "val":
            data.val_mask = supervised_mask
        elif self.active_split == "test":
            data.test_mask = supervised_mask

        if x is not None:
            data.x_0 = data.x
            data.batch_0 = torch.zeros(data.num_nodes, dtype=torch.long)
    
        # Reproduce collate_fn behavior for batch_k and cell_statistics

        # Ensure batch_0 exists for nodes:
        if hasattr(data, "batch_0"):
            pass  # respect whatever the transform set
        elif hasattr(data, "batch"):
            # Backward-compat: if someone set `batch`, alias it:
            data.batch_0 = data.batch
        else:
            # Single graph: all nodes belong to graph 0
            data.batch_0 = torch.zeros(data.num_nodes, dtype=torch.long)

        # For every x_k (k >= 1) or x_hyperedges, create corresponding batch_k
        # if it doesn't already exist. This mirrors TBDataloader.collate_fn.
        for key in list(data.keys()):
            if key.startswith("x_") and key not in ("x", "x_0"):
                if key == "x_hyperedges":
                    cell_dim = "hyperedges"
                    batch_key = "batch_hyperedges"
                else:
                    try:
                        cell_dim = int(key.split("_")[1])
                    except Exception:
                        continue
                    batch_key = f"batch_{cell_dim}"

                if not hasattr(data, batch_key):
                    num_cells = getattr(data, key).size(0)
                    device = getattr(getattr(data, key), "device", None)
                    setattr(
                        data,
                        batch_key,
                        torch.zeros(num_cells, dtype=torch.long, device=device),
                    )

        # If transform produced `shape`, map it to cell_statistics:
        if hasattr(data, "shape") and not hasattr(data, "cell_statistics"):
            shape = data.shape
            shape = torch.as_tensor(shape, dtype=torch.long)

            # ensure shape is 2D: [1, max_dim+1].
            if shape.dim() == 1:
                shape = shape.unsqueeze(0)

            data.cell_statistics = shape
            delattr(data, "shape")
            
        if self.device is not None:
            data = data.to(self.device, non_blocking=True)
        
        return data
                
# DataModule-like wrapper
class ClusterGCNDataModule(LightningDataModule):
    """Streaming DataModule for a single global Cluster-GCN partition.

    Uses one shared global partition and memmap bundle; train, validation
    and test loaders differ only in which cluster parts they cover and
    which supervision mask is active.

    Parameters
    ----------
    data_handle : dict
        Handle dictionary describing dataset paths and metadata.
    q : int, optional
        Number of clusters per mini-batch. Default is 10.
    num_workers : int, optional
        Number of worker processes for the dataloaders. Default is 0.
    pin_memory : bool, optional
        If True, pin memory in dataloaders. Default is False.
    with_edge_attr : bool, optional
        If True, batches include edge attributes. Default is False.
    eval_cover_strategy : str, optional
        Strategy for evaluation coverage (e.g. ``"all_parts"``). Default is "all_parts".
    seed : int, optional
        Random seed for part shuffling. Default is 42.
    device : torch.device or None, optional
        Device to move batches to. If ``None``, stays on CPU.
    persistent_workers : bool or None, optional
        If True, use persistent workers in dataloaders. If ``None``,
        inferred from ``num_workers``.
    post_batch_transform : callable or None, optional
        Optional transform applied to each batch after collation.
    """

    def __init__(
        self,
        *,
        data_handle: dict[str, object],
        q: int = 10,
        num_workers: int = 0,
        pin_memory: bool = False,
        with_edge_attr: bool = False,
        eval_cover_strategy: str = "all_parts",
        seed: int = 42,
        device: torch.device | None = None,
        persistent_workers: bool | None = None,
        post_batch_transform: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()

        self.handle = data_handle
        self.q = int(q)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.with_edge_attr = bool(with_edge_attr)
        self.eval_cover_strategy = str(eval_cover_strategy)
        self.seed = int(seed)
        self.device = device
        self.persistent_workers = (
            bool(persistent_workers)
            if persistent_workers is not None
            else (self.num_workers > 0)
        )

        self.ds_adapter = _HandleAdapter(self.handle)
        self._paths = self.handle.get("paths", {})
        self.post_batch_transform = post_batch_transform

        # Preload part-lists for splits if available
        self._parts_with = {}
        for split in ("train", "val", "test"):
            key = f"parts_with_{split}"
            path = self._paths.get(key, None)
            if path and osp.exists(path):
                self._parts_with[split] = np.load(path)
            else:
                self._parts_with[split] = None
        
    # in TBBlockStreamDataModule
    def _part_ids_for_split(self, split: str) -> Iterable[int]:
        """Return cluster IDs to iterate for a given split.

        Prefers precomputed ``parts_with_{split}`` collections to avoid
        batches without supervision; falls back to all parts otherwise.

        Parameters
        ----------
        split : {"train", "val", "test"}
            Split name.

        Returns
        -------
        Iterable[int]
            Iterable of part IDs for the split.
        """
        split = split.lower()

        key = None
        if split == "train":
            key = "train"
        elif split == "val":
            key = "val"
        elif split == "test":
            key = "test"

        if key is not None:
            arr = self._parts_with.get(key, None)
            if arr is not None and len(arr) > 0:
                return arr.astype(np.int64)

        # Fallback: if parts_with_* is missing, use all parts.
        return np.arange(self.ds_adapter.num_parts, dtype=np.int64)

    # internal: build one loader
    def _build_loader(self, *, split: str, shuffle: bool) -> DataLoader:
        part_ids = self._part_ids_for_split(split)
        part_ds = _PartIdListDataset(part_ids)

        collate = BlockCSRBatchCollator(
            self.ds_adapter,
            device=self.device,
            with_edge_attr=self.with_edge_attr,
            active_split=split,
            post_batch_transform = self.post_batch_transform,
        )

        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(
            part_ds,
            batch_size=self.q,            # q clusters per step
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate,
            generator=g if shuffle else None,
        )

    # Public API similar to LightningDataModule
    def train_dataloader(self) -> DataLoader:
        """Return dataloader for the training split.

        Returns
        -------
        DataLoader
            Training dataloader.
        """
        return self._build_loader(split="train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return dataloader for the validation split.

        Returns
        -------
        DataLoader
            Validation dataloader.
        """
        return self._build_loader(split="val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return dataloader for the test split.

        Returns
        -------
        DataLoader
            Test dataloader.
        """
        return self._build_loader(split="test", shuffle=False)

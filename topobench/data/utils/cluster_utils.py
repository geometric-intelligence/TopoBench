import os
import os.path as osp
from collections.abc import Callable
from typing import Any

import hydra
import numpy as np
import torch
import torch_geometric
from numpy.lib.format import open_memmap
from torch_geometric.data import Data, OnDiskDataset
from torch_geometric.loader import ClusterData


def build_cluster_transform(transforms_config) -> Callable | None:
    """Build a post-batch transform for clustered training.

    Parameters
    ----------
    transforms_config : dict or None
        Hydra-style configuration for transforms.

    Returns
    -------
    callable or None
        Composed transform or ``None`` if no transforms are defined.
    """
    # Build a post-batch transform for ClusterGCNDataModule.

    # Semantics match PreProcessor.instantiate_pre_transform.
    if not transforms_config:
        return None

    # Handle nested `liftings:` block like in the original code.
    if set(transforms_config.keys()) == {"liftings"}:
        transforms_config = transforms_config["liftings"]

    # Resolve Hydra interpolations / _target_ side effects.
    # (We ignore the output, mirroring PreProcessor.)
    hydra.utils.instantiate(transforms_config)

    from topobench.transforms.data_transform import DataTransform
    # Now wrap each config in a DataTransform, like PreProcessor does.
    transform_dict = {
        key: DataTransform(**value)
        for key, value in transforms_config.items()
    }

    if not transform_dict:
        return None

    if len(transform_dict) == 1:
        # Single transform: no need for Compose.
        return next(iter(transform_dict.values()))

    return torch_geometric.transforms.Compose(list(transform_dict.values()))


def to_bool_mask(mask: torch.Tensor, N: int) -> torch.Tensor:
    """Convert an index or score tensor to a boolean mask of length ``N``.

    Handles index lists, existing boolean masks, and length-``N`` score
    vectors.

    Parameters
    ----------
    mask : torch.Tensor
        Input mask or index tensor.
    N : int
        Desired length of the output mask.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(N,)``.
    """
    mask = mask.view(-1)

    # Case 1: already [N] bool
    if mask.dtype == torch.bool and mask.numel() == N:
        return mask

    # Case 2: index list (what load_transductive_splits gives us)
    if mask.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        out = torch.zeros(N, dtype=torch.bool)
        if mask.numel() > 0:
            out[mask.long()] = True
        return out

    # Case 3: length-N 0/1 or float scores
    if mask.numel() == N:
        return (mask != 0)
        
    # Fallback for ruff check:
    return torch.zeros(N, dtype=torch.bool)
        
def _tensor_schema_entry(t: torch.Tensor) -> dict[str, Any]:
    """Create a schema entry for a tensor value.

    Scalars are mapped to scalar Python types; higher-dimensional
    tensors are stored with shape ``(-1, ...)`` and fixed trailing
    dimensions.

    Parameters
    ----------
    t : torch.Tensor
        Example tensor.

    Returns
    -------
    dict or type
        Schema description for the tensor field.
    """
    # Make a schema entry for a tensor: (-1, ...) with fixed trailing dims.
    if t.dim() == 0:  # scalar tensor -> store as scalar type
        if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return int
        if t.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            return float
        if t.dtype == torch.bool:
            return bool
        # fallback: store as variable-length 1D
        return dict(dtype=t.dtype, size=(-1,))
    size = (-1,) + tuple(int(d) for d in t.size()[1:])
    return dict(dtype=t.dtype, size=size)

class ClusterOnDisk(OnDiskDataset):
    """On-disk storage and metadata for Cluster-GCN training.

    Builds a global partition using :class:`ClusterData`, infers a
    generic schema over all cluster subgraphs, stores them on disk, and
    writes permuted structural and feature arrays as NumPy memmaps.

    Parameters
    ----------
    root : str
        Root directory for the on-disk dataset.
    graph_getter : callable
        Callable returning the full :class:`torch_geometric.data.Data`
        graph.
    num_parts : int, optional
        Number of clusters for partitioning. Default is 10.
    recursive : bool, optional
        Whether to apply recursive partitioning. Default is False.
    keep_inter_cluster_edges : bool, optional
        If True, inter-cluster edges are kept. Default is False.
    sparse_format : {"csr"}, optional
        Sparse adjacency representation. Default is "csr".
    backend : {"sqlite", "rocksdb"}, optional
        On-disk backend. Default is "sqlite".
    transform : callable, optional
        Transform applied on loaded cluster subgraphs.
    pre_filter : callable, optional
        Filter applied before writing samples to disk.
    """
    def __init__(
        self,
        root: str,
        *,
        graph_getter: Callable[[], Data],
        num_parts: int = 10,
        recursive: bool = False,
        keep_inter_cluster_edges: bool = False,
        sparse_format: str = "csr",
        backend: str = "sqlite",
        transform=None,
        pre_filter=None,
    ) -> None:
        self._graph_getter = graph_getter
        self._cfg = dict(
            num_parts=int(num_parts),
            recursive=bool(recursive),
            keep_inter=bool(keep_inter_cluster_edges),
            sparse_format=str(sparse_format),
        )

        # Bootstrap once to know the REAL schema and partition ---
        full = self._graph_getter()

        cluster_data = ClusterData(
            full,
            num_parts=self._cfg["num_parts"],
            recursive=self._cfg["recursive"],
            keep_inter_cluster_edges=self._cfg["keep_inter"],
            sparse_format=self._cfg["sparse_format"],
            save_dir=None,
            log=False,
        )

        # Discover schema across ALL parts:
        # - edge_index is always present (2, -1)
        # - other tensor/int/float/bool fields gathered via union
        discovered: dict[str, Any] = {
            "edge_index": dict(dtype=torch.long, size=(2, -1))
        }

        for i in range(len(cluster_data)):
            part = cluster_data[i]
            if getattr(part, "edge_index", None) is None:
                raise ValueError("Cluster part without edge_index; cannot store.")
            for key, val in self._iter_data_items(part):
                if key == "edge_index":
                    continue
                self._schema_union_update(discovered, key, val)

        # Stash bootstrap so process() can reuse without recomputing:
        self._bootstrap_full = full
        self._bootstrap_cluster_data = cluster_data

        # Initialize OnDiskDataset with the discovered schema:
        super().__init__(
            root,
            transform=transform,
            pre_filter=pre_filter,
            backend=backend,
            schema=discovered,
        )
        self._meta: dict[str, Any] | None = None

    @property
    def raw_file_names(self) -> list[str]:
        """Raw files used by this dataset.

        Returns
        -------
        list of str
            Empty list (all data comes from ``graph_getter``).
        """
        # We don't rely on raw files here; everything comes from graph_getter.
        return []

    def download(self) -> None:
        """Download raw data (no-op).

        Notes
        -----
        All data are provided via ``graph_getter``, so nothing is
        downloaded.
        """
        # Nothing to download; graph_getter is user-provided.

    @staticmethod
    def _schema_union_update(
        schema: dict[str, Any],
        key: str,
        val: Any,
    ) -> None:
        """Update a schema dict with a sample field value.

        Parameters
        ----------
        schema : dict
            Mutable schema mapping field names to schema entries.
        key : str
            Field name.
        val : Any
            Sample value used to infer the entry.
        """
        # Add/validate a schema entry for key based on a sample value.
        # Skip Nones or non-serializable objects
        if val is None:
            return

        if isinstance(val, torch.Tensor):
            entry = _tensor_schema_entry(val)
            # If already present, assume trailing dims are consistent.
            schema.setdefault(key, entry)

        elif isinstance(val, (int, bool, float)):
            schema.setdefault(key, type(val))

        # Else: skip (e.g., strings, objects, SparseTensor not handled here).

    @staticmethod
    def _iter_data_items(d: Data):
        """Iterate over all public attributes of a Data object.

        Parameters
        ----------
        d : Data
            Graph data object.

        Yields
        ------
        tuple
            ``(key, value)`` pairs, including ``num_nodes`` if set.
        """
        # Yield (key, value) pairs for all public attrs in a Data,
        # including num_nodes if set.
        for k in d.keys(): # noqa: SIM118
            yield k, getattr(d, k)
        if getattr(d, "num_nodes", None) is not None:
            yield "num_nodes", int(d.num_nodes)

    # Processing: build DB + memmaps once
    def process(self) -> None:
        """Build on-disk cluster database and permutation memmaps.

        Uses the bootstrapped partition to store per-cluster subgraphs
        in the DB and writes global permuted structural and feature
        arrays as memmaps.
        """
        # Use bootstrapped objects; avoid recomputing partition.
        full: Data = self._bootstrap_full
        cluster_data: ClusterData = self._bootstrap_cluster_data

        # Write cluster subgraphs into the OnDiskDataset DB.
        buf: list[Data] = []
        for i in range(len(cluster_data)):
            buf.append(cluster_data[i])
            if (i + 1) % 1000 == 0 or (i + 1) == len(cluster_data):
                self.extend(buf)
                buf = []

        # Persist partition/meta info.
        meta = {
            "num_parts": cluster_data.num_parts,
            "recursive": cluster_data.recursive,
            "keep_inter_cluster_edges": cluster_data.keep_inter_cluster_edges,
            "sparse_format": cluster_data.sparse_format,
            # Partition object from PyG; torch.save-able and used later:
            "partition": cluster_data.partition,
        }
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(meta, self._meta_path)

        # Write global permuted CSR + X/y/edge_attr memmaps.
        self._write_perm_memmaps(full, cluster_data.partition)

        # Drop bootstrap references.
        self._bootstrap_full = None
        self._bootstrap_cluster_data = None

    # (De)serialization for OnDiskDataset
    def serialize(self, data: Data) -> dict[str, Any]:
        """Serialize a cluster subgraph into a schema-compatible row.

        Only fields present in the inferred schema are stored.

        Parameters
        ----------
        data : Data
            Cluster subgraph to serialize.

        Returns
        -------
        dict
            Dictionary of stored fields.
        """
        # Convert a Data object into a row matching the schema.
        # Only attributes present in the schema are stored.
        row: dict[str, Any] = {}
        # edge_index is mandatory
        if getattr(data, "edge_index", None) is None:
            raise ValueError("Data object without edge_index cannot be serialized.")
        row["edge_index"] = data.edge_index

        for key in self.schema: # .keys():
            if key == "edge_index":
                continue
            if hasattr(data, key):
                val = getattr(data, key)
                if isinstance(val, (torch.Tensor, int, bool, float)):
                    row[key] = val
                # else: silently skip unsupported types

        return row

    def deserialize(self, row: dict[str, Any]) -> Data:
        """Deserialize a stored row into a Data object.

        Parameters
        ----------
        row : dict
            Stored field dictionary.

        Returns
        -------
        Data
            Reconstructed cluster subgraph.
        """
        # Rebuild a Data object from a stored row.
        return Data.from_dict(row)

    # Meta properties
    @property
    def _meta_path(self) -> str:
        """Path to the stored cluster metadata file.

        Returns
        -------
        str
            Full path to ``cluster_meta.pt``.
        """
        return osp.join(self.processed_dir, "cluster_meta.pt")

    @property
    def meta(self) -> dict[str, Any]:
        """Cluster metadata dictionary.

        Returns
        -------
        dict
            Metadata including partition and configuration.
        """
        if self._meta is None:
            self._meta = torch.load(self._meta_path, map_location="cpu")
        return self._meta

    @property
    def partition(self):
        """Partition object from :class:`ClusterData`.

        Returns
        -------
        Any
            Partition object used for permuted structures.
        """
        return self.meta["partition"]

    @property
    def num_parts(self) -> int:
        """Number of cluster parts.

        Returns
        -------
        int
            Total number of partitions.
        """
        return int(self.meta["num_parts"])

    @property
    def recursive(self) -> bool:
        """Whether recursive partitioning is used.

        Returns
        -------
        bool
            True if recursive partitioning is enabled.
        """
        return bool(self.meta["recursive"])

    @property
    def keep_inter_cluster_edges(self) -> bool:
        """Whether inter-cluster edges are kept.

        Returns
        -------
        bool
            True if inter-cluster edges are preserved.
        """
        return bool(self.meta["keep_inter_cluster_edges"])

    @property
    def sparse_format(self) -> str:
        """Sparse format used for structural data.

        Returns
        -------
        str
            Name of the sparse format (e.g. ``"csr"``).
        """
        return str(self.meta["sparse_format"])

    # Memmap writers
    def _memmap_dir(self) -> str:
        """Directory for permuted memmap arrays.

        Returns
        -------
        str
            Path to the memmap directory.
        """
        return osp.join(self.processed_dir, "perm_memmap")

    def _write_perm_memmaps(self, full: Data, P: Any) -> None:
        """Write permuted structural and feature arrays as memmaps.

        Creates CSR structural arrays and permuted feature, label and
        edge-attribute arrays under ``processed/perm_memmap``.

        Parameters
        ----------
        full : Data
            Full, unpermuted input graph.
        P : Any
            Partition object with permutation and CSR attributes.
        """
        out_dir = self._memmap_dir()
        os.makedirs(out_dir, exist_ok=True)

        # Save structural arrays from Partition P (PyG's object):
        # P has: partptr, indptr, index, node_perm, edge_perm
        np.save(osp.join(out_dir, "partptr.npy"), P.partptr.cpu().numpy())
        np.save(osp.join(out_dir, "indptr.npy"),  P.indptr.cpu().numpy())
        np.save(osp.join(out_dir, "indices.npy"), P.index.cpu().numpy())

        # Node-level perm:
        node_perm = P.node_perm.cpu()
        N = int(full.num_nodes)
        
        # Save permutation maps
        # node_perm[i] = original_node_id for permuted row i
        perm_to_global = node_perm.clone().to(torch.long)
        np.save(osp.join(out_dir, "perm_to_global.npy"), perm_to_global.numpy())

        # Inverse: global_id -> perm_index (handy for debugging / analysis)
        global_to_perm = torch.empty_like(perm_to_global)
        global_to_perm[perm_to_global] = torch.arange(perm_to_global.numel(), dtype=torch.long)
        np.save(osp.join(out_dir, "global_to_perm.npy"), global_to_perm.numpy())

        # X_perm.npy (features)
        if getattr(full, "x", None) is not None and full.x.numel() > 0:
            x = full.x
            # ensure 2D
            if x.dim() == 1:
                x = x.view(-1, 1)
            F = int(x.size(1))
            X_path = osp.join(out_dir, "X_perm.npy")
            X_mm = open_memmap(
                X_path,
                dtype="float32",
                mode="w+",
                shape=(N, F),
            )
            X_mm[:] = x[node_perm].to(torch.float32).cpu().numpy()
            del X_mm  # flush & close

        # y_perm.npy (labels)
        if getattr(full, "y", None) is not None:
            y_src = full.y.view(-1)[node_perm].to(torch.int64).cpu().numpy()
            y_path = osp.join(out_dir, "y_perm.npy")
            y_mm = open_memmap(
                y_path,
                dtype="int64",
                mode="w+",
                shape=(y_src.shape[0],),
            )
            y_mm[:] = y_src
            del y_mm

        # edge_attr_perm.npy (edge features aligned with CSR edges)
        if getattr(full, "edge_attr", None) is not None:
            ea = full.edge_attr
            if ea.dim() == 1:
                ea = ea.view(-1, 1)
            ea_src = ea[P.edge_perm].to(torch.float32).cpu().numpy()
            E, F_e = ea_src.shape
            ea_path = osp.join(out_dir, "edge_attr_perm.npy")
            ea_mm = open_memmap(
                ea_path,
                dtype="float32",
                mode="w+",
                shape=(E, F_e),
            )
            ea_mm[:] = ea_src
            del ea_mm

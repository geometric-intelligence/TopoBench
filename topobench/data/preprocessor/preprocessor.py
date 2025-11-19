"""Preprocessor for datasets."""

import json
import os
import os.path as osp
import shutil
from typing import Any

import hydra
import numpy as np
import torch
import torch_geometric
from torch_geometric.io import fs

from topobench.data.utils import (
    ClusterOnDisk,
    ensure_serializable,
    load_inductive_splits,
    load_transductive_splits,
    make_hash,
    to_bool_mask,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class PreProcessor(torch_geometric.data.InMemoryDataset):
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : list
        List of data objects.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        self.dataset = dataset
        self.data_dir = data_dir
        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )
            super().__init__(
                self.processed_data_dir, None, pre_transform, **kwargs
            )
            self.save_transform_parameters()
            self.load(self.processed_paths[0])
        else:
            self.transforms_applied = False
            super().__init__(str(data_dir), None, None, **kwargs)
            self.load(data_dir + "/processed/data.pt")

        self.data_list = [self.get(idx) for idx in range(len(self))]
        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        if self.transforms_applied:
            return self.root
        else:
            return self.root + "/processed"
            
    @property
    def processed_file_names(self) -> str:
        """Return the name of the processed file.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        """Instantiate the pre-transforms.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.

        Returns
        -------
        torch_geometric.transforms.Compose
            Pre-transform object.
        """
        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings
        pre_transforms_dict = hydra.utils.instantiate(transforms_config)
        pre_transforms_dict = {
            key: DataTransform(**value)
            for key, value in transforms_config.items()
        }
        pre_transforms = torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
        )
        self.set_processed_data_dir(
            pre_transforms_dict, data_dir, transforms_config
        )
        return pre_transforms

    def set_processed_data_dir(
        self, pre_transforms_dict, data_dir, transforms_config
    ) -> None:
        """Set the processed data directory.

        Parameters
        ----------
        pre_transforms_dict : dict
            Dictionary containing the pre-transforms.
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.
        """
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            *[data_dir, repo_name, f"{params_hash}"]
        )

    def save_transform_parameters(self) -> None:
        """Save the transform parameters."""
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError(
                    "Different transform parameters for the same data_dir"
                )

            print(
                f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
            )

    def process(self) -> None:
        """Method that processes the data."""
        if isinstance(self.dataset, torch_geometric.data.Dataset):
            data_list = [
                self.dataset.get(idx) for idx in range(len(self.dataset))
            ]
        elif isinstance(self.dataset, torch.utils.data.Dataset):
            data_list = [self.dataset[idx] for idx in range(len(self.dataset))]
        elif isinstance(self.dataset, torch_geometric.data.Data):
            data_list = [self.dataset]

        self.data_list = (
            [self.pre_transform(d) for d in data_list]
            if self.pre_transform is not None
            else data_list
        )

        self._data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])

    def load(self, path: str) -> None:
        r"""Load the dataset from the file path `path`.

        Parameters
        ----------
        path : str
            The path to the processed data.
        """
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) >= 2 and len(out) <= 4
        if len(out) == 2:  # Backward compatibility (1).
            data, self.slices = out
        elif len(out) == 3:  # Backward compatibility (2).
            data, self.slices, data_cls = out
        else:  # TU Datasets store additional element (__class__) in the processed file
            data, self.slices, sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def load_dataset_splits(
        self, split_params
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load the dataset splits.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test datasets.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting == "inductive":
            return load_inductive_splits(self, split_params)
        elif split_params.learning_setting == "transductive":
            return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting.\
                Please define either 'inductive' or 'transductive'."
            )
            
    def pack_global_partition(
        self,
        *,
        split_params: dict[str, Any],
        cluster_params: dict[str, Any],
        stream_params: dict[str, Any],
        dtype_policy: str = "preserve",
        pack_db: bool = True,
        pack_memmaps: bool = True,
    ) -> dict[str, Any]:
        """
        Build a global Cluster-GCN partition for a transductive
        single-graph dataset, persist the resulting artifacts to disk, and
        return a compact handle for block-streaming loaders.

        The returned handle dictionary contains paths and metadata required by
        block-streaming dataloaders (e.g. `TBBlockStreamDataModule`) to build
        Cluster-GCN-style mini-batches directly from disk without reloading the
        full graph into memory.

        Parameters
        ----------
        split_params : dict
            Parameters for the split pipeline; must define a transductive
            single-graph setting and produce train/val/test masks.
        cluster_params : dict
            Parameters controlling graph partitioning:
            `num_parts`, `recursive`, `keep_inter_cluster_edges`,
            `sparse_format`, etc.
        stream_params : dict
            Parameters for downstream streaming, including
            `precompute_split_parts` to speed up split-aware sampling.
        dtype_policy : {"preserve", "float32"}, optional
            Policy for persisting feature/edge_attr dtypes. Recorded in meta
            for downstream consumers.
        pack_db : bool, optional
            If True, keep the `OnDiskDataset` DB of per-cluster subgraphs.
        pack_memmaps : bool, optional
            If True, write CSR and permuted feature/label/mask memmaps.

        Returns
        -------
        dict
            A handle with root/processed/memmap paths, partition metadata, and
            file locations for all relevant arrays.
        """
        # HASH: if handle.pt exists and matching, do now partition again
        cluster_config = {
            "split_params": ensure_serializable(split_params),
            "cluster_params": ensure_serializable(cluster_params),
            "stream_params": ensure_serializable(stream_params),
            "dtype_policy": dtype_policy,
        }
        config_hash = make_hash(cluster_config)

        root = self.data_dir
        processed_dir = osp.join(root, "processed")
        handle_path = osp.join(processed_dir, "handle.pt")

        # If handle exists, try to reuse
        if osp.exists(handle_path):
            old_handle = torch.load(handle_path, map_location="cpu")
            old_hash = old_handle.get("config_hash", None)
            print(old_hash, config_hash)
            if old_hash == config_hash:
                # Compatible cached partition: reuse it directly.
                return old_handle
            else:
                # Config changed: drop old processed dir and rebuild.
                shutil.rmtree(processed_dir, ignore_errors=True)
        else:
            # No handle, but there might be stale processed/ from some other run.
            if osp.exists(processed_dir):
                shutil.rmtree(processed_dir, ignore_errors=True)

        os.makedirs(root, exist_ok=True)
        _ = self.load_dataset_splits(split_params)

        full = getattr(self.dataset, "data", None)
        
        # num_nodes
        if getattr(full, "num_nodes", None) is not None:
            N = int(full.num_nodes)
        elif getattr(full, "x", None) is not None:
            N = int(full.x.size(0))
            full.num_nodes = N
        elif getattr(full, "y", None) is not None:
            N = int(full.y.size(0))
            full.num_nodes = N
        else:
            raise ValueError("Cannot infer num_nodes from full graph.")

        if getattr(full, "train_mask", None) is None:
            ds_train, ds_val, ds_test = self.load_dataset_splits(split_params)
            full = ds_train.data_lst[0]
            full.train_mask = to_bool_mask(getattr(full, "train_mask", None), N)
            full.val_mask = to_bool_mask(getattr(full, "val_mask", None), N)
            full.test_mask = to_bool_mask(getattr(full, "test_mask", None), N)
                        
        # Checks: we require a single full graph with masks.
        if getattr(full, "edge_index", None) is None:
            raise ValueError("Full graph has no edge_index.")
        if getattr(full, "train_mask", None) is None:
            raise ValueError("train_mask must exist on the full graph.")
        if getattr(full, "val_mask", None) is None:
            raise ValueError("val_mask must exist on the full graph.")
        if getattr(full, "test_mask", None) is None:
            raise ValueError("test_mask must exist on the full graph.")

        # Resolve cluster config.
        num_parts = int(cluster_params.get("num_parts", 10))
        recursive = bool(cluster_params.get("recursive", False))
        keep_inter = bool(cluster_params.get("keep_inter_cluster_edges", False))
        sparse_format = str(cluster_params.get("sparse_format", "csr"))

        # Build the ClusterOnDisk dataset (global partition).
        # Root lives under the dataset directory.
        # root = osp.join(self.data_dir, f"cluster_{num_parts}")
        root = self.data_dir # self.processed_dir
        # This version overwrite the processed .pt file.
        # shutil.rmtree(root + "/processed")
        # os.makedirs(root, exist_ok=True)
        
        ds = ClusterOnDisk(
            root=root,
            graph_getter=lambda: full,
            num_parts=num_parts,
            recursive=recursive,
            keep_inter_cluster_edges=keep_inter,
            sparse_format=sparse_format,
            backend="sqlite",
            transform=None,
            pre_filter=None,
        )
        # Touch to trigger process() if not already done.
        _ = len(ds)
        
        # Save schema for future use
        torch.save(ds.schema, osp.join(ds.processed_dir, "schema.pt"))

        # Write permuted split masks into the memmap bundle.
        mm_dir = osp.join(ds.processed_dir, "perm_memmap")
        os.makedirs(mm_dir, exist_ok=True)

        P = ds.partition
        node_perm = P.node_perm.cpu().numpy()

        def _to_numpy_bool(mask: torch.Tensor) -> np.ndarray:
            return mask.view(-1)[node_perm].to(torch.bool).cpu().numpy()

        train_mask_perm = _to_numpy_bool(full.train_mask)
        val_mask_perm   = _to_numpy_bool(full.val_mask)
        test_mask_perm  = _to_numpy_bool(full.test_mask)

        np.save(osp.join(mm_dir, "train_mask_perm.npy"), train_mask_perm)
        np.save(osp.join(mm_dir, "val_mask_perm.npy"),   val_mask_perm)
        np.save(osp.join(mm_dir, "test_mask_perm.npy"),  test_mask_perm)

        # Precompute which parts contain which split nodes.
        if bool(stream_params.get("precompute_split_parts", True)):
            partptr = np.load(osp.join(mm_dir, "partptr.npy"))

            def _parts_with(mask_perm: np.ndarray) -> np.ndarray:
                pos = np.flatnonzero(mask_perm)
                part_ids = np.searchsorted(partptr, pos, side="right") - 1
                return np.unique(part_ids.astype(np.int64))

            np.save(osp.join(mm_dir, "parts_with_train.npy"), _parts_with(train_mask_perm))
            np.save(osp.join(mm_dir, "parts_with_val.npy"),   _parts_with(val_mask_perm))
            np.save(osp.join(mm_dir, "parts_with_test.npy"),  _parts_with(test_mask_perm))

        # Record meta.
        full_N = int(getattr(full, "num_nodes", train_mask_perm.shape[0]))
        meta = {
            "num_parts": ds.num_parts,
            "recursive": ds.recursive,
            "keep_inter_cluster_edges": ds.keep_inter_cluster_edges,
            "sparse_format": ds.sparse_format,
            "dtype_policy": dtype_policy,
            "has_x": getattr(full, "x", None) is not None,
            "has_y": getattr(full, "y", None) is not None,
            "has_edge_attr": getattr(full, "edge_attr", None) is not None,
            "N": full_N,
        }
        # torch.save(meta, osp.join(ds.processed_dir, "cluster_global_meta.pt"))
        torch.save(meta, osp.join(ds.processed_dir, "cluster_meta.pt"))

        # Build and return handle for TBBlockStreamDataModule.
        handle = {
            "root": ds.root,
            "processed_dir": ds.processed_dir,
            "memmap_dir": mm_dir,
            "num_parts": int(ds.num_parts),
            "sparse_format": str(ds.sparse_format),
            "has_x": bool(meta["has_x"]),
            "has_y": bool(meta["has_y"]),
            "has_edge_attr": bool(meta["has_edge_attr"]),
            "paths": {
                "partptr": osp.join(mm_dir, "partptr.npy"),
                "indptr": osp.join(mm_dir, "indptr.npy"),
                "indices": osp.join(mm_dir, "indices.npy"),
                "X_perm": osp.join(mm_dir, "X_perm.npy"),
                "y_perm": osp.join(mm_dir, "y_perm.npy"),
                "edge_attr_perm": osp.join(mm_dir, "edge_attr_perm.npy"),
                "train_mask_perm": osp.join(mm_dir, "train_mask_perm.npy"),
                "val_mask_perm": osp.join(mm_dir, "val_mask_perm.npy"),
                "test_mask_perm": osp.join(mm_dir, "test_mask_perm.npy"),
                "parts_with_train": osp.join(mm_dir, "parts_with_train.npy"),
                "parts_with_val": osp.join(mm_dir, "parts_with_val.npy"),
                "parts_with_test": osp.join(mm_dir, "parts_with_test.npy"),
            },
        }
        # Store config hash in handle and on disk
        handle["config_hash"] = config_hash
        torch.save(handle, osp.join(ds.processed_dir, "handle.pt"))
        return handle

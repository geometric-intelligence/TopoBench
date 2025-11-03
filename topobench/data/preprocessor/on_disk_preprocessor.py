"""Preprocessor for datasets, backed by PyG OnDiskDataset."""

import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import torch_geometric
from torch_geometric.data.data import BaseData

from topobench.data.utils import (
    ensure_serializable,
    load_inductive_splits_on_disk,
    # load_transductive_splits,
    make_hash,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class _LazyDataList(Sequence):
    """
    Lazy list-like wrapper around an OnDiskDataset to preserve compatibility
    with TopoBench split utilities that expect `dataset.data_list`.
    Does NOT load the whole dataset into memory.
    """
    def __init__(self, dataset):
        self._ds = dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._ds[i] for i in range(start, stop, step)]
        return self._ds[idx]  # fetches on demand from disk

    def __iter__(self):
        for i in range(len(self)):
            yield self._ds[i]


class OnDiskPreProcessor(torch_geometric.data.OnDiskDataset):
    def __init__(
        self,
        dataset,
        data_dir: str,
        transforms_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # Keep a handle on the upstream dataset
        self.dataset = dataset
        self.transforms_applied = transforms_config is not None

        # Defaults:
        self.heavy_transforms_config: dict[str, Any] = {}
        self.easy_transforms_config: dict[str, Any] = {}
        self.heavy_transforms: Callable | None = None
        self.transforms_parameters: dict[str, Any] = {}
        
        if self.transforms_applied:
            (
                self.heavy_transforms_config,
                self.easy_transforms_config,
            ) = self._split_transforms(transforms_config)

            heavy_transform_dict, _ = self._build_transform_dict(
                self.heavy_transforms_config
            )
            self.heavy_transforms = self._compose_from_dict(heavy_transform_dict)

            self.transforms_parameters = ensure_serializable(
                {
                    name: transform.parameters
                    for name, transform in heavy_transform_dict.items()
                }
            )
            params_hash = make_hash(self.transforms_parameters)
            repo_name = "_".join(list(self.heavy_transforms_config.keys()))

            if repo_name:
                root_for_this_version = os.path.join(
                    data_dir, repo_name, f"{params_hash}"
                )
            else:
                root_for_this_version = os.path.join(data_dir, f"{params_hash}")
            super().__init__(
            root=root_for_this_version,
            transform=None,  # Online transforms are set manually later
            pre_filter=None,
            **kwargs,
            )
        else:
            root_for_this_version = data_dir
            super().__init__(
            root=root_for_this_version,
            transform=None,  # Online transforms are set manually later
            pre_filter=None,
            **kwargs,
            )
            # Attach dataset to OnDiskPreProcessor class
            if len(self) == 0:
                for sample in self.dataset:
                    self.append(sample)
                
        # Prepare online transforms and save metadata.
        # The `self.transform` attribute is used by PyG during data loading.
        self.transform = self._prepare_online_transforms()
        if self.transforms_applied:
            self.save_transform_parameters()

        # Preserve split_idx if the original dataset had fixed splits.
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
            return os.path.join(self.root, "processed")
                    
    @property
    def data_list(self):
        """
        Compatibility with original PreProcessor.
        Returns a sequence-like object, but keeps data on disk.
        """
        if not hasattr(self, "_lazy_data_list"):
            self._lazy_data_list = _LazyDataList(self)
        return self._lazy_data_list

    def process(self) -> None:
        """
        Called (once per cache dir) by Dataset._process() if the DB does not
        already exist. We iterate over `self.dataset`, apply heavy transforms,
        optionally apply `pre_filter`, and append each sample to the on-disk DB.
        """
        for sample in self.dataset:
            if not isinstance(sample, BaseData):
                raise TypeError(
                    f"Source dataset must yield PyG BaseData objects, "
                    f"but got {type(sample)}"
                )

            if self.heavy_transforms is not None:
                sample = self.heavy_transforms(sample)

            # Respect PyG-style pre_filter semantics if provided:
            if getattr(self, "pre_filter", None) is not None and not self.pre_filter(sample):
                continue

            # Insert into the underlying DB (self.db):
            self.append(sample)

    def save_transform_parameters(self) -> None:
        """
        Save (or verify) the parameters of the heavy/offline transforms.
        Raises if an existing cache directory corresponds to a different set
        of transform parameters.
        """
        # self.processed_dir now correctly points to the hashed directory
        os.makedirs(self.processed_dir, exist_ok=True)

        path_transform_parameters = os.path.join(
            self.processed_dir, "path_transform_parameters_dict.json"
        )

        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
        else:
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError(
                    "Different transform parameters for the same data_dir"
                )

            print(
                "Transform parameters are the same, "
                f"using existing data_dir: {self.processed_dir}"
            )

    def _prepare_online_transforms(self) -> Callable | None:
        """
        Compose lightweight/easy transforms (feature tweaks, etc.) with the
        dataset's intrinsic transform, and return a callable.
        This becomes `self.transform`, which PyG applies at access time.
        """
        transforms_list: Sequence[Callable] = []

        # 1. Dataset's intrinsic transform, if present:
        if hasattr(self.dataset, "transform") and self.dataset.transform is not None:
            transforms_list.append(self.dataset.transform)

        # 2. Easy transforms from config:
        easy_transform_dict, _ = self._build_transform_dict(
            self.easy_transforms_config
        )
        easy_compose = self._compose_from_dict(easy_transform_dict)
        if easy_compose is not None:
            transforms_list.append(easy_compose)

        # Return None / single transform / composed pipeline:
        if not transforms_list:
            return None
        if len(transforms_list) == 1:
            return transforms_list[0]
        return torch_geometric.transforms.Compose(list(transforms_list))

    # Transform config utilities
    def _split_transforms(
        self, transforms_config: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Split a Hydra-style transforms config into:
          heavy_config = {name: cfg} for lifting / feature_lifting
          easy_config  = {name: cfg} for the rest

        We support:
          - dict-of-dicts
          - or a single dict with keys like "transform_name", "transform_type".
        """
        heavy_config: dict[str, Any] = {}
        easy_config: dict[str, Any] = {}
        heavy_types = {"lifting", "feature_lifting"}

        # Normalize to {name: cfg} form:
        if "transform_name" in transforms_config:
            cfg_items = {"single": transforms_config}
        else:
            cfg_items = transforms_config

        for name, cfg in cfg_items.items():
            ttype = cfg.get("transform_type")
            if ttype in heavy_types:
                heavy_config[name] = cfg
            else:
                easy_config[name] = cfg

        return heavy_config, easy_config

    def _build_transform_dict(
        self, subset_cfg: dict[str, Any]
    ) -> tuple[dict[str, DataTransform], dict[str, Any]]:
        """
        Instantiate DataTransform objects from a subset config.
        Returns both:
          transform_dict: {name: DataTransform(...)}
          params_dict:    {name: parameters}  # for hashing/metadata
        """
        transform_dict: dict[str, DataTransform] = {}
        params_dict: dict[str, Any] = {}

        for name, cfg in subset_cfg.items():
            transform = DataTransform(**cfg)
            transform_dict[name] = transform
            params_dict[name] = transform.parameters

        return transform_dict, params_dict

    @staticmethod
    def _compose_from_dict(
        transform_dict: dict[str, Callable]
    ) -> Callable | None:
        """
        Turn {name: transform} into:
          None  (if empty),
          that single transform (if only one),
          or a torch_geometric.transforms.Compose([...]) (if multiple).
        """
        if len(transform_dict) == 0:
            return None
        if len(transform_dict) == 1:
            return list(transform_dict.values())[0]
        return torch_geometric.transforms.Compose(
            list(transform_dict.values())
        )

    # Split loading
    def load_dataset_splits(
        self, split_params: dict[str, Any]
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """
        Load or generate train/val/test splits using project utilities.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")
        if split_params.learning_setting == "inductive":
            return load_inductive_splits_on_disk(self, split_params)
        # elif split_params.learning_setting == "transductive":
        #     return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting. "
                "Please define either 'inductive' or 'transductive'."
            )

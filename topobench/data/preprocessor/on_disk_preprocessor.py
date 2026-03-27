"""Preprocessor for datasets using PyG OnDiskDataset."""

import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import torch_geometric
from torch_geometric.data.data import BaseData

from topobench.data.utils import (
    ensure_serializable,
    load_inductive_splits_on_disk,
    make_hash,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class _LazyDataList(Sequence):
    """Lazy sequence wrapper around an OnDiskDataset.

    Provides a list-like interface over data stored on disk, avoiding
    loading the whole dataset into memory.

    Parameters
    ----------
    dataset : OnDiskDataset
        Underlying on-disk dataset accessed on demand.
    """

    def __init__(self, dataset):
        self._ds = dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._ds[i] for i in range(start, stop, step)]
        return self._ds[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self._ds[i]


class OnDiskPreProcessor(torch_geometric.data.OnDiskDataset):
    """Preprocess datasets and persist results using OnDiskDataset.

    Applies heavy (offline) transforms once and stores processed samples
    on disk. Lightweight (online) transforms are composed and applied at
    access time through ``self.transform``.

    Parameters
    ----------
    dataset : Dataset
        Source dataset yielding PyG ``BaseData`` objects.
    data_dir : str
        Directory where processed data versions are stored.
    transforms_config : dict, optional
        Hydra-style configuration for heavy and easy transforms.
    **kwargs
        Additional arguments forwarded to ``OnDiskDataset``.
    """

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
                {name: transform.parameters for name, transform in heavy_transform_dict.items()}
            )
            params_hash = make_hash(self.transforms_parameters)
            repo_name = "_".join(list(self.heavy_transforms_config.keys()))

            if repo_name:
                root_for_this_version = os.path.join(data_dir, repo_name, f"{params_hash}")
            else:
                root_for_this_version = os.path.join(data_dir, f"{params_hash}")

            super().__init__(
                root=root_for_this_version,
                transform=None,
                pre_filter=None,
                **kwargs,
            )
        else:
            root_for_this_version = data_dir
            super().__init__(
                root=root_for_this_version,
                transform=None,
                pre_filter=None,
                **kwargs,
            )
            # Attach dataset to OnDiskPreProcessor class
            if len(self) == 0:
                for sample in self.dataset:
                    self.append(sample)
                
        # Prepare online transforms and save metadata.
        self.transform = self._prepare_online_transforms()
        if self.transforms_applied:
            self.save_transform_parameters()

        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

    @property
    def processed_dir(self) -> str:
        """Return directory containing processed samples.

        Returns
        -------
        str
            Path to processed directory.
        """
        if self.transforms_applied:
            return self.root
        else:
            return os.path.join(self.root, "processed")
                    
    @property
    def data_list(self):
        """Return a lazy list-like interface to on-disk samples.

        Returns
        -------
        _LazyDataList
            Sequence-like wrapper over stored data.
        """
        if not hasattr(self, "_lazy_data_list"):
            self._lazy_data_list = _LazyDataList(self)
        return self._lazy_data_list

    def process(self) -> None:
        """Apply heavy transforms and store processed samples.

        Iterates over the source dataset, applies heavy transforms if
        defined, applies optional ``pre_filter``, and appends results to
        the on-disk database.

        Raises
        ------
        TypeError
            If a sample is not a ``BaseData`` instance.
        """
        for sample in self.dataset:
            if not isinstance(sample, BaseData):
                raise TypeError(
                    f"Source dataset must yield PyG BaseData objects, but got {type(sample)}"
                )

            if self.heavy_transforms is not None:
                sample = self.heavy_transforms(sample)

            if getattr(self, "pre_filter", None) is not None and not self.pre_filter(sample):
                continue

            self.append(sample)

    def save_transform_parameters(self) -> None:
        """Save or validate heavy transform parameters on disk.

        Ensures reproducibility by writing parameters to a JSON file.
        If an existing file contains mismatched parameters, an error is raised.

        Raises
        ------
        ValueError
            If stored parameters differ from current parameters.
        """
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
                raise ValueError("Different transform parameters for the same data_dir")

            print(
                "Transform parameters are the same, using existing data_dir: "
                f"{self.processed_dir}"
            )

    def _prepare_online_transforms(self) -> Callable | None:
        """Compose online (easy) transforms with dataset-level transforms.

        Returns
        -------
        callable or None
            One transform, composed transforms, or None if no online
            transforms exist.
        """
        transforms_list: Sequence[Callable] = []

        if hasattr(self.dataset, "transform") and self.dataset.transform is not None:
            transforms_list.append(self.dataset.transform)

        easy_transform_dict, _ = self._build_transform_dict(self.easy_transforms_config)
        easy_compose = self._compose_from_dict(easy_transform_dict)
        if easy_compose is not None:
            transforms_list.append(easy_compose)

        if not transforms_list:
            return None
        if len(transforms_list) == 1:
            return transforms_list[0]
        return torch_geometric.transforms.Compose(list(transforms_list))

    def _split_transforms(
        self, transforms_config: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split transform config into heavy and easy components.

        Parameters
        ----------
        transforms_config : dict
            Raw transform configuration.

        Returns
        -------
        tuple of dict
            ``(heavy_config, easy_config)``.
        """
        heavy_config: dict[str, Any] = {}
        easy_config: dict[str, Any] = {}
        heavy_types = {"lifting", "feature_lifting"}

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
        """Build transform instances from config.

        Parameters
        ----------
        subset_cfg : dict
            Mapping from transform names to configs.

        Returns
        -------
        tuple of dict
            ``(transform_dict, params_dict)``.
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
        """Compose transforms stored in a mapping.

        Parameters
        ----------
        transform_dict : dict
            Mapping from names to transform callables.

        Returns
        -------
        callable or None
            Composed transform or single transform.
        """
        if len(transform_dict) == 0:
            return None
        if len(transform_dict) == 1:
            return list(transform_dict.values())[0]
        return torch_geometric.transforms.Compose(list(transform_dict.values()))

    def load_dataset_splits(
        self, split_params: dict[str, Any]
    ) -> tuple[DataloadDataset, DataloadDataset | None, DataloadDataset | None]:
        """Load or generate dataset splits.

        Supports inductive learning settings backed by on-disk splits.

        Parameters
        ----------
        split_params : dict
            Dictionary containing at least ``learning_setting``.

        Returns
        -------
        tuple
            ``(train, val, test)`` datasets.

        Raises
        ------
        ValueError
            If learning setting is missing or invalid.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")
        if split_params.learning_setting == "inductive":
            return load_inductive_splits_on_disk(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting. "
                "Please define either 'inductive' or 'transductive'."
            )

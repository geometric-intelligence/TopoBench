"""Dataset loader for OMol25 metals in hypergraph domain."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from topobench.data.datasets.omol25_metals_dataset import OMol25MetalsDataset


@dataclass
class OMol25MetalsDatasetLoader:
    """Loader for the OMol25 metal-complex subset.

    Attributes
    ----------
    data_domain : str
        Domain string used in configs (for example ``"hypergraph"``).
    data_type : str
        Dataset family identifier (for example ``"omol25"``).
    data_name : str
        Specific dataset name (for example ``"omol25_metals"``).
    data_dir : str
        Base directory where the data live. The dataset root is
        ``data_dir / data_name`` and is expected to contain a
        ``processed/data.pt`` file.
    """

    data_domain: str
    data_type: str
    data_name: str
    data_dir: str

    def _get_dataset_root(self) -> Path:
        """Resolve the root path passed to :class:`OMol25MetalsDataset`.

        Returns
        -------
        pathlib.Path
            Root directory that contains the ``processed`` folder.
        """
        base = Path(self.data_dir).expanduser().resolve()
        return base / self.data_name

    def _load_full_dataset(self) -> OMol25MetalsDataset:
        """Instantiate the underlying :class:`OMol25MetalsDataset`.

        Returns
        -------
        OMol25MetalsDataset
            The loaded dataset instance.
        """
        root = self._get_dataset_root()
        return OMol25MetalsDataset(root=str(root))

    def get_splits(
        self,
        split_params: Mapping[str, Any],
    ) -> dict[str, torch.utils.data.Dataset]:
        """Create train, validation, and test splits.

        Parameters
        ----------
        split_params : Mapping[str, Any]
            Dictionary describing the split configuration. Expected keys
            include ``"learning_setting"``, ``"split_type"``, ``"data_seed"``,
            ``"train_prop"``, and optionally ``"val_prop"``.

        Returns
        -------
        dict of str to Dataset
            Mapping with keys ``"train"``, ``"val"``, and ``"test"``.
        """
        dataset = self._load_full_dataset()
        n_total = len(dataset)

        learning_setting = split_params.get("learning_setting", "inductive")
        split_type = split_params.get("split_type", "random_in_train")

        if learning_setting != "inductive":
            msg = (
                "OMol25MetalsDatasetLoader currently supports only "
                f'learning_setting="inductive", got "{learning_setting}".'
            )
            raise NotImplementedError(msg)

        if split_type not in {"random_in_train", "random"}:
            msg = (
                "OMol25MetalsDatasetLoader currently supports only random "
                f'splits, got split_type="{split_type}".'
            )
            raise NotImplementedError(msg)

        train_prop = float(split_params.get("train_prop", 0.8))
        val_prop = split_params.get("val_prop")
        val_prop = None if val_prop is None else float(val_prop)

        if not 0.0 < train_prop < 1.0:
            msg = f"train_prop must be in (0, 1), got {train_prop}."
            raise ValueError(msg)

        if val_prop is not None and not 0.0 <= val_prop < 1.0:
            msg = f"val_prop must be in [0, 1), got {val_prop}."
            raise ValueError(msg)

        if val_prop is None:
            val_prop = 0.1

        if train_prop + val_prop >= 1.0:
            msg = (
                "train_prop + val_prop must be < 1.0, "
                f"got {train_prop + val_prop}."
            )
            raise ValueError(msg)

        n_train = int(round(train_prop * n_total))
        n_val = int(round(val_prop * n_total))
        n_test = n_total - n_train - n_val

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            msg = (
                "Invalid split sizes with "
                f"n_total={n_total}, train={n_train}, "
                f"val={n_val}, test={n_test}."
            )
            raise ValueError(msg)

        generator = torch.Generator()
        generator.manual_seed(int(split_params.get("data_seed", 0)))

        ds_train, ds_val, ds_test = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=generator,
        )

        return {
            "train": ds_train,
            "val": ds_val,
            "test": ds_test,
        }

    def get_dataloaders(
        self,
        split_params: Mapping[str, Any],
        dataloader_params: Mapping[str, Any],
    ) -> dict[str, DataLoader]:
        """Wrap the splits into PyG :class:`DataLoader` objects.

        Parameters
        ----------
        split_params : Mapping[str, Any]
            Split configuration dictionary (same as for :meth:`get_splits`).
        dataloader_params : Mapping[str, Any]
            Dictionary with loader settings such as ``"batch_size"``,
            ``"num_workers"``, ``"pin_memory"``, and ``"persistent_workers"``.

        Returns
        -------
        dict of str to DataLoader
            Mapping with keys ``"train"``, ``"val"``, and ``"test"``.
        """
        splits = self.get_splits(split_params)

        batch_size = int(dataloader_params.get("batch_size", 64))
        num_workers = int(dataloader_params.get("num_workers", 0))
        pin_memory = bool(dataloader_params.get("pin_memory", False))
        persistent_workers = bool(
            dataloader_params.get("persistent_workers", False)
            and num_workers > 0
        )

        loaders: dict[str, DataLoader] = {}
        for split_name, split_dataset in splits.items():
            shuffle = split_name == "train"
            loaders[split_name] = DataLoader(
                split_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

        return loaders

    def __call__(
        self,
        split_params: Mapping[str, Any],
        dataloader_params: Mapping[str, Any],
    ) -> tuple[dict[str, torch.utils.data.Dataset], dict[str, DataLoader]]:
        """Return dataset splits and dataloaders.

        Parameters
        ----------
        split_params : Mapping[str, Any]
            Split configuration dictionary.
        dataloader_params : Mapping[str, Any]
            Loader configuration dictionary.

        Returns
        -------
        tuple
            Two-element tuple ``(splits, loaders)`` where both entries are
            dictionaries keyed by ``"train"``, ``"val"``, and ``"test"``.
        """
        splits = self.get_splits(split_params)
        loaders = self.get_dataloaders(split_params, dataloader_params)
        return splits, loaders

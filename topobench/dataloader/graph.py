"""Native PyG batching for homogeneous graph data."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

from lightning import LightningDataModule
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

LearningSetting = Literal["inductive", "transductive"]


def _validate_integer(
    name: str,
    value: int,
    *,
    minimum: int,
    range_message: str,
) -> int:
    """Validate a loader integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(range_message)
    return normalized


def _require_non_empty(name: str, dataset: Dataset[Data]) -> None:
    """Reject an empty phase dataset with a stable boundary error."""
    if len(dataset) == 0:
        raise ValueError(f"dataset_{name} must not be empty")

def loader_worker_options(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> dict[str, int | bool]:
    """Return the worker lifecycle options shared by native graph loaders."""
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }


class GraphDataModule(LightningDataModule):
    """Batch native homogeneous PyG graphs for one learning setting.

    Inductive phases are separate, non-empty dataset views and only training is
    shuffled. Transductive loading accepts one source graph, exposes that same
    singleton dataset in every phase, and never accepts phase-specific copies.

    Parameters
    ----------
    dataset_train : Dataset[Data]
        Training view, or the one source dataset in transductive mode.
    dataset_val : Dataset[Data] | None, optional
        Non-empty validation view required only for inductive mode.
    dataset_test : Dataset[Data] | None, optional
        Non-empty test view required only for inductive mode.
    learning_setting : {"inductive", "transductive"}
        Explicit graph learning mode.
    batch_size : int, default=1
        Positive number of graphs per batch. Must be one for transductive mode.
    num_workers : int, default=0
        Non-negative PyTorch data-loading worker count.
    pin_memory : bool, default=False
        Forwarded to the PyG loader.
    persistent_workers : bool, default=False
        Forwarded when workers are enabled; forced off for zero workers.
    """

    def __init__(
        self,
        dataset_train: Dataset[Data],
        dataset_val: Dataset[Data] | None = None,
        dataset_test: Dataset[Data] | None = None,
        *,
        learning_setting: LearningSetting,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        batch_size = _validate_integer(
            "batch_size",
            batch_size,
            minimum=1,
            range_message="batch_size must be positive",
        )
        num_workers = _validate_integer(
            "num_workers",
            num_workers,
            minimum=0,
            range_message="num_workers must be non-negative",
        )
        _require_non_empty("train", dataset_train)

        if learning_setting == "transductive":
            if len(dataset_train) != 1:
                raise ValueError(
                    "transductive graph loading requires exactly one graph"
                )
            if dataset_val is not None or dataset_test is not None:
                raise ValueError(
                    "transductive phases must reuse the source graph"
                )
            if batch_size != 1:
                raise ValueError(
                    "transductive graph loading requires batch_size=1"
                )
            dataset_val = dataset_test = dataset_train
        elif learning_setting == "inductive":
            if dataset_val is None or dataset_test is None:
                raise ValueError(
                    "inductive loading requires train, validation, and test views"
                )
            if not all(
                isinstance(dataset, Subset)
                for dataset in (dataset_train, dataset_val, dataset_test)
            ):
                raise ValueError(
                    "inductive loading requires index-backed Subset views"
                )
            if not (
                dataset_train.dataset
                is dataset_val.dataset
                is dataset_test.dataset
            ):
                raise ValueError(
                    "inductive phase views must share one source dataset"
                )
            _require_non_empty("validation", dataset_val)
            _require_non_empty("test", dataset_test)
        else:
            raise ValueError(
                f"unsupported learning_setting: {learning_setting!r}"
            )

        self.learning_setting = learning_setting
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.batch_size = batch_size
        self.loader_kwargs = loader_worker_options(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training graph loader."""
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.learning_setting == "inductive",
            **self.loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the deterministic validation graph loader."""
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            **self.loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the deterministic test graph loader."""
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            **self.loader_kwargs,
        )

"TBDataloader class."

from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from topobench.data.batching import NeighborCellsLoader
from topobench.dataloader.dataload_dataset import DataloadDataset
from topobench.dataloader.utils import collate_fn


class TBDataloader(LightningDataModule):
    r"""This class takes care of returning the dataloaders for the training, validation, and test datasets.

    It also handles the collate function. The class is designed to work with the `torch` dataloaders.

    Parameters
    ----------
    dataset_train : DataloadDataset
        The training dataset.
    dataset_val : DataloadDataset, optional
        The validation dataset (default: None).
    dataset_test : DataloadDataset, optional
        The test dataset (default: None).
    batch_size : int, optional
        The batch size for the dataloader (default: 1).
    rank : int, optional
        The rank of the cells to consider when batching in the transductive setting (default: 0).
    num_neighbors : list[int], optional
        The number of neighbors to sample in the transductive setting. To consider n-hop neighborhoods this list should contain n elements. Care should be taken to check that the number of hops is appropriate for your model. With topological models the number of layers might not be enough to determine how far information is propagated.  (default: [-1]).
    num_workers : int, optional
        The number of worker processes to use for data loading (default: 0).
    pin_memory : bool, optional
        If True, the data loader will copy tensors into pinned memory before returning them (default: False).
    **kwargs : optional
        Additional arguments.

    References
    ----------
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset_train: DataloadDataset,
        dataset_val: DataloadDataset = None,
        dataset_test: DataloadDataset = None,
        batch_size: int = 1,
        rank: int = 0,
        num_neighbors: list[int] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["dataset_train", "dataset_val", "dataset_test"],
        )
        self.dataset_train = dataset_train
        self.batch_size = batch_size
        self.transductive = False
        self.rank = rank
        self.num_neighbors = (
            num_neighbors if num_neighbors is not None else [-1]
        )

        if dataset_val is None and dataset_test is None:
            # Transductive setting
            self.dataset_val = dataset_train
            self.dataset_test = dataset_train
            self.transductive = True
        else:
            self.dataset_val = dataset_val
            self.dataset_test = dataset_test

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = kwargs.get("persistent_workers", False)
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset_train={self.dataset_train}, dataset_val={self.dataset_val}, dataset_test={self.dataset_test}, batch_size={self.batch_size})"

    def _get_dataloader(self, split: str) -> DataLoader | NeighborCellsLoader:
        r"""Create and return the dataloader for the specified split.

        Parameters
        ----------
        split : str
            The split to create the dataloader for.

        Returns
        -------
        torch.utils.data.DataLoader | NeighborCellsLoader
            The dataloader for the specified split.
        """
        shuffle = split == "train"

        if not self.transductive or self.batch_size == "full":
            batch_size = self.batch_size if self.batch_size != "full" else 1

            return DataLoader(
                dataset=getattr(self, f"dataset_{split}"),
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=shuffle,
                collate_fn=collate_fn,
                persistent_workers=self.persistent_workers,
                **self.kwargs,
            )

        mask_idx = self.dataset_train[0][1].index(f"{split}_mask")
        mask = self.dataset_train[0][0][mask_idx]

        return NeighborCellsLoader(
            data=getattr(self, f"dataset_{split}"),
            rank=self.rank,
            num_neighbors=self.num_neighbors,
            input_nodes=mask,  # We have to specify split mask as we have to provide node indices to be sampled
            batch_size=self.batch_size,
            shuffle=shuffle,
            split=split,  # We need split to get the correct batch index mtching
            **self.kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        r"""Create and return the train dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The train dataloader.
        """
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        r"""Create and return the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        return self._get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        r"""Create and return the test dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The test dataloader.
        """
        if self.dataset_test is None:
            raise ValueError("There is no test dataloader.")
        return self._get_dataloader("test")

    def teardown(self, stage: str | None = None) -> None:
        r"""Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        Parameters
        ----------
        stage : str, optional
            The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"` (default: None).
        """

    def state_dict(self) -> dict[Any, Any]:
        r"""Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns
        -------
        dict
            A dictionary containing the datamodule state that you want to save.
        """
        return {}

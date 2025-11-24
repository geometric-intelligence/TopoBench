"""OMol25 metals dataset integration for TopoBench."""

import os
from collections.abc import Callable

import torch
from torch_geometric.data import InMemoryDataset, download_url


class OMol25MetalsDataset(InMemoryDataset):
    r"""Metal-complex subset of OMol25 as a PyG dataset.

    The dataset stores the preprocessed file ``processed/data.pt`` under
    ``root``. The file is produced by an external pipeline that converts
    OMol25 molecules into :class:`torch_geometric.data.Data` objects and
    serializes them with :class:`torch_geometric.data.InMemoryDataset`.

    Parameters
    ----------
    root : str
        Root directory for the dataset. The file ``processed/data.pt`` will
        be stored under this directory.
    transform : callable, optional
        Callable applied to each data object on-the-fly.
    pre_transform : callable, optional
        Callable applied before saving data objects to disk.
    """

    url: str = (
        "https://github.nrel.gov/yqin/omol25_metals/raw/main/"
        "data/omol25_metals/subset/processed/data.pt"
    )

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root, transform=transform, pre_transform=pre_transform
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return list of raw file names.

        Returns
        -------
        list of str
            Empty list, because raw OMol25 files are not stored locally.
        """
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Return list of processed file names.

        Returns
        -------
        list of str
            List containing ``"data.pt"``.
        """
        return ["data.pt"]

    def download(self) -> None:
        """Download the preprocessed file into ``processed/data.pt``.

        The file is fetched from the internal NREL GitHub URL defined in
        :attr:`url`.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        local_path = download_url(self.url, self.processed_dir)
        target_path = self.processed_paths[0]
        if os.path.abspath(local_path) != os.path.abspath(target_path):
            os.replace(local_path, target_path)

    def process(self) -> None:
        """Convert raw data into processed form.

        All preprocessing is performed externally, so this method is a no-op.
        It is only defined to satisfy the :class:`InMemoryDataset` interface.
        """
        # Nothing to do: ``download`` already creates ``processed/data.pt``.
        return

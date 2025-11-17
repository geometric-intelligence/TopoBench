"""Dataset class for OGBN-Arxiv dataset."""

import os
from typing import ClassVar

from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs


class OgbnArxivDataset(InMemoryDataset):
    r"""Dataset class for OGBN-Arxiv dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.
    """

    URLS: ClassVar = {
        "ogbn-arxiv": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv",
    }

    FILE_FORMAT: ClassVar = {
        "ogbn-arxiv": "auto",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters
        super().__init__(root)

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return f"{self.name}(root={self.root}, name={self.name}, parameters={self.parameters})"

    def download(self) -> None:
        """Download the dataset via OGB API (automatically handled)."""
        _ = PygNodePropPredDataset(name=self.name, root=self.root)

    def process(self) -> None:
        """Transform the raw dataset into TopoBench format."""
        dataset = PygNodePropPredDataset(name=self.name, root=self.root)
        data = dataset[0]

        # OGB provides y as shape [num_nodes, 1] — flatten it
        data.y = data.y.view(-1)

        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None

        # Save the processed dataset
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

    @property
    def raw_file_names(self):
        """Return files required in raw_dir to skip download().

        Returns
        -------
        list
            Empty list (OGB handles this internally).
        """
        return []  # OGB handles this internally

    @property
    def processed_file_names(self):
        """Return processed file name.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    @property
    def processed_paths(self):
        """Return processed path list.

        Returns
        -------
        list
            List of processed file paths.
        """
        return [
            os.path.join(self.root, "processed", self.processed_file_names)
        ]

    @property
    def processed_root(self):
        """Return processed root path.

        Returns
        -------
        str
            Path to the processed root directory.
        """
        return self.root

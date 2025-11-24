"""Dataset class for Semantic Representations."""

import os.path as osp

import torch
from datasets import concatenate_datasets, load_dataset
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from tqdm.auto import tqdm


class SemanticDataset(InMemoryDataset):
    r"""Dataset class for semantic representation of real datasets.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    parameters : DictConfig
        Configuration parameters for the dataset.
    """

    parameters = DictConfig(
        {
            "data_name": None,
            "n_subsampling": None,
            "models": None,
        }
    )

    def __init__(
        self,
        root: str,
        parameters: DictConfig,
    ) -> None:
        print(parameters)
        self.parameters.update(parameters)

        # Unpack parameters
        self.data_name = self.parameters.data_name
        self.models = self.parameters.models
        self.n_subsampling = self.parameters.n_subsampling

        assert self.data_name is not None, (
            'The "data_name" parameter must be set to a dataset name.'
        )
        assert self.models is not None, (
            'The "models" parameter must be set to a list of model names.'
        )

        # HF Repository URL
        self.url: str = f"spaicom-lab/semantic-{self.data_name}"

        # Call the super init
        super().__init__(
            root,
        )

        out = fs.torch_load(self.processed_paths[0])

        data, self.slices, self.sizes, data_cls = out

        self.data = data_cls.from_dict(data)
        # print(f"{self.sizes=}")
        # if self.n_subsampling is not None:
        #     data_list = []
        #     for i in range(len(self.models)):
        #         d = self.get(i)
        #         n = min(self.n_subsampling, d.x.size(0))
        #         d.x = d.x[:n]
        #         d.y = d.y[:n]
        #         data_list.append(d)

        #     self.data, self.slices = self.collate(data_list)
        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return f"{self.data_name}(self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.data_name, "raw")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return []

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, self.data_name, "processed")

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        r"""Download the dataset from the HF repository."""
        # Step 1: Download data from the source
        for model in tqdm(self.models, desc="Loading Models"):
            load_dataset(self.url, model, cache_dir=self.raw_dir)

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the semantic representation data, and saves the processed data
        to the appropriate location.
        """
        data_list = []
        for model in tqdm(self.models, desc="Preprocessing Models"):
            dataset = load_dataset(self.url, model, cache_dir=self.raw_dir)

            dataset = concatenate_datasets([dataset["train"], dataset["test"]])
            if self.n_subsampling is not None:
                n_sub = min(self.n_subsampling, len(dataset))
                dataset = dataset.select(range(n_sub))

            label = torch.tensor(dataset["label"])
            embedding = torch.tensor(dataset["embedding"])

            data = Data(x=embedding, y=label)
            data.model = model

            data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        self._data.dataset = self.data_name
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )


if __name__ == "__main__":
    # Some Variables
    root: str = "example"
    parameters: dict[str] = DictConfig(
        {
            "data_name": "cifar10",
            "models": [
                "aimv2_1b_patch14_224.apple_pt",
                "aimv2_1b_patch14_336.apple_pt",
                "aimv2_1b_patch14_448.apple_pt",
            ],
        }
    )

    # Initialize a Semantic Dataset
    dataset = SemanticDataset(root=root, parameters=parameters)

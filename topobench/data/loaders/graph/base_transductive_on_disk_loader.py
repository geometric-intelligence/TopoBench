from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets import TransductiveOnDiskDataset
from topobench.data.loaders.base import AbstractLoader


class OnDiskDatasetLoader(AbstractLoader):
    """
    Generic dataloader for partitioned on-disk datasets.

    Attributes
    ----------
    URLS : dict
        Mapping of dataset names to their download URLs.
    """

    URLS = {
        "Texas_on_disk": "https://github.com/dleko11/TopoBench/blob/main/for_download/WebKB/Texas/processed.zip?raw=1",
        "Cornell_on_disk": "https://github.com/dleko11/TopoBench/blob/main/for_download/WebKB/Cornell/processed.zip?raw=1",
        "Wisconsin_on_disk": "https://github.com/dleko11/TopoBench/blob/main/for_download/WebKB/Wisconsin/processed.zip?raw=1",
        "PubMed_on_disk": "https://github.com/dleko11/TopoBench/blob/main/for_download/cocitation/PubMed/processed.zip?raw=1",
        "Cora_on_disk": "https://github.com/dleko11/TopoBench/blob/main/for_download/cocitation/Cora/processed.zip?raw=1",
        "citeseer_on_disk": "https://github.com/dleko11/TopoBench/blob/main/for_download/cocitation/citeseer/processed.zip?raw=1",
        # "Reddit_on_disk": "https://github.com/dleko11/TopoBench/blob/main/for_download/Reddit/Reddit/processed.zip?raw=1",
        # Add more datasets here...
    }

    def __init__(self, parameters: DictConfig) -> None:
        """
        Initialize the loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration containing dataset name, backend, and paths.
        """
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """
        Load an on-disk dataset using the configured parameters.

        Returns
        -------
        Dataset
            A fully initialized `TransductiveOnDiskDataset` instance.

        Raises
        ------
        ValueError
            If the requested dataset name is not defined in `URLS`.
        RuntimeError
            If loading the dataset fails.
        """
        data_name = self.parameters.get("data_name", None)
        backend = self.parameters.get("backend", "sqlite")

        if data_name not in self.URLS:
            raise ValueError(
                f"Dataset '{data_name}' not found in URL registry. "
                f"Available keys: {list(self.URLS.keys())}"
            )

        url = self.URLS[data_name]

        try:
            dataset = TransductiveOnDiskDataset(
                root=str(self.root_data_dir),
                name=data_name,
                backend=backend,
                url=url,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load OnDiskDataset from {self.root_data_dir}"
            ) from e

        return dataset

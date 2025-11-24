"""Loaders for Semantic representation dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import SemanticDataset
from topobench.data.loaders.base import AbstractLoader


class SemanticDatasetLoader(AbstractLoader):
    """Load Semantic representations dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data.
            - data_name: Name of the dataset.
            - models: a list of (neural) model names used to build semantic representations.
            - other relevant parameters.
    """

    def __init__(
        self,
        parameters: DictConfig,
    ) -> None:
        super().__init__(parameters)
        self.parameters = parameters

    def load_dataset(self) -> SemanticDataset:
        """Load the Semantic dataset.

        Returns
        -------
        SemanticDataset
            The loaded a Semantic dataset with the appropriate `name`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> SemanticDataset:
        """Initialize the Semantic dataset.

        Returns
        -------
        SemanticDataset
            The initialized dataset instance.
        """
        return SemanticDataset(
            root=str(self.root_data_dir),
            parameters=self.parameters,
        )


if __name__ == "__main__":
    # Some Variables
    parameters: dict[str] = DictConfig(
        {
            "data_dir": "example",
            "data_name": "cifar10",
            "models": [
                "aimv2_1b_patch14_224.apple_pt",
                "aimv2_1b_patch14_336.apple_pt",
                "aimv2_1b_patch14_448.apple_pt",
            ],
        }
    )

    # Initialize a Semantic Dataset
    dataloader = SemanticDatasetLoader(parameters=parameters)

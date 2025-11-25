"""Loader for Conjugated Molecule dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import ConjugatedMoleculeDataset
from topobench.data.loaders.base import AbstractLoader


class ConjugatedMoleculeDatasetLoader(AbstractLoader):
    """Load Conjugated Molecule dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - split: Split of the dataset (optional, for OPV)
            - other relevant parameters
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self, **kwargs) -> ConjugatedMoleculeDataset:
        """Load the Conjugated Molecule dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        ConjugatedMoleculeDataset
            The loaded Conjugated Molecule dataset.
        """
        dataset = self._initialize_dataset()

        # Handle slicing if requested (e.g. for testing long-running datasets)
        if "slice" in kwargs:
            dataset = dataset[: kwargs["slice"]]

        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> ConjugatedMoleculeDataset:
        """Initialize the Conjugated Molecule dataset.

        Returns
        -------
        ConjugatedMoleculeDataset
            The initialized dataset instance.
        """
        # Check if split is in parameters, default to None
        split = self.parameters.get("split", None)

        return ConjugatedMoleculeDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            split=split,
            # Pass other parameters if needed, e.g. transforms from config
        )

"""Loaders for expressivity datasets."""

from omegaconf import DictConfig

from topobench.data.datasets import BRECDataset
from topobench.data.loaders.base import AbstractLoader


class ExpressivityDatasetLoader(AbstractLoader):
    """Load expressivity datasets (BREC) with configurable parameters.

     Parameters
     ----------
     parameters : DictConfig
         Configuration parameters containing:
             - data_dir: Root directory for data
             - data_name: Name of the dataset
             - other relevant parameters

    **kwargs : dict
         Additional keyword arguments.
    """

    def __init__(self, parameters: DictConfig, **kwargs) -> None:
        super().__init__(parameters, **kwargs)

    def load_dataset(self, **kwargs):
        """Load the expressivity dataset dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for dataset initialization.

        Returns
        -------
        CitationHypergraphDataset
            The loaded Citation Hypergraph dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset(**kwargs)
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self, **kwargs):
        """Initialize the expressivity dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for dataset initialization.

        Returns
        -------
        ExpressivityDataset
            The initialized dataset instance.
        """
        if self.parameters.data_type == "BREC":
            return BRECDataset(
                root=str(self.root_data_dir),
                name=self.parameters.data_name,
                parameters=self.parameters,
                load_as_graph=True,
                **kwargs,
            )
        else:
            raise RuntimeError(
                f"Dataset {self.parameters.data_name} not supported."
            )

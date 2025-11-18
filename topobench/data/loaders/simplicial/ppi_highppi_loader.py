"""Loader for PPI dataset (HIGH-PPI variant) with CORUM complexes."""

from omegaconf import DictConfig

from topobench.data.datasets.ppi_highppi_dataset import PPIHighPPIDataset
from topobench.data.loaders.base import AbstractLoader


class PPIHighPPIDatasetLoader(AbstractLoader):
    """Load HIGH-PPI SHS27k dataset with CORUM topological enrichment.

    This loader creates a hybrid simplicial complex from:
    - HIGH-PPI's SHS27k PPI network (labeled edges)
    - CORUM protein complexes (unlabeled higher-order cells)

    Task: Edge-level multi-label classification (7 interaction types)

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - min_complex_size: Minimum CORUM complex size
            - max_complex_size: Maximum CORUM complex size
            - max_rank: Maximum simplicial rank
            - use_official_split: Use HIGH-PPI's train/val split
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, parameters: DictConfig, **kwargs) -> None:
        super().__init__(parameters, **kwargs)

    def load_dataset(self, **kwargs) -> PPIHighPPIDataset:
        """Load the dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to dataset initialization.

        Returns
        -------
        PPIHighPPIDataset
            Dataset with HIGH-PPI network and CORUM complexes.
        """
        dataset = self._initialize_dataset(**kwargs)
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self, **kwargs) -> PPIHighPPIDataset:
        """Initialize the HIGH-PPI SHS27k dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for dataset initialization.

        Returns
        -------
        PPIHighPPIDataset
            The initialized dataset instance.
        """
        self.dataset = PPIHighPPIDataset(
            root=str(self.root_data_dir),
            name=self.parameters.get("data_name", "highppi_shs27k"),
            parameters=self.parameters,
            **kwargs,
        )
        return self.dataset

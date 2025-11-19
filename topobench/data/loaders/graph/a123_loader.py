"""A123 dataset loader module."""

import torch
from omegaconf import DictConfig

from topobench.data.datasets.a123 import A123CortexMDataset
from topobench.data.loaders.base import AbstractLoader


class A123DatasetLoader(AbstractLoader):
    """Loader for A123 mouse auditory cortex dataset.

    Implements the AbstractLoader interface: accepts a DictConfig `parameters`
    and implements `load_dataset()` which returns a dataset object.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters for the dataset.
    **overrides
        Additional keyword arguments to override parameters.
    """

    def __init__(self, parameters: DictConfig, **overrides):
        """Initialize the A123 dataset loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration parameters for the dataset.
        **overrides
            Additional keyword arguments to override parameters.
        """
        # Initialize AbstractLoader (sets self.parameters and self.root_data_dir)
        super().__init__(parameters)

        # hyperparameters can come from the DictConfig or be passed as overrides
        params = parameters if parameters is not None else {}

        def _get(k, default):
            """Get parameter value from DictConfig or overrides.

            Parameters
            ----------
            k : str
                Parameter key.
            default : Any
                Default value if key not found.

            Returns
            -------
            Any
                Parameter value from DictConfig or overrides, or default.
            """
            try:
                return params.get(k, overrides.get(k, default))
            except Exception:
                # DictConfig may use attribute access
                return getattr(params, k, overrides.get(k, default))

        self.batch_size = int(_get("batch_size", 32))
        # dataset will be created when load_dataset() is called
        self.dataset = None

    def load_dataset(self) -> torch.utils.data.Dataset:
        """Instantiate and return the underlying dataset.

        Returns a `A123CortexMDataset` instance constructed from the loader's
        parameters and root data directory.

        Returns
        -------
        torch.utils.data.Dataset
            A123CortexMDataset instance.
        """
        # determine dataset name from parameters, fallback to expected id
        name = self.parameters.data_name

        # root path for dataset: use the root_data_dir (Path) as string
        root = str(self.root_data_dir)

        # Construct dataset; A123CortexMDataset expects (root, name, parameters)
        self.dataset = A123CortexMDataset(
            root=root, name=name, parameters=self.parameters
        )

        return self.dataset

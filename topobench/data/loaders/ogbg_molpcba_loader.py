"""Loader for ogbg-molpcba and mock molecular datasets.

This module provides loaders for the OGBG-molpcba dataset (437K molecular graphs)
and a mock dataset for testing pipelines without downloading large data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

from topobench.data.datasets.ogbg_molpcba import (
    MockMolecularDataset,
    OGBGMolPCBADataset,
)
from topobench.data.loaders.base import AbstractLoader


class OGBGMolPCBALoader(AbstractLoader):
    """Loader for OGBG-molpcba molecular property prediction dataset.

    OGBG-molpcba is a large-scale molecular property prediction dataset with:
    - 437,929 molecular graphs
    - 128 binary classification tasks (multi-label)
    - Average ~26 nodes per molecule
    - 9-dimensional node features
    - 3-dimensional edge features

    This loader supports:
    - Full dataset or subset for testing
    - Mock dataset for pipeline verification
    - On-disk preprocessing via OnDiskInductivePreprocessor

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters including:
        - data_dir: Directory for storing the dataset
        - data_name: Name identifier (default: "ogbg-molpcba")
        - subset_size: Optional, use only first N samples (default: None = all)
        - split: Dataset split - "train", "valid", or "test" (default: "train")
        - use_mock: If True, use MockMolecularDataset instead (default: False)

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from topobench.data.loaders import OGBGMolPCBALoader
    >>>
    >>> # Load full training set
    >>> config = OmegaConf.create({
    ...     "data_dir": "./data/ogbg_molpcba",
    ...     "data_name": "ogbg-molpcba",
    ...     "split": "train"
    ... })
    >>> loader = OGBGMolPCBALoader(config)
    >>> dataset, data_dir = loader.load()
    >>>
    >>> # Load subset for testing
    >>> config.subset_size = 100
    >>> loader = OGBGMolPCBALoader(config)
    >>> dataset, data_dir = loader.load()
    >>>
    >>> # Use mock dataset (no download needed)
    >>> config.use_mock = True
    >>> loader = OGBGMolPCBALoader(config)
    >>> dataset, data_dir = loader.load()

    Notes
    -----
    - Requires ogb package: `pip install ogb`
    - First run downloads dataset (~500MB)
    - Use with OnDiskInductivePreprocessor for memory-efficient training
    - Mock dataset is perfect for testing pipelines on memory-constrained machines

    See Also
    --------
    topobench.data.datasets.ogbg_molpcba.OGBGMolPCBADataset : The dataset class
    topobench.data.datasets.ogbg_molpcba.MockMolecularDataset : Mock dataset for testing
    topobench.data.preprocessor.ondisk_inductive.OnDiskInductivePreprocessor : For preprocessing
    """

    def __init__(self, parameters: DictConfig):
        """Initialize OGBG-molpcba loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration with data_dir, split, subset_size, and use_mock options.
        """
        super().__init__(parameters)
        self.subset_size = parameters.get("subset_size", None)
        self.split = parameters.get("split", "train")
        self.use_mock = parameters.get("use_mock", False)
        self.dataset = None

    def load_dataset(self):
        """Load the OGBG-molpcba or mock dataset.

        Returns
        -------
        Dataset
            Either OGBGMolPCBADataset or MockMolecularDataset depending on use_mock.

        Raises
        ------
        ImportError
            If ogb package is not installed (when use_mock=False).
        """
        if self.dataset is None:
            if self.use_mock:
                self.dataset = MockMolecularDataset(
                    root=self.root_data_dir,
                    num_samples=self.subset_size or 100,
                    cache_samples=False,  # Disable to prevent disk bloat (use OnDisk preprocessor instead)
                )
            else:
                try:
                    self.dataset = OGBGMolPCBADataset(
                        root=self.root_data_dir,
                        split=self.split,
                        subset_size=self.subset_size,
                        cache_samples=False,  # CRITICAL: Disable to prevent 171GB disk cache!
                    )
                except ImportError as e:
                    raise ImportError(
                        "ogb package is required for OGBG-molpcba dataset. "
                        "Install with: pip install ogb\n"
                        "Or use use_mock=True to test with synthetic data."
                    ) from e
        return self.dataset

    def load(self):
        """Load dataset and return with data directory.

        Returns
        -------
        tuple
            (dataset, data_dir) tuple.
        """
        dataset = self.load_dataset()
        return dataset, str(self.root_data_dir)

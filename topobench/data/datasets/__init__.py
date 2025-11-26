"""Dataset module with automated exports."""

import inspect
import sys
from importlib import util
from pathlib import Path
from typing import ClassVar

from torch_geometric.data import InMemoryDataset

# Import lazy splits for O(1) memory usage
from ._lazy import LazyDataloadDataset, LazySubset

# Import adapters for converting existing PyG datasets
from .adapters import (
    PyGDatasetAdapter,
    adapt_dataset,
    adapt_tu_dataset,
)

# Import our high-performance base classes
from .base_inductive import (
    BaseOnDiskInductiveDataset,
    FileBasedInductiveDataset,
    GeneratedInductiveDataset,
)


class DatasetManager:
    """Manages automatic discovery and registration of dataset classes."""

    # Static dataset definitions
    PLANETOID_DATASETS: ClassVar[list[str]] = [
        "Cora",
        "citeseer",
        "PubMed",
    ]

    TU_DATASETS: ClassVar[list[str]] = [
        "MUTAG",
        "ENZYMES",
        "PROTEINS",
        "COLLAB",
        "IMDB-BINARY",
        "IMDB-MULTI",
        "REDDIT-BINARY",
        "NCI1",
        "NCI109",
    ]

    FIXED_SPLITS_DATASETS: ClassVar[list[str]] = ["ZINC", "AQSOL"]

    HETEROPHILIC_DATASETS: ClassVar[list[str]] = [
        "amazon_ratings",
        "questions",
        "minesweeper",
        "roman_empire",
        "tolokers",
    ]

    @classmethod
    def discover_datasets(
        cls, package_path: str
    ) -> dict[str, type[InMemoryDataset]]:
        """Dynamically discover all dataset classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        Dict[str, Type[InMemoryDataset]]
            Dictionary mapping class names to their corresponding class objects.
        """
        datasets = {}

        # Get the directory containing the dataset modules
        package_dir = Path(package_path).parent

        # Iterate through all .py files in the directory
        for file_path in package_dir.glob("*.py"):
            if file_path.stem == "__init__":
                continue

            # Import the module
            module_name = f"{__name__}.{file_path.stem}"
            spec = util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Register in sys.modules for correct pickling
                sys.modules[module_name] = module

                # Find all dataset classes in the module
                new_datasets = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        inspect.isclass(obj)
                        and obj.__module__ == module.__name__
                        and not name.startswith("_")
                        and issubclass(obj, InMemoryDataset)
                        and obj != InMemoryDataset
                    )
                }
                datasets.update(new_datasets)
        return datasets

    @classmethod
    def get_pyg_datasets(cls) -> list[str]:
        """Get combined list of all PyG datasets.

        Returns
        -------
        List[str]
            List of all PyG datasets.
        """
        return (
            cls.PLANETOID_DATASETS
            + cls.TU_DATASETS
            + cls.FIXED_SPLITS_DATASETS
            + cls.HETEROPHILIC_DATASETS
        )


# Create the dataset manager
manager = DatasetManager()

# Automatically discover and populate datasets
MANUAL_DATASETS = manager.discover_datasets(__file__)

# Create other dataset collections
PYG_DATASETS = manager.get_pyg_datasets()
PLANETOID_DATASETS = manager.PLANETOID_DATASETS
TU_DATASETS = manager.TU_DATASETS
FIXED_SPLITS_DATASETS = manager.FIXED_SPLITS_DATASETS
HETEROPHILIC_DATASETS = manager.HETEROPHILIC_DATASETS

# Automatically generate __all__
__all__ = [
    # High-performance base classes for custom datasets
    "BaseOnDiskInductiveDataset",
    "FileBasedInductiveDataset",
    "GeneratedInductiveDataset",
    # Lazy splits for O(1) memory usage
    "LazySubset",
    "LazyDataloadDataset",
    # Adapters for existing PyG datasets
    "PyGDatasetAdapter",
    "adapt_dataset",
    "adapt_tu_dataset",
    # Dataset collections
    "PYG_DATASETS",
    "PLANETOID_DATASETS",
    "TU_DATASETS",
    "FIXED_SPLITS_DATASETS",
    "HETEROPHILIC_DATASETS",
    "MANUAL_DATASETS",
    # Discovered dataset classes
    *MANUAL_DATASETS.keys(),
]

# For backwards compatibility, create individual imports
locals().update(**MANUAL_DATASETS)

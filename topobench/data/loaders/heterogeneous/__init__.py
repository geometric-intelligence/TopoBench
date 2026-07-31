"""Native heterogeneous graph dataset loaders."""

from .dblp import DBLPDatasetLoader
from .synthetic import SyntheticHeterogeneousDatasetLoader

__all__ = [
    "DBLPDatasetLoader",
    "SyntheticHeterogeneousDatasetLoader",
]

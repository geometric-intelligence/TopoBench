"""Native heterogeneous graph dataset loaders."""

from .dblp import DBLPDatasetLoader
from .ogb_mag import OGBMAGDatasetLoader
from .synthetic import SyntheticHeterogeneousDatasetLoader

__all__ = [
    "DBLPDatasetLoader",
    "OGBMAGDatasetLoader",
    "SyntheticHeterogeneousDatasetLoader",
]

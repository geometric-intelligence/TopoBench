"""Dataset loaders for the heterogeneous graph domain."""

from .dblp import DBLPDatasetLoader
from .ogb_mag import OGBMAGDatasetLoader
from .synthetic import SyntheticHeterogeneousDatasetLoader

HETEROGENEOUS_LOADERS = dict(
    sorted(
        {
            loader_class.__name__: loader_class
            for loader_class in (
                DBLPDatasetLoader,
                OGBMAGDatasetLoader,
                SyntheticHeterogeneousDatasetLoader,
            )
        }.items()
    )
)

globals().update(HETEROGENEOUS_LOADERS)

__all__ = [*HETEROGENEOUS_LOADERS, "HETEROGENEOUS_LOADERS"]

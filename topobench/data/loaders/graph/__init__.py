"""Dataset loaders for the graph domain."""

from .adme_datasets import ADMEDatasetLoader
from .graph_universe_loader import GraphUniverseDatasetLoader
from .hetero_datasets import HeterophilousGraphDatasetLoader
from .manual_graph_dataset_loader import ManualGraphDatasetLoader
from .molecule_datasets import MoleculeDatasetLoader
from .ogbg_datasets import OGBGDatasetLoader
from .planetoid_datasets import PlanetoidDatasetLoader
from .tu_datasets import TUDatasetLoader
from .us_county_demos_dataset_loader import USCountyDemosDatasetLoader

GRAPH_LOADERS = dict(
    sorted(
        {
            loader_class.__name__: loader_class
            for loader_class in (
                ADMEDatasetLoader,
                GraphUniverseDatasetLoader,
                HeterophilousGraphDatasetLoader,
                ManualGraphDatasetLoader,
                MoleculeDatasetLoader,
                OGBGDatasetLoader,
                PlanetoidDatasetLoader,
                TUDatasetLoader,
                USCountyDemosDatasetLoader,
            )
        }.items()
    )
)

globals().update(GRAPH_LOADERS)

__all__ = [*GRAPH_LOADERS, "GRAPH_LOADERS"]

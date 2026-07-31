"""Dataset loaders for the hypergraph domain."""

from .citation_hypergraph_dataset_loader import (
    CitationHypergraphDatasetLoader,
)
from .hypergraph_dataset_loader import HypergraphDatasetLoader
from .synthetic import SyntheticHypergraphDatasetLoader

HYPERGRAPH_LOADERS = dict(
    sorted(
        {
            loader_class.__name__: loader_class
            for loader_class in (
                CitationHypergraphDatasetLoader,
                HypergraphDatasetLoader,
                SyntheticHypergraphDatasetLoader,
            )
        }.items()
    )
)

globals().update(HYPERGRAPH_LOADERS)

__all__ = [*HYPERGRAPH_LOADERS, "HYPERGRAPH_LOADERS"]

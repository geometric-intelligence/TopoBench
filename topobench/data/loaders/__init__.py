"""Dataset loaders exposed by the supported data domains."""

from .base import AbstractLoader
from .graph import GRAPH_LOADERS
from .heterogeneous import HETEROGENEOUS_LOADERS
from .hypergraph import HYPERGRAPH_LOADERS

LOADER_CLASSES = dict(
    sorted(
        {
            **GRAPH_LOADERS,
            **HETEROGENEOUS_LOADERS,
            **HYPERGRAPH_LOADERS,
        }.items()
    )
)

globals().update(LOADER_CLASSES)

__all__ = ["AbstractLoader", *LOADER_CLASSES, "LOADER_CLASSES"]

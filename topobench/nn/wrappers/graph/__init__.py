"""Wrappers for the graph domain."""

from .gnn_wrapper import GNNWrapper
from .graph_mlp_wrapper import GraphMLPWrapper

WRAPPER_CLASSES = dict(
    sorted(
        {
            wrapper_class.__name__: wrapper_class
            for wrapper_class in (GNNWrapper, GraphMLPWrapper)
        }.items()
    )
)

globals().update(WRAPPER_CLASSES)

__all__ = [*WRAPPER_CLASSES, "WRAPPER_CLASSES"]

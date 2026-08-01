"""Wrappers exposed by the supported data domains."""

from .graph import GNNWrapper, GraphMLPWrapper
from .heterogeneous import HeterogeneousWrapper
from .hypergraph import HypergraphWrapper

WRAPPER_CLASSES = {
    "GNNWrapper": GNNWrapper,
    "GraphMLPWrapper": GraphMLPWrapper,
    "HeterogeneousWrapper": HeterogeneousWrapper,
    "HypergraphWrapper": HypergraphWrapper,
}

__all__ = [
    "GNNWrapper",
    "GraphMLPWrapper",
    "HeterogeneousWrapper",
    "HypergraphWrapper",
    "WRAPPER_CLASSES",
]

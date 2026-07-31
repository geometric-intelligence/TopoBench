"""Explicit native graph wrapper registry."""

from .gnn_wrapper import GNNWrapper
from .graph_mlp_wrapper import GraphMLPWrapper

WRAPPER_CLASSES = {
    "GNNWrapper": GNNWrapper,
    "GraphMLPWrapper": GraphMLPWrapper,
}

__all__ = [*WRAPPER_CLASSES, "WRAPPER_CLASSES"]

"""Wrappers for graph models."""

from .gnn_wrapper import GNNWrapper
from .graph_mlp_wrapper import GraphMLPWrapper

# Export all wrappers
__all__ = [
    "GNNWrapper",
    "GraphMLPWrapper",
]

"""Backbones for the graph domain."""

from .gcn_dgm import GCNDGM
from .gps import GPSEncoder
from .graph_mlp import GraphMLP
from .nsd import NSDEncoder

BACKBONE_CLASSES = {
    "GCNDGM": GCNDGM,
    "GPSEncoder": GPSEncoder,
    "GraphMLP": GraphMLP,
    "NSDEncoder": NSDEncoder,
}

__all__ = [
    "GCNDGM",
    "GPSEncoder",
    "GraphMLP",
    "NSDEncoder",
    "BACKBONE_CLASSES",
]

"""Backbones for the graph domain."""

from .gcn_dgm import GCNDGM
from .gps import GPSEncoder
from .graph_mlp import GraphMLP
from .nsd import NSDEncoder

BACKBONE_CLASSES = dict(
    sorted(
        {
            backbone_class.__name__: backbone_class
            for backbone_class in (GCNDGM, GPSEncoder, GraphMLP, NSDEncoder)
        }.items()
    )
)

globals().update(BACKBONE_CLASSES)

__all__ = [*BACKBONE_CLASSES, "BACKBONE_CLASSES"]

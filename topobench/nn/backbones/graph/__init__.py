"""Backbones for the graph domain."""

from .gps import GPSEncoder
from .graph_mlp import GraphMLP
from .nsd import NSDEncoder

BACKBONE_CLASSES = dict(
    sorted(
        {
            backbone_class.__name__: backbone_class
            for backbone_class in (GPSEncoder, GraphMLP, NSDEncoder)
        }.items()
    )
)

globals().update(BACKBONE_CLASSES)

__all__ = [*BACKBONE_CLASSES, "BACKBONE_CLASSES"]

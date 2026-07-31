"""Backbones for the heterogeneous graph domain."""

from .heterosage import HeteroSAGEBackbone
from .hgt import HGTBackbone

BACKBONE_CLASSES = dict(
    sorted(
        {
            backbone_class.__name__: backbone_class
            for backbone_class in (HGTBackbone, HeteroSAGEBackbone)
        }.items()
    )
)

globals().update(BACKBONE_CLASSES)

__all__ = [*BACKBONE_CLASSES, "BACKBONE_CLASSES"]

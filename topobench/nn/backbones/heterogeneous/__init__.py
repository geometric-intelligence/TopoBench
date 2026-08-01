"""Backbones for the heterogeneous graph domain."""

from .heterosage import HeteroSAGEBackbone
from .hgt import HGTBackbone

BACKBONE_CLASSES = {
    "HGTBackbone": HGTBackbone,
    "HeteroSAGEBackbone": HeteroSAGEBackbone,
}

__all__ = [
    "HGTBackbone",
    "HeteroSAGEBackbone",
    "BACKBONE_CLASSES",
]

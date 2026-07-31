"""Reusable backbones for native heterogeneous graphs."""

from topobench.nn.backbones.heterogeneous.common import (
    validate_backbone_arguments,
    validate_forward_dictionaries,
)
from topobench.nn.backbones.heterogeneous.hgt import HGTBackbone

__all__ = [
    "HGTBackbone",
    "validate_backbone_arguments",
    "validate_forward_dictionaries",
]

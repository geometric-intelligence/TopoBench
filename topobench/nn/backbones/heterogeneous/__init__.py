"""Reusable backbones for native heterogeneous graphs."""

from topobench.nn.backbones.heterogeneous.common import (
    validate_backbone_arguments,
    validate_forward_dictionaries,
)
from topobench.nn.backbones.heterogeneous.heterosage import (
    HeteroSAGEBackbone,
)
from topobench.nn.backbones.heterogeneous.hgt import HGTBackbone

__all__ = [
    "HGTBackbone",
    "HeteroSAGEBackbone",
    "validate_backbone_arguments",
    "validate_forward_dictionaries",
]

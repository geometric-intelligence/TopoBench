"""Explicit public readout registry."""

from .heterogeneous_node import HeterogeneousNodeReadout
from .identical import NoReadOut
from .mlp_readout import MLPReadout

READOUT_CLASSES = {
    "HeterogeneousNodeReadout": HeterogeneousNodeReadout,
    "MLPReadout": MLPReadout,
    "NoReadOut": NoReadOut,
}

__all__ = [
    "HeterogeneousNodeReadout",
    "MLPReadout",
    "NoReadOut",
    "READOUT_CLASSES",
]

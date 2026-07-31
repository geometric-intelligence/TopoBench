"""Explicit public readout registry."""

from .base import AbstractZeroCellReadOut
from .heterogeneous_node import HeterogeneousNodeReadout
from .hopse import HOPSEReadout
from .identical import NoReadOut
from .mlp_readout import MLPReadout
from .propagate_signal_down import PropagateSignalDown

READOUT_CLASSES = dict(
    sorted(
        {
            readout.__name__: readout
            for readout in (
                AbstractZeroCellReadOut,
                HeterogeneousNodeReadout,
                HOPSEReadout,
                MLPReadout,
                NoReadOut,
                PropagateSignalDown,
            )
        }.items()
    )
)

__all__ = [*READOUT_CLASSES, "READOUT_CLASSES"]

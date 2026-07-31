"""Init file for data module."""

from .heterogeneous import (
    HeterogeneousDataSpec,
    validate_heterogeneous_node_data,
)

__all__ = [
    "HeterogeneousDataSpec",
    "validate_heterogeneous_node_data",
]

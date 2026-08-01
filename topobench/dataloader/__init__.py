"""This module implements the dataloader for the topobench package."""

from .graph import GraphDataModule
from .heterogeneous import HeterogeneousNodeDataModule

__all__ = ["GraphDataModule", "HeterogeneousNodeDataModule"]

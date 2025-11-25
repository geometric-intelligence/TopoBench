"""Init file for load module."""

from .base import AbstractLoader
from .cell import *
from .cell import __all__ as cell_all
from .graph import *
from .graph import __all__ as graph_all
from .hypergraph import *
from .hypergraph import __all__ as hypergraph_all
from .pointcloud import *
from .pointcloud import __all__ as pointcloud_all
from .simplicial import *
from .simplicial import __all__ as simplicial_all

__all__ = [
    "AbstractLoader",
    *cell_all,
    *graph_all,
    *hypergraph_all,
    *simplicial_all,
    *pointcloud_all,
]

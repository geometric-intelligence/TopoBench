"""This module contains the transforms for the topobench package."""

from typing import Any

from topobench.transforms.data_manipulations import DATA_MANIPULATIONS
from topobench.transforms.feature_liftings import FEATURE_LIFTINGS
from topobench.transforms.liftings.graph2cell import GRAPH2CELL_LIFTINGS
from topobench.transforms.liftings.graph2combinatorial import (
    GRAPH2COMBINATORIAL_LIFTINGS,
)
from topobench.transforms.liftings.graph2hypergraph import (
    GRAPH2HYPERGRAPH_LIFTINGS,
)
from topobench.transforms.liftings.graph2simplicial import (
    GRAPH2SIMPLICIAL_LIFTINGS,
)
from topobench.transforms.liftings.hypergraph2combinatorial import (
    HYPERGRAPH2COMBINATORIAL_LIFTINGS,
)
from topobench.transforms.liftings.pointcloud2hypergraph import (
    POINTCLOUD2HYPERGRAPH_LIFTINGS,
)
from topobench.transforms.liftings.pointcloud2simplicial import (
    POINTCLOUD2SIMPLICIAL_LIFTINGS,
)
from topobench.transforms.liftings.simplicial2combinatorial import (
    SIMPLICIAL2COMBINATORIAL_LIFTINGS,
)

LIFTINGS = {
    **GRAPH2CELL_LIFTINGS,
    **GRAPH2HYPERGRAPH_LIFTINGS,
    **GRAPH2SIMPLICIAL_LIFTINGS,
    **POINTCLOUD2HYPERGRAPH_LIFTINGS,
    **POINTCLOUD2SIMPLICIAL_LIFTINGS,
    **GRAPH2COMBINATORIAL_LIFTINGS,
    **HYPERGRAPH2COMBINATORIAL_LIFTINGS,
    **SIMPLICIAL2COMBINATORIAL_LIFTINGS,
}

TRANSFORMS: dict[Any, Any] = {
    **LIFTINGS,
    **FEATURE_LIFTINGS,
    **DATA_MANIPULATIONS,
}

__all__ = [
    "DATA_MANIPULATIONS",
    "FEATURE_LIFTINGS",
    "LIFTINGS",
    "TRANSFORMS",
]

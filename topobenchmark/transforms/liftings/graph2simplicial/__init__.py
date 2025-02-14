"""Graph2SimplicialLifting module with automated exports."""

from topobenchmark.transforms._utils import discover_objs
from topobenchmark.transforms.liftings.base import LiftingMap

GRAPH2SIMPLICIAL_LIFTINGS = discover_objs(
    __file__,
    condition=lambda name, obj: issubclass(obj, LiftingMap),
)

locals().update(GRAPH2SIMPLICIAL_LIFTINGS)

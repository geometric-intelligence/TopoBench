"""Feature lifting transforms with automated exports."""

from topobenchmark.transforms._utils import discover_objs

from .base import FeatureLiftingMap

FEATURE_LIFTINGS = discover_objs(
    __file__,
    condition=lambda name, obj: issubclass(obj, FeatureLiftingMap),
)

locals().update(FEATURE_LIFTINGS)

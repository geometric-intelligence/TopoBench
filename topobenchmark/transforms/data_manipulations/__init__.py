"""Data manipulations module with automated exports."""

from topobenchmark.transforms._utils import discover_objs

DATA_MANIPULATIONS = discover_objs(__file__)

locals().update(DATA_MANIPULATIONS)

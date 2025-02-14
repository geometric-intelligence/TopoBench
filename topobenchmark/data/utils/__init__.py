"""Init file for data/utils module."""

from .adapters import *
from .domain import ComplexData, HypergraphData  # noqa: F401
from .utils import (
    ensure_serializable,  # noqa: F401
    generate_zero_sparse_connectivity,  # noqa: F401
    get_complex_connectivity,  # noqa: F401
    get_routes_from_neighborhoods,  # noqa: F401
    load_cell_complex_dataset,  # noqa: F401
    load_manual_graph,  # noqa: F401
    load_simplicial_dataset,  # noqa: F401
    make_hash,  # noqa: F401
    select_neighborhoods_of_interest,  # noqa: F401
)

utils_functions = [
    "get_complex_connectivity",
    "get_routes_from_neighborhoods",
    "generate_zero_sparse_connectivity",
    "load_cell_complex_dataset",
    "load_simplicial_dataset",
    "load_manual_graph",
    "make_hash",
    "ensure_serializable",
    "select_neighborhoods_of_interest",
]

from .split_utils import (  # noqa: E402
    load_coauthorship_hypergraph_splits,  # noqa: F401
    load_inductive_splits,  # noqa: F401
    load_transductive_splits,  # noqa: F401
)

split_helper_functions = [
    "load_coauthorship_hypergraph_splits",
    "load_inductive_splits",
    "load_transductive_splits",
]

from .io_utils import (  # noqa: E402
    download_file_from_drive,  # noqa: F401
    download_file_from_link,  # noqa: F401
    load_hypergraph_pickle_dataset,  # noqa: F401
    read_ndim_manifolds,  # noqa: F401
    read_us_county_demos,  # noqa: F401
)

io_helper_functions = [
    "load_hypergraph_pickle_dataset",
    "read_us_county_demos",
    "download_file_from_drive",
]

__all__ = utils_functions + split_helper_functions + io_helper_functions

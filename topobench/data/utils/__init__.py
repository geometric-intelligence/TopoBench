"""Dependency-light data utility exports for native loaders."""

from .common import ensure_serializable, make_hash
from .downloads import (
    download_file_from_drive,
    download_file_from_link,
    get_file_id_from_url,
)
from .hypergraph_io import (
    incidence_pairs,
    load_hypergraph_content_dataset,
    load_hypergraph_pickle_dataset,
)
from .split_utils import (
    load_coauthorship_hypergraph_splits,
    load_inductive_splits,
    load_transductive_splits,
)

__all__ = [
    "download_file_from_drive",
    "download_file_from_link",
    "ensure_serializable",
    "get_file_id_from_url",
    "incidence_pairs",
    "load_coauthorship_hypergraph_splits",
    "load_hypergraph_content_dataset",
    "load_hypergraph_pickle_dataset",
    "load_inductive_splits",
    "load_transductive_splits",
    "make_hash",
]

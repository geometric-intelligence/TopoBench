"""Dependency-light data utility exports for native loaders."""

from .common import ensure_serializable, make_hash
from .downloads import (
    ArchiveLimits,
    RemoteArchive,
    acquire_verified_archive,
)
from .hypergraph_io import (
    SAFE_HYPERGRAPH_CONVERTER_VERSION,
    SAFE_HYPERGRAPH_FORMAT,
    SAFE_HYPERGRAPH_FORMAT_VERSION,
    ContentRoleSpec,
    incidence_pairs,
    load_hypergraph_content_dataset,
    load_hypergraph_npz_dataset,
    validate_hypergraph_npz_assets,
)
from .split_utils import (
    load_coauthorship_hypergraph_splits,
    load_inductive_splits,
    load_transductive_splits,
)

__all__ = [
    "ArchiveLimits",
    "RemoteArchive",
    "acquire_verified_archive",
    "ContentRoleSpec",
    "SAFE_HYPERGRAPH_CONVERTER_VERSION",
    "SAFE_HYPERGRAPH_FORMAT",
    "SAFE_HYPERGRAPH_FORMAT_VERSION",
    "ensure_serializable",
    "incidence_pairs",
    "load_coauthorship_hypergraph_splits",
    "load_hypergraph_content_dataset",
    "load_hypergraph_npz_dataset",
    "validate_hypergraph_npz_assets",
    "load_inductive_splits",
    "load_transductive_splits",
    "make_hash",
]

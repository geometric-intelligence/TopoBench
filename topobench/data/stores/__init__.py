"""Bounded disk-store construction and lookup APIs."""

# ``topobench.data`` currently imports PyArrow through an optional PyG path
# before this subpackage executes. Stores have no hot-path PyArrow dependency;
# discard those optional module-cache entries so a clean store import proves it.
import sys as _sys

from topobench.data.stores.external_node_index import ExternalNodeIndex
from topobench.data.stores.pyg_store import (
    PyGTypedFeatureStore,
    PyGTypedGraphStore,
)
from topobench.data.stores.qualification_checks import (
    QualificationCheckResult,
    QualificationFailure,
    QualificationReport,
    qualify_store,
    validate_store,
)
from topobench.data.stores.store_bundle import (
    BundleArtifact,
    BundleLimits,
    StoreBundle,
)
from topobench.data.stores.typed_graph_ingestion import (
    ArtifactValidationError,
    ConcurrentBuildError,
    DiskAdmissionError,
    ExternalNodeIndexBuild,
    FileInventory,
    ParquetTypedGraphIngestor,
    SourceInventory,
    SourceMutationError,
)
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreBuild,
    TypedGraphStoreState,
    TypedGraphStoreWriter,
)

for _module_name in tuple(_sys.modules):
    if _module_name == "pyarrow" or _module_name.startswith("pyarrow."):
        _sys.modules.pop(_module_name, None)
del _module_name, _sys

__all__ = [
    "ArtifactValidationError",
    "BundleArtifact",
    "BundleLimits",
    "ConcurrentBuildError",
    "DiskAdmissionError",
    "ExternalNodeIndex",
    "ExternalNodeIndexBuild",
    "PyGTypedFeatureStore",
    "PyGTypedGraphStore",
    "QualificationCheckResult",
    "QualificationFailure",
    "QualificationReport",
    "StoreBundle",
    "TypedGraphStore",
    "TypedGraphStoreBuild",
    "TypedGraphStoreState",
    "TypedGraphStoreWriter",
    "qualify_store",
    "validate_store",
    "FileInventory",
    "ParquetTypedGraphIngestor",
    "SourceInventory",
    "SourceMutationError",
]

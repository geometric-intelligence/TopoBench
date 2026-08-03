"""Immutable executable qualification metadata for retained datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

QualificationGate = Literal["packaged", "download"]
TaskKind = Literal["classification", "regression"]
TaskLevel = Literal["graph", "node"]
SplitMode = Literal["inductive", "transductive"]
SplitType = Literal["fixed", "random"]
FeaturePolicy = Literal[
    "continuous",
    "categorical_one_hot",
    "degree",
    "constant",
    "continuous_per_node_type",
    "continuous_with_constant_fill",
]
EdgePolicy = Literal[
    "structural_edges",
    "edge_attr_available",
    "native_typed_relations",
    "typed_relations_with_reverse",
    "hyperedge_incidence",
]

_EVIDENCE_TEST = (
    "test/integration/test_retained_datasets.py::"
    "test_retained_dataset_lifecycle"
)
_GRAPH_FIXED_SPLITS = frozenset(
    {
        "AQSOL",
        "BBB_Martins",
        "CYP3A4_Veith",
        "Caco2_Wang",
        "Clearance_Hepatocyte_AZ",
        "ParquetTypedGraph",
        "SyntheticGraph",
        "SyntheticGraphRegression",
        "SyntheticNodeGraph",
        "ZINC",
        "ZINC_OGB",
        "ogbg-molhiv",
    }
)
_GRAPH_FEATURE_WIDTHS: Mapping[str, int] = MappingProxyType(
    {
        "AQSOL": 21,
        "BBB_Martins": 9,
        "CYP3A4_Veith": 9,
        "Caco2_Wang": 9,
        "Clearance_Hepatocyte_AZ": 9,
        "IMDB-BINARY": 136,
        "IMDB-MULTI": 89,
        "MUTAG": 7,
        "NCI1": 37,
        "NCI109": 38,
        "PROTEINS": 3,
        "ParquetTypedGraph": 2,
        "QM9": 11,
        "REDDIT-BINARY": 10,
        "SyntheticGraph": 4,
        "SyntheticGraphRegression": 4,
        "SyntheticNodeGraph": 4,
        "ZINC": 21,
        "ZINC_OGB": 21,
        "amazon_ratings": 300,
        "cocitation_citeseer": 3703,
        "cocitation_cora": 1433,
        "cocitation_pubmed": 500,
        "graphuniverse_inductive_triangle": 15,
        "graphuniverse_transductive": 15,
        "minesweeper": 7,
        "ogbg-molhiv": 9,
        "questions": 301,
        "roman_empire": 300,
        "tolokers": 10,
    }
)
_GRAPH_CLASS_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "IMDB-MULTI": 3,
        "amazon_ratings": 5,
        "cocitation_citeseer": 6,
        "cocitation_cora": 7,
        "cocitation_pubmed": 3,
        "graphuniverse_transductive": 10,
        "roman_empire": 18,
    }
)
_HETEROGENEOUS_FEATURE_WIDTHS: Mapping[str, tuple[tuple[str, int], ...]] = (
    MappingProxyType(
        {
            "DBLP": (
                ("author", 334),
                ("paper", 4231),
                ("term", 50),
                ("conference", 1),
            ),
            "OGB_MAG": (
                ("paper", 128),
                ("author", 128),
                ("institution", 128),
                ("field_of_study", 128),
            ),
            "ParquetTypedGraph": (("author", 3), ("paper", 128)),
            "SyntheticHeterogeneous": (
                ("author", 8),
                ("paper", 5),
                ("venue", 1),
            ),
        }
    )
)


@dataclass(frozen=True, slots=True)
class DatasetQualification:
    """One dataset selector's executable product-qualification contract."""

    selector: str
    loader_family: str
    gate: QualificationGate
    task: TaskKind
    task_level: TaskLevel
    split_mode: SplitMode
    split_type: SplitType
    feature_policy: FeaturePolicy
    edge_policy: EdgePolicy
    compatible_model: str
    evidence_test: str
    num_classes: int | None = None
    target_node_type: str | None = None
    feature_widths: tuple[tuple[str, int], ...] = ()


def _qualification(
    selector: str,
    *,
    loader_family: str,
    gate: QualificationGate,
    task: TaskKind,
    task_level: TaskLevel,
    split_mode: SplitMode,
    feature_policy: FeaturePolicy,
    edge_policy: EdgePolicy,
    compatible_model: str,
    num_classes: int | None = None,
    target_node_type: str | None = None,
    split_type: SplitType | None = None,
    feature_widths: tuple[tuple[str, int], ...] = (),
    evidence_test: str = _EVIDENCE_TEST,
) -> DatasetQualification:
    """Attach exact static execution evidence to one selector."""
    data_domain, _, data_name = selector.partition("/")
    if not feature_widths:
        if data_domain == "graph":
            feature_widths = (("node", _GRAPH_FEATURE_WIDTHS[data_name]),)
        elif data_domain == "heterogeneous":
            feature_widths = _HETEROGENEOUS_FEATURE_WIDTHS[data_name]
        elif data_domain == "hypergraph":
            feature_widths = (("node", 4),)
    if split_type is None:
        split_type = (
            "fixed"
            if data_domain != "graph" or data_name in _GRAPH_FIXED_SPLITS
            else "random"
        )
    if num_classes is None:
        num_classes = (
            1
            if task == "regression"
            else _GRAPH_CLASS_COUNTS.get(data_name, 2)
        )
    return DatasetQualification(
        selector=selector,
        loader_family=loader_family,
        gate=gate,
        task=task,
        task_level=task_level,
        split_mode=split_mode,
        split_type=split_type,
        feature_policy=feature_policy,
        edge_policy=edge_policy,
        compatible_model=compatible_model,
        evidence_test=evidence_test,
        num_classes=num_classes,
        target_node_type=target_node_type,
        feature_widths=feature_widths,
    )


_ROWS = (
    _qualification(
        "graph/AQSOL",
        loader_family="topobench.data.loaders.graph.MoleculeDatasetLoader",
        gate="download",
        task="regression",
        task_level="graph",
        split_mode="inductive",
        feature_policy="degree",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/BBB_Martins",
        loader_family="topobench.data.loaders.ADMEDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="categorical_one_hot",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/CYP3A4_Veith",
        loader_family="topobench.data.loaders.ADMEDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="categorical_one_hot",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/Caco2_Wang",
        loader_family="topobench.data.loaders.ADMEDatasetLoader",
        gate="download",
        task="regression",
        task_level="graph",
        split_mode="inductive",
        feature_policy="categorical_one_hot",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/Clearance_Hepatocyte_AZ",
        loader_family="topobench.data.loaders.ADMEDatasetLoader",
        gate="download",
        task="regression",
        task_level="graph",
        split_mode="inductive",
        feature_policy="categorical_one_hot",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/IMDB-BINARY",
        loader_family="topobench.data.loaders.TUDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="degree",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/IMDB-MULTI",
        loader_family="topobench.data.loaders.TUDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="degree",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/MUTAG",
        loader_family="topobench.data.loaders.TUDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="continuous",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/NCI1",
        loader_family="topobench.data.loaders.TUDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/NCI109",
        loader_family="topobench.data.loaders.TUDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/PROTEINS",
        loader_family="topobench.data.loaders.TUDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/ParquetTypedGraph",
        loader_family=(
            "topobench.data.loaders.parquet.ParquetTypedGraphLoader"
        ),
        gate="packaged",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
        num_classes=2,
        target_node_type="paper",
        evidence_test=(
            "test/data/stores/test_typed_graph_store.py::"
            "test_opens_homogeneous_and_heterogeneous_content_addressed_stores"
        ),
    ),
    _qualification(
        "graph/QM9",
        loader_family="topobench.data.loaders.graph.MoleculeDatasetLoader",
        gate="download",
        task="regression",
        task_level="graph",
        split_mode="inductive",
        feature_policy="continuous",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/REDDIT-BINARY",
        loader_family="topobench.data.loaders.TUDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="constant",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/SyntheticGraph",
        loader_family="topobench.data.loaders.SyntheticGraphDatasetLoader",
        gate="packaged",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/SyntheticGraphRegression",
        loader_family="topobench.data.loaders.SyntheticGraphDatasetLoader",
        gate="packaged",
        task="regression",
        task_level="graph",
        split_mode="inductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/SyntheticNodeGraph",
        loader_family="topobench.data.loaders.SyntheticGraphDatasetLoader",
        gate="packaged",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/ZINC",
        loader_family="topobench.data.loaders.MoleculeDatasetLoader",
        gate="download",
        task="regression",
        task_level="graph",
        split_mode="inductive",
        feature_policy="categorical_one_hot",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/ZINC_OGB",
        loader_family="topobench.data.loaders.MoleculeDatasetLoader",
        gate="download",
        task="regression",
        task_level="graph",
        split_mode="inductive",
        feature_policy="categorical_one_hot",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/amazon_ratings",
        loader_family=(
            "topobench.data.loaders.HeterophilousGraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/cocitation_citeseer",
        loader_family="topobench.data.loaders.PlanetoidDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/cocitation_cora",
        loader_family="topobench.data.loaders.PlanetoidDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/cocitation_pubmed",
        loader_family="topobench.data.loaders.PlanetoidDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/graphuniverse_inductive_triangle",
        loader_family="topobench.data.loaders.GraphUniverseDatasetLoader",
        gate="download",
        task="regression",
        task_level="graph",
        split_mode="inductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/graphuniverse_transductive",
        loader_family="topobench.data.loaders.GraphUniverseDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/minesweeper",
        loader_family=(
            "topobench.data.loaders.HeterophilousGraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/ogbg-molhiv",
        loader_family="topobench.data.loaders.OGBGDatasetLoader",
        gate="download",
        task="classification",
        task_level="graph",
        split_mode="inductive",
        feature_policy="categorical_one_hot",
        edge_policy="edge_attr_available",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/questions",
        loader_family=(
            "topobench.data.loaders.HeterophilousGraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/roman_empire",
        loader_family=(
            "topobench.data.loaders.HeterophilousGraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "graph/tolokers",
        loader_family=(
            "topobench.data.loaders.HeterophilousGraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="structural_edges",
        compatible_model="graph/gcn",
    ),
    _qualification(
        "heterogeneous/DBLP",
        loader_family="topobench.data.loaders.DBLPDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous_with_constant_fill",
        edge_policy="native_typed_relations",
        compatible_model="heterogeneous/hgt",
        num_classes=4,
        target_node_type="author",
    ),
    _qualification(
        "heterogeneous/OGB_MAG",
        loader_family="topobench.data.loaders.OGBMAGDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous_per_node_type",
        edge_policy="typed_relations_with_reverse",
        compatible_model="heterogeneous/hgt",
        num_classes=349,
        target_node_type="paper",
    ),
    _qualification(
        "heterogeneous/SyntheticHeterogeneous",
        loader_family=(
            "topobench.data.loaders.SyntheticHeterogeneousDatasetLoader"
        ),
        gate="packaged",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous_with_constant_fill",
        edge_policy="typed_relations_with_reverse",
        compatible_model="heterogeneous/hgt",
        num_classes=2,
        target_node_type="author",
    ),
    _qualification(
        "heterogeneous/ParquetTypedGraph",
        loader_family=(
            "topobench.data.loaders.parquet.ParquetTypedGraphLoader"
        ),
        gate="packaged",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous_per_node_type",
        edge_policy="native_typed_relations",
        compatible_model="heterogeneous/hgt",
        num_classes=2,
        target_node_type="paper",
        evidence_test=(
            "test/integration/test_heterogeneous_disk_resume.py::"
            "test_heterogeneous_cluster_and_neighbor_resume_exactly_at_every_boundary"
        ),
    ),
    _qualification(
        "hypergraph/SyntheticHypergraph",
        loader_family=(
            "topobench.data.loaders.hypergraph.synthetic."
            "SyntheticHypergraphDatasetLoader"
        ),
        gate="packaged",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
)

DATASET_QUALIFICATION_MANIFEST: Mapping[str, DatasetQualification] = (
    MappingProxyType({row.selector: row for row in _ROWS})
)


__all__ = [
    "DATASET_QUALIFICATION_MANIFEST",
    "DatasetQualification",
    "SplitType",
]

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


@dataclass(frozen=True, slots=True)
class DatasetQualification:
    """One dataset selector's executable product-qualification contract."""

    selector: str
    loader_family: str
    gate: QualificationGate
    task: TaskKind
    task_level: TaskLevel
    split_mode: SplitMode
    feature_policy: FeaturePolicy
    edge_policy: EdgePolicy
    compatible_model: str
    evidence_test: str
    num_classes: int | None = None
    target_node_type: str | None = None


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
    evidence_test: str = _EVIDENCE_TEST,
) -> DatasetQualification:
    """Attach the shared parameterized evidence node to one explicit row."""
    return DatasetQualification(
        selector=selector,
        loader_family=loader_family,
        gate=gate,
        task=task,
        task_level=task_level,
        split_mode=split_mode,
        feature_policy=feature_policy,
        edge_policy=edge_policy,
        compatible_model=compatible_model,
        evidence_test=evidence_test,
        num_classes=num_classes,
        target_node_type=target_node_type,
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
        "hypergraph/20newsgroup",
        loader_family="topobench.data.loaders.HypergraphDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
    _qualification(
        "hypergraph/ModelNet40",
        loader_family="topobench.data.loaders.HypergraphDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
    _qualification(
        "hypergraph/Mushroom",
        loader_family="topobench.data.loaders.HypergraphDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
    _qualification(
        "hypergraph/NTU2012",
        loader_family="topobench.data.loaders.HypergraphDatasetLoader",
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
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
    _qualification(
        "hypergraph/coauthorship_cora",
        loader_family=(
            "topobench.data.loaders.CitationHypergraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
    _qualification(
        "hypergraph/coauthorship_dblp",
        loader_family=(
            "topobench.data.loaders.CitationHypergraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
    _qualification(
        "hypergraph/cocitation_citeseer",
        loader_family=(
            "topobench.data.loaders.CitationHypergraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
    _qualification(
        "hypergraph/cocitation_cora",
        loader_family=(
            "topobench.data.loaders.CitationHypergraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
    _qualification(
        "hypergraph/cocitation_pubmed",
        loader_family=(
            "topobench.data.loaders.CitationHypergraphDatasetLoader"
        ),
        gate="download",
        task="classification",
        task_level="node",
        split_mode="transductive",
        feature_policy="continuous",
        edge_policy="hyperedge_incidence",
        compatible_model="hypergraph/edgnn",
    ),
    _qualification(
        "hypergraph/zoo",
        loader_family="topobench.data.loaders.HypergraphDatasetLoader",
        gate="download",
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
]

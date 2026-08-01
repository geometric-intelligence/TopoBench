"""Datasets exposed by the supported data domains."""

from .citation_hypergraph_dataset import CitationHypergraphDataset
from .hypergraph_datasets import HypergraphDataset
from .synthetic_graph_dataset import SyntheticGraphDataset
from .synthetic_heterogeneous_dataset import (
    SyntheticHeterogeneousDataset,
    make_synthetic_heterogeneous_data,
)
from .synthetic_hypergraph_dataset import (
    SyntheticHypergraphDataset,
    make_synthetic_hypergraph_data,
)

PLANETOID_DATASETS = ["Cora", "citeseer", "PubMed"]

TU_DATASETS = [
    "MUTAG",
    "ENZYMES",
    "PROTEINS",
    "COLLAB",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "NCI1",
    "NCI109",
]

FIXED_SPLITS_DATASETS = ["ZINC", "AQSOL"]

HETEROPHILIC_DATASETS = [
    "amazon_ratings",
    "questions",
    "minesweeper",
    "roman_empire",
    "tolokers",
]

PYG_DATASETS = [
    *PLANETOID_DATASETS,
    *TU_DATASETS,
    *FIXED_SPLITS_DATASETS,
    *HETEROPHILIC_DATASETS,
]


__all__ = [
    "CitationHypergraphDataset",
    "FIXED_SPLITS_DATASETS",
    "HETEROPHILIC_DATASETS",
    "HypergraphDataset",
    "PLANETOID_DATASETS",
    "PYG_DATASETS",
    "SyntheticGraphDataset",
    "SyntheticHeterogeneousDataset",
    "SyntheticHypergraphDataset",
    "TU_DATASETS",
    "make_synthetic_heterogeneous_data",
    "make_synthetic_hypergraph_data",
]

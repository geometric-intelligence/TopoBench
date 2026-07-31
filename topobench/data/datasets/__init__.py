"""Datasets exposed by the supported data domains."""

from .citation_hypergraph_dataset import CitationHypergraphDataset
from .hypergraph_datasets import HypergraphDataset
from .synthetic_graph_dataset import SyntheticGraphDataset
from .synthetic_heterogeneous_dataset import (
    SyntheticHeterogeneousDataset,
    make_synthetic_heterogeneous_data,
)
from .us_county_demos_dataset import USCountyDemosDataset

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

MANUAL_DATASETS = dict(
    sorted(
        {
            dataset_class.__name__: dataset_class
            for dataset_class in (
                CitationHypergraphDataset,
                HypergraphDataset,
                SyntheticGraphDataset,
                SyntheticHeterogeneousDataset,
                USCountyDemosDataset,
            )
        }.items()
    )
)

globals().update(MANUAL_DATASETS)

__all__ = [
    "PYG_DATASETS",
    "PLANETOID_DATASETS",
    "TU_DATASETS",
    "FIXED_SPLITS_DATASETS",
    "HETEROPHILIC_DATASETS",
    *MANUAL_DATASETS,
    "MANUAL_DATASETS",
    "make_synthetic_heterogeneous_data",
]

"""This module contains the implemented dataset classes for the topological benchmarking experiments."""

from .citation_hypergaph_dataset import CitationHypergraphDataset
from .us_county_demos_dataset import USCountyDemosDataset
from .h36m_dataset import H36MDataset

PLANETOID_DATASETS = [
    "Cora",
    "citeseer",
    "PubMed",
]

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

PYG_DATASETS = (
    PLANETOID_DATASETS
    + TU_DATASETS
    + FIXED_SPLITS_DATASETS
    + HETEROPHILIC_DATASETS
)

__all__ = [
    "USCountyDemosDataset",
    "PYG_DATASETS",
    "PLANETOID_DATASETS",
    "TU_DATASETS",
    "FIXED_SPLITS_DATASETS",
    "HETEROPHILIC_DATASETS",
    "CitationHypergraphDataset",
    "H36MDataset",
]

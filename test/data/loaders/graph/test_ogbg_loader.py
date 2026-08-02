"""Focused tests for molecular OGB graph loading without downloads."""

from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.data.loaders.base import (
    CACHE_SOURCE_ATTRIBUTE,
    canonical_sha256,
)
from topobench.data.loaders.graph.ogbg_datasets import OGBGDatasetLoader


class _FakeOGBGDataset:
    def __init__(self, name: str, root: str) -> None:
        self.name = name
        self.root = root
        self._data = Data(
            x=torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [118, 4, 11, 11, 9, 5, 5, 1, 1],
                ]
            ),
            y=torch.tensor([[1], [0]]),
        )

    @staticmethod
    def get_idx_split() -> dict[str, torch.Tensor]:
        return {
            "train": torch.tensor([0]),
            "valid": torch.tensor([], dtype=torch.long),
            "test": torch.tensor([1]),
        }


def test_ogbg_loader_keeps_atom_categories_compact(tmp_path) -> None:
    parameters = OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": "ogbg-molhiv",
            "data_type": "OGBGDataset",
        }
    )

    with patch(
        "topobench.data.loaders.graph.ogbg_datasets.PygGraphPropPredDataset",
        _FakeOGBGDataset,
    ):
        dataset = OGBGDatasetLoader(parameters).load_dataset()

    expected = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [118, 4, 11, 11, 9, 5, 5, 1, 1],
        ]
    )
    torch.testing.assert_close(dataset._data.x, expected)
    assert dataset._data.x.shape == (2, 9)
    assert dataset._data.x.dtype == torch.long
    assert dataset.feature_encoding == "categorical_one_hot"
    assert dataset.feature_cardinalities == (
        119,
        5,
        12,
        12,
        10,
        6,
        6,
        2,
        2,
    )
    assert dataset._data.y.shape == (2,)
    assert dataset.split_idx["train"].tolist() == [0]


def test_compact_representation_changes_processed_cache_identity(
    tmp_path,
) -> None:
    def cache_record(representation_version: str | None) -> dict:
        parameters = {
            "data_dir": str(tmp_path),
            "data_domain": "graph",
            "data_name": "ogbg-molhiv",
            "data_type": "OGBGDataset",
        }
        if representation_version is not None:
            parameters["representation_version"] = representation_version
        dataset, _ = OGBGDatasetLoader(
            OmegaConf.create(parameters)
        ).load()
        source = getattr(dataset, CACHE_SOURCE_ATTRIBUTE)
        return {
            "schema": "topobench.processed-cache",
            "schema_version": 2,
            **source,
            "transform": {"steps": []},
        }

    with patch(
        "topobench.data.loaders.graph.ogbg_datasets.PygGraphPropPredDataset",
        _FakeOGBGDataset,
    ):
        previous = cache_record(None)
        compact_first = cache_record("molecular-compact-categorical-v1")
        compact_second = cache_record("molecular-compact-categorical-v1")

    previous_key = canonical_sha256(previous)
    compact_key = canonical_sha256(compact_first)
    assert previous["versions"]["representation"] is None
    assert (
        compact_first["versions"]["representation"]
        == "molecular-compact-categorical-v1"
    )
    assert compact_first == compact_second
    assert previous_key != compact_key
    assert tmp_path / previous_key != tmp_path / compact_key

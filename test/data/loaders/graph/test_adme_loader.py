"""Tests for ``ADMEDatasetLoader`` (no network).

The real ``load_dataset`` downloads from TDC over the network, which we
don't want to hit in CI. These tests focus on the parts of the loader
that we can exercise locally: construction, ``get_data_dir``, the
classification/regression bookkeeping, and the unknown-name branch.

The ``test_load_dataset_*`` tests mock out ``tdc.single_pred.ADME`` and
``ogb.utils.smiles2graph`` so no network requests are made.
"""

import hashlib
import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from topobench.data.capabilities import GRAPH_DATASET_MANIFEST
from topobench.data.features import validate_qualified_graph_source
from topobench.data.loaders.graph.adme_datasets import ADMEDatasetLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    tmp_path,
    data_name="BBB_Martins",
    *,
    split_method="scaffold",
    split_seed=0,
):
    return OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": data_name,
            "data_type": "ADME",
            "split_method": split_method,
            "split_seed": split_seed,
        }
    )


def _fake_smiles2graph(_smiles):
    """Minimal graph dict returned by ogb's smiles2graph."""
    return {
        "num_nodes": 3,
        "node_feat": [
            [1, 2, 3, 4, 5, 1, 2, 1, 0],
            [6, 0, 0, 0, 0, 0, 0, 0, 1],
            [118, 4, 11, 11, 9, 5, 5, 1, 1],
        ],
        "edge_index": [[0, 1], [1, 0]],
        "edge_feat": [[1, 0, 0], [1, 0, 0]],
    }


def _fake_tdc_adme(name, path):
    """Return a mock ADME object whose get_split() yields tiny DataFrames."""
    df_train = pd.DataFrame({"Drug": ["C", "CC"], "Y": [1, 0]})
    df_valid = pd.DataFrame({"Drug": ["CCC"], "Y": [1]})
    df_test = pd.DataFrame({"Drug": ["CCCC"], "Y": [0]})
    mock = MagicMock()
    mock.get_split.return_value = {
        "train": df_train,
        "valid": df_valid,
        "test": df_test,
    }
    return mock


def _seeded_split(seed):
    """Return a deterministic partition of one fixed source dataset."""
    source = pd.DataFrame(
        {
            "Drug": ["C", "CC", "CCC", "CCCC"],
            "Y": [0, 1, 0, 1],
        },
        index=[101, 203, 307, 409],
    )
    order = np.random.default_rng(seed).permutation(len(source))
    return {
        "train": source.iloc[order[:2]],
        "valid": source.iloc[order[2:3]],
        "test": source.iloc[order[3:]],
    }


def _rng_mutating_tdc(name, path):
    """Emulate a dependency that consumes caller-global RNG internally."""
    del name, path
    mock = MagicMock()
    mock.version = "fake-dataset-v1"

    def get_split(*, method, seed):
        assert method == "scaffold"
        np.random.seed(seed + 100)
        np.random.random(3)
        torch.random.default_generator.manual_seed(seed + 100)
        torch.rand(3)
        return _seeded_split(seed)

    mock.get_split.side_effect = get_split
    return mock


def _assert_rng_unchanged(load):
    """Run ``load`` and assert NumPy/Torch caller streams are untouched."""
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    result = load()

    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_before)
    return result


def _canonical_digest(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Basic unit tests (no mocking needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def loader(tmp_path):
    return ADMEDatasetLoader(_make_cfg(tmp_path))


def test_repr(loader):
    assert "ADMEDatasetLoader" in repr(loader)


def test_get_data_dir_combines_root_and_name(loader, tmp_path):
    assert loader.get_data_dir() == os.path.join(str(tmp_path), "BBB_Martins")


def test_load_dataset_rejects_unknown_name(tmp_path):
    cfg = _make_cfg(tmp_path, data_name="TotallyMadeUp")
    with pytest.raises(ValueError, match="Unknown ADME dataset"):
        ADMEDatasetLoader(cfg).load_dataset()


def test_load_dataset_rejects_non_scaffold_method(tmp_path):
    cfg = _make_cfg(tmp_path, split_method="random")
    with pytest.raises(ValueError, match="split_method.*scaffold"):
        ADMEDatasetLoader(cfg)


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
def test_load_dataset_calls_scaffold_split_with_configured_seed(
    mock_s2g, tmp_path
):
    tdc_dataset = _fake_tdc_adme("BBB_Martins", str(tmp_path))
    with patch(
        "topobench.data.loaders.graph.adme_datasets.ADME",
        return_value=tdc_dataset,
    ):
        dataset = ADMEDatasetLoader(
            _make_cfg(tmp_path, split_seed=37)
        ).load_dataset()

    assert len(dataset) == 4
    tdc_dataset.get_split.assert_called_once_with(
        method="scaffold",
        seed=37,
    )


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_rng_mutating_tdc,
)
def test_cache_miss_and_hit_preserve_rng_and_reuse_matching_provenance(
    mock_adme, mock_s2g, tmp_path
):
    cfg = _make_cfg(tmp_path, split_seed=7)

    cache_miss = _assert_rng_unchanged(
        lambda: ADMEDatasetLoader(cfg).load_dataset()
    )
    conversion_calls_after_miss = mock_s2g.call_count
    cache_hit = _assert_rng_unchanged(
        lambda: ADMEDatasetLoader(cfg).load_dataset()
    )

    assert mock_adme.call_count == 2
    assert conversion_calls_after_miss == 4
    assert mock_s2g.call_count == conversion_calls_after_miss
    assert cache_hit.split_provenance == cache_miss.split_provenance
    assert cache_hit.split_fingerprint == cache_miss.split_fingerprint


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_rng_mutating_tdc,
)
def test_valid_read_only_cache_hit_does_not_chmod_or_mutate_artifacts(
    mock_adme, mock_s2g, tmp_path
):
    del mock_adme
    cfg = _make_cfg(tmp_path, split_seed=7)
    first = ADMEDatasetLoader(cfg).load_dataset()
    processed_path = Path(first.processed_paths[0])
    provenance_path = Path(first.provenance_path)
    os.chmod(processed_path, 0o444)
    os.chmod(provenance_path, 0o444)

    def snapshot(path):
        file_stat = path.stat()
        return (
            stat.S_IMODE(file_stat.st_mode),
            file_stat.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    before = {
        path: snapshot(path) for path in (processed_path, provenance_path)
    }
    conversion_calls_after_miss = mock_s2g.call_count

    with patch.object(
        Path,
        "chmod",
        side_effect=AssertionError("cache hit attempted chmod"),
    ):
        cache_hit = _assert_rng_unchanged(
            lambda: ADMEDatasetLoader(cfg).load_dataset()
        )

    assert cache_hit.split_fingerprint == first.split_fingerprint
    assert mock_s2g.call_count == conversion_calls_after_miss
    assert {
        path: snapshot(path) for path in (processed_path, provenance_path)
    } == before


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_rng_mutating_tdc,
)
def test_split_fingerprint_is_seed_sensitive_and_repeat_stable(
    mock_adme, mock_s2g, tmp_path
):
    del mock_adme, mock_s2g
    seed_7 = _assert_rng_unchanged(
        lambda: ADMEDatasetLoader(
            _make_cfg(tmp_path, split_seed=7)
        ).load_dataset()
    )
    seed_11 = _assert_rng_unchanged(
        lambda: ADMEDatasetLoader(
            _make_cfg(tmp_path, split_seed=11)
        ).load_dataset()
    )
    seed_7_repeated = _assert_rng_unchanged(
        lambda: ADMEDatasetLoader(
            _make_cfg(tmp_path, split_seed=7)
        ).load_dataset()
    )

    assert seed_7.split_fingerprint != seed_11.split_fingerprint
    assert seed_7.split_fingerprint == seed_7_repeated.split_fingerprint
    assert (
        seed_7.split_provenance["source"]["data_digest"]
        == seed_11.split_provenance["source"]["data_digest"]
    )
    assert (
        seed_7.split_provenance["split"]["phase_index_digests"]
        != seed_11.split_provenance["split"]["phase_index_digests"]
    )
    assert (
        seed_7.split_provenance["split"]["phase_data_digests"]
        != seed_11.split_provenance["split"]["phase_data_digests"]
    )


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_rng_mutating_tdc,
)
def test_split_provenance_is_versioned_complete_and_non_executable(
    mock_adme, mock_s2g, tmp_path
):
    del mock_adme, mock_s2g
    seed = 7
    expected_split = _seeded_split(seed)
    dataset = ADMEDatasetLoader(
        _make_cfg(tmp_path, split_seed=seed)
    ).load_dataset()
    provenance = dataset.split_provenance

    assert provenance["provenance_version"] == 2
    assert provenance["source"]["provider"] == "PyTDC"
    assert "provider_version" in provenance["source"]
    assert provenance["source"]["dataset_name"] == "BBB_Martins"
    assert provenance["source"]["dataset_version"] == "fake-dataset-v1"
    assert len(provenance["source"]["data_digest"]) == 64
    assert provenance["split"]["method"] == "scaffold"
    assert provenance["split"]["seed"] == seed
    assert provenance["split"]["phase_index_digests"] == {
        phase: _canonical_digest(frame.index.tolist())
        for phase, frame in expected_split.items()
    }
    assert provenance["split"]["phase_data_digests"] == {
        phase: _canonical_digest(frame.to_dict(orient="records"))
        for phase, frame in expected_split.items()
    }
    assert provenance["representation"] == {
        "node_feature_encoding": "categorical_one_hot",
        "node_feature_cardinalities": [119, 5, 12, 12, 10, 6, 6, 2, 2],
        "stored_node_feature_width": 9,
        "encoded_node_feature_width": 174,
    }

    provenance_path = Path(dataset.provenance_path)
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == provenance
    assert provenance_path.stat().st_mode & (
        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    ) == 0
    assert stat.S_IMODE(provenance_path.stat().st_mode) == 0o644


@pytest.mark.parametrize(
    ("field_path", "stale_value"),
    [
        (("provenance_version",), 0),
        (("source", "provider"), "stale-provider"),
        (("source", "provider_version"), "stale-provider-version"),
        (("source", "dataset_name"), "stale-name"),
        (("source", "dataset_version"), "stale-dataset-version"),
        (("source", "data_digest"), "stale-data"),
        (("split", "method"), "random"),
        (("split", "seed"), -1),
        (
            ("split", "phase_index_digests", "train"),
            "stale-phase-index",
        ),
        (
            ("split", "phase_data_digests", "valid"),
            "stale-phase-data",
        ),
        (
            ("representation", "node_feature_encoding"),
            "stale-encoding",
        ),
        (
            ("representation", "node_feature_cardinalities"),
            [1],
        ),
    ],
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_rng_mutating_tdc,
)
def test_any_stale_provenance_field_forces_cache_rebuild(
    mock_adme,
    mock_s2g,
    tmp_path,
    field_path,
    stale_value,
):
    del mock_adme
    cfg = _make_cfg(tmp_path, split_seed=7)
    first = ADMEDatasetLoader(cfg).load_dataset()
    provenance_path = Path(first.provenance_path)
    stale = json.loads(provenance_path.read_text(encoding="utf-8"))
    target = stale
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = stale_value
    provenance_path.write_text(json.dumps(stale), encoding="utf-8")
    mock_s2g.reset_mock()

    rebuilt = _assert_rng_unchanged(
        lambda: ADMEDatasetLoader(cfg).load_dataset()
    )

    assert mock_s2g.call_count == 4
    assert rebuilt.split_fingerprint == first.split_fingerprint
    assert (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        == first.split_provenance
    )


# ---------------------------------------------------------------------------
# Mocked load_dataset tests
# ---------------------------------------------------------------------------


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_fake_tdc_adme,
)
def test_load_dataset_classification(mock_adme, mock_s2g, tmp_path):
    """load_dataset builds int labels for a classification dataset."""
    loader = ADMEDatasetLoader(_make_cfg(tmp_path, "BBB_Martins"))
    dataset = loader.load_dataset()

    assert len(dataset) == 4  # 2 train + 1 valid + 1 test
    # Classification labels should be long tensors
    assert dataset[0].y.dtype == torch.long
    assert hasattr(dataset, "split_idx")
    assert "train" in dataset.split_idx
    assert len(dataset.split_idx["train"]) == 2
    assert (
        validate_qualified_graph_source(
            dataset,
            capability=GRAPH_DATASET_MANIFEST["BBB_Martins"],
            configured_num_classes=2,
            total_num_features=9,
        )
        is dataset
    )


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_fake_tdc_adme,
)
def test_load_dataset_regression(mock_adme, mock_s2g, tmp_path):
    """load_dataset builds float labels for a regression dataset."""
    loader = ADMEDatasetLoader(_make_cfg(tmp_path, "Caco2_Wang"))
    dataset = loader.load_dataset()

    assert len(dataset) == 4
    assert dataset[0].y.dtype == torch.float
    assert dataset[0].y.shape == (1,)
    assert (
        validate_qualified_graph_source(
            dataset,
            capability=GRAPH_DATASET_MANIFEST["Caco2_Wang"],
            configured_num_classes=1,
            total_num_features=9,
        )
        is dataset
    )


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_fake_tdc_adme,
)
def test_load_dataset_keeps_node_categories_compact(
    mock_adme, mock_s2g, tmp_path
):
    """Cached graphs retain compact integral OGB atom columns."""
    loader = ADMEDatasetLoader(_make_cfg(tmp_path, "BBB_Martins"))
    dataset = loader.load_dataset()

    graph = dataset[0]
    expected = torch.tensor(
        [
            [1, 2, 3, 4, 5, 1, 2, 1, 0],
            [6, 0, 0, 0, 0, 0, 0, 0, 1],
            [118, 4, 11, 11, 9, 5, 5, 1, 1],
        ]
    )
    torch.testing.assert_close(graph.x, expected)
    assert graph.x.shape == (3, 9)
    assert graph.x.dtype == torch.long
    assert graph.edge_attr.shape[1] == 3
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
    assert dataset.split_provenance["representation"] == {
        "node_feature_encoding": "categorical_one_hot",
        "node_feature_cardinalities": [119, 5, 12, 12, 10, 6, 6, 2, 2],
        "stored_node_feature_width": 9,
        "encoded_node_feature_width": 174,
    }


@patch(
    "topobench.data.loaders.graph.adme_datasets.smiles2graph",
    side_effect=_fake_smiles2graph,
)
@patch(
    "topobench.data.loaders.graph.adme_datasets.ADME",
    side_effect=_fake_tdc_adme,
)
def test_load_dataset_split_indices_partition_dataset(
    mock_adme, mock_s2g, tmp_path
):
    """train/valid/test split indices together cover the whole dataset."""
    loader = ADMEDatasetLoader(_make_cfg(tmp_path, "BBB_Martins"))
    dataset = loader.load_dataset()

    idx = dataset.split_idx
    all_indices = torch.cat([idx["train"], idx["valid"], idx["test"]])
    assert len(all_indices) == len(dataset)

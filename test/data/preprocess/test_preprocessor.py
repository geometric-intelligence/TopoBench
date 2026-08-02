"""Comprehensive unit tests for the PreProcessor class.

This test file provides extensive coverage of the PreProcessor class functionality,
including initialization, data transformations, split loading, and edge cases.
"""

import copy
import json
import random
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import torch
import torch_geometric.data
from omegaconf import DictConfig
from torch_geometric.data import Data, HeteroData

from topobench.data.datasets import (
    SyntheticGraphDataset,
    SyntheticHypergraphDataset,
    SyntheticHeterogeneousDataset,
)
from topobench.data.loaders.base import (
    AbstractLoader,
    normalize_cache_value,
)
from topobench.data.loaders.graph.synthetic import (
    SyntheticGraphDatasetLoader,
)
from topobench.data.loaders.hypergraph.synthetic import (
    SyntheticHypergraphDatasetLoader,
)
from topobench.data.preprocessor.preprocessor import PreProcessor


class MockTorchDataset(torch.utils.data.Dataset):
    """A mock of the torch.utils.data.Dataset class.

    Parameters
    ----------
    data : Any
        The data to store in the dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        """Return the length of the data.

        Returns
        -------
        int
            The length of the data.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Return the data at the given index.

        Parameters
        ----------
        idx : int
            The index of the data to return.

        Returns
        -------
        Any
            The data at the given index.
        """
        return self.data[idx]


def _process_only_preprocessor(
    dataset,
    tmp_path,
    *,
    pre_transform=None,
) -> PreProcessor:
    """Build a preprocessor whose process output cannot be persisted."""
    preprocessor = object.__new__(PreProcessor)
    preprocessor.dataset = dataset
    preprocessor.pre_transform = pre_transform
    preprocessor.root = str(tmp_path)
    preprocessor.save = MagicMock()
    return preprocessor


def _opposite_data_family(data):
    """Return the opposite supported PyG representation."""
    return HeteroData() if isinstance(data, Data) else Data()


def _assert_synthetic_heterodata_equal(
    expected: HeteroData,
    actual: HeteroData,
) -> None:
    """Assert all persisted fields in the canonical fixture are identical."""
    assert expected.metadata() == actual.metadata()
    for attribute in ("x", "y", "train_mask", "val_mask", "test_mask"):
        assert torch.equal(
            expected["author"][attribute],
            actual["author"][attribute],
        )
    assert torch.equal(expected["paper"].x, actual["paper"].x)
    assert expected["venue"].num_nodes == actual["venue"].num_nodes
    assert "x" not in actual["venue"]
    for edge_type in expected.edge_types:
        assert torch.equal(
            expected[edge_type].edge_index,
            actual[edge_type].edge_index,
        )

class _CacheRecordLoader(AbstractLoader):
    """Return a local fixture while exercising the production loader seam."""

    def load_dataset(self):
        return SyntheticGraphDataset(
            task="graph_classification",
            seed=int(self.parameters.seed),
        )


class _AlternateCacheRecordLoader(_CacheRecordLoader):
    """Semantically identical loader with a distinct exact target."""


_GENERATION_PARAMETERS = {
    "task": "triangle_counting",
    "universe_parameters": {
        "K": 20,
        "feature_dim": 15,
        "center_variance": 0.2,
        "cluster_variance": 0.4,
        "edge_propensity_variance": 1.0,
        "seed": 42,
    },
    "family_parameters": {
        "n_graphs": 1000,
        "n_nodes_range": [50, 200],
        "n_communities_range": [3, 7],
        "homophily_range": [0.4, 0.8],
        "avg_degree_range": [1.0, 2.0],
        "degree_separation_range": [0.5, 1.0],
        "power_law_exponent_range": [1.5, 2.5],
        "seed": 42,
    },
}


def _cache_parameters(tmp_path) -> dict:
    """Return one fully resolved content-affecting loader configuration."""
    return {
        "data_dir": str(tmp_path / "cache"),
        "data_domain": "graph",
        "data_type": "synthetic",
        "data_name": "SyntheticGraph",
        "seed": 17,
        "num_nodes": 12,
        "num_hyperedges": 5,
        "generation_parameters": copy.deepcopy(_GENERATION_PARAMETERS),
        "feature_policy": "continuous",
        "representation_version": "pyg-data-v1",
        "parser_version": "synthetic-parser-v1",
    }


def _replace_nested(mapping: dict, path: str, value) -> None:
    """Replace exactly one dotted configuration field."""
    components = path.split(".")
    current = mapping
    for component in components[:-1]:
        current = current[component]
    current[components[-1]] = value


def _processed_fixture(
    tmp_path,
    *,
    mutation: tuple[str, object] | None = None,
    loader_type=_CacheRecordLoader,
    transform_name=None,
    transform_value=1,
    transform_key="identity",
) -> PreProcessor:
    """Load and process one cache-record fixture through the real seam."""
    parameters = _cache_parameters(tmp_path)
    if mutation is not None:
        _replace_nested(parameters, *mutation)
    loader = loader_type(DictConfig(parameters))
    dataset, dataset_dir = loader.load()
    transforms = DictConfig(
        {
            transform_key: {
                "transform_name": transform_name,
                "value": transform_value,
            }
        }
    )
    return PreProcessor(dataset, dataset_dir, transforms)

def _processed_transform_config(
    tmp_path,
    transform_config: dict,
) -> PreProcessor:
    """Process one fixture using an exact transform configuration."""
    loader = _CacheRecordLoader(DictConfig(_cache_parameters(tmp_path)))
    dataset, dataset_dir = loader.load()
    return PreProcessor(dataset, dataset_dir, DictConfig(transform_config))


def _processed_direct(dataset, data_dir: Path) -> PreProcessor:
    """Process a supported direct dataset under one shared cache root."""
    return PreProcessor(
        dataset,
        str(data_dir),
        DictConfig({"identity": {"transform_name": None, "value": 1}}),
    )


def _rng_states() -> tuple[object, tuple, torch.Tensor]:
    """Snapshot all caller-global CPU RNG streams."""
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _assert_rng_states_equal(
    before: tuple[object, tuple, torch.Tensor],
    after: tuple[object, tuple, torch.Tensor],
) -> None:
    """Compare Python, NumPy, and Torch RNG snapshots exactly."""
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


class TestPreProcessorBasic:
    """Test basic PreProcessor functionality."""

    def test_init_without_transforms(self):
        """Test PreProcessor initialization without transforms."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([
            torch_geometric.data.Data(x=torch.randn(3, 4)),
            torch_geometric.data.Data(x=torch.randn(5, 4)),
        ]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    assert preprocessor.transforms_applied == False
                    assert hasattr(preprocessor, 'data_list')

    def test_init_preserves_split_idx(self):
        """Test that split_idx is preserved from dataset."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.split_idx = {"train": [0, 1], "val": [2], "test": [3]}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    assert hasattr(preprocessor, "split_idx")
                    assert preprocessor.split_idx == mock_dataset.split_idx

    def test_processed_file_names(self):
        """Test the processed_file_names property."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    assert preprocessor.processed_file_names == "data.pt"

    @patch("topobench.data.preprocessor.preprocessor.load_inductive_splits")
    def test_load_dataset_splits_inductive(self, mock_load_inductive_splits):
        """Test loading dataset splits for inductive learning.

        Parameters
        ----------
        mock_load_inductive_splits : MagicMock
            Mock of the load_inductive_splits function.
        """
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    split_params = DictConfig({"learning_setting": "inductive"})
                    preprocessor.load_dataset_splits(split_params)

                    mock_load_inductive_splits.assert_called_once_with(
                        preprocessor, split_params
                    )

    @patch("topobench.data.preprocessor.preprocessor.load_transductive_splits")
    def test_load_dataset_splits_transductive(self, mock_load_transductive_splits):
        """Test loading dataset splits for transductive learning.

        Parameters
        ----------
        mock_load_transductive_splits : MagicMock
            Mock of the load_transductive_splits function.
        """
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    split_params = DictConfig({"learning_setting": "transductive"})
                    preprocessor.load_dataset_splits(split_params)

                    mock_load_transductive_splits.assert_called_once_with(
                        preprocessor, split_params
                    )

    def test_invalid_learning_setting(self):
        """Test error with invalid learning setting."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    split_params = DictConfig({"learning_setting": "invalid"})
                    with pytest.raises(ValueError, match="Invalid.*learning setting"):
                        preprocessor.load_dataset_splits(split_params)

    def test_no_learning_setting_error(self):
        """Test error when no learning setting is specified."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    # Test with no learning_setting key
                    split_params = DictConfig({})
                    with pytest.raises(ValueError, match="No learning setting specified"):
                        preprocessor.load_dataset_splits(split_params)

                    # Test with learning_setting = False
                    split_params = DictConfig({"learning_setting": False})
                    with pytest.raises(ValueError, match="No learning setting specified"):
                        preprocessor.load_dataset_splits(split_params)


class TestPreProcessorProcessing:
    """Test PreProcessor data processing methods."""

    def test_preprocessor_preserves_heterodata_on_process_and_reload(
        self,
        tmp_path,
    ):
        """A processed native heterogeneous graph reloads without flattening."""
        dataset = SyntheticHeterogeneousDataset(seed=7)
        identity = DictConfig({"transform_name": "IdentityTransform"})

        original = dataset[0]
        first = PreProcessor(dataset, tmp_path, transforms_config=identity)
        reloaded = PreProcessor(dataset, tmp_path, transforms_config=identity)

        assert isinstance(first[0], HeteroData)
        assert isinstance(reloaded[0], HeteroData)
        _assert_synthetic_heterodata_equal(original, first[0])
        _assert_synthetic_heterodata_equal(first[0], reloaded[0])

    def test_preprocessor_preserves_data_on_process_and_reload(self, tmp_path):
        """A processed homogeneous graph remains homogeneous after reload."""
        data = Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )
        identity = DictConfig({"transform_name": "IdentityTransform"})

        first = PreProcessor(data, tmp_path, transforms_config=identity)
        reloaded = PreProcessor(data, tmp_path, transforms_config=identity)

        assert isinstance(first[0], Data)
        assert not isinstance(first[0], HeteroData)
        assert isinstance(reloaded[0], Data)
        assert not isinstance(reloaded[0], HeteroData)
        assert torch.equal(first[0].x, reloaded[0].x)

    def test_preprocessor_persists_direct_heterodata_input(self, tmp_path):
        """Direct HeteroData input follows the same persisted path as datasets."""
        data = SyntheticHeterogeneousDataset(seed=11)[0]
        identity = DictConfig({"transform_name": "IdentityTransform"})

        first = PreProcessor(data, tmp_path, transforms_config=identity)
        reloaded = PreProcessor(data, tmp_path, transforms_config=identity)

        assert isinstance(first[0], HeteroData)
        assert isinstance(reloaded[0], HeteroData)
        assert torch.equal(
            first[0]["author"].test_mask,
            reloaded[0]["author"].test_mask,
        )

    def test_preprocessor_rejects_non_pyg_dataset_item(
        self,
        tmp_path,
    ):
        """Every dataset item must use a supported PyG representation."""
        preprocessor = _process_only_preprocessor(
            MockTorchDataset([object()]),
            tmp_path,
        )

        with pytest.raises(TypeError) as error:
            preprocessor.process()

        assert str(error.value) == (
            "Dataset item 0 must be Data or HeteroData; received object"
        )
        preprocessor.save.assert_not_called()
        assert not any(tmp_path.rglob("data.pt"))

    def test_preprocessor_rejects_unsupported_transform_result(
        self,
        tmp_path,
    ):
        """Transforms may not replace PyG data with an arbitrary object."""
        preprocessor = _process_only_preprocessor(
            Data(x=torch.ones(2, 1)),
            tmp_path,
            pre_transform=lambda _: object(),
        )

        with pytest.raises(TypeError) as error:
            preprocessor.process()

        assert str(error.value) == (
            "Pre-transform result for dataset item 0 must be Data or "
            "HeteroData; received object"
        )
        preprocessor.save.assert_not_called()
        assert not any(tmp_path.rglob("data.pt"))

    @pytest.mark.parametrize(
        "data",
        [
            Data(x=torch.ones(2, 1)),
            HeteroData({"author": {"x": torch.ones(2, 1)}}),
        ],
        ids=["data-to-heterodata", "heterodata-to-data"],
    )
    def test_preprocessor_rejects_data_heterodata_representation_change(
        self,
        data,
        tmp_path,
    ):
        """Transforms must preserve the homogeneous/heterogeneous family."""
        preprocessor = _process_only_preprocessor(
            data,
            tmp_path,
            pre_transform=_opposite_data_family,
        )

        with pytest.raises(TypeError) as error:
            preprocessor.process()

        assert str(error.value) == (
            "Pre-transform changed representation for dataset item 0: "
            f"expected {type(data).__name__}, received "
            f"{type(_opposite_data_family(data)).__name__}"
        )
        preprocessor.save.assert_not_called()

    @pytest.mark.parametrize(
        ("data_list", "expected_family", "received_family"),
        [
            (
                [HeteroData(), Data()],
                "HeteroData",
                "Data",
            ),
            (
                [Data(), HeteroData()],
                "Data",
                "HeteroData",
            ),
        ],
        ids=["heterodata-then-data", "data-then-heterodata"],
    )
    def test_preprocessor_rejects_mixed_dataset_representations(
        self,
        data_list,
        expected_family,
        received_family,
        tmp_path,
    ):
        """One persisted dataset cannot mix PyG representation families."""
        preprocessor = _process_only_preprocessor(
            MockTorchDataset(data_list),
            tmp_path,
        )

        with pytest.raises(TypeError) as error:
            preprocessor.process()

        assert str(error.value) == (
            "Dataset item 1 has mixed representation: "
            f"expected {expected_family}, received {received_family}"
        )
        preprocessor.save.assert_not_called()
        assert not any(tmp_path.rglob("data.pt"))

    def test_preprocessor_rejects_invalid_top_level_input(self, tmp_path):
        """The preprocessor reports unsupported input containers clearly."""
        preprocessor = _process_only_preprocessor(object(), tmp_path)

        with pytest.raises(TypeError) as error:
            preprocessor.process()

        assert str(error.value) == (
            "PreProcessor expects a PyG/PyTorch dataset, Data, or HeteroData; "
            "received object"
        )
        preprocessor.save.assert_not_called()

    def test_process_with_torch_utils_dataset(self):
        """Test process method with torch.utils.data.Dataset."""
        mock_data = [
            torch_geometric.data.Data(x=torch.randn(3, 4)),
            torch_geometric.data.Data(x=torch.randn(5, 4)),
        ]
        mock_dataset = MockTorchDataset(mock_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.dataset = mock_dataset
                preprocessor.pre_transform = None
                preprocessor.collate = MagicMock(
                    return_value=(torch_geometric.data.Data(), {})
                )
                preprocessor.save = MagicMock()

                # Mock the processed_paths property
                with patch.object(type(preprocessor), 'processed_paths', new_callable=lambda: property(lambda self: [f"{tmpdir}/data.pt"])):
                    preprocessor.process()

                    assert len(preprocessor.data_list) == len(mock_data)
                    preprocessor.collate.assert_called_once()
                    preprocessor.save.assert_called_once()

    def test_process_with_torch_geometric_data(self):
        """Test process method with torch_geometric.data.Data."""
        mock_data = torch_geometric.data.Data(x=torch.randn(3, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.dataset = mock_data
                preprocessor.pre_transform = None
                preprocessor.collate = MagicMock(
                    return_value=(torch_geometric.data.Data(), {})
                )
                preprocessor.save = MagicMock()

                # Mock the processed_paths property
                with patch.object(type(preprocessor), 'processed_paths', new_callable=lambda: property(lambda self: [f"{tmpdir}/data.pt"])):
                    preprocessor.process()

                    assert preprocessor.data_list == [mock_data]
                    preprocessor.collate.assert_called_once_with([mock_data])

    def test_process_with_pre_transform(self):
        """Test process method with a pre_transform applied."""
        mock_data = [
            torch_geometric.data.Data(x=torch.randn(3, 4)),
            torch_geometric.data.Data(x=torch.randn(5, 4))
        ]
        mock_dataset = MockTorchDataset(mock_data)
        mock_pre_transform = MagicMock(side_effect=lambda x: x)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.dataset = mock_dataset
                preprocessor.pre_transform = mock_pre_transform
                preprocessor.collate = MagicMock(
                    return_value=(torch_geometric.data.Data(), {})
                )
                preprocessor.save = MagicMock()

                # Mock the processed_paths property
                with patch.object(type(preprocessor), 'processed_paths', new_callable=lambda: property(lambda self: [f"{tmpdir}/data.pt"])):
                    preprocessor.process()

                    # Verify pre_transform was called for each data item
                    assert mock_pre_transform.call_count == len(mock_data)


class TestPreProcessorLoad:
    """Test PreProcessor load method."""

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_backward_compatibility_2_elements(self, mock_torch_load):
        """Test load method with 2 elements (backward compatibility).

        Parameters
        ----------
        mock_torch_load : MagicMock
            Mock of the torch_load function.
        """
        mock_data = torch_geometric.data.Data()
        mock_slices = {"x": torch.tensor([0, 3])}
        mock_torch_load.return_value = (mock_data, mock_slices)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.load("/fake/path")

                # Use _data as that's what the actual code uses
                assert preprocessor._data == mock_data
                assert preprocessor.slices == mock_slices

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_backward_compatibility_3_elements(self, mock_torch_load):
        """Test load method with 3 elements (backward compatibility).

        Parameters
        ----------
        mock_torch_load : MagicMock
            Mock of the torch_load function.
        """
        mock_data = torch_geometric.data.Data()
        mock_slices = {"x": torch.tensor([0, 3])}
        mock_data_cls = torch_geometric.data.Data
        mock_torch_load.return_value = (mock_data, mock_slices, mock_data_cls)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.load("/fake/path")

                assert preprocessor._data == mock_data
                assert preprocessor.slices == mock_slices

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_with_4_elements(self, mock_torch_load):
        """Test load method with 4 elements (TU Datasets format).

        Parameters
        ----------
        mock_torch_load : MagicMock
            Mock of the torch_load function.
        """
        mock_data = torch_geometric.data.Data()
        mock_slices = {"x": torch.tensor([0, 3])}
        mock_sizes = {"x": 3}
        mock_data_cls = torch_geometric.data.Data
        mock_torch_load.return_value = (mock_data, mock_slices, mock_sizes, mock_data_cls)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.load("/fake/path")

                assert preprocessor._data == mock_data
                assert preprocessor.slices == mock_slices

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_with_dict_data(self, mock_torch_load):
        """Test load method when data is a dictionary.

        Parameters
        ----------
        mock_torch_load : MagicMock
            Mock of the torch_load function.
        """
        mock_data_dict = {
            "x": torch.randn(3, 4),
            "edge_index": torch.tensor([[0, 1], [1, 2]])
        }
        mock_slices = {"x": torch.tensor([0, 3])}
        mock_torch_load.return_value = (
            mock_data_dict,
            mock_slices,
            Data,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.load("/fake/path")

                assert isinstance(preprocessor._data, Data)
                assert torch.equal(
                    preprocessor._data.x,
                    mock_data_dict["x"],
                )
                assert torch.equal(
                    preprocessor._data.edge_index,
                    mock_data_dict["edge_index"],
                )
                assert preprocessor.slices == mock_slices

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_legacy_two_element_dict_defaults_to_data(
        self,
        mock_torch_load,
    ):
        """A legacy dictionary payload without class metadata reconstructs Data."""
        data_dict = {"x": torch.randn(3, 2)}
        slices = {"x": torch.tensor([0, 3])}
        mock_torch_load.return_value = (data_dict, slices)
        preprocessor = object.__new__(PreProcessor)

        preprocessor.load("/fake/path")

        assert isinstance(preprocessor._data, Data)
        assert torch.equal(preprocessor._data.x, data_dict["x"])
        assert preprocessor.slices == slices

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_rejects_non_tuple_artifact(self, mock_torch_load):
        """Corrupt cache containers raise an optimized-safe exception."""
        mock_torch_load.return_value = ["data", {}]
        preprocessor = object.__new__(PreProcessor)

        with pytest.raises(TypeError) as error:
            preprocessor.load("/fake/path")

        assert str(error.value) == (
            "Processed data artifact must be a tuple; received list"
        )

    @pytest.mark.parametrize("length", [1, 5])
    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_rejects_unsupported_tuple_length(
        self,
        mock_torch_load,
        length,
    ):
        """Only documented two-, three-, and four-element caches are valid."""
        mock_torch_load.return_value = tuple(range(length))
        preprocessor = object.__new__(PreProcessor)

        with pytest.raises(ValueError) as error:
            preprocessor.load("/fake/path")

        assert str(error.value) == (
            "Processed data artifact must contain 2, 3, or 4 elements; "
            f"received {length}"
        )

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_rejects_unsupported_data_payload(self, mock_torch_load):
        """A tuple alone does not make an arbitrary payload a valid cache."""
        mock_torch_load.return_value = (42, None)
        preprocessor = object.__new__(PreProcessor)

        with pytest.raises(TypeError) as error:
            preprocessor.load("/fake/path")

        assert str(error.value) == (
            "Processed data payload must be Data, HeteroData, or a "
            "dictionary; received int"
        )

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_rejects_unsupported_slices_payload(self, mock_torch_load):
        """Slice metadata must retain the structure expected by PyG."""
        mock_torch_load.return_value = (Data(), [])
        preprocessor = object.__new__(PreProcessor)

        with pytest.raises(TypeError) as error:
            preprocessor.load("/fake/path")

        assert str(error.value) == (
            "Processed data slices must be a dictionary or None; "
            "received list"
        )

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_rejects_invalid_data_class_for_dict(
        self,
        mock_torch_load,
    ):
        """Dictionary cache payloads require a supported reconstruction class."""
        mock_torch_load.return_value = ({"x": torch.ones(1, 1)}, None, object)
        preprocessor = object.__new__(PreProcessor)

        with pytest.raises(TypeError) as error:
            preprocessor.load("/fake/path")

        assert str(error.value) == (
            "Processed data class must be Data or HeteroData for a "
            "dictionary payload; received object"
        )


class TestPreProcessorTransforms:
    """Test PreProcessor with transforms."""

    def test_readable_cache_record_is_stored_beside_digest(self, tmp_path):
        """The digest directory contains the exact canonical readable record."""
        preprocessor = _processed_fixture(tmp_path)

        record_path = Path(preprocessor.cache_record_path)
        assert record_path.parent == Path(preprocessor.processed_data_dir)
        assert record_path.name == "cache_record.json"
        assert record_path.parent.name == preprocessor.cache_identity
        assert len(preprocessor.cache_identity) == 64
        assert json.loads(record_path.read_text(encoding="utf-8")) == (
            preprocessor.cache_record
        )
        assert list(record_path.parent.glob(".cache_record.json.*")) == []
        assert preprocessor.cache_record == {
            "schema": "topobench.processed-cache",
            "schema_version": 2,
            "dataset_selector": {
                "data_domain": "graph",
                "data_type": "synthetic",
                "data_name": "SyntheticGraph",
            },
            "loader": {
                "target": (
                    "test.data.preprocess.test_preprocessor."
                    "_CacheRecordLoader"
                ),
                "parameters": {
                    **_cache_parameters(tmp_path),
                    "task": "graph_classification",
                },
            },
            "transform": {
                "steps": [
                    {
                        "name": "identity",
                        "target": None,
                        "parameters": {
                            "preprocessor_device": "cpu",
                            "transform_name": None,
                            "value": 1,
                        },
                    }
                ]
            },
            "feature_policy": "continuous",
            "versions": {
                "representation": "pyg-data-v1",
                "parser": "synthetic-parser-v1",
            },
        }

    @pytest.mark.parametrize(
        ("field", "different_value"),
        [
            ("data_name", "SyntheticGraphAlternate"),
            ("seed", 18),
            ("num_nodes", 13),
            ("num_hyperedges", 6),
            ("generation_parameters.task", "community_detection"),
            ("generation_parameters.universe_parameters.K", 21),
            ("generation_parameters.universe_parameters.feature_dim", 16),
            (
                "generation_parameters.universe_parameters.center_variance",
                0.3,
            ),
            (
                "generation_parameters.universe_parameters.cluster_variance",
                0.5,
            ),
            (
                "generation_parameters.universe_parameters."
                "edge_propensity_variance",
                0.9,
            ),
            ("generation_parameters.universe_parameters.seed", 43),
            ("generation_parameters.family_parameters.n_graphs", 1001),
            (
                "generation_parameters.family_parameters.n_nodes_range",
                [51, 200],
            ),
            (
                "generation_parameters.family_parameters.n_communities_range",
                [4, 7],
            ),
            (
                "generation_parameters.family_parameters.homophily_range",
                [0.5, 0.8],
            ),
            (
                "generation_parameters.family_parameters.avg_degree_range",
                [1.1, 2.0],
            ),
            (
                "generation_parameters.family_parameters."
                "degree_separation_range",
                [0.6, 1.0],
            ),
            (
                "generation_parameters.family_parameters."
                "power_law_exponent_range",
                [1.6, 2.5],
            ),
            ("generation_parameters.family_parameters.seed", 43),
            ("feature_policy", "degree"),
            ("representation_version", "pyg-data-v2"),
            ("parser_version", "synthetic-parser-v2"),
        ],
    )
    def test_each_content_field_changes_cache_identity(
        self,
        tmp_path,
        field,
        different_value,
    ):
        """Every independently varied source field receives a new identity."""
        baseline = _processed_fixture(tmp_path)
        varied = _processed_fixture(
            tmp_path,
            mutation=(field, different_value),
        )

        assert varied.cache_identity != baseline.cache_identity
        assert varied.processed_data_dir != baseline.processed_data_dir

    def test_exact_loader_target_changes_cache_identity(self, tmp_path):
        """Runtime loader implementation is part of cache provenance."""
        baseline = _processed_fixture(tmp_path)
        varied = _processed_fixture(
            tmp_path,
            loader_type=_AlternateCacheRecordLoader,
        )

        assert varied.cache_identity != baseline.cache_identity
        assert (
            varied.cache_record["loader"]["target"]
            != baseline.cache_record["loader"]["target"]
        )

    @pytest.mark.parametrize(
        ("option", "different_value"),
        [("target", "IdentityTransform"), ("value", 2)],
    )
    def test_transform_target_and_value_change_cache_identity(
        self,
        tmp_path,
        option,
        different_value,
    ):
        """Transform implementation and parameters are both fingerprinted."""
        baseline = _processed_fixture(tmp_path)
        kwargs = (
            {"transform_name": different_value}
            if option == "target"
            else {"transform_value": different_value}
        )
        varied = _processed_fixture(tmp_path, **kwargs)

        assert varied.cache_identity != baseline.cache_identity

    def test_effective_transform_defaults_define_identity(self, tmp_path):
        """Omitted and explicit-equivalent constructor defaults are identical."""
        omitted = _processed_transform_config(
            tmp_path,
            {"degrees": {"transform_name": "NodeDegrees"}},
        )
        explicit = _processed_transform_config(
            tmp_path,
            {
                "degrees": {
                    "transform_name": "NodeDegrees",
                    "selected_fields": ["edge_index"],
                }
            },
        )

        assert omitted.cache_identity == explicit.cache_identity
        assert omitted.cache_record["transform"] == {
            "steps": [
                {
                    "name": "degrees",
                    "target": (
                        "topobench.transforms.data_manipulations."
                        "node_degrees.NodeDegrees"
                    ),
                    "parameters": {
                        "preprocessor_device": "cpu",
                        "selected_fields": ["edge_index"],
                        "transform_name": "NodeDegrees",
                    },
                }
            ]
        }
        for left, right in zip(omitted, explicit, strict=True):
            assert torch.equal(left.node_degrees, right.node_degrees)

        changed = _processed_transform_config(
            tmp_path,
            {
                "degrees": {
                    "transform_name": "NodeDegrees",
                    "selected_fields": [],
                }
            },
        )
        assert changed.cache_identity != omitted.cache_identity

    def test_transform_reserved_and_variadic_parameters_are_preserved(
        self,
        tmp_path,
    ):
        """Canonical transform steps retain all supplied variadic parameters."""
        processed = _processed_transform_config(
            tmp_path,
            {
                "identity": {
                    "transform_name": "IdentityTransform",
                    "_partial_": False,
                    "_target_": "configured.target",
                    "behavior_flag": {"enabled": True},
                }
            },
        )

        assert processed.cache_record["transform"]["steps"][0][
            "parameters"
        ] == {
            "_partial_": False,
            "_target_": "configured.target",
            "behavior_flag": {"enabled": True},
            "preprocessor_device": "cpu",
            "transform_name": "IdentityTransform",
        }

    def test_direct_dataset_effective_defaults_share_readable_identity(
        self,
        tmp_path,
    ):
        """Direct datasets fingerprint normalized effective content defaults."""
        cases = [
            (
                SyntheticGraphDataset(),
                SyntheticGraphDataset(
                    task="graph_classification",
                    seed=0,
                ),
                {
                    "task": "graph_classification",
                    "seed": 0,
                },
            ),
            (
                SyntheticHypergraphDataset(),
                SyntheticHypergraphDataset(
                    seed=0,
                    num_nodes=12,
                    num_hyperedges=5,
                ),
                {
                    "seed": 0,
                    "num_nodes": 12,
                    "num_hyperedges": 5,
                },
            ),
        ]

        for index, (omitted, explicit, expected) in enumerate(cases):
            data_dir = tmp_path / f"direct-defaults-{index}"
            omitted_processed = _processed_direct(omitted, data_dir)
            explicit_processed = _processed_direct(explicit, data_dir)

            assert (
                omitted_processed.cache_identity
                == explicit_processed.cache_identity
            )
            assert omitted_processed.cache_record["loader"] == {
                "target": None,
                "parameters": expected,
            }
            assert json.loads(
                Path(omitted_processed.cache_record_path).read_text(
                    encoding="utf-8"
                )
            ) == omitted_processed.cache_record

    def test_direct_dataset_content_parameters_prevent_collisions(
        self,
        tmp_path,
    ):
        """Graph tasks and hypergraph seeds select distinct direct caches."""
        transforms_root = tmp_path / "direct-content"
        graph_classification = _processed_direct(
            SyntheticGraphDataset(
                task="graph_classification",
                seed=11,
            ),
            transforms_root,
        )
        graph_regression = _processed_direct(
            SyntheticGraphDataset(task="graph_regression", seed=11),
            transforms_root,
        )
        hypergraph_seed_11 = _processed_direct(
            SyntheticHypergraphDataset(seed=11),
            transforms_root,
        )
        hypergraph_seed_12 = _processed_direct(
            SyntheticHypergraphDataset(seed=12),
            transforms_root,
        )

        assert (
            graph_classification.cache_identity
            != graph_regression.cache_identity
        )
        assert (
            graph_classification.cache_record["loader"]["parameters"]["task"]
            == "graph_classification"
        )
        assert (
            graph_regression.cache_record["loader"]["parameters"]["task"]
            == "graph_regression"
        )
        assert (
            hypergraph_seed_11.cache_identity
            != hypergraph_seed_12.cache_identity
        )

    def test_same_config_reuses_identity_and_tensors(self, tmp_path):
        """A cache hit reproduces the exact miss identity and tensors."""
        miss = _processed_fixture(tmp_path)
        hit = _processed_fixture(tmp_path)

        assert hit.cache_identity == miss.cache_identity
        assert hit.processed_data_dir == miss.processed_data_dir
        assert len(hit) == len(miss)
        for left, right in zip(miss, hit, strict=True):
            assert left.to_dict().keys() == right.to_dict().keys()
            for key in left.to_dict():
                assert torch.equal(left[key], right[key])

    def test_cache_miss_and_hit_preserve_all_global_rng_states(
        self,
        tmp_path,
    ):
        """Loading and preprocessing never consumes caller-global RNG streams."""
        for _ in ("miss", "hit"):
            random.seed(901)
            np.random.seed(902)
            torch.manual_seed(903)
            before = _rng_states()

            _processed_fixture(tmp_path)

            _assert_rng_states_equal(before, _rng_states())

    def test_existing_mismatched_record_is_rejected_without_overwrite(
        self,
        tmp_path,
    ):
        """A digest collision cannot silently replace existing provenance."""
        first = _processed_fixture(tmp_path)
        record_path = Path(first.cache_record_path)
        mismatch = {"schema": "mismatched", "schema_version": -1}
        record_path.write_text(json.dumps(mismatch), encoding="utf-8")

        with pytest.raises(ValueError, match="cache identity collision"):
            _processed_fixture(tmp_path)

        assert json.loads(record_path.read_text(encoding="utf-8")) == mismatch

    def test_digest_component_prevents_transform_path_traversal(self, tmp_path):
        """Untrusted transform labels never become path components."""
        preprocessor = _processed_fixture(
            tmp_path,
            transform_key="../../escape",
        )
        expected_parent = (
            tmp_path / "cache" / "SyntheticGraph"
        ).resolve()

        assert Path(preprocessor.processed_data_dir).parent.resolve() == (
            expected_parent
        )

    def test_dataset_selector_cannot_escape_loader_root(self, tmp_path):
        """A selector is provenance, never an unchecked filesystem path."""
        parameters = _cache_parameters(tmp_path)
        parameters["data_name"] = "../../escape"

        with pytest.raises(
            ValueError,
            match="data_name must remain within data_dir",
        ):
            _CacheRecordLoader(DictConfig(parameters)).load()

    @pytest.mark.parametrize("different_value", ["1000", 1000.0])
    def test_canonical_identity_preserves_primitive_types(
        self,
        tmp_path,
        different_value,
    ):
        """Canonical encoding never stringifies distinct primitive types."""
        baseline = _processed_fixture(tmp_path)
        varied = _processed_fixture(
            tmp_path,
            mutation=(
                "generation_parameters.family_parameters.n_graphs",
                different_value,
            ),
        )

        assert varied.cache_identity != baseline.cache_identity

    def test_canonical_container_tags_cannot_collide_with_mappings(self):
        """A readable mapping cannot impersonate a tagged tuple."""
        tuple_value = normalize_cache_value((1, 2), path="tuple")
        mapping_value = normalize_cache_value(
            {"__cache_type__": "tuple", "items": [1, 2]},
            path="mapping",
        )

        assert mapping_value != tuple_value

    @pytest.mark.parametrize(
        ("loader_type", "domain", "data_name", "defaults"),
        [
            (
                SyntheticGraphDatasetLoader,
                "graph",
                "SyntheticGraph",
                {"seed": 0},
            ),
            (
                SyntheticHypergraphDatasetLoader,
                "hypergraph",
                "SyntheticHypergraph",
                {"seed": 0, "num_nodes": 12, "num_hyperedges": 5},
            ),
        ],
    )
    def test_omitted_and_explicit_synthetic_defaults_share_identity(
        self,
        tmp_path,
        loader_type,
        domain,
        data_name,
        defaults,
    ):
        """Effective content defaults are readable and canonical before hashing."""
        base_parameters = {
            "data_dir": str(tmp_path / "defaults"),
            "data_domain": domain,
            "data_type": "synthetic",
            "data_name": data_name,
        }
        transforms = DictConfig(
            {"identity": {"transform_name": None, "value": 1}}
        )
        results = []
        for parameters in (
            base_parameters,
            {**base_parameters, **defaults},
        ):
            dataset, data_dir = loader_type(DictConfig(parameters)).load()
            results.append(PreProcessor(dataset, data_dir, transforms))

        omitted, explicit = results
        assert omitted.cache_identity == explicit.cache_identity
        for key, value in defaults.items():
            assert omitted.cache_record["loader"]["parameters"][key] == value

    def test_unsupported_loader_value_is_rejected_contextually(self, tmp_path):
        """Unsupported values identify their exact configuration path."""
        parameters = _cache_parameters(tmp_path)
        parameters["unsupported"] = object()
        config = DictConfig(parameters, flags={"allow_objects": True})

        with pytest.raises(
            TypeError,
            match=r"loader\.parameters\.unsupported.*object",
        ):
            _CacheRecordLoader(config).load()

    def test_unresolved_loader_config_is_rejected_contextually(self, tmp_path):
        """Missing OmegaConf interpolation is never hashed as source text."""
        parameters = _cache_parameters(tmp_path)
        parameters["unresolved"] = "${missing.value}"

        with pytest.raises(
            ValueError,
            match="loader parameters could not be fully resolved",
        ):
            _CacheRecordLoader(DictConfig(parameters)).load()

    def test_instantiate_pre_transform_with_liftings(self):
        """Test instantiate_pre_transform with liftings config."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}

        transforms_config = DictConfig({
            "liftings": {
                "transform1": {"transform_name": "DummyTransform", "param1": "value1"}
            }
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create preprocessor instance
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                # Mock DataTransform to avoid needing real transforms
                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # Check that a Compose object was created
                    assert hasattr(pre_transform, '__call__')

    def test_instantiate_pre_transform_multiple_transforms(self):
        """Test instantiate_pre_transform with multiple transforms (else branch)."""
        transforms_config = DictConfig({
            "transform1": {"transform_name": "Transform1", "param1": "value1"},
            "transform2": {"transform_name": "Transform2", "param2": "value2"}
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                # Mock DataTransform
                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()

                    # Mock set_processed_data_dir
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # DataTransform should be called for each transform
                    assert mock_dt.call_count == 2
                    assert hasattr(pre_transform, '__call__')

    def test_instantiate_pre_transform_single_transform(self):
        """Test instantiate_pre_transform with single transform (if branch)."""
        transforms_config = DictConfig({
            "transform_name": "SingleTransform",
            "param1": "value1",
            "param2": 42
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                # Mock DataTransform
                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    # Mock DataTransform to return a mock object
                    mock_transform = MagicMock()
                    mock_dt.return_value = mock_transform

                    # Mock set_processed_data_dir
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # DataTransform should be called once with the entire config
                    assert mock_dt.call_count == 1
                    # Should be called with all the config parameters
                    mock_dt.assert_called_once_with(**transforms_config)

                    # Verify the pre_transform is a Compose object
                    assert isinstance(
                        pre_transform,
                        torch_geometric.transforms.Compose
                    )

    def test_instantiate_pre_transform_calls_set_processed_data_dir(self):
        """Test that instantiate_pre_transform calls set_processed_data_dir."""
        transforms_config = DictConfig({
            "transform1": {"transform_name": "Transform1", "param1": "value1"}
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()
                    # Mock set_processed_data_dir
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    call_args = (
                        preprocessor.set_processed_data_dir.call_args.args
                    )
                    assert call_args[0] == tmpdir
                    assert call_args[1][0]["name"] == "transform1"
                    assert call_args[1][0]["parameters"] == {
                        "param1": "value1",
                        "preprocessor_device": "cpu",
                        "transform_name": "Transform1",
                    }

    def test_instantiate_pre_transform_returns_compose(self):
        """Test that instantiate_pre_transform returns a Compose object."""
        transforms_config = DictConfig({
            "transform1": {"transform_name": "Transform1", "param1": "value1"}
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # Check it's a Compose instance
                    assert isinstance(
                        pre_transform,
                        torch_geometric.transforms.Compose
                    )

    def test_instantiate_pre_transform_single_vs_multiple(self):
        """Test that the method correctly distinguishes between single and multiple transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.set_processed_data_dir = MagicMock()

                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()

                    # Test single transform (has transform_name key)
                    single_config = DictConfig({
                        "transform_name": "SingleTransform",
                        "param1": "value1"
                    })

                    preprocessor.instantiate_pre_transform(tmpdir, single_config)

                    # Should be called once with all parameters
                    assert mock_dt.call_count == 1
                    mock_dt.assert_called_with(**single_config)

                    # Reset mock
                    mock_dt.reset_mock()

                    # Test multiple transforms (no transform_name key at top level)
                    multiple_config = DictConfig({
                        "transform1": {"transform_name": "Transform1", "param1": "value1"},
                        "transform2": {"transform_name": "Transform2", "param2": "value2"}
                    })

                    preprocessor.instantiate_pre_transform(tmpdir, multiple_config)

                    # Should be called twice, once for each transform
                    assert mock_dt.call_count == 2


class TestPreProcessorEdgeCases:
    """Test edge cases and error handling."""

    def test_process_with_empty_dataset(self, tmp_path):
        """Test process method with an empty dataset."""
        mock_dataset = MockTorchDataset([])
        preprocessor = _process_only_preprocessor(mock_dataset, tmp_path)

        with pytest.raises(ValueError) as error:
            preprocessor.process()

        assert str(error.value) == (
            "PreProcessor requires at least one Data or HeteroData item"
        )
        preprocessor.save.assert_not_called()
        assert not any(tmp_path.rglob("data.pt"))

    def test_processed_dir_property(self):
        """Test the processed_dir property returns correct paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without transforms
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.root = tmpdir
                preprocessor.transforms_applied = False

                assert preprocessor.processed_dir == tmpdir

            # With transforms
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.root = tmpdir
                preprocessor.transforms_applied = True

                assert preprocessor.processed_dir == tmpdir

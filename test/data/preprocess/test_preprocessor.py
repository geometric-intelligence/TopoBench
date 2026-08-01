"""Comprehensive unit tests for the PreProcessor class.

This test file provides extensive coverage of the PreProcessor class functionality,
including initialization, data transformations, split loading, and edge cases.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_geometric.data
from omegaconf import DictConfig
from torch_geometric.data import Data, HeteroData

from topobench.data.datasets import SyntheticHeterogeneousDataset
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

    def test_save_transform_parameters_new_file(self):
        """Test saving transform parameters when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.processed_data_dir = tmpdir
                preprocessor.transforms_parameters = {
                    "transform1": {"param": "value"}
                }

                preprocessor.save_transform_parameters()

                # Check if file was created
                param_file = os.path.join(
                    tmpdir, "path_transform_parameters_dict.json"
                )
                assert os.path.exists(param_file)

                # Check file contents
                with open(param_file, 'r') as f:
                    saved_params = json.load(f)
                assert saved_params == preprocessor.transforms_parameters

    def test_save_transform_parameters_existing_same(self, capsys):
        """Test saving transform parameters when file exists with same params.

        Parameters
        ----------
        capsys : pytest.CaptureFixture
            Pytest fixture to capture stdout/stderr output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing params file
            params = {"transform1": {"param": "value"}}
            param_file = os.path.join(
                tmpdir, "path_transform_parameters_dict.json"
            )
            with open(param_file, 'w') as f:
                json.dump(params, f)

            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.processed_data_dir = tmpdir
                preprocessor.transforms_parameters = params

                preprocessor.save_transform_parameters()

                # Check that message was printed
                captured = capsys.readouterr()
                assert "Transform parameters are the same" in captured.out

    def test_save_transform_parameters_existing_different(self):
        """Test error when saving different transform parameters to same path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing params file with different params
            existing_params = {"transform1": {"param": "old_value"}}
            param_file = os.path.join(
                tmpdir, "path_transform_parameters_dict.json"
            )
            with open(param_file, 'w') as f:
                json.dump(existing_params, f)

            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.processed_data_dir = tmpdir
                preprocessor.transforms_parameters = {
                    "transform1": {"param": "new_value"}
                }

                with pytest.raises(ValueError, match="Different transform parameters"):
                    preprocessor.save_transform_parameters()

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

                    # Verify set_processed_data_dir was called
                    preprocessor.set_processed_data_dir.assert_called_once()
                    call_args = preprocessor.set_processed_data_dir.call_args
                    assert call_args[0][1] == tmpdir
                    assert call_args[0][2] == transforms_config

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

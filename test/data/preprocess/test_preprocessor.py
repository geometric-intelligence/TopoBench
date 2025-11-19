""" Test the PreProcessor class."""

import os
import os.path as osp
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
import torch_geometric.data
from omegaconf import DictConfig

import topobench.data.preprocessor.preprocessor as preproc_mod
from topobench.data.preprocessor.preprocessor import PreProcessor

from ..._utils.flow_mocker import FlowMocker


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

@pytest.mark.usefixtures("mocker_fixture")
class TestPreProcessor:
    """Test the PreProcessor class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, mocker_fixture):
        """Test setup.
        
        Parameters
        ----------
        mocker_fixture : MockerFixture
            A fixture that provides a mocker object.
        """
        mocker = mocker_fixture

        # Setup test parameters
        self.dataset = MagicMock(spec=torch_geometric.data.Dataset)
        self.data_dir = "fake/path"
        self.transforms_config = DictConfig(
            {"transform": {"transform_name": "CellCycleLifting"}}
        )

        params = [
            {
                "mock_inmemory_init": "torch_geometric.data.InMemoryDataset.__init__"
            },
            {
                "mock_save_transform": (
                    PreProcessor,
                    "save_transform_parameters",
                )
            },
            {"mock_load": (PreProcessor, "load")},
            {
                "mock_len": (PreProcessor, "__len__"),
                "init_args": {"return_value": 3},
            },
            {
                "mock_getitem": (PreProcessor, "get"),
                "init_args": {"return_value": "0"},
            },
        ]
        self.flow_mocker = FlowMocker(mocker, params)

        # Initialize PreProcessor
        self.preprocessor = PreProcessor(self.dataset, self.data_dir, None)
        
    def teardown_method(self):
        """Test teardown."""
        del self.preprocessor
        del self.flow_mocker

    def test_init(self):
        """Test the initialization of the PreProcessor class."""
        self.flow_mocker.get("mock_inmemory_init").assert_called_once_with(
            self.data_dir, None, None
        )
        self.flow_mocker.get("mock_load").assert_called_once_with(
            self.data_dir + "/processed/data.pt"
        )
        assert not self.preprocessor.transforms_applied
        assert self.preprocessor.data_list == ["0", "0", "0"]

    def test_init_with_transform(self, mocker_fixture):
        """Test the initialization of the PreProcessor class with transforms.
        
        Parameters
        ----------
        mocker_fixture : MockerFixture
            A fixture that provides a mocker object.
        """
        mocker = mocker_fixture
        val_processed_paths = ["/some/path"]
        params = [
            {"assert_args": ("created_property", "processed_data_dir")},
            {"assert_args": ("created_property", "processed_data_dir")},
            {
                "mock_inmemory_init": "torch_geometric.data.InMemoryDataset.__init__",
                "assert_args": ("called_once_with", ANY, None, ANY),
            },
            {
                "mock_processed_paths": (PreProcessor, "processed_paths"),
                "init_args": {"property_val": val_processed_paths},
            },
            {
                "mock_save_transform": (
                    PreProcessor,
                    "save_transform_parameters",
                ),
                "assert_args": ("created_property", "processed_paths"),
            },
            {
                "mock_load": (PreProcessor, "load"),
                "assert_args": ("called_once_with", val_processed_paths[0]),
            },
            {"mock_len": (PreProcessor, "__len__")},
            {"mock_getitem": (PreProcessor, "get")},
        ]
        self.flow_mocker = FlowMocker(mocker, params)
        self.preprocessor_with_tranform = PreProcessor(
            self.dataset, self.data_dir, self.transforms_config
        )
        self.flow_mocker.assert_all(self.preprocessor_with_tranform)
        
        transforms_config_liftings = DictConfig(
            {"liftings": {"transform": {"transform_name": "CellCycleLifting"}}}
        )
        _ = self.preprocessor.instantiate_pre_transform(self.data_dir, transforms_config_liftings)

    @patch("topobench.data.preprocessor.preprocessor.load_inductive_splits")
    def test_load_dataset_splits_inductive(self, mock_load_inductive_splits):
        """Test loading dataset splits for inductive learning.
        
        Parameters
        ----------
        mock_load_inductive_splits : MagicMock
            A mock of the load_inductive_splits function.
        """
        split_params = DictConfig({"learning_setting": "inductive"})
        self.preprocessor.load_dataset_splits(split_params)
        mock_load_inductive_splits.assert_called_once_with(
            self.preprocessor, split_params
        )

    @patch(
        "topobench.data.preprocessor.preprocessor.load_transductive_splits"
    )
    def test_load_dataset_splits_transductive(
        self, mock_load_transductive_splits
    ):
        """Test loading dataset splits for transductive learning.
        
        Parameters
        ----------
        mock_load_transductive_splits : MagicMock
            A mock of the load_transductive_splits function.
        """
        split_params = DictConfig({"learning_setting": "transductive"})
        self.preprocessor.load_dataset_splits(split_params)
        mock_load_transductive_splits.assert_called_once_with(
            self.preprocessor, split_params
        )

    def test_invalid_learning_setting(self):
        """Test an invalid learning setting."""
        split_params = DictConfig({"learning_setting": "invalid"})
        with pytest.raises(ValueError):
            self.preprocessor.load_dataset_splits(split_params)

    def test_process_torch_utils_dataset(self):
        """Test the process method with torch.utils.data.Dataset."""
        mock_data = [1, 2, 3]
        mock_dataset = MockTorchDataset(mock_data)
        self.preprocessor.dataset = mock_dataset
        self.preprocessor.pre_transform = None
        self.preprocessor.collate = MagicMock(return_value=(torch_geometric.data.Data(), MagicMock())) # Corrected line
        self.preprocessor.save = MagicMock()

        # Mock the processed_paths property
        with patch.object(PreProcessor, "processed_paths", new_callable=PropertyMock) as mock_processed_paths:
            mock_processed_paths.return_value = ["/fake/path"]
            self.preprocessor.process()

        assert self.preprocessor.data_list == mock_data
        self.preprocessor.collate.assert_called_once_with(mock_data)
        self.preprocessor.save.assert_called_once()

    def test_process_torch_geometric_data_data(self):
        """Test the process method with torch_geometric.data.Data."""
        mock_data = torch_geometric.data.Data()
        self.preprocessor.dataset = mock_data
        self.preprocessor.pre_transform = None
        self.preprocessor.collate = MagicMock(return_value=(torch_geometric.data.Data(), MagicMock())) # Corrected line
        self.preprocessor.save = MagicMock()

        # Mock the processed_paths property
        with patch.object(PreProcessor, "processed_paths", new_callable=PropertyMock) as mock_processed_paths:
            mock_processed_paths.return_value = ["/fake/path"]
            self.preprocessor.process()

        assert self.preprocessor.data_list == [mock_data]
        self.preprocessor.collate.assert_called_once_with([mock_data])
        self.preprocessor.save.assert_called_once()
        
def test_pack_global_partition(tmp_path, monkeypatch):
    """
    Test pack_global_partition builds a valid handle and memmaps.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for data_dir.
    monkeypatch : pytest.MonkeyPatch
        Fixture to patch ClusterOnDisk.
    """
    # Fake ClusterOnDisk with minimal behavior
    class FakePartition:
        def __init__(self):
            # Two parts, four nodes: [0,2), [2,4)
            self.partptr = torch.tensor([0, 2, 4], dtype=torch.long)
            self.indptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
            self.index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            self.node_perm = torch.arange(4, dtype=torch.long)
            self.edge_perm = torch.arange(4, dtype=torch.long)

    class FakeClusterOnDisk:
        def __init__(
            self,
            root,
            *,
            graph_getter,
            num_parts,
            recursive,
            keep_inter_cluster_edges,
            sparse_format,
            backend,
            transform,
            pre_filter,
        ):
            self.root = root
            self.processed_dir = osp.join(root, "processed")
            os.makedirs(self.processed_dir, exist_ok=True)

            self.num_parts = int(num_parts)
            self.recursive = bool(recursive)
            self.keep_inter_cluster_edges = bool(keep_inter_cluster_edges)
            self.sparse_format = str(sparse_format)

            mm_dir = osp.join(self.processed_dir, "perm_memmap")
            os.makedirs(mm_dir, exist_ok=True)

            partptr = np.array([0, 2, 4], dtype=np.int64)
            indptr = np.array([0, 1, 2, 3, 4], dtype=np.int64)
            indices = np.array([0, 1, 2, 3], dtype=np.int64)
            np.save(osp.join(mm_dir, "partptr.npy"), partptr)
            np.save(osp.join(mm_dir, "indptr.npy"), indptr)
            np.save(osp.join(mm_dir, "indices.npy"), indices)

            self._partition = FakePartition()
            self.schema = {"edge_index": dict(dtype=torch.long, size=(2, -1))}

        @property
        def partition(self):
            """Return fake partition."""
            return self._partition

        def __len__(self):
            """Return number of cluster parts."""
            return self.num_parts

    # Patch ClusterOnDisk used inside preprocessor module
    monkeypatch.setattr(preproc_mod, "ClusterOnDisk", FakeClusterOnDisk)

    # Build a tiny full graph with masks
    edge_index = torch.tensor(
        [[0, 1, 2, 3],
         [1, 2, 3, 0]],
        dtype=torch.long,
    )
    x = torch.randn(4, 3)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False])
    val_mask = torch.tensor([False, False, True, False])
    test_mask = torch.tensor([False, False, False, True])

    full = torch_geometric.data.Data(
        edge_index=edge_index,
        x=x,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    class DatasetWrapper:
        def __init__(self, data):
            self.data = data

    dataset = DatasetWrapper(full)

    # Create a minimal PreProcessor-like instance without running __init__
    pre = object.__new__(PreProcessor)
    pre.dataset = dataset
    pre.data_dir = str(tmp_path)
    pre.load_dataset_splits = MagicMock(return_value=(None, None, None))

    split_params = DictConfig({"learning_setting": "transductive"})
    cluster_params = {
        "num_parts": 2,
        "recursive": False,
        "keep_inter_cluster_edges": False,
        "sparse_format": "csr",
    }
    stream_params = {"precompute_split_parts": True}

    handle = pre.pack_global_partition(
        split_params=split_params,
        cluster_params=cluster_params,
        stream_params=stream_params,
        dtype_policy="preserve",
    )

    # Basic handle structure checks
    assert isinstance(handle, dict)
    assert handle["root"] == pre.data_dir
    assert handle["num_parts"] == 2
    assert handle["sparse_format"] == "csr"
    assert handle["has_x"] is True
    assert handle["has_y"] is True

    memmap_dir = handle["memmap_dir"]
    assert osp.isdir(memmap_dir)

    # Mask memmaps must exist and have correct length
    for key in ("train_mask_perm", "val_mask_perm", "test_mask_perm"):
        path = handle["paths"][key]
        assert osp.exists(path)
        arr = np.load(path)
        assert arr.shape == (4,)

    # Cached handle should be reused on second call
    call_count_before = pre.load_dataset_splits.call_count
    handle2 = pre.pack_global_partition(
        split_params=split_params,
        cluster_params=cluster_params,
        stream_params=stream_params,
        dtype_policy="preserve",
    )
    call_count_after = pre.load_dataset_splits.call_count

    assert handle2["config_hash"] == handle["config_hash"]
    # No extra split loading on cached path
    assert call_count_after == call_count_before

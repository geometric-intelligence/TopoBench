"""Test the GraphLoader class."""

from unittest.mock import ANY, MagicMock, patch

import pytest
import torch_geometric
from omegaconf import DictConfig

from topobenchmark.data.preprocessor import PreProcessor

from ..._utils.flow_mocker import FlowMocker


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
        assert self.preprocessor.transforms_applied == False
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

    @patch("topobenchmark.data.preprocessor.preprocessor.load_inductive_splits")
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
        "topobenchmark.data.preprocessor.preprocessor.load_transductive_splits"
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

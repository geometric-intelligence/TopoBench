"""Tests for the OnDiskPreProcessor class."""

from unittest.mock import MagicMock, patch

import pytest
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data

from topobench.data.preprocessor.on_disk_preprocessor import (
    OnDiskPreProcessor,
    _LazyDataList,
)


@pytest.mark.usefixtures("mocker_fixture")
class TestOnDiskPreProcessor:
    """
    Test suite for OnDiskPreProcessor.
    """

    @pytest.fixture(autouse=True)
    def setup_method(self, mocker_fixture):
        """
        Set up a fresh OnDiskPreProcessor for each test.

        Parameters
        ----------
        mocker_fixture : pytest_mock.MockerFixture
            Mocker fixture used to patch internals.
        """
        mocker = mocker_fixture

        # Upstream dataset mocked as a regular PyG dataset.
        self.dataset = MagicMock(spec=torch_geometric.data.Dataset)
        self.data_dir = "fake/path"

        # Iteration over the upstream dataset should yield Data objects.
        self.dataset.__iter__.return_value = iter([Data(), Data(), Data()])
        self.dataset.transform = None

        # Patch the underlying OnDiskDataset.__init__ to avoid real I/O / DB.
        self.mock_ondisk_init = mocker.patch(
            "torch_geometric.data.OnDiskDataset.__init__",
            return_value=None,
        )

        # OnDiskPreProcessor.__init__ checks `len(self)` to decide whether
        # to append from the upstream dataset.
        self.mock_len = mocker.patch.object(
            OnDiskPreProcessor, "__len__", return_value=0
        )

        # Append should not touch the real on-disk DB.
        self.mock_append = mocker.patch.object(OnDiskPreProcessor, "append")

        # Instantiate without transforms.
        self.preprocessor = OnDiskPreProcessor(
            self.dataset, self.data_dir, None
        )

    def teardown_method(self):
        """
        Tear down state after each test.
        """
        del self.preprocessor

    def test_init_without_transforms(self):
        """
        Test initialization without transform configuration.

        Checks that OnDiskDataset.__init__ is called correctly, that
        transforms_applied is False and that samples are appended.
        """
        # super().__init__ should be called with the raw data_dir
        self.mock_ondisk_init.assert_called_once_with(
            root=self.data_dir, transform=None, pre_filter=None
        )

        # No transforms_config -> transforms_applied should be False
        assert self.preprocessor.transforms_applied is False

        # Since len(self) == 0 (mocked), we should have appended all
        # items from the upstream dataset.
        self.mock_len.assert_called_once()
        # Three samples in the iterator => 3 appends
        assert self.mock_append.call_count == 3

    def test_data_list_is_lazy_view(self):
        """
        Test that data_list is lazy and delegates to the dataset.

        Ensures that data_list returns a _LazyDataList and that
        _LazyDataList calls the underlying dataset __getitem__.
        """
        # 1) Ensure OnDiskPreProcessor.data_list returns a _LazyDataList
        dl = self.preprocessor.data_list
        assert isinstance(dl, _LazyDataList)

        # 2) Separately test that _LazyDataList delegates to the underlying
        #    dataset without involving PyG Dataset internals.
        class DummyDataset:
            """
            Minimal dataset used to validate delegation logic.
            """

            def __init__(self):
                """Initialize the dummy dataset."""
                self.calls = []

            def __len__(self):
                """
                Return dataset length.

                Returns
                -------
                int
                    Fixed length value.
                """
                return 3

            def __getitem__(self, idx):
                """
                Return a synthetic item.

                Parameters
                ----------
                idx : int
                    Index of the requested element.

                Returns
                -------
                str
                    Encoded index string.
                """
                self.calls.append(idx)
                return f"graph-{idx}"

        dummy = DummyDataset()
        lazy = _LazyDataList(dummy)

        out = lazy[0]
        assert out == "graph-0"
        assert dummy.calls == [0]

    def test_init_with_transforms(self, mocker_fixture):
        """
        Test initialization when transforms_config is provided.

        Verifies heavy and easy transforms are built and composed, and that
        transform parameters are saved.
        """
        mocker = mocker_fixture

        # Repatch OnDiskDataset.__init__ just to be explicit for this test
        mock_ondisk_init = mocker.patch(
            "torch_geometric.data.OnDiskDataset.__init__",
            return_value=None,
        )

        # Upstream dataset with an intrinsic transform
        dataset = MagicMock(spec=torch_geometric.data.Dataset)
        dataset.__iter__.return_value = iter([Data(), Data()])
        dataset.transform = "intrinsic_transform"

        # Heavy & easy transforms (we only care that they flow through)
        heavy_transform = MagicMock()
        heavy_transform.parameters = {"alpha": 1}
        easy_transform = MagicMock()
        easy_transform.parameters = {"beta": 2}

        heavy_dict = {"heavy": heavy_transform}
        easy_dict = {"easy": easy_transform}

        # _build_transform_dict is called twice:
        #   1) for heavy transforms in __init__
        #   2) for easy transforms in _prepare_online_transforms
        build_dict_mock = mocker.patch.object(
            OnDiskPreProcessor,
            "_build_transform_dict",
            side_effect=[(heavy_dict, {}), (easy_dict, {})],
        )

        # _compose_from_dict is called twice:
        #   1) once for the heavy/offline pipeline
        #   2) once for the easy/online part
        heavy_pipeline = MagicMock(name="heavy_pipeline")
        easy_pipeline = MagicMock(name="easy_pipeline")
        compose_mock = mocker.patch.object(
            OnDiskPreProcessor,
            "_compose_from_dict",
            side_effect=[heavy_pipeline, easy_pipeline],
        )

        save_params_mock = mocker.patch.object(
            OnDiskPreProcessor, "save_transform_parameters"
        )

        # Config with one heavy and one easy transform type
        transforms_config = {
            "lift": {"transform_type": "lifting", "foo": "bar"},
            "small": {"transform_type": "something_else", "baz": "qux"},
        }

        preproc = OnDiskPreProcessor(dataset, self.data_dir, transforms_config)

        # The dataset is created for a hashed subdirectory under data_dir.
        mock_ondisk_init.assert_called_once()
        _, kwargs = mock_ondisk_init.call_args
        assert "root" in kwargs
        assert kwargs["root"].startswith(self.data_dir)

        # Heavy transforms should be applied
        assert preproc.transforms_applied is True
        assert preproc.heavy_transforms is heavy_pipeline

        # _build_transform_dict called once for heavy, once for easy
        assert build_dict_mock.call_count == 2

        # We set heavy_pipeline and easy_pipeline as side effects:
        assert compose_mock.call_count == 2

        # Online transform should be a Compose(...) of
        #   [dataset.transform, easy_pipeline]
        from torch_geometric.transforms import Compose

        assert isinstance(preproc.transform, Compose)

        # Heavy transform parameters must have been persisted/validated
        save_params_mock.assert_called_once()

    def test_process_applies_heavy_and_respects_pre_filter(self):
        """
        Test the process method with heavy transforms and pre_filter.

        Uses a bare instance created with __new__ and a small in-memory
        dataset.
        """
        # Build a "bare" instance without running __init__
        proc = OnDiskPreProcessor.__new__(OnDiskPreProcessor)

        # Small in-memory source dataset
        data1 = Data()
        data2 = Data()
        proc.dataset = [data1, data2]

        # Heavy transforms pipeline (offline)
        heavy_transform = MagicMock(side_effect=lambda d: d)
        proc.heavy_transforms = heavy_transform

        # pre_filter that keeps every sample
        proc.pre_filter = lambda d: True

        # append should just be a mock (no real DB)
        proc.append = MagicMock()

        # Run process
        OnDiskPreProcessor.process(proc)

        # Both samples passed through heavy_transforms and appended
        assert heavy_transform.call_count == 2
        assert proc.append.call_count == 2

    def test_process_raises_on_non_basedata(self):
        """
        Test that process raises on non-BaseData samples.

        Ensures that TypeError is raised when dataset elements are not
        PyG BaseData instances.

        Raises
        ------
        TypeError
            If a non-BaseData object is processed.
        """
        proc = OnDiskPreProcessor.__new__(OnDiskPreProcessor)

        proc.dataset = [object()]  # not a BaseData
        proc.heavy_transforms = None
        proc.pre_filter = None
        proc.append = MagicMock()

        with pytest.raises(TypeError):
            OnDiskPreProcessor.process(proc)

    @patch(
        "topobench.data.preprocessor.on_disk_preprocessor."
        "load_inductive_splits_on_disk"
    )
    def test_load_dataset_splits_inductive(
        self, mock_load_inductive_splits_on_disk
    ):
        """
        Test loading dataset splits for inductive learning.

        Parameters
        ----------
        mock_load_inductive_splits_on_disk : unittest.mock.MagicMock
            Mock for load_inductive_splits_on_disk.
        """
        split_params = DictConfig({"learning_setting": "inductive"})
        self.preprocessor.load_dataset_splits(split_params)
        mock_load_inductive_splits_on_disk.assert_called_once_with(
            self.preprocessor, split_params
        )

    def test_invalid_learning_setting_raises(self):
        """
        Test that invalid learning settings raise ValueError.

        Raises
        ------
        ValueError
            If learning_setting is not supported.
        """
        split_params = DictConfig({"learning_setting": "transductive"})
        with pytest.raises(ValueError):
            self.preprocessor.load_dataset_splits(split_params)

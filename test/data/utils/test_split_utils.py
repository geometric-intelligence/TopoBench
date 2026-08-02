"""Unit tests for split utilities."""

import os
import re
import tempfile
import shutil
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig
from torch.utils.data import Subset
from torch_geometric.data import Data

from topobench.data.splits import (
    apply_transductive_split,
    indices_to_mask,
    inductive_split_views,
    validate_transductive_masks,
)

from topobench.data.utils.split_utils import (
    k_fold_split,
    k_fold_split_fixed,
    load_coauthorship_hypergraph_splits,
    load_inductive_splits,
    load_transductive_splits,
    random_splitting,
    stratified_splitting,
)


class TestLoadInductiveSplits:
    """Test load_inductive_splits function."""

    def setup_method(self):
        """Setup method for each test."""
        # Create temporary directory for test splits
        self.test_dir = tempfile.mkdtemp(prefix=".topobench_test_tmp_")

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_mock_dataset(self, n_graphs, label_shapes, has_get_data_dir=True):
        """Create a mock dataset with specified label shapes.

        Parameters
        ----------
        n_graphs : int
            Number of graphs in the dataset.
        label_shapes : list
            List of tuples representing label shapes for each graph.
        has_get_data_dir : bool
            Whether the dataset has get_data_dir method.

        Returns
        -------
        MagicMock
            Mock dataset object.
        """
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=n_graphs)

        # Create mock graphs with different label shapes
        mock_graphs = []
        for i, shape in enumerate(label_shapes):
            mock_graph = MagicMock()
            # Create labels with specified shape
            if len(shape) == 0:
                labels = np.array([i % 3])  # Single label
            else:
                labels = np.random.randint(0, 3, size=shape)
            mock_graph.y.squeeze.return_value.numpy.return_value = labels
            mock_graphs.append(mock_graph)

        mock_dataset.__getitem__ = lambda self, idx: mock_graphs[idx]
        mock_dataset.__iter__ = lambda self: iter(mock_graphs)

        # Setup dataset.dataset.get_data_dir()
        if has_get_data_dir:
            mock_dataset.dataset.get_data_dir.return_value = self.test_dir
        else:
            mock_dataset.dataset = MagicMock(spec=[])

        return mock_dataset

    def test_uniform_label_shapes_random_split(self):
        """Test with uniform label shapes using random split."""
        # Create dataset with uniform label shapes (all graphs have 1 label)
        n_graphs = 20
        label_shapes = [()] * n_graphs  # All single labels
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        # Verify splits exist and are non-empty
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

        # Verify total equals original
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_graphs

    @pytest.mark.parametrize("split_type", ["k-fold", "k-fold-fixed"])
    def test_unqualified_kfold_rejected_before_dataset_loading(
        self,
        split_type,
    ):
        """Unqualified nested CV fails before the source dataset is read."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.side_effect = AssertionError(
            "dataset loading must not start"
        )
        parameters = DictConfig({"split_type": split_type})

        with pytest.raises(
            ValueError,
            match=(
                rf"^split_type {re.escape(repr(split_type))} is not qualified: "
                r"nested cross-validation requires an outer held-out test "
                r"partition$"
            ),
        ):
            load_inductive_splits(mock_dataset, parameters)

        mock_dataset.__len__.assert_not_called()

    def test_stratified_split(self):
        """Test with stratified split."""
        n_graphs = 30
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "stratified",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_graphs

    def test_ragged_label_shapes_random_split(self):
        """Test with ragged label shapes (different sizes) using random split."""
        # Create dataset with varying label shapes
        n_graphs = 15
        label_shapes = [()] * 5 + [(2,)] * 5 + [(3,)] * 5  # Mix of shapes
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_graphs


    def test_fixed_split_type(self):
        """Test with fixed split type."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        # Add split_idx attribute
        split_idx = {
            "train": np.arange(12),
            "valid": np.arange(12, 16),
            "test": np.arange(16, 20)
        }
        mock_dataset.split_idx = split_idx

        parameters = DictConfig({
            "split_type": "fixed",
            "data_seed": 0,
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) == 12
        assert len(val_ds) == 4
        assert len(test_ds) == 4


    def test_fixed_split_type_without_split_idx_raises_error(self):
        """Test that fixed split without split_idx raises error."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)
        # Ensure split_idx attribute doesn't exist
        if hasattr(mock_dataset, 'split_idx'):
            delattr(mock_dataset, 'split_idx')

        parameters = DictConfig({
            "split_type": "fixed",
            "data_seed": 0,
        })

        with pytest.raises(NotImplementedError):
            load_inductive_splits(mock_dataset, parameters)

    def test_invalid_split_type_raises_error(self):
        """Test that invalid split type raises NotImplementedError."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "invalid_split_type",
            "data_seed": 0,
        })

        with pytest.raises(NotImplementedError, match="not valid"):
            load_inductive_splits(mock_dataset, parameters)

    def test_single_graph_raises_value_error(self):
        """An inductive source must contain multiple graphs."""
        n_graphs = 1
        label_shapes = [()]
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        with pytest.raises(ValueError, match="more than one graph"):
            load_inductive_splits(mock_dataset, parameters)

    def test_without_get_data_dir(self):
        """Test when dataset doesn't have get_data_dir method."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes, has_get_data_dir=False)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        # Should work fine without get_data_dir
        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_splits_are_index_backed_without_graph_masks(self):
        """Inductive phases remain lazy views over the original dataset."""
        n_graphs = 10
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)
        mock_dataset.split_idx = {
            "train": np.arange(6),
            "valid": np.arange(6, 8),
            "test": np.arange(8, 10),
        }

        train_ds, val_ds, test_ds = load_inductive_splits(
            mock_dataset,
            DictConfig({"split_type": "fixed"}),
        )

        assert all(
            isinstance(dataset, Subset)
            for dataset in (train_ds, val_ds, test_ds)
        )
        assert train_ds.dataset is mock_dataset
        assert val_ds.dataset is mock_dataset
        assert test_ds.dataset is mock_dataset
        assert list(train_ds.indices) == list(range(6))
        assert list(val_ds.indices) == [6, 7]
        assert list(test_ds.indices) == [8, 9]

    def test_different_data_seeds_produce_different_splits(self):
        """Test that different data seeds produce different splits."""
        n_graphs = 20
        label_shapes = [()] * n_graphs

        # First split
        mock_dataset1 = self.create_mock_dataset(n_graphs, label_shapes)
        parameters1 = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })
        train_ds1, _, _ = load_inductive_splits(mock_dataset1, parameters1)

        # Second split with different seed
        mock_dataset2 = self.create_mock_dataset(n_graphs, label_shapes)
        parameters2 = DictConfig({
            "split_type": "random",
            "data_seed": 1,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })
        train_ds2, _, _ = load_inductive_splits(mock_dataset2, parameters2)

        # Splits should have same size but potentially different composition
        assert len(train_ds1) == len(train_ds2)

    def test_multidimensional_ragged_labels(self):
        """Test with multidimensional ragged labels."""
        n_graphs = 12
        # Mix of different multidimensional shapes
        label_shapes = [(5,)] * 4 + [(10,)] * 4 + [(15,)] * 4
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.5,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_graphs


class TestRejectedKFoldSplits:
    """Direct k-fold helpers never return validation aliased as test."""

    @pytest.mark.parametrize(
        ("helper", "split_type"),
        [
            (k_fold_split, "k-fold"),
            (k_fold_split_fixed, "k-fold-fixed"),
        ],
    )
    def test_direct_helper_rejects_without_creating_split_artifacts(
        self,
        helper,
        split_type,
        tmp_path,
    ):
        """Legacy helper entry points explain the missing outer test."""
        parameters = DictConfig({
            "k": 2,
            "data_seed": 0,
            "data_split_dir": str(tmp_path / "data_splits"),
        })
        args = (
            (np.array([0, 1]), parameters)
            if helper is k_fold_split
            else (
                torch.tensor([0, 1]),
                parameters,
                {
                    "train": [np.array([0])],
                    "valid": [np.array([1])],
                    "test": [np.array([1])],
                },
            )
        )

        with pytest.raises(
            ValueError,
            match=(
                rf"^split_type {re.escape(repr(split_type))} is not qualified: "
                r"nested cross-validation requires an outer held-out test "
                r"partition$"
            ),
        ):
            helper(*args)

        assert not (tmp_path / "data_splits").exists()


class TestRandomSplitting:
    """Test random_splitting function."""

    def setup_method(self):
        """Setup method for each test."""
        self.test_dir = tempfile.mkdtemp(prefix=".topobench_test_tmp_")

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_basic_random_split(self):
        """Test basic random splitting."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        parameters = DictConfig({
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        split_idx = random_splitting(labels, parameters)

        assert "train" in split_idx
        assert "valid" in split_idx
        assert "test" in split_idx

        total = len(split_idx["train"]) + len(split_idx["valid"]) + len(split_idx["test"])
        assert total == len(labels)

    def test_random_split_proportions(self):
        """Test that random split respects train_prop."""
        labels = np.array([0, 1, 2] * 100)  # 300 samples
        parameters = DictConfig({
            "data_seed": 0,
            "train_prop": 0.7,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        split_idx = random_splitting(labels, parameters)

        train_ratio = len(split_idx["train"]) / len(labels)
        # Should be approximately 0.7
        assert 0.65 < train_ratio < 0.75

    def test_random_split_with_custom_seed(self):
        """Test random splitting with custom global seed."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        parameters = DictConfig({
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        split_idx = random_splitting(labels, parameters, global_data_seed=999)

        assert "train" in split_idx
        # Verify split directory reflects custom seed
        split_dir = os.path.join(self.test_dir, "data_splits", "train_prop=0.6_global_seed=999")
        assert os.path.exists(split_dir)


class TestStratifiedSplitting:
    """Test stratified_splitting function."""

    def setup_method(self):
        """Setup method for each test."""
        self.test_dir = tempfile.mkdtemp(prefix=".topobench_test_tmp_")

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_basic_stratified_split(self):
        """Test basic stratified splitting."""
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        parameters = DictConfig({
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        split_idx = stratified_splitting(labels, parameters)

        assert "train" in split_idx
        assert "valid" in split_idx
        assert "test" in split_idx

        total = len(split_idx["train"]) + len(split_idx["valid"]) + len(split_idx["test"])
        assert total == len(labels)




class TestLoadTransductiveSplits:
    """Test native transductive splitting and canonical masks."""

    def setup_method(self):
        """Setup method for each test."""
        self.test_dir = tempfile.mkdtemp(prefix=".topobench_test_tmp_")

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @staticmethod
    def dataset(labels: torch.Tensor) -> list[Data]:
        """Build one native graph dataset."""
        return [Data(x=torch.randn(labels.numel(), 4), y=labels)]

    def test_transductive_indices_become_boolean_masks(self):
        """Index arrays become disjoint, complete, full-length masks."""
        data = self.dataset(torch.tensor([0, 1, 0, 1]))[0]

        apply_transductive_split(
            data,
            train=[0, 2],
            val=[1],
            test=[3],
        )

        assert data.train_mask.dtype is torch.bool
        assert data.train_mask.tolist() == [True, False, True, False]
        assert not torch.any(data.train_mask & data.val_mask)
        assert torch.all(
            data.train_mask | data.val_mask | data.test_mask
        )

    @pytest.mark.parametrize(
        ("train", "val", "test", "message"),
        [
            ([0], [], [1, 2], "split masks must be non-empty"),
            ([0, 1], [1], [2], "split masks must be disjoint"),
            ([0], [1], [2], "split masks must cover all labeled nodes"),
        ],
    )
    def test_transductive_split_invariants(
        self,
        train,
        val,
        test,
        message,
    ):
        """Invalid split masks fail at the representation boundary."""
        data = self.dataset(torch.tensor([0, 1, 0, 1]))[0]

        with pytest.raises(ValueError, match=f"^{message}$"):
            apply_transductive_split(
                data,
                train=train,
                val=val,
                test=test,
            )

    def test_mask_validation_rejects_shape_and_dtype(self):
        """Canonical masks are one-dimensional, boolean, and full-length."""
        data = self.dataset(torch.tensor([0, 1, 0]))[0]
        data.train_mask = torch.tensor([[True, False, False]])
        data.val_mask = torch.tensor([False, True, False])
        data.test_mask = torch.tensor([False, False, True])

        with pytest.raises(
            ValueError,
            match="^train_mask must be a rank-1 boolean mask$",
        ):
            validate_transductive_masks(data)

        data.train_mask = torch.tensor([1, 0, 0])
        with pytest.raises(
            ValueError,
            match="^train_mask must be a rank-1 boolean mask$",
        ):
            validate_transductive_masks(data)

    def test_indices_to_mask_validates_index_contract(self):
        """Index conversion rejects non-integral, repeated, and out-of-range indices."""
        assert indices_to_mask([0, 2], 3).tolist() == [True, False, True]
        with pytest.raises(TypeError, match="^indices must contain integers$"):
            indices_to_mask([0.5], 3)
        with pytest.raises(ValueError, match="^indices must be unique$"):
            indices_to_mask([0, 0], 3)
        with pytest.raises(ValueError, match="^indices must be in \\[0, 3\\)$"):
            indices_to_mask([3], 3)

    @pytest.mark.parametrize("split_type", ["random", "stratified"])
    def test_generated_transductive_splits_are_canonical(self, split_type):
        """All supported index algorithms feed the same mask boundary."""
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        parameters = {
            "split_type": split_type,
            "data_seed": 0,
            "data_split_dir": self.test_dir,
        }
        parameters["train_prop"] = 0.5

        dataset, val, test = load_transductive_splits(
            self.dataset(labels),
            DictConfig(parameters),
        )

        assert val is None
        assert test is None
        assert isinstance(dataset, list)
        assert len(dataset) == 1
        validate_transductive_masks(dataset[0])

    def test_fixed_transductive_split_validates_existing_masks(self):
        """A packaged fixed node split follows the same canonical contract."""
        data = self.dataset(torch.tensor([0, 1, 0]))[0]
        apply_transductive_split(data, train=[0], val=[1], test=[2])

        dataset, val, test = load_transductive_splits(
            [data],
            DictConfig({"split_type": "fixed"}),
        )

        assert dataset == [data]
        assert val is None
        assert test is None

    def test_fixed_index_masks_are_canonicalized(self):
        """Legacy fixed index fields are converted at the split boundary."""
        data = self.dataset(torch.tensor([0, 1, 0, 1]))[0]
        data.train_mask = torch.tensor([0, 2])
        data.val_mask = torch.tensor([1])
        data.test_mask = torch.tensor([3])

        dataset, _, _ = load_transductive_splits(
            [data],
            DictConfig({"split_type": "fixed"}),
        )

        assert dataset[0].train_mask.dtype is torch.bool
        assert dataset[0].train_mask.tolist() == [True, False, True, False]




    def test_invalid_split_type_raises_error(self):
        """Unsupported choices list only qualified split strategies."""
        with pytest.raises(
            NotImplementedError,
            match=(
                r"^split_type invalid not valid\. Choose either "
                r"'random', 'stratified', or 'fixed'$"
            ),
        ):
            load_transductive_splits(
                self.dataset(torch.tensor([0, 1])),
                DictConfig({"split_type": "invalid"}),
            )


class TestLoadCoauthorshipHypergraphSplits:
    """Test load_coauthorship_hypergraph_splits function."""

    def setup_method(self):
        """Setup method for each test."""
        self.test_dir = tempfile.mkdtemp(prefix=".topobench_test_tmp_")

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_coauthorship_splits(self):
        """Test loading coauthorship hypergraph splits."""
        data = MagicMock()
        data.num_nodes = 10

        train_prop = 0.5
        parameters = DictConfig({
            "data_seed": 0,
            "data_split_dir": self.test_dir
        })

        # Create the expected directory and file
        split_dir = os.path.join(self.test_dir, f"train_prop={train_prop}")
        os.makedirs(split_dir, exist_ok=True)

        split_path = os.path.join(split_dir, "split_0.npz")
        split_idx = {
            "train": np.array([0, 1, 2, 3, 4]),
            "valid": np.array([5, 6, 7]),
            "test": np.array([8, 9])
        }
        np.savez(split_path, **split_idx)

        dataset, _, _ = load_coauthorship_hypergraph_splits(data, parameters, train_prop=train_prop)

        assert len(dataset) == 1
        assert torch.equal(data.train_mask, torch.from_numpy(split_idx["train"]))


class TestInductiveSubsetViews:
    """Test the narrow inductive representation boundary."""
    @pytest.mark.parametrize(
        ("split_idx", "message"),
        [
            (
                {"train": [0, 1], "valid": [1], "test": [2, 3]},
                "train, valid, and test splits must be pairwise disjoint",
            ),
            (
                {"train": [0], "valid": [1], "test": [2]},
                "train, valid, and test splits must cover every source index "
                "exactly once",
            ),
            (
                {"train": [0, 0], "valid": [1], "test": [2, 3]},
                "train split indices must be unique",
            ),
            (
                {"train": [0], "valid": [1], "test": [2, 4]},
                "test split indices must be in [0, 4)",
            ),
            (
                {"train": [0], "valid": [], "test": [1, 2, 3]},
                "valid split must not be empty",
            ),
            (
                {"train": [0], "valid": [1.5], "test": [2, 3]},
                "valid split indices must contain integers",
            ),
        ],
        ids=[
            "phase-overlap",
            "incomplete-coverage",
            "duplicate-within-phase",
            "out-of-range",
            "empty-phase",
            "non-integral",
        ],
    )
    def test_invalid_partition_fails_with_context(
        self,
        split_idx,
        message,
    ):
        """Malformed partitions name the phase or cross-phase invariant."""
        with pytest.raises(
            ValueError,
            match=rf"^{re.escape(message)}$",
        ):
            inductive_split_views(list(range(4)), split_idx)

    def test_unordered_valid_partition_preserves_phase_order(self):
        """Unordered indices are valid and retain their lazy view order."""
        source = list(range(6))

        train_ds, val_ds, test_ds = inductive_split_views(
            source,
            {
                "train": [2, 0],
                "valid": [5, 3],
                "test": [4, 1],
            },
        )

        assert list(train_ds.indices) == [2, 0]
        assert list(val_ds.indices) == [5, 3]
        assert list(test_ds.indices) == [4, 1]

    def test_complete_fixed_split_views_share_source_without_item_access(self):
        """Complete fixed phases create lazy views over one source dataset."""

        class ItemAccessTrackingDataset:
            def __init__(self, size):
                self.size = size
                self.item_accesses = 0

            def __len__(self):
                return self.size

            def __getitem__(self, index):
                self.item_accesses += 1
                raise AssertionError(
                    f"split construction retrieved graph item {index}"
                )

        source = ItemAccessTrackingDataset(10)
        train_ds, val_ds, test_ds = inductive_split_views(
            source,
            {
                "train": [0, 1, 2, 3, 4],
                "valid": [5, 6, 7],
                "test": [8, 9],
            },
        )

        assert isinstance(train_ds, Subset)
        assert isinstance(val_ds, Subset)
        assert isinstance(test_ds, Subset)
        assert train_ds.dataset is val_ds.dataset is test_ds.dataset is source
        assert source.item_accesses == 0



"""Unit tests for ClassifierWrapper class."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.classification import (
    ClassifierWrapper,
)


class TestClassifierWrapperInitialization:
    """Test ClassifierWrapper initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        assert wrapper.backbone == backbone
        assert wrapper.use_embeddings is True
        assert wrapper.use_node_features is True
        assert wrapper.sampler is None

    def test_init_with_sampler(self):
        """Test initialization with sampler."""
        backbone = LogisticRegression()
        sampler = MagicMock()
        wrapper = ClassifierWrapper(backbone, sampler=sampler)
        assert wrapper.sampler == sampler


class TestClassifierWrapperInitTargets:
    """Test _init_targets method."""

    def test_init_targets_binary(self):
        """Test _init_targets with binary classification."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        y_train = np.array([0, 1, 1, 0, 1])
        wrapper._init_targets(y_train)

        assert wrapper.num_classes_ == 2
        assert len(wrapper.classes_) == 2
        assert wrapper.most_common_class_ == 1
        assert np.allclose(wrapper.class_distribution_, [0.4, 0.6])
        assert np.allclose(wrapper.uniform_, [0.5, 0.5])

    def test_init_targets_multiclass(self):
        """Test _init_targets with multiclass classification."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        y_train = np.array([0, 1, 2, 0, 1, 2, 0])
        wrapper._init_targets(y_train)

        assert wrapper.num_classes_ == 3
        assert len(wrapper.classes_) == 3
        assert wrapper.most_common_class_ == 0
        assert np.allclose(wrapper.uniform_, [1/3, 1/3, 1/3])


class TestClassifierWrapperNoNeighbors:
    """Test _no_neighbors method."""

    def test_no_neighbors_binary(self):
        """Test _no_neighbors for binary classification."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1, 1, 0]))

        batch_size = 5
        probs, preds = wrapper._no_neighbors(batch_size)

        assert probs.shape == (batch_size, 2)
        assert len(preds) == batch_size
        assert all(p == wrapper.most_common_class_ for p in preds)
        np.testing.assert_array_almost_equal(probs[0], wrapper.class_distribution_)

    def test_no_neighbors_multiclass(self):
        """Test _no_neighbors for multiclass classification."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1, 2, 0, 1]))

        batch_size = 3
        probs, preds = wrapper._no_neighbors(batch_size)

        assert probs.shape == (batch_size, 3)
        assert len(preds) == batch_size


class TestClassifierWrapperOneNeighbor:
    """Test _one_neighbor method."""

    def test_one_neighbor_binary(self):
        """Test _one_neighbor for binary classification."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1, 1, 0]))

        labels = np.array([1])
        batch_size = 3
        probs, preds = wrapper._one_neighbor(labels, batch_size)

        assert probs.shape == (batch_size, 2)
        assert len(preds) == batch_size
        assert all(p == 1 for p in preds)
        np.testing.assert_array_equal(probs[0], [0.0, 1.0])

    def test_one_neighbor_multiclass(self):
        """Test _one_neighbor for multiclass classification."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1, 2]))

        labels = np.array([2])
        batch_size = 4
        probs, preds = wrapper._one_neighbor(labels, batch_size)

        assert probs.shape == (batch_size, 3)
        assert all(p == 2 for p in preds)
        np.testing.assert_array_equal(probs[0], [0.0, 0.0, 1.0])


class TestClassifierWrapperAllFeaturesConstant:
    """Test _all_features_constant method."""

    def test_all_features_constant(self):
        """Test _all_features_constant."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1, 1, 0, 1]))

        labels = np.array([1, 1, 1])
        batch_size = 2
        probs, preds = wrapper._all_features_constant(labels, batch_size)

        assert probs.shape == (batch_size, 2)
        assert len(preds) == batch_size
        assert all(p == 1 for p in preds)

    def test_all_features_constant_empty_labels(self):
        """Test _all_features_constant with empty labels."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1]))

        labels = np.array([])
        batch_size = 2
        probs, preds = wrapper._all_features_constant(labels, batch_size)

        assert probs.shape == (batch_size, 2)
        assert len(preds) == batch_size


class TestClassifierWrapperGetPredictions:
    """Test _get_predictions method."""

    def test_get_predictions_with_predict_proba(self):
        """Test _get_predictions with model that has predict_proba."""
        backbone = LogisticRegression()
        backbone.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
        backbone.classes_ = np.array([0, 1])

        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1]))

        X_test = np.random.randn(5, 5)
        probs, preds = wrapper._get_predictions(backbone, X_test)

        assert len(probs) == 5
        assert len(preds) == 5
        assert all(isinstance(p, (int, np.integer)) for p in preds)

    def test_get_predictions_without_predict_proba(self):
        """Test _get_predictions with model without predict_proba."""
        backbone = DecisionTreeClassifier()
        backbone.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))
        backbone.classes_ = np.array([0, 1])

        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1]))

        X_test = np.random.randn(3, 5)
        # Mock predict_proba to raise AttributeError
        with patch.object(backbone, 'predict_proba', side_effect=AttributeError()):
            probs, preds = wrapper._get_predictions(backbone, X_test)

        assert len(probs) == 3
        assert len(preds) == 3

    def test_get_predictions_class_mapping(self):
        """Test _get_predictions handles class ordering correctly."""
        backbone = LogisticRegression()
        # Train with non-contiguous classes
        X_train = np.random.randn(10, 5)
        y_train = np.array([0, 2, 0, 2, 1, 1, 0, 2, 1, 0])
        backbone.fit(X_train, y_train)
        backbone.classes_ = np.array([0, 1, 2])

        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(y_train)

        X_test = np.random.randn(3, 5)
        probs, preds = wrapper._get_predictions(backbone, X_test)

        assert len(probs) == 3
        assert len(preds) == 3
        # Check that probabilities sum to 1
        for prob_row in probs:
            assert isinstance(prob_row, (list, np.ndarray))
            if isinstance(prob_row, np.ndarray):
                assert np.allclose(np.sum(prob_row), 1.0)


class TestClassifierWrapperProcessOutput:
    """Test _process_output method."""

    def test_process_output(self):
        """Test _process_output."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1]))

        output_tensor = torch.randn(5, 2)
        num_dataset_points = 10

        empty_tensor, processed_output = wrapper._process_output(
            output_tensor, num_dataset_points
        )

        assert empty_tensor.shape == (10, 2)
        assert processed_output.shape == (5, 2)
        assert empty_tensor.device == output_tensor.device
        assert processed_output.device == output_tensor.device


class TestClassifierWrapperFullGraphTraining:
    """Test _full_graph_training method."""

    def test_full_graph_training_with_predict_proba(self):
        """Test _full_graph_training with predict_proba."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1]))

        X_train = np.random.randn(10, 5)
        y_train = np.random.randint(0, 2, 10)
        X_test = np.random.randn(5, 5)
        device = torch.device("cpu")

        output = wrapper._full_graph_training(X_train, y_train, X_test, device)

        assert output.shape == (5, 2)
        assert isinstance(output, torch.Tensor)
        assert output.device == device

    def test_full_graph_training_without_predict_proba(self):
        """Test _full_graph_training without predict_proba."""
        backbone = DecisionTreeClassifier()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1]))

        X_train = np.random.randn(10, 5)
        y_train = np.random.randint(0, 2, 10)
        X_test = np.random.randn(3, 5)
        device = torch.device("cpu")

        # Mock predict_proba to raise AttributeError
        with patch.object(backbone, 'predict_proba', side_effect=AttributeError()):
            output = wrapper._full_graph_training(X_train, y_train, X_test, device)

        assert output.shape == (3, 2)
        assert isinstance(output, torch.Tensor)


class TestClassifierWrapperUpdateProgress:
    """Test _update_progress_and_results method."""

    def test_update_progress_and_results(self):
        """Test _update_progress_and_results."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone)
        wrapper._init_targets(np.array([0, 1]))

        probs = [[0.3, 0.7], [0.8, 0.2]]
        predictions = [1, 0]
        true_labels = torch.tensor([1, 0])
        outputs = []
        preds = []
        trues = []
        pbar = MagicMock()

        wrapper._update_progress_and_results(
            probs, predictions, true_labels, outputs, preds, trues, pbar
        )

        assert len(outputs) == 2
        assert len(preds) == 2
        assert len(trues) == 2
        assert pbar.update.called
        assert pbar.set_postfix.called


class TestClassifierWrapperForward:
    """Test forward method integration."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        num_nodes = 10
        batch = {
            "x_0": torch.randn(num_nodes, 5),
            "y": torch.randint(0, 2, (num_nodes,)),
            "edge_index": torch.randint(0, num_nodes, (2, 20)),
            "batch_0": torch.zeros(num_nodes, dtype=torch.long),
            "train_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "val_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "test_mask": torch.zeros(num_nodes, dtype=torch.bool),
        }
        batch["train_mask"][:6] = True
        batch["test_mask"][6:] = True
        return batch

    def test_forward_without_sampler(self, sample_batch):
        """Test forward pass without sampler."""
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone, sampler=None)

        output = wrapper.forward(sample_batch)

        assert "labels" in output
        assert "batch_0" in output
        assert "x_0" in output
        assert output["x_0"].shape[0] == len(sample_batch["y"])

    def test_forward_with_sampler(self, sample_batch):
        """Test forward pass with sampler."""
        backbone = LogisticRegression()
        sampler = MagicMock()
        sampler.fit = MagicMock()
        sampler.sample = MagicMock(return_value=np.array([0, 1, 2]))

        wrapper = ClassifierWrapper(backbone, sampler=sampler, num_test_nodes=1)

        # Mock the safe predictor
        wrapper.safe.predict_batch = MagicMock(
            return_value=(
                [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]],
                [0, 1, 0, 1],
                "normal",
            )
        )

        output = wrapper.forward(sample_batch)

        assert "labels" in output
        assert "batch_0" in output
        assert "x_0" in output
        assert sampler.fit.called
        assert sampler.sample.called

    def test_forward_with_index_format_masks(self):
        """Test forward when train/val/test are index tensors (e.g. cocitation_cora)."""
        num_nodes = 10
        batch = {
            "x_0": torch.randn(num_nodes, 5),
            "y": torch.randint(0, 2, (num_nodes,)),
            "edge_index": torch.randint(0, num_nodes, (2, 20)),
            "batch_0": torch.zeros(num_nodes, dtype=torch.long),
            "train_mask": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
            "val_mask": torch.tensor([6, 7], dtype=torch.long),
            "test_mask": torch.tensor([8, 9], dtype=torch.long),
        }
        backbone = LogisticRegression()
        wrapper = ClassifierWrapper(backbone, sampler=None)
        output = wrapper.forward(batch)
        assert "x_0" in output
        assert output["x_0"].shape[0] == num_nodes

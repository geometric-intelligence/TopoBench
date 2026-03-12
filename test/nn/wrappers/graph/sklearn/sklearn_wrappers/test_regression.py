"""Unit tests for RegressorWrapper class."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.regression import (
    RegressorWrapper,
)


class TestRegressorWrapperInitialization:
    """Test RegressorWrapper initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        assert wrapper.backbone == backbone
        assert wrapper.use_embeddings is True
        assert wrapper.use_node_features is True
        assert wrapper.sampler is None
        assert wrapper.global_mean_ == 0.0

    def test_init_with_sampler(self):
        """Test initialization with sampler."""
        backbone = LinearRegression()
        sampler = MagicMock()
        wrapper = RegressorWrapper(backbone, sampler=sampler)
        assert wrapper.sampler == sampler


class TestRegressorWrapperInitTargets:
    """Test _init_targets method."""

    def test_init_targets(self):
        """Test _init_targets computes global mean."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wrapper._init_targets(y_train)

        assert wrapper.global_mean_ == 3.0

    def test_init_targets_single_value(self):
        """Test _init_targets with single value."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        y_train = np.array([5.5])
        wrapper._init_targets(y_train)

        assert wrapper.global_mean_ == 5.5


class TestRegressorWrapperNoNeighbors:
    """Test _no_neighbors method."""

    def test_no_neighbors(self):
        """Test _no_neighbors returns global mean."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        wrapper._init_targets(np.array([1.0, 2.0, 3.0]))

        batch_size = 5
        probs, preds = wrapper._no_neighbors(batch_size)

        assert probs.shape == (batch_size,)
        assert preds.shape == (batch_size,)
        assert np.allclose(probs, wrapper.global_mean_)
        assert np.allclose(preds, wrapper.global_mean_)
        np.testing.assert_array_equal(probs, preds)


class TestRegressorWrapperOneNeighbor:
    """Test _one_neighbor method."""

    def test_one_neighbor(self):
        """Test _one_neighbor returns neighbor label."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        wrapper._init_targets(np.array([1.0, 2.0, 3.0]))

        labels = np.array([2.5])
        batch_size = 4
        probs, preds = wrapper._one_neighbor(labels, batch_size)

        assert probs.shape == (batch_size,)
        assert preds.shape == (batch_size,)
        assert np.allclose(probs, 2.5)
        assert np.allclose(preds, 2.5)
        np.testing.assert_array_equal(probs, preds)


class TestRegressorWrapperAllFeaturesConstant:
    """Test _all_features_constant method."""

    def test_all_features_constant(self):
        """Test _all_features_constant returns mean of labels."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        wrapper._init_targets(np.array([1.0, 2.0, 3.0]))

        labels = np.array([2.0, 3.0, 4.0])
        batch_size = 3
        probs, preds = wrapper._all_features_constant(labels, batch_size)

        assert probs.shape == (batch_size,)
        assert preds.shape == (batch_size,)
        expected_mean = np.mean(labels)
        assert np.allclose(probs, expected_mean)
        assert np.allclose(preds, expected_mean)
        np.testing.assert_array_equal(probs, preds)


class TestRegressorWrapperGetPredictions:
    """Test _get_predictions method."""

    def test_get_predictions(self):
        """Test _get_predictions."""
        backbone = LinearRegression()
        backbone.fit(np.random.randn(10, 5), np.random.randn(10))

        wrapper = RegressorWrapper(backbone)
        wrapper._init_targets(np.random.randn(10))

        X_test = np.random.randn(5, 5)
        probs, preds = wrapper._get_predictions(backbone, X_test)

        assert len(probs) == 5
        assert len(preds) == 5
        # For regression, probs and preds should be the same
        np.testing.assert_array_equal(probs, preds)
        assert all(isinstance(p, (int, float, np.number)) for p in preds)


class TestRegressorWrapperProcessOutput:
    """Test _process_output method."""

    def test_process_output(self):
        """Test _process_output."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        wrapper._init_targets(np.array([1.0, 2.0, 3.0]))

        output_tensor = torch.randn(5, 1)
        num_dataset_points = 10

        empty_tensor, processed_output = wrapper._process_output(
            output_tensor, num_dataset_points
        )

        assert empty_tensor.shape == (10, 1)
        assert processed_output.shape == (5, 1)
        assert empty_tensor.device == output_tensor.device
        assert processed_output.device == output_tensor.device

    def test_process_output_reshapes_correctly(self):
        """Test _process_output handles different input shapes."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        wrapper._init_targets(np.array([1.0, 2.0]))

        # Test with 1D tensor
        output_tensor = torch.randn(5)
        num_dataset_points = 8

        empty_tensor, processed_output = wrapper._process_output(
            output_tensor, num_dataset_points
        )

        assert empty_tensor.shape == (8, 1)
        assert processed_output.shape == (5, 1)


class TestRegressorWrapperFullGraphTraining:
    """Test _full_graph_training method."""

    def test_full_graph_training(self):
        """Test _full_graph_training."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        wrapper._init_targets(np.array([1.0, 2.0, 3.0]))

        X_train = np.random.randn(10, 5)
        y_train = np.random.randn(10)
        X_test = np.random.randn(5, 5)
        device = torch.device("cpu")

        output = wrapper._full_graph_training(X_train, y_train, X_test, device)

        assert output.shape == (5, 1)
        assert isinstance(output, torch.Tensor)
        assert output.device == device


class TestRegressorWrapperUpdateProgress:
    """Test _update_progress_and_results method."""

    def test_update_progress_and_results(self):
        """Test _update_progress_and_results."""
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone)
        wrapper._init_targets(np.array([1.0, 2.0, 3.0]))

        probs = [1.5, 2.5, 3.5]
        predictions = [1.5, 2.5, 3.5]
        true_labels = torch.tensor([1.0, 2.0, 3.0])
        outputs = []
        preds = []
        trues = []
        pbar = MagicMock()

        wrapper._update_progress_and_results(
            probs, predictions, true_labels, outputs, preds, trues, pbar
        )

        assert len(outputs) == 3
        assert len(preds) == 3
        assert len(trues) == 3
        assert pbar.update.called
        assert pbar.set_postfix.called
        # Check that MSE is in the postfix
        call_args = pbar.set_postfix.call_args[0][0]
        assert "MSE" in call_args


class TestRegressorWrapperForward:
    """Test forward method integration."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        num_nodes = 10
        batch = {
            "x_0": torch.randn(num_nodes, 5),
            "y": torch.randn(num_nodes),
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
        backbone = LinearRegression()
        wrapper = RegressorWrapper(backbone, sampler=None)

        output = wrapper.forward(sample_batch)

        assert "labels" in output
        assert "batch_0" in output
        assert "x_0" in output
        assert output["x_0"].shape[0] == len(sample_batch["y"])
        assert output["x_0"].shape[1] == 1  # Regression output is 1D

    def test_forward_with_sampler(self, sample_batch):
        """Test forward pass with sampler."""
        backbone = LinearRegression()
        sampler = MagicMock()
        sampler.fit = MagicMock()
        sampler.sample = MagicMock(return_value=np.array([0, 1, 2]))

        wrapper = RegressorWrapper(backbone, sampler=sampler, num_test_nodes=1)

        # Mock the safe predictor
        wrapper.safe.predict_batch = MagicMock(
            return_value=(
                [1.5, 2.5, 3.5, 4.5],
                [1.5, 2.5, 3.5, 4.5],
                "normal",
            )
        )

        output = wrapper.forward(sample_batch)

        assert "labels" in output
        assert "batch_0" in output
        assert "x_0" in output
        assert sampler.fit.called
        assert sampler.sample.called
        assert output["x_0"].shape[1] == 1  # Regression output is 1D

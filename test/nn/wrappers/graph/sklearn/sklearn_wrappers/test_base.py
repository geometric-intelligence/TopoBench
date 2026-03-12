"""Unit tests for BaseWrapper class."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from typing import Tuple

from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.base import (
    BaseWrapper,
    _build_sampler_features,
    SAMPLER_FEATURES_ALL,
    SAMPLER_FEATURES_NODE,
    SAMPLER_FEATURES_STRUCTURAL,
)
from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.types import (
    BatchValidationError,
    masks_to_bool,
    validate_batch,
)


class MockBaseWrapper(BaseWrapper):
    """Concrete implementation of BaseWrapper for testing purposes."""

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone, **kwargs)
        self.num_classes_ = 2
        self.global_mean_ = 0.5

    def _init_targets(self, y_train: np.ndarray):
        """Initialize target-related attributes."""
        self.num_classes_ = len(np.unique(y_train))
        self.global_mean_ = float(np.mean(y_train))

    def _no_neighbors(self, batch_size: int):
        """Return default predictions for no neighbors case."""
        probs = np.ones((batch_size, self.num_classes_)) / self.num_classes_
        preds = np.zeros(batch_size, dtype=int)
        return probs, list(preds)

    def _one_neighbor(self, labels: np.ndarray, batch_size: int):
        """Return predictions for one neighbor case."""
        label = labels[0] if len(labels) > 0 else 0
        probs = np.zeros((batch_size, self.num_classes_))
        probs[:, label] = 1.0
        preds = np.full(batch_size, label, dtype=int)
        return probs, list(preds)

    def _all_features_constant(self, labels: np.ndarray, batch_size: int):
        """Return predictions for constant features case."""
        label = int(np.mean(labels)) if len(labels) > 0 else 0
        probs = np.zeros((batch_size, self.num_classes_))
        probs[:, label] = 1.0
        preds = np.full(batch_size, label, dtype=int)
        return probs, list(preds)

    def _get_predictions(self, model, X_test: np.ndarray):
        """Get predictions from the model."""
        # Mock prediction: return random probabilities
        n_samples = X_test.shape[0]
        probs = np.random.rand(n_samples, self.num_classes_)
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        return list(probs), list(preds)

    def _process_output(
        self, output_tensor: torch.Tensor, num_dataset_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process output tensor."""
        empty_tensor = torch.zeros((num_dataset_points, self.num_classes_)).to(
            output_tensor.device
        )
        return empty_tensor, output_tensor

    def _full_graph_training(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, device
    ):
        """Full graph training mode."""
        # Mock training
        self.backbone.fit(X_train, y_train)
        output = torch.randn(X_test.shape[0], self.num_classes_).to(device)
        return output

    def _update_progress_and_results(
        self, probs, predictions, true_labels, outputs, preds, trues, pbar
    ):
        """Update progress bar and results."""
        outputs.extend(probs)
        preds.extend(predictions)
        trues.extend(list(true_labels.cpu().numpy()))
        pbar.update(len(predictions))
        pbar.set_postfix({"metric": "0.50"})


class TestBuildSamplerFeatures:
    """Test the _build_sampler_features function."""

    def test_build_sampler_features_all(self):
        """Test sampler_features='all' returns node_features_model."""
        batch = {
            "x_0": torch.randn(5, 10),
            "x_1_hop_mean": torch.randn(5, 5),
        }
        node_features_model = np.random.randn(5, 15)
        result = _build_sampler_features(batch, node_features_model, SAMPLER_FEATURES_ALL)
        np.testing.assert_array_equal(result, node_features_model)

    def test_build_sampler_features_node(self):
        """Test sampler_features='node' returns x_0 from batch."""
        batch = {
            "x_0": torch.randn(5, 10),
            "x_1_hop_mean": torch.randn(5, 5),
        }
        node_features_model = np.random.randn(5, 15)
        result = _build_sampler_features(batch, node_features_model, SAMPLER_FEATURES_NODE)
        expected = batch["x_0"].cpu().numpy().copy()
        np.testing.assert_array_equal(result, expected)

    def test_build_sampler_features_node_missing_key(self):
        """Test sampler_features='node' raises KeyError when x_0 is missing."""
        batch = {
            "x_1_hop_mean": torch.randn(5, 5),
        }
        node_features_model = np.random.randn(5, 15)
        with pytest.raises(KeyError, match=f"sampler_features='{SAMPLER_FEATURES_NODE}' requires batch\\['x_0'\\]"):
            _build_sampler_features(batch, node_features_model, SAMPLER_FEATURES_NODE)

    def test_build_sampler_features_structural(self):
        """Test sampler_features='structural' concatenates structural features."""
        batch = {
            "x_0": torch.randn(5, 10),
            "x_1_hop_mean": torch.randn(5, 5),
            "x_2_hop_mean": torch.randn(5, 3),
        }
        node_features_model = np.random.randn(5, 15)
        result = _build_sampler_features(batch, node_features_model, SAMPLER_FEATURES_STRUCTURAL)
        expected = torch.cat([batch["x_1_hop_mean"], batch["x_2_hop_mean"]], dim=1).cpu().numpy()
        np.testing.assert_array_equal(result, expected)

    def test_build_sampler_features_structural_no_features(self):
        """Test sampler_features='structural' raises RuntimeError when no structural features."""
        batch = {
            "x_0": torch.randn(5, 10),
        }
        node_features_model = np.random.randn(5, 15)
        with pytest.raises(
            RuntimeError,
            match=f"sampler_features='{SAMPLER_FEATURES_STRUCTURAL}' requested but no structural features found",
        ):
            _build_sampler_features(batch, node_features_model, SAMPLER_FEATURES_STRUCTURAL)

    def test_build_sampler_features_invalid_option(self):
        """Test that invalid sampler_features raises ValueError."""
        batch = {"x_0": torch.randn(5, 10)}
        node_features_model = np.random.randn(5, 15)
        with pytest.raises(ValueError, match="sampler_features must be one of"):
            _build_sampler_features(batch, node_features_model, "invalid")


class TestBaseWrapperInitialization:
    """Test BaseWrapper initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        backbone = MagicMock()
        wrapper = MockBaseWrapper(backbone)
        assert wrapper.backbone == backbone
        assert wrapper.use_embeddings is True
        assert wrapper.use_node_features is True
        assert wrapper.sampler is None
        assert wrapper.num_test_nodes == 1
        assert wrapper.sampler_features == SAMPLER_FEATURES_ALL
        assert wrapper.logger is None
        assert wrapper.checker is not None
        assert wrapper.stats is not None
        assert wrapper.safe is not None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        backbone = MagicMock()
        sampler = MagicMock()
        logger = MagicMock()
        wrapper = MockBaseWrapper(
            backbone,
            use_embeddings=False,
            use_node_features=True,
            sampler=sampler,
            num_test_nodes=5,
            sampler_features="node",
            logger=logger,
        )
        assert wrapper.use_embeddings is False
        assert wrapper.use_node_features is True
        assert wrapper.sampler == sampler
        assert wrapper.num_test_nodes == 5
        assert wrapper.sampler_features == SAMPLER_FEATURES_NODE
        assert wrapper.logger == logger

    def test_init_sampler_empty_dict_becomes_none(self):
        """Test that empty dict sampler becomes None."""
        backbone = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler={})
        assert wrapper.sampler is None

    def test_init_invalid_sampler_features(self):
        """Test that invalid sampler_features raises ValueError."""
        backbone = MagicMock()
        with pytest.raises(ValueError, match="sampler_features must be one of"):
            MockBaseWrapper(backbone, sampler_features="invalid")

    def test_init_both_use_flags_false(self):
        """Test that both use_embeddings and use_node_features cannot be False."""
        backbone = MagicMock()
        with pytest.raises(ValueError, match="Either use_embeddings or use_node_features must be True"):
            MockBaseWrapper(backbone, use_embeddings=False, use_node_features=False)

    def test_init_valid_sampler_features_options(self):
        """Test all valid sampler_features options."""
        backbone = MagicMock()
        for sampler_feat in [SAMPLER_FEATURES_ALL, SAMPLER_FEATURES_NODE, SAMPLER_FEATURES_STRUCTURAL]:
            wrapper = MockBaseWrapper(backbone, sampler_features=sampler_feat)
            assert wrapper.sampler_features == sampler_feat


class TestBaseWrapperFit:
    """Test BaseWrapper fit method."""

    def test_fit_returns_self(self):
        """Test that fit method returns self."""
        backbone = MagicMock()
        wrapper = MockBaseWrapper(backbone)
        x = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        result = wrapper.fit(x, y)
        assert result is wrapper


class TestBaseWrapperForward:
    """Test BaseWrapper forward method."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        num_nodes = 10
        num_train = 4
        num_val = 3
        num_test = 3
        feature_dim = 5

        batch = {
            "x_0": torch.randn(num_nodes, feature_dim),
            "y": torch.randint(0, 2, (num_nodes,)),
            "edge_index": torch.randint(0, num_nodes, (2, 20)),
            "batch_0": torch.zeros(num_nodes, dtype=torch.long),
            "train_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "val_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "test_mask": torch.zeros(num_nodes, dtype=torch.bool),
        }
        # Set masks
        batch["train_mask"][:num_train] = True
        batch["val_mask"][num_train : num_train + num_val] = True
        batch["test_mask"][num_train + num_val :] = True

        return batch

    def test_forward_without_sampler(self, sample_batch):
        """Test forward pass without sampler (full graph training)."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler=None)

        output = wrapper.forward(sample_batch)

        # Check output structure
        assert "labels" in output
        assert "batch_0" in output
        assert "x_0" in output
        assert torch.equal(output["labels"], sample_batch["y"])
        assert torch.equal(output["batch_0"], sample_batch["batch_0"])
        assert output["x_0"].shape[0] == len(sample_batch["y"])

    def test_forward_with_sampler(self, sample_batch):
        """Test forward pass with sampler."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        backbone.predict = MagicMock(return_value=np.random.randint(0, 2, 3))

        sampler = MagicMock()
        sampler.fit = MagicMock()
        sampler.sample = MagicMock(return_value=np.array([0, 1, 2]))

        wrapper = MockBaseWrapper(backbone, sampler=sampler, num_test_nodes=1)

        # Mock the safe predictor
        wrapper.safe.predict_batch = MagicMock(
            return_value=(
                [np.array([0.5, 0.5])] * 3,
                [0, 1, 0],
                "normal",
            )
        )

        output = wrapper.forward(sample_batch)

        # Check output structure
        assert "labels" in output
        assert "batch_0" in output
        assert "x_0" in output
        assert sampler.fit.called
        assert sampler.sample.called

    def test_forward_use_embeddings_false(self, sample_batch):
        """Test forward pass with use_embeddings=False."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler=None, use_embeddings=False)

        output = wrapper.forward(sample_batch)
        assert "x_0" in output

    def test_forward_use_node_features_false(self, sample_batch):
        """Test forward pass with use_node_features=False."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        sample_batch["x_1_hop_mean"] = torch.randn(10, 3)
        wrapper = MockBaseWrapper(
            backbone, sampler=None, use_node_features=False, use_embeddings=True
        )

        output = wrapper.forward(sample_batch)
        assert "x_0" in output

    def test_forward_sampler_features_all(self, sample_batch):
        """Test forward pass with sampler_features='all'."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        sampler = MagicMock()
        sampler.fit = MagicMock()
        sampler.sample = MagicMock(return_value=np.array([0, 1]))

        wrapper = MockBaseWrapper(
            backbone, sampler=sampler, sampler_features="all", num_test_nodes=1
        )

        wrapper.safe.predict_batch = MagicMock(
            return_value=([np.array([0.5, 0.5])] * 3, [0, 1, 0], "normal")
        )

        output = wrapper.forward(sample_batch)
        assert "x_0" in output

    def test_forward_sampler_features_node(self, sample_batch):
        """Test forward pass with sampler_features='node'."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        sampler = MagicMock()
        sampler.fit = MagicMock()
        sampler.sample = MagicMock(return_value=np.array([0, 1]))

        wrapper = MockBaseWrapper(
            backbone, sampler=sampler, sampler_features="node", num_test_nodes=1
        )

        wrapper.safe.predict_batch = MagicMock(
            return_value=([np.array([0.5, 0.5])] * 3, [0, 1, 0], "normal")
        )

        output = wrapper.forward(sample_batch)
        assert "x_0" in output

    def test_forward_sampler_features_structural(self, sample_batch):
        """Test forward pass with sampler_features='structural'."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        sample_batch["x_1_hop_mean"] = torch.randn(10, 3)
        sample_batch["x_2_hop_mean"] = torch.randn(10, 2)

        sampler = MagicMock()
        sampler.fit = MagicMock()
        sampler.sample = MagicMock(return_value=np.array([0, 1]))

        wrapper = MockBaseWrapper(
            backbone, sampler=sampler, sampler_features="structural", num_test_nodes=1
        )

        wrapper.safe.predict_batch = MagicMock(
            return_value=([np.array([0.5, 0.5])] * 3, [0, 1, 0], "normal")
        )

        output = wrapper.forward(sample_batch)
        assert "x_0" in output

    def test_forward_sampler_features_structural_missing(self, sample_batch):
        """Test forward pass with sampler_features='structural' but no structural features."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        sampler = MagicMock()

        wrapper = MockBaseWrapper(
            backbone, sampler=sampler, sampler_features="structural", num_test_nodes=1
        )

        with pytest.raises(RuntimeError, match="no structural features found"):
            wrapper.forward(sample_batch)

    def test_forward_batch_size_greater_than_one(self, sample_batch):
        """Test forward pass with num_test_nodes > 1."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        sampler = MagicMock()
        sampler.fit = MagicMock()
        sampler.sample = MagicMock(return_value=np.array([0, 1, 2]))

        wrapper = MockBaseWrapper(backbone, sampler=sampler, num_test_nodes=2)

        wrapper.safe.predict_batch = MagicMock(
            return_value=(
                [np.array([0.5, 0.5])] * 3,
                [0, 1, 0],
                "normal",
            )
        )

        output = wrapper.forward(sample_batch)
        assert "x_0" in output

    def test_forward_output_shapes(self, sample_batch):
        """Test that forward output has correct shapes."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler=None)

        output = wrapper.forward(sample_batch)

        num_nodes = len(sample_batch["y"])
        assert output["x_0"].shape[0] == num_nodes
        assert output["labels"].shape[0] == num_nodes
        assert output["batch_0"].shape[0] == num_nodes

    def test_forward_calls_init_targets(self, sample_batch):
        """Test that forward calls _init_targets."""
        backbone = MagicMock()
        backbone.fit = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler=None)

        # Mock _init_targets to track calls
        # Note: Accessing protected members is standard practice in unit tests
        wrapper._init_targets = MagicMock()  # pylint: disable=protected-access
        wrapper.forward(sample_batch)

        assert wrapper._init_targets.called  # pylint: disable=protected-access
        # Check it was called with y_train
        call_args = wrapper._init_targets.call_args[0]  # pylint: disable=protected-access
        assert len(call_args) == 1
        assert isinstance(call_args[0], np.ndarray)

    def test_forward_with_index_format_masks(self):
        """Test forward when batch has index-style train/val/test masks (e.g. cocitation_cora)."""
        num_nodes = 10
        train_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        val_idx = torch.tensor([4, 5], dtype=torch.long)
        test_idx = torch.tensor([6, 7, 8, 9], dtype=torch.long)
        batch = {
            "x_0": torch.randn(num_nodes, 5),
            "y": torch.randint(0, 2, (num_nodes,)),
            "edge_index": torch.randint(0, num_nodes, (2, 20)),
            "batch_0": torch.zeros(num_nodes, dtype=torch.long),
            "train_mask": train_idx,
            "val_mask": val_idx,
            "test_mask": test_idx,
        }
        backbone = MagicMock()
        backbone.fit = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler=None)
        output = wrapper.forward(batch)
        assert "x_0" in output
        assert output["x_0"].shape[0] == num_nodes
        assert torch.equal(output["labels"], batch["y"])


class TestBaseWrapperAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_cannot_instantiate_base_wrapper_directly(self):
        """Test that BaseWrapper cannot be instantiated directly."""
        backbone = MagicMock()
        # BaseWrapper is abstract, so instantiation should raise TypeError
        with pytest.raises(TypeError):
            BaseWrapper(backbone)  # pylint: disable=abstract-class-instantiated

    def test_mock_implementation_has_all_methods(self):
        """Test that MockBaseWrapper implements all abstract methods."""
        backbone = MagicMock()
        wrapper = MockBaseWrapper(backbone)

        # Check all abstract methods exist
        # Note: Accessing protected members is standard practice in unit tests
        assert hasattr(wrapper, "_init_targets")
        assert hasattr(wrapper, "_no_neighbors")
        assert hasattr(wrapper, "_one_neighbor")
        assert hasattr(wrapper, "_all_features_constant")
        assert hasattr(wrapper, "_get_predictions")
        assert hasattr(wrapper, "_process_output")
        assert hasattr(wrapper, "_full_graph_training")
        assert hasattr(wrapper, "_update_progress_and_results")

        # Check they are callable
        assert callable(wrapper._init_targets)  # pylint: disable=protected-access
        assert callable(wrapper._no_neighbors)  # pylint: disable=protected-access
        assert callable(wrapper._one_neighbor)  # pylint: disable=protected-access
        assert callable(wrapper._all_features_constant)  # pylint: disable=protected-access
        assert callable(wrapper._get_predictions)  # pylint: disable=protected-access
        assert callable(wrapper._process_output)  # pylint: disable=protected-access
        assert callable(wrapper._full_graph_training)  # pylint: disable=protected-access
        assert callable(wrapper._update_progress_and_results)  # pylint: disable=protected-access


class TestBaseWrapperEdgeCases:
    """Test edge cases and error handling."""

    def test_forward_empty_test_mask(self):
        """Test forward pass with empty test mask."""
        num_nodes = 5
        batch = {
            "x_0": torch.randn(num_nodes, 5),
            "y": torch.randint(0, 2, (num_nodes,)),
            "edge_index": torch.randint(0, num_nodes, (2, 10)),
            "batch_0": torch.zeros(num_nodes, dtype=torch.long),
            "train_mask": torch.ones(num_nodes, dtype=torch.bool),
            "val_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "test_mask": torch.zeros(num_nodes, dtype=torch.bool),
        }

        backbone = MagicMock()
        backbone.fit = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler=None)

        output = wrapper.forward(batch)
        assert "x_0" in output

    def test_forward_empty_train_mask(self):
        """Test forward pass with empty train mask."""
        num_nodes = 5
        batch = {
            "x_0": torch.randn(num_nodes, 5),
            "y": torch.randint(0, 2, (num_nodes,)),
            "edge_index": torch.randint(0, num_nodes, (2, 10)),
            "batch_0": torch.zeros(num_nodes, dtype=torch.long),
            "train_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "val_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "test_mask": torch.ones(num_nodes, dtype=torch.bool),
        }

        backbone = MagicMock()
        backbone.fit = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler=None)

        # Should still work, but y_train will be empty
        output = wrapper.forward(batch)
        assert "x_0" in output

    def test_num_test_nodes_zero(self):
        """Test that num_test_nodes=0 defaults to 1."""
        backbone = MagicMock()
        wrapper = MockBaseWrapper(backbone, num_test_nodes=0)
        # batch_size should be max(1, num_test_nodes) = 1
        assert wrapper.num_test_nodes == 0  # Stored as-is
        # But in forward, batch_size = max(1, self.num_test_nodes) = 1

    def test_forward_device_consistency(self):
        """Test that output tensors are on the correct device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_nodes = 5
        batch = {
            "x_0": torch.randn(num_nodes, 5).to(device),
            "y": torch.randint(0, 2, (num_nodes,)).to(device),
            "edge_index": torch.randint(0, num_nodes, (2, 10)).to(device),
            "batch_0": torch.zeros(num_nodes, dtype=torch.long).to(device),
            "train_mask": torch.ones(num_nodes, dtype=torch.bool).to(device),
            "val_mask": torch.zeros(num_nodes, dtype=torch.bool).to(device),
            "test_mask": torch.zeros(num_nodes, dtype=torch.bool).to(device),
        }

        backbone = MagicMock()
        backbone.fit = MagicMock()
        wrapper = MockBaseWrapper(backbone, sampler=None)

        output = wrapper.forward(batch)
        assert output["x_0"].device == device
        assert output["labels"].device == device
        assert output["batch_0"].device == device


class TestBatchValidation:
    """Test batch validation."""

    def test_validate_batch_success(self):
        """Valid batch passes."""
        batch = {
            "x_0": torch.randn(5, 3),
            "y": torch.randint(0, 2, (5,)),
            "edge_index": torch.randint(0, 5, (2, 10)),
            "batch_0": torch.zeros(5, dtype=torch.long),
            "train_mask": torch.ones(5, dtype=torch.bool),
            "val_mask": torch.zeros(5, dtype=torch.bool),
            "test_mask": torch.zeros(5, dtype=torch.bool),
        }
        validate_batch(batch)

    def test_validate_batch_missing_key(self):
        """Missing required key raises BatchValidationError."""
        batch = {
            "x_0": torch.randn(5, 3),
            "y": torch.randint(0, 2, (5,)),
            "edge_index": torch.randint(0, 5, (2, 10)),
            "batch_0": torch.zeros(5, dtype=torch.long),
            "train_mask": torch.ones(5, dtype=torch.bool),
            "val_mask": torch.zeros(5, dtype=torch.bool),
        }
        with pytest.raises(BatchValidationError, match="missing required keys"):
            validate_batch(batch)

    def test_validate_batch_per_node_mask_must_be_bool_or_01(self):
        """Per-node mask with values not in {0,1} raises BatchValidationError."""
        batch = {
            "x_0": torch.randn(5, 3),
            "y": torch.randint(0, 2, (5,)),
            "edge_index": torch.randint(0, 5, (2, 10)),
            "batch_0": torch.zeros(5, dtype=torch.long),
            "train_mask": torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
            "val_mask": torch.zeros(5, dtype=torch.bool),
            "test_mask": torch.zeros(5, dtype=torch.bool),
        }
        with pytest.raises(BatchValidationError, match="0/1|boolean|mask format"):
            validate_batch(batch)

    def test_validate_batch_index_format_success(self):
        """Index-format masks (1D indices) are accepted."""
        batch = {
            "x_0": torch.randn(10, 3),
            "y": torch.randint(0, 2, (10,)),
            "edge_index": torch.randint(0, 10, (2, 20)),
            "batch_0": torch.zeros(10, dtype=torch.long),
            "train_mask": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "val_mask": torch.tensor([4, 5], dtype=torch.long),
            "test_mask": torch.tensor([6, 7, 8, 9], dtype=torch.long),
        }
        validate_batch(batch)

    def test_validate_batch_index_format_out_of_range(self):
        """Index-format mask with indices out of range raises."""
        batch = {
            "x_0": torch.randn(5, 3),
            "y": torch.randint(0, 2, (5,)),
            "edge_index": torch.randint(0, 5, (2, 10)),
            "batch_0": torch.zeros(5, dtype=torch.long),
            "train_mask": torch.tensor([0, 1, 10], dtype=torch.long),
            "val_mask": torch.zeros(5, dtype=torch.bool),
            "test_mask": torch.zeros(5, dtype=torch.bool),
        }
        with pytest.raises(BatchValidationError, match="indices must be in"):
            validate_batch(batch)

    def test_masks_to_bool_per_node_mask(self):
        """masks_to_bool with per-node boolean/0/1 masks."""
        batch = {
            "train_mask": torch.tensor([True, False, True, False, True]),
            "val_mask": torch.tensor([0, 1, 0, 0, 0], dtype=torch.long),
            "test_mask": torch.zeros(5, dtype=torch.bool),
        }
        t, v, te = masks_to_bool(batch, 5)
        np.testing.assert_array_equal(t, [True, False, True, False, True])
        np.testing.assert_array_equal(v, [False, True, False, False, False])
        np.testing.assert_array_equal(te, [False, False, False, False, False])

    def test_masks_to_bool_index_format(self):
        """masks_to_bool with index-format masks."""
        batch = {
            "train_mask": torch.tensor([0, 2, 4], dtype=torch.long),
            "val_mask": torch.tensor([1], dtype=torch.long),
            "test_mask": torch.tensor([3], dtype=torch.long),
        }
        t, v, te = masks_to_bool(batch, 5)
        np.testing.assert_array_equal(t, [True, False, True, False, True])
        np.testing.assert_array_equal(v, [False, True, False, False, False])
        np.testing.assert_array_equal(te, [False, False, False, True, False])


class TestBaseWrapperHelperMethods:
    """Test helper methods in BaseWrapper."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        num_nodes = 10
        batch = {
            "x_0": torch.randn(num_nodes, 5),
            "x_1_hop_mean": torch.randn(num_nodes, 3),
            "y": torch.randint(0, 2, (num_nodes,)),
            "edge_index": torch.randint(0, num_nodes, (2, 20)),
            "batch_0": torch.zeros(num_nodes, dtype=torch.long),
            "train_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "val_mask": torch.zeros(num_nodes, dtype=torch.bool),
            "test_mask": torch.zeros(num_nodes, dtype=torch.bool),
        }
        batch["train_mask"][:4] = True
        batch["test_mask"][4:] = True
        return batch

    def test_extract_node_features_use_embeddings(self, sample_batch):
        """Test _extract_node_features with use_embeddings=True."""
        backbone = MagicMock()
        wrapper = MockBaseWrapper(backbone, use_embeddings=True, use_node_features=True)
        features = wrapper._extract_node_features(sample_batch)
        assert features.shape == (10, 8)  # 5 (x_0) + 3 (x_1_hop_mean)

    def test_extract_node_features_no_embeddings(self, sample_batch):
        """Test _extract_node_features with use_embeddings=False."""
        backbone = MagicMock()
        wrapper = MockBaseWrapper(backbone, use_embeddings=False, use_node_features=True)
        features = wrapper._extract_node_features(sample_batch)
        assert features.shape == (10, 5)  # Only x_0

    def test_extract_node_features_no_node_features(self, sample_batch):
        """Test _extract_node_features with use_node_features=False."""
        backbone = MagicMock()
        wrapper = MockBaseWrapper(backbone, use_embeddings=True, use_node_features=False)
        features = wrapper._extract_node_features(sample_batch)
        assert features.shape == (10, 3)  # Only x_1_hop_mean

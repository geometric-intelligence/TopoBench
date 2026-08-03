"""Tests for typed homogeneous and heterogeneous supervision selection."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, HeteroData

from topobench.model import (
    DefaultSupervisionAdapter,
    HeterogeneousNodeSupervisionAdapter,
    SupervisedBatch,
)


def _model_out(num_nodes: int = 6) -> dict[str, object]:
    """Return deterministic classification outputs."""
    return {
        "logits": torch.arange(num_nodes * 3, dtype=torch.float32).reshape(
            num_nodes, 3
        ),
        "labels": torch.arange(num_nodes),
        "untouched": object(),
    }


def _homogeneous_batch() -> Data:
    """Return a homogeneous node batch with disjoint phase masks."""
    return Data(
        train_mask=torch.tensor([True, True, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, False, False, False]),
        test_mask=torch.tensor([False, False, False, True, True, True]),
    )


def _heterogeneous_batch() -> HeteroData:
    """Return a target store and an adversarial context store."""
    batch = HeteroData()
    batch["paper"].x = torch.arange(12).reshape(6, 2)
    batch["paper"].train_mask = torch.tensor(
        [True, True, False, False, False, False]
    )
    batch["paper"].val_mask = torch.tensor(
        [False, False, True, False, False, False]
    )
    batch["paper"].test_mask = torch.tensor(
        [False, False, False, True, True, True]
    )

    # These masks are deliberately all true: target supervision must never
    # consult a context node store.
    batch["author"].x = torch.arange(8).reshape(4, 2)
    batch["author"].train_mask = torch.ones(4, dtype=torch.bool)
    batch["author"].val_mask = torch.ones(4, dtype=torch.bool)
    batch["author"].test_mask = torch.ones(4, dtype=torch.bool)
    return batch


class TestSupervisedBatch:
    """Validate the public immutable selection result."""

    def test_accepts_tensor_fields_and_positive_builtin_count(self):
        """Valid direct construction normalizes no public fields."""
        logits = torch.randn(2, 3)
        targets = torch.tensor([0, 1])

        selected = SupervisedBatch(logits, targets, 2)

        assert selected.logits is logits
        assert selected.targets is targets
        assert type(selected.num_examples) is int

    @pytest.mark.parametrize("num_examples", [0, -1, True, np.int64(2), 1.5])
    def test_rejects_nonpositive_or_non_builtin_count(self, num_examples):
        """Direct results always carry a positive built-in integer count."""
        with pytest.raises(
            (TypeError, ValueError),
            match="num_examples must be a positive built-in int",
        ):
            SupervisedBatch(
                torch.randn(2, 3),
                torch.tensor([0, 1]),
                num_examples,
            )

    @pytest.mark.parametrize(
        ("logits", "targets"),
        [
            ([[1.0]], torch.tensor([0])),
            (torch.tensor([[1.0]]), [0]),
        ],
    )
    def test_rejects_non_tensor_fields(self, logits, targets):
        """Direct construction enforces the public tensor field types."""
        with pytest.raises(
            TypeError, match="logits and targets must be tensors"
        ):
            SupervisedBatch(logits, targets, 1)

    def test_is_frozen(self):
        """Selection results cannot be mutated after construction."""
        selected = SupervisedBatch(torch.randn(1, 2), torch.tensor([0]), 1)

        with pytest.raises(FrozenInstanceError):
            selected.num_examples = 2


class TestDefaultSupervisionAdapter:
    """Enforce native homogeneous classification and regression contracts."""

    @pytest.mark.parametrize("task_level", [None, 1, "", "   ", "edge"])
    def test_rejects_invalid_task_level(self, task_level):
        """Only supported homogeneous task levels can select supervision."""
        with pytest.raises(
            (TypeError, ValueError),
            match="task_level",
        ):
            DefaultSupervisionAdapter(task_level)

    @pytest.mark.parametrize("batch_size", [4, 1])
    def test_graph_classification_keeps_rank_one_targets_and_counts_graphs(
        self, batch_size
    ):
        """Graph classification reports one supervised example per graph."""
        model_out = {
            "logits": torch.randn(batch_size, 3),
            "labels": torch.arange(batch_size, dtype=torch.long) % 3,
        }

        selected = DefaultSupervisionAdapter("graph").select(
            model_out, Data(), "Training"
        )

        assert selected.logits is model_out["logits"]
        assert selected.targets is model_out["labels"]
        assert selected.targets.shape == (batch_size,)
        assert selected.num_examples == batch_size

    @pytest.mark.parametrize("batch_size", [4, 1])
    @pytest.mark.parametrize("target_rank", [1, 2])
    def test_graph_regression_normalizes_once_and_counts_graphs(
        self, batch_size, target_rank
    ):
        """Scalar source labels become exactly [B, 1], including B=1."""
        targets = torch.linspace(0.0, 1.0, batch_size)
        if target_rank == 2:
            targets = targets.unsqueeze(1)
        model_out = {
            "logits": torch.randn(batch_size, 1),
            "labels": targets,
        }

        selected = DefaultSupervisionAdapter("graph").select(
            model_out, Data(), "Training"
        )

        assert selected.logits.shape == (batch_size, 1)
        assert selected.targets.shape == (batch_size, 1)
        assert selected.num_examples == batch_size

    @pytest.mark.parametrize(
        ("logits", "labels", "message"),
        [
            (torch.randn(3, 4), torch.zeros(3, 1, dtype=torch.long), "classification"),
            (torch.randn(3, 4), torch.zeros(3, 1, 1, dtype=torch.long), "classification"),
            (torch.randn(3, 1), torch.zeros(3, 1, 1), "regression"),
            (torch.randn(3, 1), torch.zeros(3, dtype=torch.long), "floating"),
            (
                torch.randn(3, 1),
                torch.tensor([0.0, float("nan"), 1.0]),
                "finite",
            ),
            (
                torch.randn(3, 1),
                torch.tensor([0.0, float("inf"), 1.0]),
                "finite",
            ),
            (torch.randn(3), torch.zeros(3), "logits"),
        ],
    )
    def test_rejects_broadcasting_or_invalid_targets(
        self, logits, labels, message
    ):
        """Shape, dtype, and finite-value failures occur before loss."""
        with pytest.raises((TypeError, ValueError), match=message):
            DefaultSupervisionAdapter("graph").select(
                {"logits": logits, "labels": labels},
                Data(),
                "Training",
            )

    @pytest.mark.parametrize(
        ("phase", "mask_name"),
        [
            ("Training", "train_mask"),
            ("Validation", "val_mask"),
            ("Test", "test_mask"),
        ],
    )
    def test_node_classification_selects_rank_one_targets_and_mask_count(
        self, phase, mask_name
    ):
        """Transductive node weighting is the selected node-label count."""
        batch = _homogeneous_batch()
        model_out = _model_out()
        original = dict(model_out)
        mask = getattr(batch, mask_name)

        selected = DefaultSupervisionAdapter("node").select(
            model_out, batch, phase
        )

        assert torch.equal(selected.logits, original["logits"][mask])
        assert torch.equal(selected.targets, original["labels"][mask])
        assert selected.targets.ndim == 1
        assert selected.num_examples == int(mask.sum())
        assert model_out == original

    def test_node_classification_rejects_rank_two_targets(self):
        """A broadcast-compatible [N, 1] node target is never flattened."""
        model_out = _model_out()
        model_out["labels"] = model_out["labels"].unsqueeze(1)

        with pytest.raises(ValueError, match="classification"):
            DefaultSupervisionAdapter("node").select(
                model_out, _homogeneous_batch(), "Training"
            )

    def test_node_rejects_invalid_case_sensitive_phase(self):
        """Phase names retain their exact case-sensitive contract."""
        with pytest.raises(ValueError, match="Invalid state_str: training"):
            DefaultSupervisionAdapter("node").select(
                _model_out(), _homogeneous_batch(), "training"
            )

    def test_node_rejects_empty_supervision(self):
        """Empty homogeneous masks fail before loss computation."""
        batch = _homogeneous_batch()
        batch.train_mask.zero_()

        with pytest.raises(
            ValueError, match="No supervised examples for Training"
        ):
            DefaultSupervisionAdapter("node").select(
                _model_out(), batch, "Training"
            )

    @pytest.mark.parametrize(
        "model_out",
        [
            {"labels": torch.tensor([0])},
            {"logits": torch.randn(1, 2)},
            {"logits": [[1.0, 2.0]], "labels": torch.tensor([0])},
            {"logits": torch.randn(1, 2), "labels": [0]},
        ],
    )
    def test_rejects_missing_or_non_tensor_model_fields(self, model_out):
        """Selection requires tensor logits and labels."""
        with pytest.raises(
            TypeError, match="model_out must contain tensor logits and labels"
        ):
            DefaultSupervisionAdapter("graph").select(
                model_out, Data(), "Training"
            )

    def test_rejects_mismatched_model_field_counts(self):
        """Logits and labels must describe the same examples."""
        with pytest.raises(
            ValueError,
            match="Logit and label counts must match before selection",
        ):
            DefaultSupervisionAdapter("graph").select(
                {
                    "logits": torch.randn(2, 3),
                    "labels": torch.tensor([0]),
                },
                Data(),
                "Training",
            )

class TestHeterogeneousNodeSupervisionAdapter:
    """Test target-store masks and strict seed-only neighbor semantics."""

    @pytest.mark.parametrize("target_node_type", [None, 1, "", "   "])
    def test_rejects_invalid_target_node_type(self, target_node_type):
        """The target node type is a required, meaningful string."""
        with pytest.raises(
            (TypeError, ValueError),
            match="target_node_type must be a non-empty string",
        ):
            HeterogeneousNodeSupervisionAdapter(target_node_type, "full_batch")

    @pytest.mark.parametrize("mode", [None, 1, "", "sampled"])
    def test_rejects_invalid_mode(self, mode):
        """Only the two declared sampling modes are accepted."""
        with pytest.raises(
            (TypeError, ValueError), match="Unsupported sampling mode"
        ):
            HeterogeneousNodeSupervisionAdapter("paper", mode)

    @pytest.mark.parametrize(
        ("phase", "mask_name"),
        [
            ("Training", "train_mask"),
            ("Validation", "val_mask"),
            ("Test", "test_mask"),
        ],
    )
    def test_full_batch_selects_exact_target_store_phase_mask(
        self, phase, mask_name
    ):
        """Full-batch supervision uses only the configured target store."""
        batch = _heterogeneous_batch()
        model_out = _model_out()
        original = dict(model_out)
        mask = batch["paper"][mask_name]

        selected = HeterogeneousNodeSupervisionAdapter(
            "paper", "full_batch"
        ).select(model_out, batch, phase)

        assert torch.equal(selected.logits, original["logits"][mask])
        assert torch.equal(selected.targets, original["labels"][mask])
        assert selected.num_examples == int(mask.sum())
        assert model_out == original

    def test_full_batch_mask_preserves_gradient_connection(self):
        """Boolean-mask selection remains connected to the original logits."""
        batch = _heterogeneous_batch()
        logits = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        logits.requires_grad_()

        selected = HeterogeneousNodeSupervisionAdapter(
            "paper", "full_batch"
        ).select(
            {"logits": logits, "labels": torch.arange(6)},
            batch,
            "Training",
        )
        selected.logits.sum().backward()

        expected_gradient = torch.zeros_like(logits)
        expected_gradient[batch["paper"].train_mask] = 1
        assert torch.equal(logits.grad, expected_gradient)

    @pytest.mark.parametrize("phase", ["Training", "Validation", "Test"])
    def test_neighbor_selects_only_leading_target_seeds_for_every_phase(
        self, phase
    ):
        """Neighbor batches never supervise sampled context target nodes."""
        batch = _heterogeneous_batch()
        batch["paper"].batch_size = 2
        # Context targets have adversarial labels and all phase masks true.
        model_out = _model_out()
        model_out["labels"] = torch.tensor([0, 1, 97, 98, 99, 100])

        selected = HeterogeneousNodeSupervisionAdapter(
            "paper", "neighbor"
        ).select(model_out, batch, phase)

        assert torch.equal(selected.logits, model_out["logits"][:2])
        assert torch.equal(selected.targets, torch.tensor([0, 1]))
        assert selected.num_examples == 2

    def test_neighbor_accepts_integral_seed_count_and_normalizes_to_int(self):
        """Non-boolean Integral counts from loaders are normalized safely."""
        batch = _heterogeneous_batch()
        batch["paper"].batch_size = np.int64(2)

        selected = HeterogeneousNodeSupervisionAdapter(
            "paper", "neighbor"
        ).select(_model_out(), batch, "Training")

        assert selected.num_examples == 2
        assert type(selected.num_examples) is int

    def test_neighbor_slice_preserves_gradient_connection(self):
        """Leading-seed slicing remains connected to the original logits."""
        batch = _heterogeneous_batch()
        batch["paper"].batch_size = 2
        logits = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        logits.requires_grad_()

        selected = HeterogeneousNodeSupervisionAdapter(
            "paper", "neighbor"
        ).select(
            {"logits": logits, "labels": torch.arange(6)},
            batch,
            "Validation",
        )
        selected.logits.sum().backward()

        expected_gradient = torch.zeros_like(logits)
        expected_gradient[:2] = 1
        assert torch.equal(logits.grad, expected_gradient)

    def test_rejects_non_heterogeneous_batch(self):
        """Heterogeneous supervision cannot silently accept homogeneous data."""
        with pytest.raises(
            TypeError, match="Heterogeneous supervision requires HeteroData"
        ):
            HeterogeneousNodeSupervisionAdapter("paper", "full_batch").select(
                _model_out(), Data(), "Training"
            )

    def test_rejects_invalid_phase_in_neighbor_mode(self):
        """Seed-only mode still validates the training phase contract."""
        batch = _heterogeneous_batch()
        batch["paper"].batch_size = 2

        with pytest.raises(ValueError, match="Invalid state_str: Predict"):
            HeterogeneousNodeSupervisionAdapter("paper", "neighbor").select(
                _model_out(), batch, "Predict"
            )

    def test_rejects_missing_target_store(self):
        """A misspelled target type produces an actionable error."""
        with pytest.raises(
            ValueError, match="Batch has no target store 'paper'"
        ):
            HeterogeneousNodeSupervisionAdapter("paper", "full_batch").select(
                _model_out(), HeteroData(), "Training"
            )

    @pytest.mark.parametrize(
        ("batch_size", "message"),
        [
            (None, "missing target-store batch_size"),
            (True, "non-boolean integer"),
            (torch.tensor(2), "non-boolean integer"),
            (1.5, "non-boolean integer"),
            (0, "Invalid target seed count: 0"),
            (-1, "Invalid target seed count: -1"),
            (7, "Invalid target seed count: 7"),
        ],
    )
    def test_rejects_missing_wrong_or_out_of_range_seed_count(
        self, batch_size, message
    ):
        """Neighbor seed counts are typed and bounded by target outputs."""
        batch = _heterogeneous_batch()
        if batch_size is not None:
            batch["paper"].batch_size = batch_size

        with pytest.raises((TypeError, ValueError), match=message):
            HeterogeneousNodeSupervisionAdapter("paper", "neighbor").select(
                _model_out(), batch, "Validation"
            )

    def test_rejects_empty_full_batch_supervision(self):
        """A phase with no target nodes fails explicitly."""
        batch = _heterogeneous_batch()
        batch["paper"].val_mask.zero_()

        with pytest.raises(
            ValueError, match="No supervised target nodes for Validation"
        ):
            HeterogeneousNodeSupervisionAdapter("paper", "full_batch").select(
                _model_out(), batch, "Validation"
            )

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda batch: batch["paper"].__delattr__("train_mask"),
                "missing boolean train_mask",
            ),
            (
                lambda batch: setattr(
                    batch["paper"],
                    "train_mask",
                    torch.tensor([1, 1, 0, 0, 0, 0]),
                ),
                "train_mask must be a rank-1 bool tensor",
            ),
            (
                lambda batch: setattr(
                    batch["paper"],
                    "train_mask",
                    torch.ones(2, 3, dtype=torch.bool),
                ),
                "train_mask must be a rank-1 bool tensor",
            ),
            (
                lambda batch: setattr(
                    batch["paper"],
                    "train_mask",
                    torch.tensor([True, False]),
                ),
                "train_mask count must match target outputs",
            ),
        ],
    )
    def test_full_batch_validates_target_mask(self, mutate, message):
        """Target phase masks have a strict type, rank, and count contract."""
        batch = _heterogeneous_batch()
        mutate(batch)

        with pytest.raises((TypeError, ValueError), match=message):
            HeterogeneousNodeSupervisionAdapter("paper", "full_batch").select(
                _model_out(), batch, "Training"
            )

    @pytest.mark.parametrize(
        ("logits", "labels"),
        [
            (torch.randn(6), torch.arange(6)),
            (torch.randn(6, 2, 1), torch.arange(6)),
            (torch.randn(6, 2), torch.arange(6).reshape(6, 1)),
        ],
    )
    def test_rejects_wrong_heterogeneous_output_ranks(self, logits, labels):
        """Heterogeneous classification has explicit output ranks."""
        with pytest.raises(
            ValueError,
            match=r"expects \[N, C\] logits and \[N\] labels",
        ):
            HeterogeneousNodeSupervisionAdapter("paper", "full_batch").select(
                {"logits": logits, "labels": labels},
                _heterogeneous_batch(),
                "Training",
            )


def test_graph_selection_returns_loss_owning_tensor_aliases_without_copy() -> None:
    """Graph supervision preserves the exact prediction and target objects."""
    logits = torch.randn(3, 2, requires_grad=True)
    targets = torch.tensor([0, 1, 0])
    selected = DefaultSupervisionAdapter("graph").select(
        {"logits": logits, "labels": targets},
        Data(),
        "Training",
    )

    assert selected.logits is logits
    assert selected.targets is targets
    assert selected.num_examples == 3

"""Unit and integration tests for TBModel supervision handling."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from lightning import Trainer
import pytest
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader

from topobench.model import (
    DefaultSupervisionAdapter,
    HeterogeneousNodeSupervisionAdapter,
    SupervisedBatch,
    TBModel,
)


def make_model(task_level):
    """Instantiate TBModel with mocked dependencies for a given task_level.

    Parameters
    ----------
    task_level : str
        The task level to assign to the readout mock.

    Returns
    -------
    TBModel
        A TBModel instance with mocked backbone, readout, loss, evaluator and optimizer.
    """
    backbone = MagicMock()
    backbone.parameters.return_value = []

    readout = MagicMock()
    readout.task_level = task_level
    readout.parameters.return_value = []

    loss = MagicMock()

    evaluator = MagicMock()

    optimizer = MagicMock()
    optimizer.configure_optimizer.return_value = {"optimizer": MagicMock()}

    feature_encoder = MagicMock()
    feature_encoder.parameters.return_value = []

    model = TBModel(
        backbone=backbone,
        readout=readout,
        loss=loss,
        feature_encoder=feature_encoder,
        evaluator=evaluator,
        optimizer=optimizer,
    )
    return model


class TestProcessOutputs:
    """Tests for TBModel.process_outputs covering all branches."""

    def _make_batch(self):
        """Create a homogeneous batch with train/val/test masks.

        Returns
        -------
        Data
            Homogeneous batch with boolean masks.
        """
        # First 6 are train, next 2 val, last 2 test
        return Data(
            train_mask=torch.tensor(
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                dtype=torch.bool,
            ),
            val_mask=torch.tensor(
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                dtype=torch.bool,
            ),
            test_mask=torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                dtype=torch.bool,
            ),
        )

    def _make_model_out(self, n=10):
        """Create a sample model output dict.

        Parameters
        ----------
        n : int
            Number of nodes.

        Returns
        -------
        dict
            Model output with logits, labels, and an extra key.
        """
        return {
            "logits": torch.randn(n, 3),
            "labels": torch.randint(0, 3, (n,)),
            "x": torch.randn(n, 8),
        }

    # ------------------------------------------------------------------
    # node-level masking
    # ------------------------------------------------------------------

    def test_node_training_filters_by_train_mask(self):
        """process_outputs with task_level='node' and state 'Training' filters by train_mask."""
        model = make_model("node")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_train = batch.train_mask.sum().item()
        assert result["logits"].shape[0] == n_train
        assert result["labels"].shape[0] == n_train
        assert result["num_supervised_examples"] == n_train
        # non-masked keys are untouched
        assert result["x"].shape[0] == 10

    def test_node_validation_filters_by_val_mask(self):
        """process_outputs with task_level='node' and state 'Validation' filters by val_mask."""
        model = make_model("node")
        model.state_str = "Validation"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_val = batch.val_mask.sum().item()
        assert result["logits"].shape[0] == n_val
        assert result["labels"].shape[0] == n_val
        assert result["num_supervised_examples"] == n_val

    def test_node_test_filters_by_test_mask(self):
        """process_outputs with task_level='node' and state 'Test' filters by test_mask."""
        model = make_model("node")
        model.state_str = "Test"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_test = batch.test_mask.sum().item()
        assert result["logits"].shape[0] == n_test
        assert result["labels"].shape[0] == n_test
        assert result["num_supervised_examples"] == n_test

    def test_node_invalid_state_raises_value_error(self):
        """process_outputs with task_level='node' and an invalid state_str raises ValueError."""
        model = make_model("node")
        model.state_str = "Invalid"
        batch = self._make_batch()
        model_out = self._make_model_out()

        with pytest.raises(ValueError, match="Invalid state_str"):
            model.process_outputs(model_out, batch)

    # ------------------------------------------------------------------
    # no-op task levels
    # ------------------------------------------------------------------

    def test_graph_level_returns_unchanged(self):
        """process_outputs with task_level='graph' returns model_out unchanged."""
        model = make_model("graph")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()
        original_logits = model_out["logits"].clone()

        result = model.process_outputs(model_out, batch)

        assert result["logits"].shape == original_logits.shape
        assert torch.equal(result["logits"], original_logits)
        assert result["num_supervised_examples"] == 10

    def test_node_inductive_returns_unchanged(self):
        """process_outputs with task_level='node_inductive' returns model_out unchanged (inductive bug-fix path)."""
        model = make_model("node_inductive")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()
        original_logits = model_out["logits"].clone()

        result = model.process_outputs(model_out, batch)

        assert result["logits"].shape == original_logits.shape
        assert torch.equal(result["logits"], original_logits)
        assert result["num_supervised_examples"] == 10


@dataclass
class _FalseyAdapter:
    """Minimal falsey adapter proving explicit-None fallback behavior."""

    selected: SupervisedBatch

    def __bool__(self) -> bool:
        """Return false without invalidating the adapter."""
        return False

    def select(
        self,
        model_out: dict[str, object],
        batch: Data | HeteroData,
        phase: str,
    ) -> SupervisedBatch:
        """Return the configured selection."""
        return self.selected


class TestAdapterIntegration:
    """Test adapter construction, compatibility delegation, and hparams."""

    def test_default_adapter_is_derived_from_readout_task_level(self):
        """An omitted adapter preserves existing Hydra configurations."""
        model = make_model("node")

        assert isinstance(model.supervision_adapter, DefaultSupervisionAdapter)
        assert model.supervision_adapter.task_level == "node"

    def test_falsey_custom_adapter_is_preserved_and_not_checkpointed(self):
        """A valid falsey adapter is not replaced or serialized in hparams."""
        adapter = _FalseyAdapter(
            SupervisedBatch(torch.randn(1, 2), torch.tensor([1]), 1)
        )
        model = TBModel(
            backbone=MagicMock(),
            readout=MagicMock(task_level="node"),
            loss=MagicMock(),
            evaluator=MagicMock(),
            supervision_adapter=adapter,
        )

        assert model.supervision_adapter is adapter
        assert "supervision_adapter" not in model.hparams

    def test_process_outputs_delegates_then_mutates_only_contract_fields(self):
        """The compatibility entry point applies, rather than duplicates, selection."""
        selected = SupervisedBatch(
            torch.tensor([[9.0, 1.0]]), torch.tensor([0]), 1
        )
        adapter = _FalseyAdapter(selected)
        model = TBModel(
            backbone=MagicMock(),
            readout=MagicMock(task_level="graph"),
            loss=MagicMock(),
            evaluator=MagicMock(),
            supervision_adapter=adapter,
        )
        untouched = object()
        model_out = {
            "logits": torch.randn(3, 2),
            "labels": torch.arange(3),
            "untouched": untouched,
        }
        model.state_str = "Training"

        result = model.process_outputs(model_out, Data())

        assert result is model_out
        assert result["logits"] is selected.logits
        assert result["labels"] is selected.targets
        assert result["num_supervised_examples"] == 1
        assert result["untouched"] is untouched

    def test_model_step_selects_before_loss_then_updates_evaluator(self):
        """Loss and evaluator retain their historical ordering."""
        events: list[tuple[str, int]] = []
        model = make_model("node")
        model.state_str = "Training"
        batch = Data(
            train_mask=torch.tensor([True, False, True]),
            val_mask=torch.tensor([False, True, False]),
            test_mask=torch.tensor([False, True, False]),
        )
        raw_output = {
            "logits": torch.randn(3, 2),
            "labels": torch.tensor([0, 1, 0]),
        }
        model.forward = MagicMock(return_value=raw_output)

        def loss_side_effect(
            *, model_out: dict[str, Any], batch: Data
        ) -> dict[str, Any]:
            events.append(("loss", int(model_out["num_supervised_examples"])))
            model_out["loss"] = torch.tensor(0.25)
            return model_out

        model.loss.side_effect = loss_side_effect
        model.evaluator.update.side_effect = lambda model_out: events.append(
            ("evaluator", int(model_out["num_supervised_examples"]))
        )

        result = model.model_step(batch)

        assert events == [("loss", 2), ("evaluator", 2)]
        assert result["batch"] is batch

    def test_public_adapter_exports_are_canonical(self):
        """Users can configure adapters from the stable model namespace."""
        from topobench.model.supervision import (
            DefaultSupervisionAdapter as ModuleDefault,
        )
        from topobench.model.supervision import (
            HeterogeneousNodeSupervisionAdapter as ModuleHeterogeneous,
        )

        assert DefaultSupervisionAdapter is ModuleDefault
        assert HeterogeneousNodeSupervisionAdapter is ModuleHeterogeneous


class _GraphBatchBackbone(nn.Module):
    """Emit native graph outputs with a trainable scalar dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: Data) -> dict[str, torch.Tensor]:
        graph_count = batch.y.size(0)
        logits = self.weight.expand(graph_count, 2)
        return {
            "logits": logits,
            "labels": batch.y,
            "x": logits,
            "batch": batch.batch,
        }


class _IdentityGraphReadout(nn.Module):
    """Preserve logits already produced by the focused test backbone."""

    task_level = "graph"

    def forward(
        self, model_out: dict[str, Any], batch: Data
    ) -> dict[str, Any]:
        del batch
        return model_out


class _TargetMeanLoss(nn.Module):
    """Create deliberately different batch losses for reduction testing."""

    def forward(
        self, model_out: dict[str, Any], batch: Data
    ) -> dict[str, Any]:
        del batch
        model_out["loss"] = (
            model_out["labels"].float().mean()
            + model_out["logits"].sum() * 0.0
        )
        return model_out


class _NoOpEvaluator:
    """Minimal evaluator satisfying TBModel's epoch hooks."""

    def update(self, model_out: dict[str, Any]) -> None:
        del model_out

    def compute(self) -> dict[str, torch.Tensor]:
        return {}

    def reset(self) -> None:
        return None


class _ZeroRateOptimizer:
    """Build an optimizer while keeping focused batch losses deterministic."""

    def configure_optimizer(
        self, parameters: list[nn.Parameter]
    ) -> dict[str, torch.optim.Optimizer]:
        return {"optimizer": torch.optim.SGD(parameters, lr=0.0)}


class TestLossLoggingWeights:
    """Verify all three Lightning hooks use supervised-example counts."""

    @pytest.mark.parametrize(
        ("step_name", "metric_name", "expected_return"),
        [
            ("training_step", "train/loss", True),
            ("validation_step", "val/loss", False),
            ("test_step", "test/loss", False),
        ],
    )
    @pytest.mark.parametrize(
        ("description", "num_examples"),
        [
            ("graph", 3),
            ("transductive_node", 4),
            ("heterogeneous_full", 3),
            ("heterogeneous_neighbor", 2),
        ],
    )
    def test_loss_log_uses_model_step_supervised_count(
        self,
        step_name,
        metric_name,
        expected_return,
        description,
        num_examples,
    ):
        """Epoch reduction weight is the adapter-selected example count."""
        del description
        model = make_model("graph")
        loss = torch.tensor(0.75)
        model.model_step = MagicMock(
            return_value={
                "loss": loss,
                "num_supervised_examples": num_examples,
            }
        )
        model.log = MagicMock()

        result = getattr(model, step_name)(Data(), 0)

        model.log.assert_called_once_with(
            metric_name,
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=num_examples,
        )
        if expected_return:
            assert result is loss
        else:
            assert result is None

    def test_epoch_loss_weights_full_and_smaller_final_graph_batches(self):
        """Lightning reduces batch losses by supervised graph counts."""
        dataset = [
            Data(
                x=torch.ones(2, 1),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor([label]),
            )
            for label in (0, 0, 1)
        ]
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = TBModel(
            backbone=_GraphBatchBackbone(),
            readout=_IdentityGraphReadout(),
            loss=_TargetMeanLoss(),
            evaluator=_NoOpEvaluator(),
            optimizer=_ZeroRateOptimizer(),
            compile=False,
        )
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, train_dataloaders=loader)

        assert trainer.callback_metrics["train/loss"].item() == pytest.approx(
            1 / 3
        )

    def test_full_heterogeneous_process_outputs_reports_mask_count(self):
        """Full heterogeneous weighting comes from target-store masks."""
        batch = HeteroData()
        batch["paper"].train_mask = torch.tensor([True, False, True, True])
        model = TBModel(
            backbone=MagicMock(),
            readout=MagicMock(task_level="node"),
            loss=MagicMock(),
            evaluator=MagicMock(),
            supervision_adapter=HeterogeneousNodeSupervisionAdapter(
                "paper", "full_batch"
            ),
        )
        model.state_str = "Training"

        result = model.process_outputs(
            {
                "logits": torch.randn(4, 2),
                "labels": torch.arange(4),
            },
            batch,
        )

        assert result["num_supervised_examples"] == 3

    def test_neighbor_process_outputs_reports_seed_count(self):
        """Sampled heterogeneous weighting counts target seeds only."""
        batch = HeteroData()
        batch["paper"].batch_size = 2
        model = TBModel(
            backbone=MagicMock(),
            readout=MagicMock(task_level="node"),
            loss=MagicMock(),
            evaluator=MagicMock(),
            supervision_adapter=HeterogeneousNodeSupervisionAdapter(
                "paper", "neighbor"
            ),
        )
        model.state_str = "Validation"

        result = model.process_outputs(
            {
                "logits": torch.randn(5, 2),
                "labels": torch.arange(5),
            },
            batch,
        )

        assert result["num_supervised_examples"] == 2

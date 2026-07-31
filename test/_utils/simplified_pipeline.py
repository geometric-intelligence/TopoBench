"""Test pipeline for a particular dataset and model."""

from collections.abc import Mapping
from numbers import Integral
from typing import TypedDict

import hydra
from lightning import seed_everything
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.utils import instantiate_callbacks
from topobench.utils.config_resolvers import register_all_resolvers

register_all_resolvers()


class PipelineRunResult(TypedDict):
    """Structured evidence from a simplified pipeline run."""

    epochs_completed: int
    observed_train_batch_size: int
    fit_metrics: dict[str, float]
    test_results: list[Mapping[str, float]]


def _require_positive_batch_size(
    value: object,
    *,
    field_name: str,
    batch: object,
) -> int:
    """Normalize one integral batch-size field or fail with its origin."""
    batch_type = type(batch).__name__
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{field_name} on batch type {batch_type} must be a positive "
            "integer"
        )
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(
            f"{field_name} on batch type {batch_type} must be greater than zero"
        )
    return normalized


def _infer_observed_train_batch_size(
    batch: object,
    *,
    data_spec: HeterogeneousDataSpec | None,
) -> int:
    """Return graph count or heterogeneous target-seed count.

    Native sampled heterogeneous batches take precedence via the target
    store's ``batch_size``. Full-batch heterogeneous data and ordinary PyG
    batches fall back to their positive integral ``num_graphs`` value.
    """
    batch_type = type(batch).__name__
    if data_spec is not None:
        if not isinstance(batch, HeteroData):
            raise TypeError(
                f"Unsupported batch type {batch_type}: a heterogeneous "
                "data_spec requires native HeteroData"
            )
        target_node_type = data_spec.target_node_type
        if target_node_type not in batch.node_types:
            raise ValueError(
                f"Batch type {batch_type} is missing target node store "
                f"{target_node_type!r}"
            )
        target_store = batch[target_node_type]
        if "batch_size" in target_store:
            return _require_positive_batch_size(
                target_store.batch_size,
                field_name="target seed batch_size",
                batch=batch,
            )

    if not hasattr(batch, "num_graphs"):
        raise TypeError(
            f"Unsupported batch type {batch_type}: expected a positive "
            "integral num_graphs attribute"
        )
    return _require_positive_batch_size(
        batch.num_graphs,
        field_name="num_graphs",
        batch=batch,
    )


def run(cfg: DictConfig) -> PipelineRunResult:
    """Run pipeline with given configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration.

    Returns
    -------
    PipelineRunResult
        Evidence captured from training, validation, and testing.
    """
    seed_everything(cfg.seed, workers=True)

    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    pipeline_output = pipeline.build(cfg)
    datamodule = pipeline_output.datamodule

    first_train_batch = next(iter(datamodule.train_dataloader()))
    observed_train_batch_size = _infer_observed_train_batch_size(
        first_train_batch,
        data_spec=pipeline_output.data_spec,
    )

    # Model for us is Network + logic: inputs backbone, readout, losses
    model = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    )
    fit_metrics = {
        metric_name: float(
            trainer.callback_metrics[metric_name].detach().cpu().item()
        )
        for metric_name in ("train/loss", "val/loss")
        if metric_name in trainer.callback_metrics
    }
    epochs_completed = int(trainer.current_epoch)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    test_results = trainer.test(
        model=model, datamodule=datamodule, ckpt_path=ckpt_path
    )

    return {
        "epochs_completed": epochs_completed,
        "observed_train_batch_size": observed_train_batch_size,
        "fit_metrics": fit_metrics,
        "test_results": test_results,
    }

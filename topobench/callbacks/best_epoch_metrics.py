"""Callback to track all metrics at the epoch when the monitored metric is best."""

import contextlib

from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint


class BestEpochMetricsCallback(Callback):
    """Tracks all metrics at the epoch when the monitored metric is best.

    This callback captures both training and validation metrics from the same epoch
    where the monitored metric (e.g., val/loss) achieves its best value. Unlike
    tracking the best value for each metric independently, this ensures all metrics
    are from the same checkpoint/epoch.

    The metrics are logged with the prefix 'best_epoch/' to distinguish them from
    the running metrics and independent best metrics.

    Parameters
    ----------
    monitor : str
        The metric to monitor (e.g., "val/loss").
    mode : str, optional
        Whether to minimize ("min") or maximize ("max") the monitored metric (default: "min").

    Examples
    --------
    If validation loss is the monitored metric and reaches its minimum at epoch 42,
    this callback will log:
    - best_epoch/train/loss
    - best_epoch/train/accuracy
    - best_epoch/val/loss
    - best_epoch/val/accuracy
    - best_epoch/val/f1
    etc., all from epoch 42.
    """

    def __init__(self, monitor: str, mode: str = "min"):
        super().__init__()
        self.monitor_metric = monitor
        self.mode = mode
        self.best_monitored_value = None
        self.best_epoch_metrics = {}
        self.best_epoch_number = None
        self.checkpoint_callback = None
        self.current_epoch_train_metrics = {}

    def on_train_start(self, trainer, pl_module):
        """Find and store reference to ModelCheckpoint callback for checkpoint path.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer.
        pl_module : LightningModule
            The PyTorch Lightning module being trained.
        """
        # Find the ModelCheckpoint callback (only needed for getting checkpoint path later)
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                self.checkpoint_callback = callback
                break

    def on_train_epoch_end(self, trainer, pl_module):
        """Capture training metrics at the end of training phase.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer.
        pl_module : LightningModule
            The PyTorch Lightning module being trained.
        """
        # Store all current training metrics temporarily
        self.current_epoch_train_metrics = {
            k: v.item() if hasattr(v, "item") else v
            for k, v in trainer.callback_metrics.items()
            if k.startswith("train/")
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        """Check if this is the best epoch and capture all metrics if so.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer.
        pl_module : LightningModule
            The PyTorch Lightning module being trained.
        """
        # Get current value of monitored metric
        current_value = trainer.callback_metrics.get(self.monitor_metric)

        # Convert to float if tensor
        current_value = (
            current_value.item()
            if hasattr(current_value, "item")
            else current_value
        )

        # Check if this is the best epoch
        is_best = (
            self.best_monitored_value is None
            or (
                self.mode == "min"
                and current_value < self.best_monitored_value
            )
            or (
                self.mode == "max"
                and current_value > self.best_monitored_value
            )
        )

        # If best, capture ALL current metrics (train + val)
        if is_best:
            self.best_monitored_value = current_value
            self.best_epoch_number = trainer.current_epoch

            # Combine training metrics (captured earlier) with validation metrics (current)
            val_metrics = {
                k: v.item() if hasattr(v, "item") else v
                for k, v in trainer.callback_metrics.items()
                if k.startswith("val/")
            }

            self.best_epoch_metrics = {
                **self.current_epoch_train_metrics,  # Training metrics from this epoch
                **val_metrics,  # Validation metrics from this epoch
            }

            # Log the best epoch number
            pl_module.log("best_epoch", self.best_epoch_number, prog_bar=False)

            # Log all metrics from best epoch with special prefix
            for key, value in self.best_epoch_metrics.items():
                pl_module.log(f"best_epoch/{key}", value, prog_bar=False)

    def _log_to_wandb_summary(self, pl_module, params_dict):
        """Log parameters to wandb summary for visibility.

        Parameters
        ----------
        pl_module : LightningModule
            The PyTorch Lightning module being trained.
        params_dict : dict
            Dictionary of parameters to log to wandb summary.
        """
        if pl_module.logger is not None:
            # Handle case where logger is a list
            loggers = (
                pl_module.logger
                if isinstance(pl_module.logger, list)
                else [pl_module.logger]
            )
            for logger in loggers:
                # Check if it's a WandbLogger and log to summary
                if hasattr(logger, "experiment") and hasattr(
                    logger.experiment, "summary"
                ):
                    with contextlib.suppress(Exception):
                        for key, value in params_dict.items():
                            logger.experiment.summary[key] = value

    def on_train_end(self, trainer, pl_module):
        """Log the best model checkpoint path and metadata at the end of training.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer.
        pl_module : LightningModule
            The PyTorch Lightning module being trained.
        """
        if self.checkpoint_callback is not None:
            # Prepare summary data
            summary_data = {}

            # Add monitored metric with mode
            monitored_metric_with_mode = f"{self.monitor_metric} ({self.mode})"
            summary_data["monitored_metric"] = monitored_metric_with_mode

            # Add best model checkpoint path
            best_model_path = self.checkpoint_callback.best_model_path
            if best_model_path:
                summary_data["best_epoch/checkpoint"] = best_model_path

            # Log to wandb summary
            self._log_to_wandb_summary(pl_module, summary_data)

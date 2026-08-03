"""Callback to track all metrics at the epoch when the monitored metric is best."""

import contextlib
from collections.abc import Mapping
from numbers import Real

from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

_CALLBACK_STATE_FORMAT = "best-epoch-metrics-v1"
_CALLBACK_STATE_KEYS = frozenset(
    {
        "format_version",
        "monitor_metric",
        "mode",
        "best_monitored_value",
        "best_epoch_metrics",
        "best_epoch_number",
        "current_epoch_train_metrics",
    }
)


def _metric_state(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{name} keys must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{name}[{key!r}] must be a real scalar")
        result[key] = item
    return result


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

    def state_dict(self):
        """Persist the selected epoch across production ``ckpt_path`` resume."""
        return {
            "format_version": _CALLBACK_STATE_FORMAT,
            "monitor_metric": self.monitor_metric,
            "mode": self.mode,
            "best_monitored_value": self.best_monitored_value,
            "best_epoch_metrics": dict(self.best_epoch_metrics),
            "best_epoch_number": self.best_epoch_number,
            "current_epoch_train_metrics": dict(
                self.current_epoch_train_metrics
            ),
        }

    def load_state_dict(self, state_dict):
        """Restore only state belonging to this exact monitor policy."""
        if not isinstance(state_dict, Mapping):
            raise TypeError("best-epoch callback state must be a mapping")
        actual = frozenset(state_dict)
        if actual != _CALLBACK_STATE_KEYS:
            raise ValueError(
                "best-epoch callback state keys must match exactly; "
                f"missing={sorted(_CALLBACK_STATE_KEYS - actual)!r}, "
                f"extra={sorted(actual - _CALLBACK_STATE_KEYS)!r}"
            )
        if state_dict["format_version"] != _CALLBACK_STATE_FORMAT:
            raise ValueError(
                "unsupported best-epoch callback state version "
                f"{state_dict['format_version']!r}"
            )
        if state_dict["monitor_metric"] != self.monitor_metric:
            raise ValueError("best-epoch monitor identity mismatch")
        if state_dict["mode"] != self.mode:
            raise ValueError("best-epoch mode identity mismatch")
        best_value = state_dict["best_monitored_value"]
        if best_value is not None and (
            isinstance(best_value, bool) or not isinstance(best_value, Real)
        ):
            raise TypeError("best_monitored_value must be a real scalar or None")
        best_epoch = state_dict["best_epoch_number"]
        if best_epoch is not None and (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch < 0
        ):
            raise TypeError(
                "best_epoch_number must be a non-negative integer or None"
            )
        self.best_monitored_value = best_value
        self.best_epoch_metrics = _metric_state(
            state_dict["best_epoch_metrics"],
            "best_epoch_metrics",
        )
        self.best_epoch_number = best_epoch
        self.current_epoch_train_metrics = _metric_state(
            state_dict["current_epoch_train_metrics"],
            "current_epoch_train_metrics",
        )

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

    def on_validation_end(self, trainer, pl_module):
        """Capture and log metrics after validation output is finalized."""
        current_value = trainer.callback_metrics.get(self.monitor_metric)
        current_value = (
            current_value.item()
            if hasattr(current_value, "item")
            else current_value
        )
        if current_value is None:
            return

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
        if not is_best:
            return

        self.best_monitored_value = current_value
        self.best_epoch_number = trainer.current_epoch
        val_metrics = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in trainer.callback_metrics.items()
            if key.startswith("val/")
        }
        self.best_epoch_metrics = {
            **self.current_epoch_train_metrics,
            **val_metrics,
        }

        metrics_to_log = {
            "best_epoch": self.best_epoch_number,
            **{
                f"best_epoch/{key}": value
                for key, value in self.best_epoch_metrics.items()
            },
        }
        loggers = trainer.loggers
        if isinstance(loggers, (list, tuple)):
            for logger in loggers:
                logger.log_metrics(metrics_to_log, step=trainer.global_step)

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

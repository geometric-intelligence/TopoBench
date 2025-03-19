"""TopoBench: A library for benchmarking topological models."""

import time

import lightning.pytorch as pl
import numpy as np


class PipelineTimer(pl.Callback):
    """Measures and logs average execution times of training, validation, and testing stages."""

    def __init__(self):
        """Initialize dictionaries to store accumulated times and counts."""
        self.sums = {
            "train_batch": [],
            "train_epoch": [],
            "val_batch": [],
            "val_epoch": [],
            "test_batch": [],
            "test_epoch": [],
        }
        self.counts = {key: 0 for key in self.sums}
        self.times = {}
        self.skip_first_n = 10

    def _start_timer(self, stage: str):
        """Start a timer for the given stage.

        Parameters
        ----------
        stage : str
            The name of the stage to track time for.
        """
        self.times[stage] = time.time()

    def _end_timer(self, stage: str):
        """End the timer for the given stage and accumulate elapsed time.

        Parameters
        ----------
        stage : str
            The name of the stage to track time for.
        """
        if stage in self.times:
            elapsed = time.time() - self.times[stage]
            self.sums[stage].append(elapsed)
            self.counts[stage] += 1

    def _log_averages(self, trainer):
        """Compute and log average times for all tracked stages.

        Parameters
        ----------
        trainer : object
            The PyTorch Lightning trainer instance used for logging.
        """
        avg_times = {}
        for stage in self.sums:
            if self.counts[stage] > 0:
                if "test" not in stage:
                    avg_times[f"AvgTime/{stage}_mean"] = np.mean(
                        self.sums[stage][self.skip_first_n :]
                    )
                    avg_times[f"AvgTime/{stage}_std"] = np.std(
                        self.sums[stage][self.skip_first_n :]
                    )
                else:
                    avg_times[f"AvgTime/{stage}_mean"] = np.mean(
                        self.sums[stage]
                    )
                    avg_times[f"AvgTime/{stage}_std"] = 0.0

        if trainer.logger:
            trainer.logger.log_metrics(avg_times)

    # Training Timing
    def on_train_batch_start(self, *args):
        """Start timing a training batch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._start_timer("train_batch")

    def on_train_batch_end(self, *args):
        """End timing a training batch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._end_timer("train_batch")

    def on_train_epoch_start(self, *args):
        """Start timing a training epoch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._start_timer("train_epoch")

    def on_train_epoch_end(self, *args):
        """End timing a training epoch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._end_timer("train_epoch")

    # Validation Timing
    def on_validation_batch_start(self, *args):
        """Start timing a validation batch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._start_timer("val_batch")

    def on_validation_batch_end(self, *args):
        """End timing a validation batch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._end_timer("val_batch")

    def on_validation_epoch_start(self, *args):
        """Start timing a validation epoch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._start_timer("val_epoch")

    def on_validation_epoch_end(self, *args):
        """End timing a validation epoch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._end_timer("val_epoch")

    # Testing Timing
    def on_test_batch_start(self, *args):
        """Start timing a test batch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._start_timer("test_batch")

    def on_test_batch_end(self, *args):
        """End timing a test batch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._end_timer("test_batch")

    def on_test_epoch_start(self, *args):
        """Start timing a test epoch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._start_timer("test_epoch")

    def on_test_epoch_end(self, *args):
        """End timing a test epoch.

        Parameters
        ----------
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._end_timer("test_epoch")

    def on_train_end(self, trainer, *args):
        """Log the average times at the end of training.

        Parameters
        ----------
        trainer : object
            The PyTorch Lightning trainer instance used for logging.
        *args : tuple
            Additional arguments passed by the trainer.
        """
        self._log_averages(trainer)

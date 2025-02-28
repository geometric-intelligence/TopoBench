"""Abstract class for the evaluator class."""

from abc import ABC, abstractmethod


class AbstractEvaluator(ABC):
    r"""Abstract class for the evaluator class."""

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def update(self, model_out: dict):
        r"""Update the metrics with the model output.

        Parameters
        ----------
        model_out : dict
            The model output.
        """

    @abstractmethod
    def compute(self):
        r"""Compute the metrics."""

    @abstractmethod
    def reset(self):
        """Reset the metrics."""

"""Abstract class for the loss class."""

from abc import ABC, abstractmethod

import torch_geometric


class AbstractLoss(ABC):
    r"""Abstract class for the loss class."""

    def __init__(
        self,
    ):
        super().__init__()

    def __call__(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Logic for the loss based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        return self.forward(model_out, batch)

    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.
        """

"""Loss module for the topobench package."""

import torch
import torch_geometric

from topobench.loss.base import AbstractLoss
from topobench.loss.dataset import DatasetLoss


class TBLoss(AbstractLoss):
    r"""Defines the default model loss for the given task.

    Parameters
    ----------
    dataset_loss : dict
        Dictionary containing the dataset loss information.
    modules_losses : AbstractLoss, optional
        Custom modules' losses to be used.
    """

    def __init__(self, dataset_loss, modules_losses={}):  # noqa: B006
        super().__init__()
        self.losses = []
        # Dataset loss
        self.losses.append(DatasetLoss(dataset_loss))
        # Model losses
        self.losses.extend(
            [loss for loss in modules_losses.values() if loss is not None]
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(losses={self.losses})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the loss function.

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
        losses = [loss(model_out, batch) for loss in self.losses]

        model_out["loss"] = torch.stack(losses).sum()

        return model_out

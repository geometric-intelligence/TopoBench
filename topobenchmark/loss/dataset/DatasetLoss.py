"""Loss module for the topobenchmark package."""

import torch
import torch_geometric

from topobenchmark.loss.base import AbstractLoss


class DatasetLoss(AbstractLoss):
    r"""Defines the default model loss for the given task.

    Parameters
    ----------
    dataset_loss : dict
        Dictionary containing the dataset loss information.
    """

    def __init__(self, dataset_loss):
        super().__init__()
        self.task = dataset_loss["task"]
        self.loss_type = dataset_loss["loss_type"]
        # Dataset loss
        if self.task == "classification" and self.loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.task == "regression" and self.loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.task == "regression" and self.loss_type == "mae":
            self.criterion = torch.nn.L1Loss()
        else:
            raise Exception("Loss is not defined")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, loss_type={self.loss_type})"

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
        logits = model_out["logits"]
        target = model_out["labels"]

        if self.task == "regression":
            target = target.unsqueeze(1)

        dataset_loss = self.criterion(logits, target)

        return dataset_loss

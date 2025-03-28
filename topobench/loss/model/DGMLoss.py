"""Differentiable Graph Module loss function."""

import torch
import torch_geometric

from topobench.loss.base import AbstractLoss


class DGMLoss(AbstractLoss):
    r"""DGM loss function.

    Original implementation https://github.com/lcosmo/DGM_pytorch/blob/main/DGMlib/model_dDGM_old.py

    Parameters
    ----------
    loss_weight : float, optional
        Loss weight (default: 0.5).
    """

    def __init__(self, loss_weight=0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.avg_accuracy = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> torch.Tensor:
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
        batch_keys = batch.keys()
        logprobs_keys = sorted(
            [key for key in batch_keys if "logprobs_" in key]
        )

        # Filter out the logprobs based on the model phase (Training, test)
        logprobs = []
        for key in logprobs_keys:
            # Get the correct mask
            if batch.model_state == "Training":
                mask = batch.train_mask
            elif batch.model_state == "Validation":
                mask = batch.val_mask
            elif batch.model_state == "Test":
                mask = batch.test_mask
            logprobs.append(batch[key][mask])
        logprobs = torch.stack(logprobs)

        corr_pred = (
            (model_out["logits"].argmax(-1) == model_out["labels"])
            .float()
            .detach()
        )
        if (
            self.avg_accuracy is None
            or self.avg_accuracy.shape[-1] != corr_pred.shape[-1]
        ):
            self.avg_accuracy = torch.ones_like(corr_pred) * 0.5

        point_w = self.avg_accuracy - corr_pred
        loss = (point_w * logprobs.exp().mean(-1)).mean()

        return self.loss_weight * loss

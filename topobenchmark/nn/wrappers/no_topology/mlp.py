"""MLP wrapper module."""

import torch

from topobenchmark.nn.wrappers.base import AbstractWrapper


class MLPWrapper(AbstractWrapper):
    r"""Wrapper for the MLP model.

    This wrapper defines the forward pass of the model. The MLP model returns
    the embeddings of the cells of rank 1. The embeddings of the cells of rank
    0 are computed as the sum of the embeddings of the cells of rank 1
    connected to them.
    """

    def forward(self, batch):
        r"""Forward pass for the MLP wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_0 = self.backbone(x_0=batch.x_0)
        model_out = {"labels": batch.y, "batch_0": batch.batch_0}        
        model_out["x_0"] = x_0
        return model_out

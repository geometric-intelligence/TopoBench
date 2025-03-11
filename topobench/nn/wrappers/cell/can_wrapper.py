"""CAN wrapper module."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper


class CANWrapper(AbstractWrapper):
    r"""Wrapper for the CAN model.

    This wrapper defines the forward pass of the model. The CAN model returns
    the embeddings of the cells of rank 1. The embeddings of the cells of rank
    0 are computed as the sum of the embeddings of the cells of rank 1
    connected to them.
    """

    def forward(self, batch):
        r"""Forward pass for the CAN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_1 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            adjacency_0=batch.adjacency_0.coalesce(),
            down_laplacian_1=batch.down_laplacian_1.coalesce(),
            up_laplacian_1=batch.up_laplacian_1.coalesce(),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_1"] = x_1
        model_out["x_0"] = torch.sparse.mm(batch.incidence_1, x_1)
        return model_out

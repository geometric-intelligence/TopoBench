"""Wrapper for the CCCN model."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper


class CCCNWrapper(AbstractWrapper):
    r"""Wrapper for the CCCN model.

    This wrapper defines the forward pass of the model. The CCCN model returns
    the embeddings of the cells of rank 1. The embeddings of the cells of rank
    0 are computed as the sum of the embeddings of the cells of rank 1
    connected to them.
    """

    def forward(self, batch):
        r"""Forward pass for the CCCN wrapper.

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
            batch.x_1,
            batch.down_laplacian_1.coalesce(),
            batch.up_laplacian_1.coalesce(),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}

        model_out["x_1"] = x_1
        model_out["x_0"] = torch.sparse.mm(batch.incidence_1, x_1)
        return model_out

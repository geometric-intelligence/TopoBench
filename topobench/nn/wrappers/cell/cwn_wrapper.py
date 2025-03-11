"""Wrapper for the CWN model."""

from topobench.nn.wrappers.base import AbstractWrapper


class CWNWrapper(AbstractWrapper):
    r"""Wrapper for the CWN model.

    This wrapper defines the forward pass of the model. The CWN model returns
    the embeddings of the cells of rank 0, 1, and 2.
    """

    def forward(self, batch):
        r"""Forward pass for the CWN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_0, x_1, x_2 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            x_2=batch.x_2,
            incidence_1_t=batch.incidence_1.T,
            adjacency_0=batch.adjacency_1,
            incidence_2=batch.incidence_2,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out

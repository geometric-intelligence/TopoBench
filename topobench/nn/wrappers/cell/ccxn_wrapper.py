"""Wrapper for the CCXN model."""

from topobench.nn.wrappers.base import AbstractWrapper


class CCXNWrapper(AbstractWrapper):
    r"""Wrapper for the CCXN model.

    This wrapper defines the forward pass of the model. The CCXN model returns
    the embeddings of the cells of rank 0, 1, and 2.
    """

    def forward(self, batch):
        r"""Forward pass for the CCXN wrapper.

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
            adjacency_0=batch.adjacency_0,
            incidence_2_t=batch.incidence_2.T,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out

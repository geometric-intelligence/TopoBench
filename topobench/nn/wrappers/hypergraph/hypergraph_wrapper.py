"""Wrapper for the hypergraph models."""

from topobench.nn.wrappers.base import AbstractWrapper


class HypergraphWrapper(AbstractWrapper):
    r"""Wrapper for the hypergraph models.

    This wrapper defines the forward pass of the model. The hypergraph model
    return the embeddings of the cells of rank 0, and 1 (the hyperedges).
    """

    def forward(self, batch):
        r"""Forward pass for the hypergraph wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_0, x_1 = self.backbone(batch.x_0, batch.incidence_hyperedges)
        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["hyperedge"] = x_1

        return model_out

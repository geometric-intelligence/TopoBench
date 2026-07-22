"""Wrapper for the Gauge model."""

from topobench.nn.wrappers.base import AbstractWrapper


class GaugeWrapper(AbstractWrapper):
    r"""Wrapper for the Gauge model.

    This wrapper defines the forward pass of the model. The Gauge backbone
    returns the rank-0 cell embeddings together with the per-node local frames;
    only the embeddings are propagated downstream.
    """

    def forward(self, batch):
        r"""Forward pass for the Gauge wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        z, _Q = self.backbone(batch.x_0, batch.edge_index)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = z

        return model_out

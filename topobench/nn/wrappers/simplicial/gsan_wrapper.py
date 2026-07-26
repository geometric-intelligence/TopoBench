"""Wrapper for the GSAN model."""

from topobench.nn.wrappers.base import AbstractWrapper


class GSANWrapper(AbstractWrapper):
    r"""Wrapper for the GSAN model.

    This wrapper defines the forward pass of the model. The GSAN model returns
    the embeddings of the cells of rank 0, 1, and 2.
    """

    def forward(self, batch):
        r"""Forward pass for the GSAN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_all = (batch.x_0, batch.x_1, getattr(batch, "x_2", None))
        
        incidence_all = (batch.incidence_1, getattr(batch, "incidence_2", None))
        
        x_0, x_1, x_2 = self.backbone(x_all, incidence_all)

        model_out = {"labels": getattr(batch, "y", None), "batch_0": getattr(batch, "batch_0", None)}

        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        if x_2 is not None:
            model_out["x_2"] = x_2

        return model_out

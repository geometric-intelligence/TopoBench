"""Wrapper for the TopoTune model."""

from topobench.nn.wrappers.base import AbstractWrapper


class TuneWrapper(AbstractWrapper):
    r"""Wrapper for the TopoTune model.

    This wrapper defines the forward pass of the TopoTune model.
    """

    def forward(self, batch):
        r"""Forward pass for the Tune wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x = self.backbone(batch)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}

        for key, value in x.items():
            model_out[f"x_{key}"] = value
        return model_out

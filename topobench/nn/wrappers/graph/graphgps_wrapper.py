from topobench.nn.wrappers.base import AbstractWrapper


class GraphGPSWrapper(AbstractWrapper):
    r"""Wrapper for the GraphGPS model.

    Handles the forward pass for GraphGPS, which expects the full batch.
    """

    def forward(self, batch):
        x_0, x_dis = self.backbone(batch)

        model_out = {
            "labels": batch.y,
            "batch_0": batch.batch_0,
            "x_0": x_0,
            "x_dis": x_dis,
        }
        return model_out

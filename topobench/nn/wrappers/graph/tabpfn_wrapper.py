import torch


from topobench.nn.wrappers.base import AbstractWrapper


class TabPFNWrapper(AbstractWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(self, backbone, **kwargs):
        self.backbone = backbone
        self.model_fit_flag = False
        self.lag = 0

        self.train_dataloader = kwargs["train_dataloader"]
        self.val_dataloader = kwargs["val_dataloader"]

    def __call__(self, batch):
        model_out = self.forward(batch)
        return model_out

    def fit(self, batch):
        batch_x = batch.x_0
        batch_y = batch.y

        self.backbone.fit(batch_x.cpu(), batch_y.cpu())

    def forward(self, batch):
        r"""Forward pass for the Tune wrapper.

        Parameters
        ----------self.backbone.fit(batch_x, batch_y)
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        # Option 2 (better one)
        if self.model_fit_flag == False:
            batch = next(iter(self.train_dataloader))
            self.fit(batch)

        batch_x = batch["batch_x"]
        batch_y = batch["batch_y"]
        batch_x_mark = batch["batch_x_mark"]
        batch_y_mark = batch["batch_y_mark"]

        outputs = self.backbone(batch_y, batch_x_mark)
        outputs = outputs[:, -self.kwargs.pred_len :, :]
        batch_y_subset = batch_y_subset[:, -self.kwargs.pred_len :, :]

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {"predictions": outputs, "targets": batch_y_subset}

        return model_out

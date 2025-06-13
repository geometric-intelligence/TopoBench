import torch


# from topobench.nn.wrappers.base import AbstractWrapper


class TabPFNWrapper(torch.nn.Module):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.model_fit_flag = False
        self.lag = 0

        # self.train_dataloader = kwargs["train_dataloader"]
        # self.val_dataloader = kwargs["val_dataloader"]

    def __call__(self, batch):
        model_out = self.forward(batch)
        return model_out

    def fit(self, x, y):
        self.backbone.fit(x.cpu(), y.cpu())

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

        mask = batch["train_mask"]
        x = batch.x_0[mask]
        y = batch.y[mask]
        self.fit(x, y)

        proba = self.backbone.predict_proba(batch.x_0.cpu())

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = torch.Tensor(proba).float().to(batch.y.device)

        return model_out

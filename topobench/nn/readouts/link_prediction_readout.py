"""Readout layer for edge-level link prediction."""

from typing import Any

import torch
import torch.nn as nn
import torch_geometric


class LinkPredictionReadOut(nn.Module):
    r"""Edge-level readout for link prediction.

    Scores candidate edges via dot products of node embeddings.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node embeddings from the backbone.
    out_channels : int
        Number of output channels (kept for API compatibility).
    task_level : str
        Must be ``"edge"``.
    pooling_type : str, optional
        Ignored, kept for API compatibility. Default is ``"sum"``.
    logits_linear_layer : bool, optional
        Ignored, kept for API compatibility. Default is ``True``.
    **kwargs : dict, optional
        Extra arguments ignored by this readout.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "sum",
        logits_linear_layer: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if task_level != "edge":
            raise ValueError(
                f"LinkPredictionReadOut is intended for 'edge' tasks, got '{task_level}'."
            )

        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.task_level = task_level

    def forward(
        self,
        model_out: dict[str, torch.Tensor],
        batch: torch_geometric.data.Data,
    ) -> dict[str, torch.Tensor]:
        r"""Compute edge scores and attach logits/labels.

        Parameters
        ----------
        model_out : dict
            Backbone outputs. Must contain node embeddings under ``"x_0"``.
        batch : torch_geometric.data.Data
            Batch with ``edge_label_index`` and ``edge_label``.

        Returns
        -------
        dict
            ``model_out`` updated with ``"logits"`` and ``"labels"``.
        """
        if "x_0" not in model_out:
            raise KeyError(
                "Expected node embeddings 'x_0' in model_out for LinkPredictionReadOut."
            )

        if not hasattr(batch, "edge_label_index") or not hasattr(batch, "edge_label"):
            raise AttributeError(
                "Batch must contain 'edge_label_index' and 'edge_label' for link prediction."
            )

        h = model_out["x_0"]  # [N, hidden_dim]
        edge_index = batch.edge_label_index
        labels = batch.edge_label.long()  # ensure class indices 0/1, not float

        src, dst = edge_index
        score = (h[src] * h[dst]).sum(dim=-1)  # [E]

        # Turn scalar score into 2-class logits: [E, 2]
        logits = torch.stack([-score, score], dim=-1)

        model_out["logits"] = logits
        model_out["labels"] = labels
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, "
            f"out_channels={self.out_channels}, task_level='{self.task_level}')"
        )

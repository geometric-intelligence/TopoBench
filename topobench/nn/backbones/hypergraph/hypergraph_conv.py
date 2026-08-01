"""Native dense-incidence PyG HypergraphConv backbone."""

from __future__ import annotations

from numbers import Integral, Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import HypergraphConv


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_inputs(
    x: Tensor,
    hyperedge_index: Tensor,
    *,
    in_channels: int,
) -> None:
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("x must be a rank-2 floating tensor")
    if x.size(1) != in_channels:
        raise ValueError(
            f"x must have {in_channels} features; received {x.size(1)}"
        )
    if not isinstance(hyperedge_index, Tensor):
        raise TypeError("hyperedge_index must be a torch.Tensor")
    if hyperedge_index.layout != torch.strided:
        raise TypeError("hyperedge_index must be a dense tensor")
    if hyperedge_index.ndim != 2 or hyperedge_index.size(0) != 2:
        raise ValueError("hyperedge_index must have shape [2, M]")
    if hyperedge_index.dtype is not torch.long:
        raise TypeError("hyperedge_index must use torch.long")
    if hyperedge_index.device != x.device:
        raise ValueError("x and hyperedge_index must use the same device")
    if not hyperedge_index.numel():
        return

    node_ids, hyperedge_ids = hyperedge_index
    if int(node_ids.min()) < 0 or int(node_ids.max()) >= x.size(0):
        raise ValueError("hyperedge_index contains an invalid node index")
    if int(hyperedge_ids.min()) < 0:
        raise ValueError("hyperedge_index contains a negative hyperedge ID")
    expected_ids = torch.arange(
        int(hyperedge_ids.max()) + 1,
        device=hyperedge_ids.device,
    )
    if not torch.equal(torch.unique(hyperedge_ids), expected_ids):
        raise ValueError(
            "hyperedge_index hyperedge IDs must be contiguous from zero"
        )


class HypergraphConvBackbone(nn.Module):
    """Stack PyG HypergraphConv layers over native dense incidence."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_integer("in_channels", in_channels)
        self.hidden_channels = _positive_integer(
            "hidden_channels", hidden_channels
        )
        num_layers = _positive_integer("num_layers", num_layers)
        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise TypeError("dropout must be a real number")
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        widths = [self.in_channels] + [self.hidden_channels] * num_layers
        self.convs = nn.ModuleList(
            HypergraphConv(widths[index], widths[index + 1])
            for index in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """Return one hidden embedding for every input node."""
        _validate_inputs(
            x,
            hyperedge_index,
            in_channels=self.in_channels,
        )
        for layer_index, conv in enumerate(self.convs):
            x = conv(x, hyperedge_index)
            if layer_index + 1 < len(self.convs):
                x = self.dropout(F.relu(x))
        return x


__all__ = ["HypergraphConvBackbone"]

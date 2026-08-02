"""Native homogeneous graph node feature encoder."""

from __future__ import annotations

from copy import copy
from collections.abc import Mapping, Sequence

from numbers import Integral

import torch
from omegaconf import ListConfig
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GraphNorm

from topobench.data.features import encode_categorical_columns
from topobench.dataloader.graph import (
    hypergraph_validation_context,
    mark_hypergraph_validated,
)
from topobench.nn.encoders.base import AbstractFeatureEncoder


class GraphNodeFeatureEncoder(AbstractFeatureEncoder):
    """Normalize and project native homogeneous node features.

    Parameters
    ----------
    in_channels : int
        Width of the input ``data.x`` tensor.
    out_channels : int
        Width assigned to the encoded ``data.x`` tensor.
    encoding_mode : {"continuous", "categorical_one_hot"}, default="continuous"
        Representation supplied in ``data.x`` before batch-local encoding.
    categorical_cardinalities : sequence of int, optional
        Per-column category counts. Required for categorical one-hot mode.
    dropout : float, default=0.0
        Dropout probability applied after projection and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        encoding_mode: str = "continuous",
        categorical_cardinalities: object = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        if encoding_mode not in {"continuous", "categorical_one_hot"}:
            raise ValueError(
                "encoding_mode must be 'continuous' or "
                "'categorical_one_hot'"
            )
        if encoding_mode == "categorical_one_hot":
            if categorical_cardinalities is None:
                raise ValueError(
                    "categorical_cardinalities are required for "
                    "categorical_one_hot encoding"
                )
            if (
                isinstance(
                    categorical_cardinalities,
                    (str, bytes, Mapping),
                )
                or not isinstance(
                    categorical_cardinalities,
                    (Sequence, ListConfig),
                )
            ):
                raise TypeError(
                    "categorical_cardinalities must be an ordered sequence"
                )
            cardinalities = tuple(categorical_cardinalities)
            if not cardinalities or any(
                isinstance(cardinality, bool)
                or not isinstance(cardinality, Integral)
                or cardinality <= 0
                for cardinality in cardinalities
            ):
                raise ValueError(
                    "categorical_cardinalities must contain positive integers"
                )
            cardinalities = tuple(int(value) for value in cardinalities)
            if len(cardinalities) > in_channels:
                raise ValueError(
                    "categorical_cardinalities length may not exceed "
                    "in_channels"
                )
            encoded_in_channels = (
                sum(cardinalities) + in_channels - len(cardinalities)
            )
        elif categorical_cardinalities is not None:
            raise ValueError(
                "categorical_cardinalities require categorical_one_hot "
                "encoding_mode"
            )
        else:
            cardinalities = ()
            encoded_in_channels = in_channels

        self.in_channels = in_channels
        self.encoded_in_channels = encoded_in_channels
        self.encoding_mode = encoding_mode
        self.categorical_cardinalities = cardinalities
        self.norm = GraphNorm(encoded_in_channels)
        self.projection = torch.nn.Linear(encoded_in_channels, out_channels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data: Data) -> Data:
        """Return a shallow data copy with batch-locally encoded features."""
        if not isinstance(data, Data) or isinstance(data, HeteroData):
            raise TypeError(
                "GraphNodeFeatureEncoder requires homogeneous Data"
            )
        validation_context = hypergraph_validation_context(data)

        x = data.get("x")
        if not isinstance(x, Tensor):
            raise ValueError("data.x must be a rank-2 tensor")
        if x.ndim != 2:
            raise ValueError("data.x must be a rank-2 tensor")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"data.x must have exactly {self.in_channels} input columns"
            )
        if self.encoding_mode == "categorical_one_hot":
            categorical_columns = len(self.categorical_cardinalities)
            categorical_prefix = x[:, :categorical_columns]
            encoded_prefix = encode_categorical_columns(
                categorical_prefix,
                self.categorical_cardinalities,
                context="data.x categorical prefix",
                allow_integral_float=True,
            )
            continuous_suffix = x[:, categorical_columns:]
            if continuous_suffix.shape[1]:
                if not x.is_floating_point():
                    raise TypeError(
                        "data.x continuous suffix must have a floating dtype"
                    )
                if not torch.isfinite(continuous_suffix).all():
                    raise ValueError(
                        "data.x continuous suffix must contain finite values"
                    )
                x = torch.cat((encoded_prefix, continuous_suffix), dim=1)
            else:
                x = encoded_prefix
        elif not x.is_floating_point():
            raise TypeError("data.x must have a floating dtype")
        if x.shape[1] != self.encoded_in_channels:
            raise ValueError(
                "encoded data.x must have exactly "
                f"{self.encoded_in_channels} feature columns"
            )
        batch = data.get("batch")
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        result = copy(data)
        result.x = self.dropout(
            self.activation(self.projection(self.norm(x, batch=batch)))
        )
        if validation_context is not None:
            mark_hypergraph_validated(
                result,
                selector=validation_context.selector,
                num_classes=validation_context.num_classes,
            )
        return result


__all__ = ["GraphNodeFeatureEncoder"]

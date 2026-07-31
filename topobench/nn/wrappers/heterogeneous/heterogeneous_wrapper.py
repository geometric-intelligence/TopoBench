"""Model-agnostic wrapper for native heterogeneous graph backbones."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


class HeterogeneousWrapper(torch.nn.Module):
    """Translate native :class:`HeteroData` into the backbone dictionary API.

    The wrapper deliberately leaves phase masks and neighbor seed selection on
    the batch. Supervision ownership belongs to
    ``HeterogeneousNodeSupervisionAdapter`` after readout.

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone accepting ``(x_dict, edge_index_dict)``.
    target_node_type : str
        Node type whose unfiltered labels are returned.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        target_node_type: str,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, torch.nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        if not isinstance(target_node_type, str):
            raise TypeError("target_node_type must be a non-empty string")
        if not target_node_type.strip():
            raise ValueError("target_node_type must be a non-empty string")
        self.backbone = backbone
        self.target_node_type = target_node_type

    def forward(self, batch: HeteroData) -> dict[str, object]:
        """Return complete typed embeddings and unfiltered target labels.

        Parameters
        ----------
        batch : torch_geometric.data.HeteroData
            Already encoded heterogeneous graph or sampled subgraph.

        Returns
        -------
        dict[str, object]
            ``x_dict`` from the backbone and the target store's complete
            ``y`` tensor.
        """
        labels = self._validate_batch(batch)
        raw_x_dict = self.backbone(batch.x_dict, batch.edge_index_dict)
        x_dict = self._validate_backbone_output(raw_x_dict, batch)
        return {"x_dict": x_dict, "labels": labels}

    def _validate_batch(self, batch: HeteroData) -> Tensor:
        """Validate all non-trainable inputs before calling the backbone."""
        if not isinstance(batch, HeteroData):
            raise TypeError("HeterogeneousWrapper requires native HeteroData")
        if self.target_node_type not in batch.node_types:
            raise ValueError(
                "HeterogeneousWrapper is missing target node store "
                f"{self.target_node_type!r}"
            )
        labels = batch[self.target_node_type].get("y")
        if not isinstance(labels, Tensor):
            raise TypeError(
                f"target store {self.target_node_type!r} must contain tensor y"
            )
        if labels.numel() == 0:
            raise ValueError("target tensor y must be non-empty")
        if labels.ndim != 1:
            raise ValueError("target tensor y must be rank-1")
        target_count = batch[self.target_node_type].num_nodes
        if labels.size(0) != target_count:
            raise ValueError(
                f"target tensor y count must match target nodes "
                f"({labels.size(0)} != {target_count})"
            )
        return labels

    @staticmethod
    def _validate_backbone_output(
        raw_x_dict: object,
        batch: HeteroData,
    ) -> dict[str, Tensor]:
        """Validate and canonically order the complete typed output."""
        if not isinstance(raw_x_dict, Mapping):
            raise TypeError("backbone output must be a mapping")
        if any(not isinstance(node_type, str) for node_type in raw_x_dict):
            raise TypeError("backbone output node-type keys must be strings")
        expected = set(batch.node_types)
        actual = set(raw_x_dict)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "backbone output node types must exactly match the batch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        output: dict[str, Tensor] = {}
        for node_type in batch.node_types:
            features = raw_x_dict[node_type]
            if not isinstance(features, Tensor):
                raise TypeError(
                    f"backbone output for {node_type!r} must be a tensor"
                )
            if features.ndim != 2:
                raise ValueError(
                    f"backbone output for {node_type!r} must be rank-2"
                )
            expected_count = batch[node_type].num_nodes
            if features.size(0) != expected_count:
                raise ValueError(
                    f"backbone output for {node_type!r} has node count "
                    f"{features.size(0)}; expected {expected_count}"
                )
            output[node_type] = features
        return output


__all__ = ["HeterogeneousWrapper"]

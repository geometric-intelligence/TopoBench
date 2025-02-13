"""A transform that has the GPSE_encoder model."""

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import pooling_dict, register_head
from torch_scatter import scatter


def _pad_and_stack(x1: torch.Tensor, x2: torch.Tensor, pad1: int, pad2: int):
    """
    Pad and stack two tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        First tensor to pad and stack.
    x2 : torch.Tensor
        Second tensor to pad and stack.
    pad1 : int
        Padding size for the first tensor.
    pad2 : int
        Padding size for the second tensor.

    Returns
    -------
    torch.Tensor
        Padded and stacked tensor.
    """
    padded_x1 = nn.functional.pad(x1, (0, pad2))
    padded_x2 = nn.functional.pad(x2, (pad1, 0))
    return torch.vstack([padded_x1, padded_x2])


def _apply_index(batch, virtual_node: bool, pad_node: int, pad_graph: int):
    """
    Apply index to batch data, handling virtual nodes and padding.

    Parameters
    ----------
    batch : Batch
        A batch of data containing node features, graph features, and labels.
    virtual_node : bool
        Whether to handle virtual nodes.
    pad_node : int
        Padding size for node features.
    pad_graph : int
        Padding size for graph features.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Padded and stacked predictions and true values.
    """
    graph_pred, graph_true = batch.graph_feature, batch.y_graph
    node_pred, node_true = batch.node_feature, batch.y
    if virtual_node:
        # Remove virtual node
        idx = torch.concat(
            [
                torch.where(batch.batch == i)[0][:-1]
                for i in range(batch.batch.max().item() + 1)
            ]
        )
        node_pred, node_true = node_pred[idx], node_true[idx]

    # Stack node predictions on top of graph predictions and pad with zeros
    pred = _pad_and_stack(node_pred, graph_pred, pad_node, pad_graph)
    true = _pad_and_stack(node_true, graph_true, pad_node, pad_graph)

    return pred, true


@register_head("inductive_hybrid_multi")
class GNNInductiveHybridMultiHead(nn.Module):
    """GNN prediction head for inductive node and graph prediction tasks using individual MLP for each task.

    Parameters
    ----------
    dim_in : int
        Input dimension.
    dim_out : int
        Output dimension. Not used. Use share.num_node_targets and share.num_graph_targets instead.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.node_target_dim = cfg.share.num_node_targets
        self.graph_target_dim = cfg.share.num_graph_targets
        self.virtual_node = cfg.virtual_node
        num_layers = cfg.gnn.layers_post_mp

        layer_config = new_layer_config(
            dim_in, 1, num_layers, has_act=False, has_bias=True, cfg=cfg
        )
        if cfg.gnn.multi_head_dim_inner is not None:
            layer_config.dim_inner = cfg.gnn.multi_head_dim_inner
        self.node_post_mps = nn.ModuleList(
            [MLP(layer_config) for _ in range(self.node_target_dim)]
        )

        self.graph_pooling = pooling_dict[cfg.model.graph_pooling]
        self.graph_post_mp = MLP(
            new_layer_config(
                dim_in,
                self.graph_target_dim,
                num_layers,
                has_act=False,
                has_bias=True,
                cfg=cfg,
            )
        )
        self.counter = 0

    def forward(self, batch):
        """Forward pass for the GNNInductiveHybridMultiHead.

        Parameters
        ----------
        batch : Batch
            A batch of data containing node features, edge attributes, and edge indices.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Padded and stacked predictions and true values.
        """
        batch.node_feature = torch.hstack(
            [m(batch.x) for m in self.node_post_mps]
        )
        # if self.counter < 178:
        #     self.counter += 1
        graph_emb = self.graph_pooling(batch.x, batch.batch)
        # else:
        #     pass
        batch.graph_feature = self.graph_post_mp(graph_emb)
        return _apply_index(
            batch,
            self.virtual_node,
            self.node_target_dim,
            self.graph_target_dim,
        )

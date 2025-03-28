"""Graph MLP loss function."""

import torch
import torch_geometric

from topobench.loss.base import AbstractLoss


class GraphMLPLoss(AbstractLoss):
    r"""Graph MLP loss function.

    Parameters
    ----------
    r_adj_power : int, optional
        Power of the adjacency matrix (default: 2).
    tau : float, optional
        Temperature parameter (default: 1).
    loss_weight : float, optional
        Loss weight (default: 0.5).
    """

    def __init__(self, r_adj_power=2, tau=1.0, loss_weight=0.5):
        super().__init__()
        self.r_adj_power = r_adj_power
        self.tau = tau
        self.loss_weight = loss_weight

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_adj_power={self.r_adj_power}, tau={self.tau}, loss_weight={self.loss_weight})"

    def get_power_adj(self, edge_index):
        r"""Get the power of the adjacency matrix.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor.

        Returns
        -------
        torch.Tensor
            Power of the adjacency matrix.
        """
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
        adj = torch_geometric.utils.to_dense_adj(edge_index)
        adj_power = adj.clone()
        for _ in range(self.r_adj_power - 1):
            adj_power = torch.matmul(adj_power, adj)
        return adj_power

    def graph_mlp_contrast_loss(self, x_dis, adj_label):
        """Graph MLP contrastive loss.

        Parameters
        ----------
        x_dis : torch.Tensor
            Distance matrix.
        adj_label : torch.Tensor
            Adjacency matrix.

        Returns
        -------
        torch.Tensor
            Contrastive loss.
        """
        x_dis = torch.exp(self.tau * x_dis)
        x_dis_sum = torch.sum(x_dis, 1)
        x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
        loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
        return loss

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> torch.Tensor:
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        x_dis = model_out["x_dis"]
        if x_dis is None:  # Validation and test
            return torch.tensor(0.0)
        adj_label = self.get_power_adj(batch.edge_index)
        graph_mlp_loss = self.loss_weight * self.graph_mlp_contrast_loss(
            x_dis, adj_label
        )
        return graph_mlp_loss

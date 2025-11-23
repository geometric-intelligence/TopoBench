
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import negative_sampling


class NegativeSamplingTransform(BaseTransform):
    """
    Dynamically sample negative edges and build edge labels.

    Parameters
    ----------
    neg_pos_ratio : float, optional
        Number of negative edges per positive edge. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. If given, it is set once at
        initialization and not touched afterwards.
    """

    def __init__(self, neg_pos_ratio: float = 1.0, seed: int | None = None, method: str | None = "sparse"):
        super().__init__()
        self.neg_pos_ratio = float(neg_pos_ratio)
        self.method = method
        if seed is not None:
            torch.manual_seed(int(seed))

    def forward(self, data: Data) -> Data:
        """
        Sample fresh negative edges and update edge labels.

        Parameters
        ----------
        data : Data
            Input graph. Must contain:
            - ``edge_index`` : adjacency used for message passing.
            - ``edge_label_index`` : positive edges to be used as labels.

        Returns
        -------
        Data
            A cloned ``Data`` object with updated:
            - ``edge_label_index`` : concatenated positive and negative edges.
            - ``edge_label`` : binary labels (1 for pos, 0 for neg).
        """
        if not hasattr(data, "edge_index"):
            raise AttributeError("Data object must have 'edge_index'.")
        if not hasattr(data, "edge_label_index"):
            raise AttributeError("Data object must have 'edge_label_index' with positive edges.")

        # Work on a clone to avoid in-place accumulation across epochs.
        data = data.clone()

        device = data.edge_index.device
        pos_edge_index = data.edge_label_index.to(device)
        num_pos = pos_edge_index.size(1)

        if num_pos == 0:
            raise ValueError("No positive edges found in 'edge_label_index'.")

        num_neg = max(1, int(self.neg_pos_ratio * num_pos))

        # Sample negatives w.r.t. current training adjacency.
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=num_neg,
            method=self.method,
        ).to(device)

        # Build labels.
        pos_label = torch.ones(num_pos, device=device)
        neg_label = torch.zeros(num_neg, device=device)

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat([pos_label, neg_label], dim=0)

        data.edge_label_index = edge_label_index
        data.edge_label = edge_label

        return data

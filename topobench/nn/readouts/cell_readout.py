"""Cell-level readout for simplicial complexes.

This readout layer predicts labels for valid cells of rank in target_ranks.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data


class SimplicialCellLevelReadout(nn.Module):
    """Readout for cell-level predictions on simplicial complexes.

    Takes features at each rank and predicts labels for valid cells
    at specified target ranks.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of input features on all ranks.
    out_channels : int
        Number of output classes.
    num_cell_dimensions : int
        Rank + 1 of simplicial complex.
    target_ranks : List[int]
        Which ranks have labels to predict (e.g., [2, 3, 4] for simplices with 3-5 nodes).
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        num_cell_dimensions: int,
        target_ranks: list[int],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_cell_dimensions = num_cell_dimensions
        self.target_ranks = target_ranks
        self.task_level = (
            "cell"  # For compatibility with TBModel need this attribute
        )

        # Create prediction head for each target rank
        # Each rank might have different hidden dims in the future, so use a dict
        self.predictors = nn.ModuleDict(
            {
                str(rank): nn.Linear(hidden_dim, out_channels)
                for rank in target_ranks
            }
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"target_ranks={self.target_ranks}, "
            f"out_channels={self.out_channels})"
        )

    def forward(self, model_out: dict, batch: Data) -> dict:
        """Compute cell-level predictions.

        Parameters
        ----------
        model_out : dict
            Dictionary containing x_0, x_1, ..., x_k features per rank.
        batch : Data
            Batch object containing cell_labels for each target rank and mask for valid cells.

        Returns
        -------
        dict
            Updated model_out with:
            - logits: [num_labeled_cells, out_channels]
            - cell_ranks: [num_labeled_cells] - which rank each prediction is for
            - valid_indices: [num_labeled_cells] - which cell indices are valid
        """
        all_logits = []
        all_labels = []
        all_ranks = []
        all_indices = []

        for rank in self.target_ranks:
            # Get features for this rank
            x_key = f"x_{rank}"
            if x_key not in model_out:
                continue

            x_rank = model_out[x_key]  # [num_cells_at_rank, hidden_dim]

            # Get labels for this rank
            label_key = f"cell_labels_{rank}"
            if not hasattr(batch, label_key):
                continue

            labels = getattr(batch, label_key)  # [num_cells_at_rank]

            # Filter valid cells with rank-specific mask
            valid_mask = torch.ones(
                len(labels), dtype=torch.bool, device=labels.device
            )

            mask_key = f"mask_{rank}"
            if hasattr(batch, mask_key):
                rank_mask = getattr(batch, mask_key)  # Boolean mask
                valid_mask &= rank_mask

            # Get final valid indices
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue

            # Get features and labels for valid cells
            x_labeled = x_rank[valid_indices]  # [num_labeled, hidden_dim]
            y_labeled = labels[valid_indices]  # [num_labeled]

            # Predict
            logits = self.predictors[str(rank)](
                x_labeled
            )  # [num_labeled, out_channels]

            all_logits.append(logits)
            all_labels.append(y_labeled)
            all_ranks.extend([rank] * len(valid_indices))
            all_indices.extend(valid_indices.tolist())

        # Concatenate all predictions and labels
        if len(all_logits) > 0:
            model_out["logits"] = torch.cat(all_logits, dim=0)
            model_out["labels"] = torch.cat(all_labels, dim=0)
            model_out["cell_ranks"] = torch.tensor(
                all_ranks, device=all_logits[0].device
            )
            model_out["valid_indices"] = torch.tensor(
                all_indices, device=all_logits[0].device
            )
        else:
            # No labeled cells found - use any available tensor for device
            device = None
            for key in model_out:
                if isinstance(model_out[key], torch.Tensor):
                    device = model_out[key].device
                    break
            if device is None:
                device = torch.device("cpu")

            model_out["logits"] = torch.zeros(
                0, self.out_channels, device=device
            )
            model_out["cell_ranks"] = torch.zeros(
                0, dtype=torch.long, device=device
            )
            model_out["valid_indices"] = torch.zeros(
                0, dtype=torch.long, device=device
            )

        return model_out

    def __call__(self, model_out: dict, batch: Data) -> dict:
        """Wrapper for forward to match AbstractZeroCellReadOut interface.

        Parameters
        ----------
        model_out : dict
            Dictionary containing features per rank.
        batch : Data
            Batch object containing cell labels and valid cell masks.

        Returns
        -------
        dict
            Updated model_out with:
            - logits: [num_labeled_cells, out_channels]
            - cell_ranks: [num_labeled_cells] - which rank each prediction is for
            - valid_indices: [num_labeled_cells] - which cell index within rank
        """
        return self.forward(model_out, batch)

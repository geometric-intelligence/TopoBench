"""Wrapper for SCCNN with cell-level predictions.

This wrapper is designed for transductive learning where predictions are made
on specific cells (simplices) rather than entire graphs or individual nodes.
"""

import torch
from torch_geometric.data import Data

from topobench.nn.wrappers.base import AbstractWrapper


class SCCNNCellWrapper(AbstractWrapper):
    """Wrapper for SCCNN backbone with cell-level outputs.

    Unlike standard wrappers that focus on node features (x_0), this wrapper
    preserves features at ALL ranks for cell-level prediction.

    Parameters
    ----------
    backbone : nn.Module
        The SCCNN backbone model.
    num_cell_dimensions : int
        Rank +1 of the simplicial complex.
    target_ranks : list[int]
        Which ranks have labels to predict.
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        num_cell_dimensions: int,
        target_ranks: list,
        **kwargs,
    ):
        # Ensure required parameters for base class
        if "out_channels" not in kwargs:
            kwargs["out_channels"] = 32  # Default value
        kwargs["num_cell_dimensions"] = num_cell_dimensions
        # Disable residual connections for cell-level prediction
        # (we just pass through features)
        kwargs["residual_connections"] = kwargs.get(
            "residual_connections", False
        )

        super().__init__(backbone, **kwargs)
        self.target_ranks = target_ranks
        self.num_cell_dimensions = num_cell_dimensions

    def __repr__(self):
        return f"{self.__class__.__name__}(target_ranks={self.target_ranks})"

    def forward(self, batch: Data) -> dict:
        """Forward pass preserving all rank features.

        Parameters
        ----------
        batch : Data
            Batch object containing features x_0, x_1, ..., x_k, Laplacians, and incidences.

        Returns
        -------
        dict
            The model_out containing updated features x_0, x_1, ..., x_k.
        """
        # Extract features for all ranks from 0 to num_cell_dimensions-1 = rank
        x_all = []
        for i in range(self.num_cell_dimensions):
            x_key = f"x_{i}"
            if hasattr(batch, x_key):
                x_all.append(getattr(batch, x_key))
            else:
                # If rank doesn't exist, add empty tensor
                x_all.append(torch.zeros(0, 1, device=batch.x_0.device))
        x_all = tuple(x_all)

        # Extract Laplacians
        laplacian_all = self._extract_laplacians(batch)

        # Extract incidences
        incidence_all = self._extract_incidences(batch)

        # Forward through SCCNN backbone
        x_all_out = self.backbone(x_all, laplacian_all, incidence_all)

        # Build output dictionary with features at ALL ranks
        model_out = {}
        for i, x_rank in enumerate(x_all_out):
            model_out[f"x_{i}"] = x_rank

        return model_out

    def _extract_laplacians(self, batch: Data) -> tuple:
        """Extract Laplacian matrices for all ranks.

        Expected format:
        - hodge_laplacian_0
        - down_laplacian_1, up_laplacian_1
        - down_laplacian_2, up_laplacian_2
        - ...

        Parameters
        ----------
        batch : Data
            Batch object containing features x_0, x_1, ..., x_k, Laplacians, and incidences.

        Returns
        -------
        tuple
            Tuple of Laplacian matrices for all ranks.
        """
        laplacian_all = []

        # Rank 0: Hodge Laplacian
        if hasattr(batch, "hodge_laplacian_0"):
            laplacian_all.append(batch.hodge_laplacian_0)
        else:
            laplacian_all.append(None)

        # Store down and up Laplacians for each rank
        for rank in range(1, self.num_cell_dimensions):
            down_key = f"down_laplacian_{rank}"
            up_key = f"up_laplacian_{rank}"

            if hasattr(batch, down_key):
                laplacian_all.append(getattr(batch, down_key))
            else:
                laplacian_all.append(None)

            if hasattr(batch, up_key):
                laplacian_all.append(getattr(batch, up_key))
            else:
                laplacian_all.append(None)

        return tuple(laplacian_all)

    def _extract_incidences(self, batch: Data) -> tuple:
        """Extract incidence matrices.

        Expected format:
        - incidence_1: From 0-cells to 1-cells
        - incidence_2: From 1-cells to 2-cells
        - ...

        Parameters
        ----------
        batch : Data
            Batch object containing features x_0, x_1, ..., x_k, Laplacians, and incidences.

        Returns
        -------
        tuple
            Tuple of incidence matrices for all ranks.
        """
        incidence_all = []

        # Incidences map from rank k-1 to rank k, so we go from 1 to num_cell_dimensions
        for rank in range(1, self.num_cell_dimensions + 1):
            inc_key = f"incidence_{rank}"
            if hasattr(batch, inc_key):
                incidence_all.append(getattr(batch, inc_key))
            else:
                incidence_all.append(None)

        return tuple(incidence_all)

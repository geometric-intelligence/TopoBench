"""Wrapper for the SheafTSP model.

Routes cell complex data from TopoBench's batch format into the SheafTSP
backbone, which operates on 1-cells using the down and up Laplacians.
Produces embeddings for all cell dimensions via incidence propagation.
"""

import torch
import torch.nn as nn

from topobench.nn.wrappers.base import AbstractWrapper


class SheafTSPWrapper(AbstractWrapper):
    r"""Wrapper for the SheafTSP model.

    The SheafTSP backbone operates on 1-cell (edge) features using the
    Hodge Laplacians to derive sheaf connectivity.  This wrapper:
      1. Feeds x_1, down_laplacian_1, up_laplacian_1 to the backbone.
      2. Recovers 0-cell embeddings via incidence_1 (boundary map),
         preserving the encoded 0-cell features via a residual sum so
         node-level signal is not discarded.
      3. Exposes the backbone's sheaf Dirichlet energy (Eq. 15 of
         Tandon et al.) as ``model_out["sheaf_dirichlet"]`` during
         training, for the ``SheafDirichletLoss`` regularizer.

    This follows the same pattern as the CCCNWrapper.

    Parameters
    ----------
    backbone : torch.nn.Module
        The SheafTSP backbone.
    **kwargs : dict
        Arguments for AbstractWrapper (``out_channels``,
        ``num_cell_dimensions``).
    """

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone, **kwargs)
        # Learned embedding of the rank-2 degree signal (per-node count
        # of incident (edge, 2-cell) pairs).  This is the DC component
        # of a rank-2 stalk section: with triangle 2-cells its graph sum
        # equals 6x the triangle count, so a linear readout can recover
        # the count exactly.  Zero-initialized so tasks that do not
        # benefit (e.g. community detection) start unaffected and can
        # keep it switched off.
        self.tri_embed = nn.Linear(1, kwargs["out_channels"], bias=False)
        nn.init.zeros_(self.tri_embed.weight)
        # Warm-start one channel so the count signal is readable from
        # epoch 0: with dropout in the feature path, a fully zero init
        # never matures before early stopping fires on the plateau of
        # the crude density solution.
        with torch.no_grad():
            self.tri_embed.weight[0, 0] = 0.1

    def forward(self, batch):
        r"""Forward pass for the SheafTSP wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.
            Expected attributes:
            - ``x_1``: 1-cell features.
            - ``down_laplacian_1``: Down Laplacian of rank 1.
            - ``up_laplacian_1``: Up Laplacian of rank 1.
            - ``incidence_1``: Boundary operator (0-cells × 1-cells).
            - ``y``: Labels.
            - ``batch_0``: Batch assignment for 0-cells.

        Returns
        -------
        dict
            Dictionary containing:
            - ``labels``: Ground truth labels.
            - ``batch_0``: Batch indices for 0-cells.
            - ``x_0``: 0-cell embeddings (propagated from 1-cells,
              plus the embedded rank-2 degree signal).
            - ``x_1``: 1-cell embeddings (direct backbone output).
        """
        x_1 = self.backbone(
            batch.x_1,
            batch.down_laplacian_1.coalesce(),
            batch.up_laplacian_1.coalesce(),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}

        # 1-cell embeddings from backbone
        model_out["x_1"] = x_1


        # Propagate to 0-cells via boundary map: x_0 = B_1 @ x_1,
        # plus a residual with the encoded 0-cell features so the
        # original node-level signal is preserved.
        x_0 = torch.sparse.mm(batch.incidence_1, x_1)
        if hasattr(batch, "x_0") and batch.x_0.shape == x_0.shape:
            x_0 = x_0 + batch.x_0

        # Rank-2 degree signal: t_e = |B_2| 1 counts 2-cells per edge,
        # t_v = |B_1| t_e sums those counts onto endpoints.  Injected
        # through the zero-initialized embedding (see __init__).
        if hasattr(batch, "incidence_2"):
            inc2 = batch.incidence_2.coalesce()
            if inc2.shape[1] > 0:
                ones_2 = torch.ones(
                    inc2.shape[1], 1, device=x_1.device, dtype=x_1.dtype
                )
                t_e = torch.sparse.mm(torch.abs(inc2), ones_2)
                t_v = torch.sparse.mm(
                    torch.abs(batch.incidence_1.coalesce()), t_e
                )
                x_0 = x_0 + self.tri_embed(t_v)
        model_out["x_0"] = x_0

        # Expose the sheaf Dirichlet energy (Eq. 15 regularizer) only
        # during training; val/test losses stay pure task losses.
        reg = getattr(self.backbone, "dirichlet_energy", None)
        if self.backbone.training and reg is not None:
            model_out["sheaf_dirichlet"] = reg

        return model_out

"""Precompute Random Walk PE for cellular complex ranks 0, 1, 2."""
import torch
from torch_geometric.transforms import BaseTransform
from topobench.nn.backbones.cell.cellular_transformer import random_walk_pe


class CellRandomWalkPE(BaseTransform):
    """Compute and cache RWPE for cellular complexes before batching.

    Attaches rwpe_0, rwpe_1, rwpe_2 to the data object so that
    CellularTransformer can read them directly instead of recomputing
    random_walk_pe inside forward().
    """

    def __init__(self, pe_steps: int = 8):
        self.pe_steps = pe_steps

    def __call__(self, data):
        # Only run if the cellular adjacency fields are present.
        # This lets the transform be safely included in generic pipelines.
        if hasattr(data, "adjacency_0") and data.adjacency_0 is not None:
            with torch.no_grad():
                data.rwpe_0 = random_walk_pe(data.adjacency_0, self.pe_steps)
        if hasattr(data, "coadjacency_1") and data.coadjacency_1 is not None:
            with torch.no_grad():
                data.rwpe_1 = random_walk_pe(data.coadjacency_1, self.pe_steps)
        if hasattr(data, "coadjacency_2") and data.coadjacency_2 is not None:
            with torch.no_grad():
                data.rwpe_2 = random_walk_pe(data.coadjacency_2, self.pe_steps)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(pe_steps={self.pe_steps})"

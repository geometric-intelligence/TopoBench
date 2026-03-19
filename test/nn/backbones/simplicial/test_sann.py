"""Unit tests for SANN."""

import torch
from torch_geometric.utils import get_laplacian
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobenchmarkx.nn.backbones.simplicial import SANN
from topobenchmarkx.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)
from topobenchmarkx.transforms.data_manipulations.precompute_khop_features import (
    PrecomputeKHopFeatures
)


def test_SANN(simple_graph_1):
    """Test SANN.

        Test the SANN backbone module.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            A fixture of simple graph 1.
    """
    max_hop = 2
    complex_dim = 3
    lifting_signed = SimplicialCliqueLifting(
            complex_dim=complex_dim, signed=True
        )
    precompute_k_hop = PrecomputeKHopFeatures(max_hop=max_hop, complex_dim=complex_dim)  
    data = lifting_signed(simple_graph_1)
    data = precompute_k_hop(data)
    out_dim = 4

    # Set all k-hop dimensions to 1 to standardize testing
    for i in range(max_hop+1):
        for j in range(complex_dim):
            data[f"x{j}_{i}"] = data[f"x{j}_{i}"][:, 0:1]

    x_in = tuple(tuple(data[f"x{i}_{j}"] for j in range(max_hop+1)) for i in range(complex_dim))
    expected_shapes = [(data.x.shape[0], out_dim), (data.x_1.shape[0], out_dim), (data.x_2.shape[0], out_dim)]

    auto_test = NNModuleAutoTest([
        {
            "module" : SANN, 
            "init": ((1, 1, 1), 1, 'lrelu', complex_dim, max_hop, 2),
            "forward":  (x_in, ),
            "assert_shape": expected_shapes
        },
    ])
    auto_test.run()

"""Unit tests for simplicial model wrappers."""

import torch
from torch_geometric.utils import get_laplacian
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from ...._utils.flow_mocker import FlowMocker
from topobench.nn.backbones.combinatorial.hopse import HOPSE
from topobench.nn.wrappers import (
    HOPSEWrapper,
)

class TestSimplicialWrappers:
    r"""Test simplicial model wrappers.

        Test all simplicial wrappers.
    """
            

    def test_SANNWrapper(self, sg1_clique_lifted_precompute_k_hop):
        """Test SANNWarpper.
        
        Parameters
        ----------
        sg1_clique_lifted_precompute_k_hop : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with SimlicialCliqueLifting and precomputed k-hop neighbourhood embedding.
        """
        data = sg1_clique_lifted_precompute_k_hop
        in_channels = data.x0_0.shape[1]
        out_channels = data.x_0.shape[1]
        
        wrapper = HOPSEWrapper(
            HOPSE(
                in_channels=in_channels,
                hidden_channels=out_channels
            ), 
            complex_dim=3,
            max_hop=3,
        )

        out = wrapper(data)

        for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
            assert key in out



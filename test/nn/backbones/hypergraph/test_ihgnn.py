import torch
import pytest
from topobench.nn.backbones.hypergraph.ihgnn import IHGNN

def test_ihgnn_forward():
    # Nodi: 5, Iperfacce: 3, Feature dim: 16
    num_nodes = 5
    num_hyperedges = 3
    in_channels = 16
    hidden_channels = 16
    out_channels = 16

    model = IHGNN(in_channels, hidden_channels, out_channels)
    
    # Feature matrix X (num_nodes x in_channels)
    x = torch.randn(num_nodes, in_channels)
    
    # Incidence matrix H (num_nodes x num_hyperedges)
    # 1 se il nodo è nell'iperfaccia, 0 altrimenti
    incidence_indices = torch.tensor([
        [0, 0, 1, 2, 3, 4], # node indices
        [0, 1, 1, 2, 0, 2]  # hyperedge indices
    ])
    incidence_values = torch.ones(6)
    incidence = torch.sparse_coo_tensor(incidence_indices, incidence_values, (num_nodes, num_hyperedges))

    # Test forward pass
    try:
        out = model(x, incidence)
        assert out.shape == (num_nodes, hidden_channels)
        print("✅ Forward pass completato con successo!")
    except Exception as e:
        print(f"❌ Errore durante il forward pass: {e}")
        raise e

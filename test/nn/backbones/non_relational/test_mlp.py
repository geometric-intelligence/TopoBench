"""Test MLP."""
import torch
import torch.nn as nn
import torch_geometric
import pytest
from topobench.nn.backbones.non_relational.mlp import MLP
from omegaconf import DictConfig

@pytest.mark.parametrize("batch_size,num_nodes,in_channels,hidden_layers,out_channels,final_act", [
    (1, 4, 10, [16, 8], 8, "sigmoid"),
    (2, None, 4, [8], 1, None),
])
def test_mlp_forward(batch_size, num_nodes, in_channels, hidden_layers, out_channels, final_act):
    """Test MLP.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_nodes : int
        The number of nodes.
    in_channels : int
        The number of input channels.
    hidden_layers : list
        The list of hidden channel sizes.
    out_channels : int
        The number of output channels.
    final_act : str
        The final activation function.
    """
    x = torch.randn(batch_size, in_channels)
    model = MLP(
        in_channels=in_channels,
        hidden_layers=hidden_layers,
        out_channels=out_channels,
        final_act=final_act,
        num_nodes=num_nodes,
    )
    output = model.forward(x, batch_size)
    if num_nodes is not None:
        assert output.shape == (batch_size, num_nodes, out_channels//num_nodes) if batch_size > 1 else (num_nodes, out_channels)
    else:
        assert output.shape == (batch_size, out_channels) if batch_size > 1 else (out_channels)
    # Test __call__ method
    model_out = torch_geometric.data.Data(x_0=x, batch_size=batch_size)
    result = model.__call__(model_out)
    assert "x_0" in result and "logits" in result
    assert torch.allclose(result["x_0"], result["logits"])

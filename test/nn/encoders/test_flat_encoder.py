"""Test FlatEncoder."""
import torch
import torch_geometric
from torch_geometric.data import Data
from topobench.nn.encoders.flat_encoder import FlatEncoder

def test_flat_encoder_forward():
    """Test the forward pass of the FlatEncoder."""
    batch_size = 2
    num_nodes = 3
    in_channels = 4
    out_channels = 6
    x = torch.randn(batch_size, num_nodes, in_channels)
    y = torch.randint(0, 2, (batch_size, num_nodes))
    data = Data(x=x, y=y)
    data.batch_size = batch_size
    encoder = FlatEncoder(in_channels=in_channels, out_channels=out_channels)
    out_data = encoder.forward(data)
    assert hasattr(out_data, "x_0")
    assert hasattr(out_data, "labels")
    assert out_data.x_0.shape[0] == batch_size
    assert out_data.x_0.ndim == 2
    assert torch.equal(out_data.labels, y)

if __name__ == "__main__":
    test_flat_encoder_forward()
    print("FlatEncoder test passed.")

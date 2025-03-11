"""Test DataloadDataset class."""

import torch
from torch_geometric.data import Data

from topobench.dataloader import DataloadDataset


class TestDataloadDataset:
    """Test DataloadDataset class."""
    def setup_method(self):
        """Set up."""
        self.data_list = [
            Data(x=torch.randn(4, 6), edge_index=torch.randint(0, 4, (2, 4))),
            Data(
                x=torch.randn(3, 6),
                edge_index=torch.randint(0, 3, (2, 3)),
                x_1=torch.randn(5, 6),
            ),
        ]
        self.dataset = DataloadDataset(self.data_list)

    def teardown(self):
        """Clean up."""
        del self.data_list
        del self.dataset

    def test_len(self):
        """Test len method."""
        assert len(self.dataset) == 2

    def test_get(self):
        """Test get method."""
        for i in range(len(self.data_list)):
            data, keys = self.dataset.get(i)

            expected_keys = list(self.data_list[i].keys())
            expected_data = [self.data_list[i][key] for key in expected_keys]

            assert keys == expected_keys

        for d, expected in zip(data, expected_data, strict=False):
            assert torch.equal(d, expected)

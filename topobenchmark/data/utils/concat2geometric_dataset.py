from torch_geometric.data import Data, Dataset


class ConcatToGeometricDataset(Dataset):
    def __init__(self, concat_dataset):
        super().__init__()
        self.concat_dataset = concat_dataset

    def len(self):
        return len(self.concat_dataset)

    def get(self, idx):
        data = self.concat_dataset[idx]

        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        y = data.y
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=1)
        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr.unsqueeze(dim=1)
        if len(y.shape) == 1:
            y = y.unsqueeze(dim=1)

        # Construct PyTorch Geometric Data object
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

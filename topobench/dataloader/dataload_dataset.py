"""Dataset class compatible with TBDataloader."""

import torch_geometric


class DataloadDataset(torch_geometric.data.Dataset):
    """Custom dataset to return all the values added to the dataset object.

    Parameters
    ----------
    data_lst : list[torch_geometric.data.Data]
        List of torch_geometric.data.Data objects.
    """

    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.data_lst)})"

    def get(self, idx):
        """Get data object from data list.

        Parameters
        ----------
        idx : int
            Index of the data object to get.

        Returns
        -------
        tuple
            Tuple containing a list of all the values for the data and the corresponding keys.
        """
        data = self.data_lst[idx]
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        """Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_lst)

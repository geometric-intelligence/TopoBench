"""CityNetwork dataset implementation."""

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import CityNetwork as PyGCityNetwork


class CityNetworkDataset(InMemoryDataset):
    """
    CityNetwork dataset from PyG for node classification.

    Supports four cities: paris, shanghai, la, london.
    Task is node classification based on eccentricity quantiles.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    name : str, default="paris"
        Name of the city. Must be one of ["paris", "shanghai", "la", "london"].
    augmented : bool, default=True
        Whether to use augmented version of the dataset.
    transform : callable, optional
        A function/transform that takes in a Data object and returns a transformed version.
    pre_transform : callable, optional
        A function/transform that takes in a Data object and returns a transformed version.
    """

    # Map the four cities
    NAMES = ["paris", "shanghai", "la", "london"]

    def __init__(
        self,
        root: str,
        name: str = "paris",
        augmented: bool = True,
        transform=None,
        pre_transform=None,
    ):
        self.name = name.lower()
        self.augmented = augmented
        if self.name not in self.NAMES:
            raise ValueError(f"Name must be one of {self.NAMES}")
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        Return list of raw file names.

        Returns
        -------
        list
            Empty list as PyG handles downloads automatically.
        """
        return []  # We let PyG handle download

    @property
    def processed_file_names(self):
        """
        Return list of processed file names.

        Returns
        -------
        list
            List containing the processed file name.
        """
        aug = "_aug" if self.augmented else ""
        return [f"citynetwork_{self.name}{aug}.pt"]

    def download(self):
        """Download the dataset."""
        # PyG will download automatically when we instantiate it in process()

    def process(self):
        """Load and process raw data and save it."""
        try:
            pyg_dataset = PyGCityNetwork(
                root=self.raw_dir, name=self.name, augmented=self.augmented
            )
            data = pyg_dataset[0]  # single graph

            # Ensure node labels are long (required by some losses)
            if hasattr(data, "y"):
                data.y = data.y.long()

            self.save([data], self.processed_paths[0])
        except Exception as e:
            raise RuntimeError(
                f"Failed to process CityNetwork dataset: {e}"
            ) from e

    def __repr__(self):
        """Return string representation."""
        return f"CityNetwork({self.name}, augmented={self.augmented})"

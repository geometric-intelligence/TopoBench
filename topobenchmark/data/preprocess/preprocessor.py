import json
import os

import torch_geometric

from modules.data.utils.utils import ensure_serializable, make_hash
from modules.transforms.data_transform import DataTransform


class PreProcessor(torch_geometric.data.InMemoryDataset):
    r"""Preprocessor for datasets.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the data.
    data_list : list
        List of data objects.
    transforms_config : DictConfig | dict
        Configuration parameters for the transforms.
    **kwargs: optional
        Additional arguments.
    """

    def __init__(self, data_list, transforms_config, data_dir, **kwargs):
        if isinstance(data_list, torch_geometric.data.Dataset):
            data_list = [data_list.get(idx) for idx in range(len(data_list))]
        elif isinstance(data_list, torch_geometric.data.Data):
            data_list = [data_list]
        self.data_list = data_list
        pre_transform = self.instantiate_pre_transform(
            data_dir, transforms_config
        )
        super().__init__(
            self.processed_data_dir, None, pre_transform, **kwargs
        )
        self.save_transform_parameters()
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        r"""Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return self.root

    @property
    def processed_file_names(self) -> str:
        r"""Return the name of the processed file.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        r"""Instantiate the pre-transforms.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.

        Returns
        -------
        torch_geometric.transforms.Compose
            Pre-transform object.
        """
        pre_transforms_dict = {
            key: DataTransform(**value)
            for key, value in transforms_config.items()
        }
        pre_transforms = torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
        )
        self.set_processed_data_dir(
            pre_transforms_dict, data_dir, transforms_config
        )
        return pre_transforms

    def set_processed_data_dir(
        self, pre_transforms_dict, data_dir, transforms_config
    ) -> None:
        r"""Set the processed data directory.

        Parameters
        ----------
        pre_transforms_dict : dict
            Dictionary containing the pre-transforms.
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.
        """
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            *[data_dir, repo_name, f"{params_hash}"]
        )

    def save_transform_parameters(self) -> None:
        r"""Save the transform parameters."""
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError(
                    "Different transform parameters for the same data_dir"
                )

            print(
                f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
            )

    def process(self) -> None:
        r"""Process the data."""
        self.data_list = [self.pre_transform(d) for d in self.data_list]

        self._data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])

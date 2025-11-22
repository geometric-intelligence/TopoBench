"""Preprocessor for datasets."""

import json
import os

import torch
import torch_geometric
from torch_geometric.io import fs

from topobench.data.utils import (
    ensure_serializable,
    load_inductive_splits,
    load_transductive_splits,
    make_hash,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class PreProcessor(torch_geometric.data.InMemoryDataset):
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : list
        List of data objects.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        self.dataset = dataset
        self._skip_processing = (
            False  # Flag to skip processing for no-transform case
        )

        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )
            super().__init__(
                self.processed_data_dir, None, pre_transform, **kwargs
            )
            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.save_transform_parameters()
            self.load(self.processed_paths[0])
            self.data_list = [data for data in self]
        else:
            print(
                "No transforms to apply, using dataset directly (skipping processing)..."
            )
            self.transforms_applied = False
            self._skip_processing = True  # Skip parent class processing

            # Call parent init but it should skip processing
            super().__init__(data_dir, None, None, **kwargs)

            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            # Check if dataset is an InMemoryDataset with _data and slices
            if hasattr(dataset, "_data") and hasattr(dataset, "slices"):
                # Directly use the dataset's data and slices
                self._data, self.slices = dataset._data, dataset.slices
                # Make data_list creation lazy to avoid loading large datasets into memory
                self._data_list = None
                self._is_inmemory = True
            else:
                # For non-InMemoryDataset (like LMDB-based datasets), we can't use _data/slices
                # The dataset will be accessed directly via indexing
                self._data = None
                self.slices = None
                self._data_list = None
                self._is_inmemory = False

        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

    @property
    def data_list(self):
        """Lazy loading of data_list to avoid loading large datasets into memory.

        Returns
        -------
        list
            List of data objects when transforms are not applied; otherwise the processed data list.
        """
        if not self.transforms_applied and self._data_list is None:
            # Only create data_list when actually needed
            print(
                "Warning: Creating data_list from large dataset - this may take a while..."
            )
            self._data_list = [data for data in self.dataset]
        return self._data_list

    @data_list.setter
    def data_list(self, value):
        """Setter for data_list.

        Parameters
        ----------
        value : list
            New list of data objects to use as the dataset's data_list.
        """
        self._data_list = value

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        if not self.transforms_applied and not self._is_inmemory:
            # For non-InMemoryDataset, delegate to the wrapped dataset
            return len(self.dataset)
        else:
            # For InMemoryDataset or transformed data, use parent implementation
            return super().__len__()

    def __getitem__(self, idx):
        """Get item at index.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve.

        Returns
        -------
        Data or list[Data]
            Data object(s) at the given index/slice.
        """
        if not self.transforms_applied and not self._is_inmemory:
            # For non-InMemoryDataset, delegate to the wrapped dataset
            return self.dataset[idx]
        else:
            # For InMemoryDataset or transformed data, use parent implementation
            return super().__getitem__(idx)

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        if not self.transforms_applied:
            return self.root
        else:
            return self.root + "/processed"

    @property
    def processed_file_names(self) -> str | list[str]:
        """Return the name of the processed file.

        Returns
        -------
        str | list[str]
            Name of the processed file, or empty list to skip processing.
        """
        # If no transforms, return empty list to skip processing check
        if hasattr(self, "_skip_processing") and self._skip_processing:
            return []
        return "data.pt"

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        """Instantiate the pre-transforms.

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
        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings
        # Check if this is a single transform config (has transform_name key)
        # or multiple transforms config (each value is a dict with transform_name)
        if "transform_name" in transforms_config:
            # Single transform configuration
            pre_transforms_dict = {
                transforms_config.transform_name: DataTransform(
                    **transforms_config
                )
            }
        else:
            # Multiple transforms configuration
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
        """Set the processed data directory.

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

        # Include dataset size in hash to avoid reusing cached data from different dataset sizes
        # This is crucial when max_samples changes
        transforms_parameters["_dataset_size"] = len(self.dataset)

        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            *[data_dir, repo_name, f"{params_hash}"]
        )

    def save_transform_parameters(self) -> None:
        """Save the transform parameters."""
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
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
        """Method that processes the data.

        Returns
        -------
        None
            Writes processed data to disk as a side effect.
        """
        # Skip processing if no transforms are applied
        if hasattr(self, "_skip_processing") and self._skip_processing:
            return

        from tqdm import tqdm

        # If dataset has split_idx, only process those samples (for efficiency with large datasets)
        if hasattr(self.dataset, "split_idx") and self.dataset.split_idx:
            # Collect all unique indices from all splits
            all_indices = []
            for split_name in ["train", "valid", "test"]:
                if split_name in self.dataset.split_idx:
                    indices = self.dataset.split_idx[split_name]
                    # Convert tensor to list if needed
                    if hasattr(indices, "tolist"):
                        indices = indices.tolist()
                    elif hasattr(indices, "__iter__"):
                        indices = list(indices)
                    all_indices.extend(indices)
            # Remove duplicates and sort
            all_indices = sorted(set(all_indices))

            print(
                f"Processing dataset with {len(all_indices)} samples (from split_idx)..."
            )

            # Load only the samples specified in split_idx
            if len(all_indices) > 1000:
                print(
                    f"Loading {len(all_indices)} graphs from split indices (this may take a while)..."
                )
                data_list = [
                    self.dataset[idx]
                    for idx in tqdm(all_indices, desc="Loading graphs")
                ]
            else:
                data_list = [self.dataset[idx] for idx in all_indices]
        else:
            # No split_idx, process all samples
            print(f"Processing dataset with {len(self.dataset)} samples...")

            if isinstance(
                self.dataset,
                (torch_geometric.data.Dataset, torch.utils.data.Dataset),
            ):
                # Use tqdm to show progress for large datasets
                if len(self.dataset) > 1000:
                    print(
                        f"Loading {len(self.dataset)} graphs (this may take a while)..."
                    )
                    data_list = [
                        data
                        for data in tqdm(self.dataset, desc="Loading graphs")
                    ]
                else:
                    data_list = [data for data in self.dataset]
            elif isinstance(self.dataset, torch_geometric.data.Data):
                data_list = [self.dataset]

        if self.pre_transform is not None:
            print(f"Applying transforms to {len(data_list)} graphs...")
            transformed_data_list = [
                self.pre_transform(d)
                for d in tqdm(data_list, desc="Applying transforms")
            ]
        else:
            transformed_data_list = data_list

        print("Collating data...")
        self._data, self.slices = self.collate(transformed_data_list)

        # If we processed only samples from split_idx, remap split_idx to new indices
        if hasattr(self.dataset, "split_idx") and self.dataset.split_idx:
            print("Remapping split_idx to new indices after processing...")
            # Create mapping from old indices to new indices
            old_to_new = {
                old_idx: new_idx for new_idx, old_idx in enumerate(all_indices)
            }

            # Remap split_idx
            new_split_idx = {}
            for split_name in ["train", "valid", "test"]:
                if split_name in self.dataset.split_idx:
                    old_indices = self.dataset.split_idx[split_name]
                    # Convert tensor to list if needed
                    if hasattr(old_indices, "tolist"):
                        old_indices = old_indices.tolist()
                    elif hasattr(old_indices, "__iter__"):
                        old_indices = list(old_indices)
                    else:
                        old_indices = [old_indices]

                    # Map old indices to new indices
                    new_indices = [
                        old_to_new[idx]
                        for idx in old_indices
                        if idx in old_to_new
                    ]
                    new_split_idx[split_name] = new_indices

            # Store the remapped split_idx on the dataset itself
            self.dataset.split_idx = new_split_idx
            print(
                f"Remapped split_idx: train={len(new_split_idx.get('train', []))}, "
                f"valid={len(new_split_idx.get('valid', []))}, test={len(new_split_idx.get('test', []))}"
            )

        assert isinstance(self._data, torch_geometric.data.Data)
        # Guard against empty processed_paths
        if self.processed_paths and len(self.processed_paths) > 0:
            print(f"Saving processed data to {self.processed_paths[0]}...")
            self.save(transformed_data_list, self.processed_paths[0])
        else:
            print("Warning: No processed paths available, skipping save.")

        # Reset cache after saving
        self._data_list = None

    def load(self, path: str) -> None:
        r"""Load the dataset from the file path `path`.

        Parameters
        ----------
        path : str
            The path to the processed data.
        """
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) >= 2 and len(out) <= 4
        if len(out) == 2:  # Backward compatibility (1).
            data, self.slices = out
        elif len(out) == 3:  # Backward compatibility (2).
            data, self.slices, data_cls = out
        else:  # TU Datasets store additional element (__class__) in the processed file
            data, self.slices, sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def load_dataset_splits(
        self, split_params
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load the dataset splits.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test datasets.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting == "inductive":
            return load_inductive_splits(self, split_params)
        elif split_params.learning_setting == "transductive":
            return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting.\
                Please define either 'inductive' or 'transductive'."
            )

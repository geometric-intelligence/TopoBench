"""
Data loader for the Bowen et al. mouse auditory cortex calcium imaging dataset.

This script downloads and processes the original dataset introduced in:

[Citation] Bowen et al. (2024), "Fractured columnar small-world functional network
organization in volumes of L2/3 of mouse auditory cortex," PNAS Nexus, 3(2): pgae074.
https://doi.org/10.1093/pnasnexus/pgae074

We apply the preprocessing and graph-construction steps defined in this module to obtain
a representation of neuronal activity suitable for our experiments.

Please cite the original paper when using this dataset or any derivatives.
"""

import os.path as osp

import torch
from omegaconf import DictConfig
from torch_geometric.io import fs

from topobench.data.datasets.a123 import A123CortexMDataset
from topobench.data.loaders.base import AbstractLoader


class A123DatasetLoader(AbstractLoader):
    """Loader for A123 mouse auditory cortex dataset.

    Implements the AbstractLoader interface: accepts a DictConfig `parameters`
    and implements `load_dataset()` which returns a dataset object.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters for the dataset.
    **overrides
        Additional keyword arguments to override parameters.
    """

    def __init__(self, parameters: DictConfig, **overrides):
        """Initialize the A123 dataset loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration parameters for the dataset.
        **overrides
            Additional keyword arguments to override parameters.
        """
        # Initialize AbstractLoader (sets self.parameters and self.root_data_dir)
        super().__init__(parameters)

        # hyperparameters can come from the DictConfig or be passed as overrides
        params = parameters if parameters is not None else {}

        def _get(k, default):
            """Get parameter value from DictConfig or overrides.

            Parameters
            ----------
            k : str
                Parameter key.
            default : Any
                Default value if key not found.

            Returns
            -------
            Any
                Parameter value from DictConfig or overrides, or default.
            """
            try:
                return params.get(k, overrides.get(k, default))
            except Exception:
                # DictConfig may use attribute access
                return getattr(params, k, overrides.get(k, default))

        self.batch_size = int(_get("batch_size", 32))
        # dataset will be created when load_dataset() is called
        self.dataset = None

    def load_dataset(
        self,
    ) -> torch.utils.data.Dataset:
        """Instantiate and return the underlying dataset.

        Returns a `A123CortexMDataset` instance constructed from the loader's
        parameters and root data directory.

        Returns
        -------
        torch.utils.data.Dataset
            A123CortexMDataset instance or triangle dataset.
        """
        # determine dataset name from parameters, fallback to expected id
        name = self.parameters.data_name
        task_type = str(
            getattr(self.parameters, "specific_task", "classification")
        )

        # root path for dataset: use the parent of root_data_dir since the dataset
        # constructs its own subdirectory based on name
        root = str(self.root_data_dir.parent)

        # Construct dataset; A123CortexMDataset expects (root, name, parameters)
        self.dataset = A123CortexMDataset(
            root=root, name=name, parameters=self.parameters
        )

        # If triangle task requested, load triangle dataset instead
        if task_type == "triangle_classification":
            # Load triangle classification dataset
            processed_dir = self.dataset.processed_dir
            triangle_data_path = osp.join(processed_dir, "data_triangles.pt")

            if osp.exists(triangle_data_path):
                # Load triangle data
                out = fs.torch_load(triangle_data_path)
                assert len(out) == 4
                data, slices, sizes, data_cls = out

                if not isinstance(data, dict):
                    self.dataset.data = data
                else:
                    self.dataset.data = data_cls.from_dict(data)

                self.dataset.slices = slices
                print(
                    "[A123 Loader] Loaded triangle classification task dataset"
                )
            else:
                print(
                    f"[A123 Loader] Warning: Triangle dataset not found at {triangle_data_path}. "
                    f"Ensure triangle_task.enabled=true in config and dataset has been processed."
                )

        # Triangle common-neighbours task
        if task_type == "triangle_common_neighbors":
            processed_dir = self.dataset.processed_dir
            triangle_cn_path = osp.join(
                processed_dir, "data_triangles_common_neighbors.pt"
            )

            if osp.exists(triangle_cn_path):
                out = fs.torch_load(triangle_cn_path)
                assert len(out) == 4
                data, slices, sizes, data_cls = out

                if not isinstance(data, dict):
                    self.dataset.data = data
                else:
                    self.dataset.data = data_cls.from_dict(data)

                self.dataset.slices = slices
                print(
                    "[A123 Loader] Loaded triangle common-neighbours task dataset"
                )
            else:
                print(
                    f"[A123 Loader] Warning: Triangle CN dataset not found at {triangle_cn_path}. "
                    f"Ensure triangle_common_task.enabled=true in config and dataset has been processed."
                )

        return self.dataset

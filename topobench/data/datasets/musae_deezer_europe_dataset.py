"""Dataset class for MUSAE Deezer Europe dataset."""

import json
import os
import os.path as osp
import shutil
from typing import ClassVar

import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix

from topobench.data.utils import (
    download_file_from_link,
)


class MusaeDeezerEuropeDataset(InMemoryDataset):
    r"""Dataset class for MUSAE Deezer Europe dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {
        "musae_deezer_europe": "https://snap.stanford.edu/data/deezer_europe.zip",
    }

    FILE_FORMAT: ClassVar = {
        "musae_deezer_europe": "zip",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.raw_name = "deezer_europe"
        self.parameters = parameters
        self.graph_representation = parameters.graph_representation
        self.feature_matrix_type = parameters.features.matrix_type
        self.apply_dim_reduction = parameters.features.dim_reduction.apply
        if self.apply_dim_reduction:
            self.reduced_dim = parameters.features.dim_reduction.reduced_dim
            self.svd_iter = parameters.features.dim_reduction.svd_iter
        super().__init__(
            root,
        )

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, " \
               f"self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        if self.apply_dim_reduction:
            feat_dim_opts = "_".join([
                "reducedFeatDim" + str(self.reduced_dim),
                "SVDnIter" + str(self.svd_iter)
            ])
        else:
            feat_dim_opts = "allFeat"
        self.processed_root = osp.join(
            self.root,
            self.name,
            "_".join([self.feature_matrix_type, feat_dim_opts]),
        )
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return ["deezer_europe_edges.csv", "deezer_europe_features.json", "deezer_europe_target.csv"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Download data from the source
        self.url = self.URLS[self.name]
        self.file_format = self.FILE_FORMAT[self.name]
        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.raw_name,
            file_format=self.file_format,
        )

        # Extract zip file
        folder = self.raw_dir
        filename = f"{self.raw_name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        # Delete zip file
        os.unlink(path)

        # Move files from osp.join(folder, name_download) to folder
        for file in os.listdir(osp.join(folder, self.raw_name)):
            shutil.move(osp.join(folder, self.raw_name, file), folder)
        # Delete osp.join(folder, self.name) dir
        shutil.rmtree(osp.join(folder, self.raw_name))
    
    def _reduce_feature_dimensionality(self, x) -> np.ndarray:
        """Reduce dimension using Truncated SVD.
        
        Parameters
        ----------
        x : np.ndarray, Scipy COO or torch.Tensor
            Wide feature matix
        
        Returns
        -------
        np.ndarray
            Reduced feature matrix.
        """
        svd = TruncatedSVD(
            n_components = self.reduced_dim,
            n_iter = self.svd_iter,
            random_state = 42,#svd_seed,
        )
        x = svd.fit_transform(x)#.astype(np.float32)
        return x

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the US county demographics data, applies any pre-
        processing transformations if specified, and saves the processed data
        to the appropriate location.
        """
        # Step 1: Load raw data files
        folder = self.raw_dir
        # Edges:
        tmp = pd.read_csv(osp.join(folder, "deezer_europe_edges.csv"))[["node_1","node_2"]].to_numpy()
        edge_index = torch.tensor(tmp, dtype=torch.long).t().contiguous()
        # Targets:
        tmp = pd.read_csv(osp.join(folder,"deezer_europe_target.csv")).sort_values("id")["target"].to_numpy()
        y = torch.tensor(pd.Categorical(tmp, set(tmp)).codes, dtype=torch.long)
        # Node features:
        with open(osp.join(folder,"deezer_europe_features.json")) as infile:
            featdict = json.load(infile)

        # Build feature matrix type
        if self.feature_matrix_type.lower() == "dense":
            unique_values_set = set()
            for values_list in featdict.values():
                unique_values_set.update(values_list)
            num_features = len(list(unique_values_set))
            node_features = []
            for ind in range(len(list(featdict.keys()))):
                # Convert features to one-hot encoded vector
                one_hot_features = [1 if i in featdict[str(ind)] else 0 for i in range(num_features)]
                node_features.append(one_hot_features)
            x = torch.tensor(node_features, dtype=torch.float)
        else:     # Default to sparse
            row = []
            col = []
            values = []
            for node_id_str, feature_list in featdict.items():
                node_id = int(node_id_str)
                for feature_id in feature_list:
                    row.append(node_id)
                    col.append(int(feature_id))
                    values.append(1)
            row = np.array(row, dtype=int)
            col = np.array(col, dtype=int)
            values = np.array(values, dtype=int)
            node_count = row.max() + 1
            feature_count = col.max() + 1
            shape = (node_count, feature_count)
            x = coo_matrix((values, (row, col)), shape)

        # Reduce feature matrix dimension
        if self.apply_dim_reduction:
            x = self._reduce_feature_dimensionality(x)
            x = torch.from_numpy(x)

        # PyG or NetworkX graph representation
        #assert x.shape[1] == 64, "Non matching sizes"
        data = Data(x=x, y=y, edge_index=edge_index)
        data_list = [data]

        # Step 2: collate the graphs
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.

        # Step 3: save processed data
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

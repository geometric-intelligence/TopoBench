"""Dataset class for Chordonomicon dataset."""

import ast
import os
import os.path as osp

import numpy as np
import pandas as pd
import requests
import torch
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs


class ChordonomiconDataset(InMemoryDataset):
    """Dataset class for Chordonomicon dataset.

    Parameters
    ----------
    data_dir : str
        Directory where the dataset will be stored, raw
        and processed will be subdirectories.
    data_name : str
        Name of the dataset (e.g., 'Chordonomicon').
    """

    URL = "https://huggingface.co/datasets/PierrickLeKing/topobench-music-synergy/resolve/main/dataframe.zip"  # pylint: disable=line-too-long
    RAW_FILE_NAMES = ["dataframe.csv"]

    def __init__(self, data_dir, data_name):
        self.name = data_name
        self.data_dir = data_dir
        self.folder_chordonomicon = osp.join(self.data_dir, self.name)
        self.root = osp.join(data_dir, data_name)
        super().__init__(self.root)

    def download(self) -> None:
        """Download the Chordonomicon dataset.

        Raises:
            requests.exceptions.HTTPError: If the download fails.
        """
        print("Downloading...")
        r = requests.get(self.URL, timeout=30)
        r.raise_for_status()
        with open(
            osp.join(self.folder_chordonomicon, "dataframe.zip"), "wb"
        ) as f:
            f.write(r.content)
        extract_zip(
            osp.join(self.folder_chordonomicon, "dataframe.zip"),
            osp.join(self.folder_chordonomicon, "raw"),
        )
        os.unlink(osp.join(self.folder_chordonomicon, "dataframe.zip"))

    def process(self) -> None:
        """Handle the Chordonomicon dataset.

        Convert the raw data into a PyTorch Geometric Data object and save it.
        """
        df = pd.read_csv(
            osp.join(self.folder_chordonomicon, "raw", self.RAW_FILE_NAMES[0])
        )
        df["chords"] = (
            df["chords"].apply(ast.literal_eval).apply(list).apply(np.array)
        )
        t1 = torch.from_numpy(np.concatenate(df["chords"].values))
        t2 = torch.tensor(df["chords"].apply(len).values)
        indices = torch.stack(
            (t1, torch.repeat_interleave(torch.arange(len(t2)), t2))
        )
        incidence_hyperedges = torch.sparse_coo_tensor(
            indices, torch.ones(indices.shape[1])
        ).coalesce()
        x_hyperedges = torch.tensor(df["frequency"].values).unsqueeze(1)
        y_hyperedges = torch.tensor(df["local_o_info"].values)
        data = Data(
            incidence_hyperedges=incidence_hyperedges,
            num_hyperedges=incidence_hyperedges.size(1),
            x_hyperedges=x_hyperedges,
            y_hyperedges=y_hyperedges,
        )
        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        fs.torch_save(
            (
                self._data.to_dict(),
                self.slices,
                {},
                self._data.__class__,
            ),
            self.processed_paths[0],
        )

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return self.RAW_FILE_NAMES

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

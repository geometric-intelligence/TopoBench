from pandas.api.types import is_integer_dtype

import os

import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data

from .repository.zenodo import ZenodoZip

# Optional: if you want to parse CSVs
import pandas as pd

ZENODO_RECORD_ID = "16895532"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/"

class GraphlandDataset(InMemoryDataset):
    """
    Example InMemoryDataset that:
      - (Optionally) downloads and extracts a ZIP (e.g., from Zenodo) into raw_dir
      - Reads nodes.csv (features + labels) and edges.csv (src, dst)
      - Builds a single large graph (Data) and stores it as processed <split>.pt
      - Supports pre_filter and pre_transform hooks

    Directory layout (after download/extract):
      root/
        raw/
          nodes.csv
          edges.csv
          READY  (sentinel file created after successful download/extract)
        processed/
          full.pt  (or <split>.pt if you change `split`)
    """

    def __init__(
        self,
        root: str | os.PathLike,
        name: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.zip_url = f"{ZENODO_BASE}{name}.zip?"
        self.name = name
        super().__init__(os.path.join(root, name), transform, pre_transform, pre_filter)

        # After super().__init__, processed file must exist. Load it:
        self.data, self.slices = torch.load(os.path.join(self.processed_paths[0], self.processed_file_names))


    def download(self) -> None:
        """
        If `raw_file_names` are missing, PyG calls this. 
        """
        downloader = ZenodoZip(
            url = self.zip_url
        )
        data = downloader.fetch()

        os.makedirs(self.raw_dir, exist_ok=True)

        for file_path, binary_content in data.items():
            # taking only the filename
            filename = file_path.split('/')[-1]
            complete_file_path = os.path.join(self.raw_dir, filename)
            with open(complete_file_path, "wb") as f:   # use wb since values are binary
                f.write(binary_content)

        

    # ---------- Building the graph(s) ----------

    def process(self) -> None:
        """
        Create and save processed tensors. For InMemoryDataset
        """
        # Reading 'csv's
        edges_df = pd.read_csv(os.path.join(self.raw_dir, 'edgelist.csv'))
        feats_df = pd.read_csv(os.path.join(self.raw_dir, 'features.csv'), index_col='node_id')
        targs_df = pd.read_csv(os.path.join(self.raw_dir, 'targets.csv'))

        # creating the edge indexes
        src = edges_df['source'].to_numpy()
        dst = edges_df['target'].to_numpy()
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # creating X tensor 
        x = torch.tensor(feats_df.values, dtype=torch.float)

        # y: node-level o graph-level
        if 'node_id' in targs_df.columns:
            # node-level
            targs_df = targs_df.set_index('node_id')
            targ_values = targs_df.squeeze().fillna(0)
            # inferring data type (NaN cannot be integer)
            if is_integer_dtype(targ_values) or targ_values.apply(float.is_integer).all():
                y = torch.tensor(targs_df.values, dtype=torch.long).squeeze() # classification
            else: 
                y = torch.tensor(targs_df.values, dtype=torch.double).squeeze() # regression
        else:
            # graph-level 
            ValueError("Not implemented for graph level task")

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_filter is not None and not self.pre_filter(data):
            data_list = []
        else:
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list = [data]

        # collate
        data_big, slices = self.collate(data_list)

        # creating the folder
        os.makedirs(self.processed_paths[0], exist_ok=True)

        #saving
        torch.save((data_big, slices),  os.path.join(self.processed_paths[0], self.processed_file_names))


    # ---------- Required properties ----------

    @property
    def raw_file_names(self):
        """
        Files that must be present in raw_dir for "download() not needed".
        If you don't know them upfront (e.g., a ZIP with many files), use a
        sentinel. We'll create 'READY' after extracting to signal completeness.
        """
        return ["edgelist.csv", "features.csv", "targets.csv"]  # sentinel; created by download() or by you manually

    @property
    def processed_paths(self):
        """The processed path to avoid processing."""
        return [ os.path.join(self.root, "processed" ) ]
    
    @property
    def processed_file_names(self):
        """The processed file produced by `process()`."""
        return "data.pt"
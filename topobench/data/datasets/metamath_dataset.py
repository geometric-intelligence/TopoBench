# metamath_dataset.py

import os
import os.path as osp
from typing import ClassVar

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobench.data.utils import download_file_from_link


class MetamathDataset(InMemoryDataset):
    """
    Metamath proof graph dataset backed by a precomputed data.pt file.

    The Hugging Face data.pt is expected to have the form:

        {
            "data": data,          # PyG Data object from collate(...)
            "slices": slices,      # slices dict from collate(...)
            "train_idx": ...,      # 1D indices of train graphs
            "val_idx": ...,        # 1D indices of val graphs
            "test_idx": ...,       # 1D indices of test graphs
        }

    This class simply:
      - downloads data.pt from HF (into raw_dir),
      - copies it into processed_dir,
      - loads it and exposes:
          * self.data, self.slices
          * self.split_idx = {"train", "valid", "test"}
    """

    HF_BASE: ClassVar[str] = "https://huggingface.co/datasets"
    HF_REPO: ClassVar[str] = "jableable/metamath-proof-graphs"
    HF_FILENAME: ClassVar[str] = "data.pt"

    def __init__(self, root: str, name: str, parameters) -> None:
        self.name = name
        self.parameters = parameters

        super().__init__(root)

        out = fs.torch_load(self.processed_paths[0])

        if not isinstance(out, dict):
            raise TypeError(
                f"Expected dict in {self.processed_paths[0]}, got {type(out)}"
            )

        data = out["data"]
        self.slices = out["slices"]

        # Rebuild Data from dict if needed
        if isinstance(data, dict):
            data = Data.from_dict(data)

        self.data = data

        # Expose fixed splits for TopoBench
        train_idx = out.get("train_idx", None)
        val_idx = out.get("val_idx", None)
        test_idx = out.get("test_idx", None)

        if (
            train_idx is not None
            and val_idx is not None
            and test_idx is not None
        ):
            # Convert to numpy arrays for split_utils
            if isinstance(train_idx, torch.Tensor):
                train_idx = train_idx.cpu().numpy()
                val_idx = val_idx.cpu().numpy()
                test_idx = test_idx.cpu().numpy()

            self.split_idx = {
                "train": np.array(train_idx, dtype=int),
                "valid": np.array(val_idx, dtype=int),
                "test": np.array(test_idx, dtype=int),
            }

    # -------------------------------------------------------------------------
    # Directory layout
    # -------------------------------------------------------------------------

    @property
    def raw_dir(self) -> str:
        # <root>/<name>/raw
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        # <root>/<name>/processed
        return osp.join(self.root, self.name, "processed")

    # -------------------------------------------------------------------------
    # File naming
    # -------------------------------------------------------------------------

    @property
    def raw_file_names(self) -> list[str]:
        # We only expect a single raw artifact: data.pt
        return [self.HF_FILENAME]

    @property
    def processed_file_names(self) -> str:
        # Single processed file, also called data.pt
        return "data.pt"

    # -------------------------------------------------------------------------
    # Download from Hugging Face
    # -------------------------------------------------------------------------

    def download(self) -> None:
        """
        Download data.pt from Hugging Face into raw_dir.

        Expected HF layout:
          https://huggingface.co/datasets/jableable/metamath-proof-graphs/resolve/main/data/data.pt
        """
        os.makedirs(self.raw_dir, exist_ok=True)

        url = f"{self.HF_BASE}/{self.HF_REPO}/resolve/main/data/{self.HF_FILENAME}"
        dataset_name, file_format = os.path.splitext(self.HF_FILENAME)
        file_format = file_format.lstrip(".")

        download_file_from_link(
            file_link=url,
            path_to_save=self.raw_dir,
            dataset_name=dataset_name,
            file_format=file_format,
        )

    # -------------------------------------------------------------------------
    # Process: copy / normalize the HF data.pt to processed_dir
    # -------------------------------------------------------------------------

    def process(self) -> None:
        """Load raw data.pt, fix dtypes, and save processed data.pt as a dict."""
        raw_pt = osp.join(self.raw_dir, "data.pt")
        obj = torch.load(raw_pt, weights_only=False)

        raw_data = obj["data"]
        raw_slices = obj["slices"]
        train_idx = obj["train_idx"]
        val_idx = obj["val_idx"]
        test_idx = obj["test_idx"]

        # Temporary dataset to reconstruct individual graphs
        class _Tmp(InMemoryDataset):
            def __init__(self, data, slices):
                super().__init__(".")
                self.data = data
                self.slices = slices

            def _download(self):
                pass

            def _process(self):
                pass

        tmp = _Tmp(raw_data, raw_slices)

        graphs = []
        for i in range(len(tmp)):
            g = tmp[i]

            # 🔧 Critical fix: ensure edge_index is integer
            if hasattr(g, "edge_index"):
                g.edge_index = g.edge_index.long()

            graphs.append(g)

        # Re-collate into a clean storage
        data_fixed, slices_fixed = tmp.collate(graphs)

        out = {
            "data": data_fixed,
            "slices": slices_fixed,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

        fs.torch_save(out, self.processed_paths[0])

    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MetamathDataset(root={self.root}, name={self.name}, "
            f"num_graphs={len(self)}, "
            f"has_split_idx={'split_idx' in self.__dict__})"
        )

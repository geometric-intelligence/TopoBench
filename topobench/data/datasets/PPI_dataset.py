# topobench/data/datasets/PPI_dataset.py

import json
import os
import os.path as osp
from collections.abc import Callable
from itertools import product

import numpy as np
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import remove_self_loops


class PPI(InMemoryDataset):
    r"""Protein-Protein Interaction (PPI) dataset.

    This is a local copy of the PyTorch Geometric PPI dataset implementation,
    with a small compatibility fix for newer versions of NetworkX
    (we do *not* pass ``edges="links"`` to ``node_link_graph``).

    The dataset contains positional gene sets, motif gene sets and
    immunological signatures as features (50 in total) and gene ontology
    sets as labels (121 in total).

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    split : {"train", "val", "test"}, optional
        Which split to load. Default is ``"train"``.
    transform : callable, optional
        Transform applied on each access.
    pre_transform : callable, optional
        Transform applied before saving to disk.
    pre_filter : callable, optional
        Filter deciding which graphs to keep.
    force_reload : bool, optional
        Whether to re-process the dataset. Default is ``False``.
    """

    url = "https://data.dgl.ai/dataset/ppi.zip"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
    ) -> None:
        assert split in ["train", "val", "test"]

        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        if split == "train":
            self.load(self.processed_paths[0])
        elif split == "val":
            self.load(self.processed_paths[1])
        elif split == "test":
            self.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> list[str]:
        splits = ["train", "valid", "test"]
        files = ["feats.npy", "graph_id.npy", "graph.json", "labels.npy"]
        return [f"{split}_{name}" for split, name in product(splits, files)]

    @property
    def processed_file_names(self) -> list[str]:
        return ["train.pt", "val.pt", "test.pt"]

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import networkx as nx
        from networkx.readwrite import json_graph

        for s, split in enumerate(["train", "valid", "test"]):
            path = osp.join(self.raw_dir, f"{split}_graph.json")
            with open(path) as f:
                # Do NOT pass edges="links" (not supported in NetworkX >= 3)
                G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

            x = np.load(osp.join(self.raw_dir, f"{split}_feats.npy"))
            x = torch.from_numpy(x).to(torch.float)

            y = np.load(osp.join(self.raw_dir, f"{split}_labels.npy"))
            y = torch.from_numpy(y).to(torch.float)

            data_list = []
            path = osp.join(self.raw_dir, f"{split}_graph_id.npy")
            idx = torch.from_numpy(np.load(path)).to(torch.long)
            idx = idx - idx.min()

            for i in range(int(idx.max()) + 1):
                mask = idx == i

                G_s = G.subgraph(
                    mask.nonzero(as_tuple=False).view(-1).tolist()
                )
                edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
                edge_index = edge_index - edge_index.min()
                edge_index, _ = remove_self_loops(edge_index)

                data = Data(edge_index=edge_index, x=x[mask], y=y[mask])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            self.save(data_list, self.processed_paths[s])

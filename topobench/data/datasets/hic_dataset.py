import itertools
import os.path as osp

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected


class HICDataset(InMemoryDataset):
    """
    PyG loader for HIC hypergraph/graph classification datasets.
    Builds:
      - x: one-hot vertex features from vertex labels (dataset-wide vocabulary)
      - y: int class (or multi-hot for multi-label datasets)
      - edge_index: clique-expanded edges for hypergraphs; raw edges for graphs
      - hyperedge_index (optional, for hypergraph datasets): [2, num_incidence] (node, hyperedge_id)
      - num_hyperedges (optional): int
    """
    base_url = "https://raw.githubusercontent.com/iMoonLab/HIC/main/data/hypergraph"

    folder_map = {
        "RHG_3": "RHG",
        "RHG_10": "RHG",
        "RHG_table": "RHG",
        "RHG_pyramid": "RHG",
        "IMDB_dir_form": "IMDB",
        "IMDB_dir_genre": "IMDB",
        "IMDB_wri_form": "IMDB",
        "IMDB_wri_genre": "IMDB",
        "IMDB_dir_genre_m": "IMDB",
        "IMDB_wri_genre_m": "IMDB",
        "stream_player": "STEAM",
        "twitter_friend": "TWITTER",
        # Also present in the repo (graph classification):
        "RG_macro": "RG",
        "RG_sub": "RG",
        "MUTAG": "MUTAG",
        "NCI1": "NCI1",
        "PROTEINS": "PROTEINS",
        "IMDBMULTI": "IMDBMULTI",
        "IMDBBINARY": "IMDBBINARY",
    }

    # Which datasets are multi-label (graph label is multi-hot)
    multi_label_names = {"IMDB_dir_genre_m", "IMDB_wri_genre_m"}

    # Which datasets are *hypergraphs* (others are simple graphs)
    hypergraph_names = {
        "RHG_3", "RHG_10", "RHG_table", "RHG_pyramid",
        "IMDB_dir_form", "IMDB_dir_genre", "IMDB_wri_form", "IMDB_wri_genre",
        "IMDB_dir_genre_m", "IMDB_wri_genre_m",
        "stream_player", "twitter_friend",
    }

    def __init__(self, root: str, name: str, use_degree_as_tag: bool=False,
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.use_degree_as_tag = use_degree_as_tag
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.name}.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def download(self):
        folder = self.folder_map.get(self.name)
        if folder is None:
            raise ValueError(f"Dataset '{self.name}' not recognised.")
        url = f"{self.base_url}/{folder}/{self.name}.txt"
        download_url(url, self.raw_dir)

    @staticmethod
    def _parse_vertex_labels(line: str) -> list[list[int] | int]:
        # Split by space to get per-vertex “label tokens”.
        # Each token is either "a" or "a/b/c".
        toks = line.strip().split()
        out = []
        for t in toks:
            if "/" in t:
                out.append([int(x) for x in t.split("/") if x != ""])
            else:
                out.append(int(t))
        return out

    @staticmethod
    def _clique_expand_edges(hyperedges: list[list[int]], num_v: int) -> torch.Tensor:
        # Build 2-combinations per hyperedge, both directions; dedup with to_undirected
        ei = []
        for hed in hyperedges:
            if len(hed) < 2:
                continue
            for u, v in itertools.combinations(hed, 2):
                ei.append([u, v])
                ei.append([v, u])
        if len(ei) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index = torch.tensor(ei, dtype=torch.long).t()
        return to_undirected(edge_index, num_nodes=num_v)

    @staticmethod
    def _build_incidence(hyperedges: list[list[int]], num_v: int) -> tuple[torch.Tensor, int]:
        # Returns hyperedge_index: [2, num_incidence] where row0=node, row1=hyperedge_id
        rows = []
        cols = []
        for e_id, hed in enumerate(hyperedges):
            for v in hed:
                rows.append(v)
                cols.append(e_id)
        if not rows:
            return torch.empty((2, 0), dtype=torch.long), 0
        he_index = torch.tensor([rows, cols], dtype=torch.long)
        return he_index, len(hyperedges)

    def _read_all_graphs(self, path: str):
        """First pass: read raw structure into memory (no remapping yet)."""
        with open(path) as f:
            n_graphs = int(f.readline().strip())

            entries = []  # temp storage per graph
            v_label_universe = set()
            g_label_universe = set()

            for _ in range(n_graphs):
                # line: num_v num_e g_lbl...
                parts = f.readline().strip().split()
                num_v, num_e = int(parts[0]), int(parts[1])
                g_lbl_raw = [int(x) for x in parts[2:]]  # could be multi-label

                # vertex labels (might be multi-label per-vertex)
                v_lbl_raw = self._parse_vertex_labels(f.readline())

                # edges/hyperedges
                hyperedges = []
                for _ in range(num_e):
                    row = [int(x) for x in f.readline().strip().split()]
                    hyperedges.append(row)

                # collect universes for remapping
                # (vertex labels can be list[int] or int)
                for lab in v_lbl_raw:
                    if isinstance(lab, list):
                        v_label_universe.update(lab)
                    else:
                        v_label_universe.add(lab)
                g_label_universe.update(g_lbl_raw)

                entries.append({
                    "num_v": num_v,
                    "num_e": num_e,
                    "g_lbl": g_lbl_raw,
                    "v_lbl": v_lbl_raw,
                    "hyperedges": hyperedges,
                })

        return entries, v_label_universe, g_label_universe

    def process(self):
        raw_path = self.raw_paths[0]

        # First pass: parse everything and build vocabularies (mirrors HIC utils.py).
        entries, v_universe, g_universe = self._read_all_graphs(raw_path)

        # Dataset-wide remaps (stable, sorted)
        v_map = {lab: i for i, lab in enumerate(sorted(v_universe))}
        g_map = {lab: i for i, lab in enumerate(sorted(g_universe))}
        ft_dim = len(v_map)
        is_hyper = self.name in self.hypergraph_names
        is_multi = self.name in self.multi_label_names

        data_list: list[Data] = []

        for e in entries:
            num_v = e["num_v"]
            v_lbl = e["v_lbl"]
            g_lbl = e["g_lbl"]
            hedges = e["hyperedges"]

            if self.use_degree_as_tag:
                # degree in hypergraph sense: count memberships across hyperedges
                deg = [0] * num_v
                for hed in hedges:
                    for v in hed:
                        deg[v] += 1
                v_lbl = [int(d) for d in deg]  # single label per vertex

                # expand v_map / ft_dim if new degrees appear (rare if used consistently)
                # In practice, we freeze ft_dim to dataset-wide initial map like HIC.
                # Any unseen label will KeyError; guard:
                for d in v_lbl:
                    if d not in v_map:
                        v_map[d] = len(v_map)
                ft_dim = len(v_map)

            # One-hot x (dataset-wide)
            x = torch.zeros((num_v, ft_dim), dtype=torch.float)
            if isinstance(v_lbl[0], list):
                # multi-label per vertex
                for vid, labs in enumerate(v_lbl):
                    for lab in labs:
                        x[vid, v_map[lab]] = 1.0
            else:
                for vid, lab in enumerate(v_lbl):
                    x[vid, v_map[lab]] = 1.0

            # Graph label y
            if is_multi:
                n_classes = len(g_map)
                y = torch.zeros(n_classes, dtype=torch.long)
                for lab in g_lbl:
                    y[g_map[lab]] = 1
            else:
                y = torch.tensor(g_map[g_lbl[0]], dtype=torch.long)

            # Topology: edge_index and (optionally) hyperedge_index
            if is_hyper:
                edge_index = self._clique_expand_edges(hedges, num_v)
                hyperedge_index, num_hyperedges = self._build_incidence(hedges, num_v)
                data = Data(
                    x=x,
                    y=y,
                    edge_index=edge_index,
                    hyperedge_index=hyperedge_index,
                    num_hyperedges=num_hyperedges,
                    num_nodes=num_v,
                )
            else:
                # Simple graph: each row in hedges is a pair (u v) or possibly more; guard
                edges = []
                for row in hedges:
                    if len(row) == 2:
                        u, v = row
                        edges.append([u, v])
                        edges.append([v, u])
                    elif len(row) > 2:
                        # If a “graph” line has >2 vertices, treat as a clique just in case.
                        for u, v in itertools.combinations(row, 2):
                            edges.append([u, v])
                            edges.append([v, u])
                edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
                edge_index = to_undirected(edge_index, num_nodes=num_v)
                data = Data(x=x, y=y, edge_index=edge_index, num_nodes=num_v)

            # Standard PyG hooks
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

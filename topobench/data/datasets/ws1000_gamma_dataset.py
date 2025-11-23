"""
Location:
topobench/data/datasets/ws1000_gamma_dataset.py

Implemented a dataset from
@misc{katsman2024revisitingnecessitygraphlearning,
      title={Revisiting the Necessity of Graph Learning and Common Graph Benchmarks},
      author={Isay Katsman and Ethan Lou and Anna Gilbert},
      year={2024},
      eprint={2412.06173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.06173},
}
Note that we do not evaluate on edge prediction, instead we evaluate node lenght classification.
Hopefully, edge prediction will be added.
"""

import os
import os.path as osp
from typing import List

import torch
from torch_geometric.data import InMemoryDataset, Data
from collections import deque
import random


class WS1000GammaDataset(InMemoryDataset):
    r"""Synthetic Watts–Strogatz dataset WS1000_gamma.

    - Graph: Watts–Strogatz (N nodes, mean degree K, rewiring prob beta)
      constructed exactly as in Watts & Strogatz (1998):
        1) regular ring lattice
        2) rewire each edge (i, i+j) with prob beta, keeping i fixed
    - Features: R^d, generated via BFS "parental dependence"
        x_root ~ N(0, I_d)
        x_child = gamma * x_parent + noise_scale * z,  z ~ N(0, I_d)

    """

    def __init__(
        self,
        root: str,
        name: str = "WS1000-gamma",
        parameters=None,
        transform=None,
        pre_transform=None,
    ) -> None:
        self.name = name
        self.parameters = parameters

        # Defaults, can be overridden from Hydra DictConfig
        self.num_nodes = 1000
        self.feature_dim = 1000
        self.mean_degree = 4      # K in WS model
        self.beta = 0.5           # rewiring probability
        self.gamma = 0.0          # parental coefficient
        self.noise_scale = 1.0
        self.seed = 0

        if parameters is not None:
            if "num_nodes" in parameters:
                self.num_nodes = int(parameters.num_nodes)
            if "feature_dim" in parameters:
                self.feature_dim = int(parameters.feature_dim)
            if "mean_degree" in parameters:
                self.mean_degree = int(parameters.mean_degree)
            if "beta" in parameters:
                self.beta = float(parameters.beta)
            if "gamma" in parameters:
                self.gamma = float(parameters.gamma)
            if "noise_scale" in parameters:
                self.noise_scale = float(parameters.noise_scale)
            if "seed" in parameters:
                self.seed = int(parameters.seed)

        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

        # Load processed data (super() will call process() the first time)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ---------------------------------------------------------------------
    # Required PyG properties
    # ---------------------------------------------------------------------
    @property
    def raw_file_names(self) -> List[str]:
        # Dummy file to satisfy InMemoryDataset's bookkeeping.
        return ["synthetic.done"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data_v1.pt"]

    # ---------------------------------------------------------------------
    # Download: here we don't download anything.
    # ---------------------------------------------------------------------
    def download(self) -> None:
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        os.makedirs(self.raw_dir, exist_ok=True)
        with open(raw_path, "w") as f:
            f.write("synthetic ws1000_gamma marker\n")

    # ---------------------------------------------------------------------
    # Process: generate WS graph + WS1000_gamma features and save.
    # ---------------------------------------------------------------------
    def process(self) -> None:
        data = self._generate_ws1000_gamma()
        data_list = [data]
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

    # ---------------------------------------------------------------------
    # Helper: Watts–Strogatz graph + gamma-based features
    # ---------------------------------------------------------------------
    def _generate_ws1000_gamma(self) -> Data:
        N = self.num_nodes
        K = self.mean_degree
        beta = self.beta
        d = self.feature_dim
        gamma = self.gamma
        noise_scale = self.noise_scale
        seed = self.seed

        assert K % 2 == 0, "mean_degree K must be even for Watts–Strogatz ring construction."

        # --- Seed everything deterministically
        random.seed(seed)
        torch.manual_seed(seed)

        # --- 1) Build regular ring lattice
        # neighbors: undirected adjacency; edges: undirected edge set
        neighbors = {i: set() for i in range(N)}
        edges = set()

        half_k = K // 2

        ring_edges_oriented = []
        for j in range(1, half_k + 1):      # distance layer outer
            for i in range(N):              # then each vertex
                v = (i + j) % N
                ring_edges_oriented.append((i, v))
                u_min, u_max = (i, v) if i < v else (v, i)
                if (u_min, u_max) not in edges:
                    edges.add((u_min, u_max))
                    neighbors[i].add(v)
                    neighbors[v].add(i)
        # --- 2) Rewire edges in Watts–Strogatz style (exactly as in the paper)
        # For each original ring edge (i, i+j) in clockwise sense, with probability beta,
        # rewire the endpoint i+j to a new node w chosen uniformly at random
        for (i, v) in ring_edges_oriented:
            if random.random() < beta:
                # Candidates: all nodes except i and current neighbours of i
                possible_nodes = [w for w in range(N)
                                  if w != i and w not in neighbors[i]]
                if not possible_nodes:
                    # No valid candidate; skip rewiring for this edge
                    continue

                w = random.choice(possible_nodes)

                # Remove old edge (i, v) if it still exists
                if v in neighbors[i]:
                    neighbors[i].remove(v)
                    neighbors[v].remove(i)
                    edges.discard((i, v) if i < v else (v, i))

                # Add new edge (i, w)
                neighbors[i].add(w)
                neighbors[w].add(i)
                edges.add((i, w) if i < w else (w, i))


        # --- 3) Convert to undirected edge_index with both directions
        edge_list = []
        for (u, v) in edges:
            edge_list.append((u, v))
            edge_list.append((v, u))
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # --- 4) Generate features with BFS parental dependence
        # Use neighbors directly as adjacency (adj = neighbors)
        x = torch.empty((N, d), dtype=torch.float)

        root = 0
        queue = deque([root])

        # root feature
        x[root] = torch.randn(d)
        dist = torch.full((N,), -1, dtype=torch.long)
        dist[root] = 0

        while queue:
            u = queue.popleft()
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    noise = torch.randn(d)
                    x[v] = gamma * x[u] + noise_scale * noise

        # For unvisited (disconnected) nodes:
        for i in range(N):
            if dist[i] == -1:
                x[i] = torch.randn(d)
                

        data = Data(
            x=x,
            edge_index=edge_index,
            y=dist,
        )
        # Metadata
        data.num_nodes = N
        data.gamma = gamma
        data.beta = beta
        data.mean_degree = K
        data.feature_dim = d
        data.seed = seed

        return data

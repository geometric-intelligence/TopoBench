"""
Dataset class for the Bowen et al. mouse auditory cortex calcium imaging dataset.

This script downloads and processes the original dataset introduced in:

[Citation] Bowen et al. (2024), "Fractured columnar small-world functional network
organization in volumes of L2/3 of mouse auditory cortex," PNAS Nexus, 3(2): pgae074.
https://doi.org/10.1093/pnasnexus/pgae074

We apply the preprocessing and graph-construction steps defined in this module to obtain
a representation of neuronal activity suitable for our experiments.

Please cite the original paper when using this dataset or any derivatives.
"""

import os
import os.path as osp
import shutil
from typing import ClassVar

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs
from torch_geometric.utils import to_undirected

from topobench.data.utils import download_file_from_link
from topobench.data.utils.io_utils import collect_mat_files, process_mat


class TriangleClassifier:
    """Helper class for extracting and classifying triangles in correlation graphs.

    Parameters
    ----------
    min_weight : float, optional
        Minimum correlation to consider as edge, by default 0.2.
    """

    def __init__(self, min_weight: float = 0.2):
        """Initialize triangle classifier.

        Parameters
        ----------
        min_weight : float, optional
            Minimum correlation to consider as edge, by default 0.2
        """
        self.min_weight = min_weight

    def extract_triangles(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        num_nodes: int,
    ) -> list:
        """Extract all triangles from graph.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge connectivity, shape (2, num_edges).
        edge_weights : torch.Tensor
            Correlation values for each edge, shape (num_edges,).
        num_nodes : int
            Number of nodes.

        Returns
        -------
        list of dict
            Each dict contains {'nodes': (a,b,c), 'edge_weights': [w1,w2,w3], 'role': str, 'label': int}.
        """
        # Build networkx graph for easy triangle finding
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for i in range(edge_index.shape[1]):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            w = edge_weights[i].item()
            G.add_edge(u, v, weight=w)

        # Find all triangles
        triangles = list(nx.enumerate_all_cliques(G))
        triangles = [t for t in triangles if len(t) == 3]

        # Classify each triangle
        triangle_data = []
        for nodes in triangles:
            a, b, c = nodes

            # Get edge weights
            w_ab = G[a][b].get("weight", self.min_weight)
            w_bc = G[b][c].get("weight", self.min_weight)
            w_ac = G[a][c].get("weight", self.min_weight)
            edge_weights_tri = [w_ab, w_bc, w_ac]

            # Classify role
            role = self._classify_role(G, nodes, edge_weights_tri)

            triangle_data.append(
                {
                    "nodes": nodes,
                    "edge_weights": edge_weights_tri,
                    "role": role,
                    "label": self._role_to_label(role),
                }
            )

        return triangle_data

    def _classify_role(
        self, G: nx.Graph, nodes: tuple, edge_weights: list
    ) -> str:
        """Classify role of triangle based on edge weights and embedding.

        Parameters
        ----------
        G : nx.Graph
            The correlation graph.
        nodes : tuple
            Three node indices forming the triangle.
        edge_weights : list
            Three edge weights.

        Returns
        -------
        str
            Role string in format "{embedding_class}_{weight_class}".
        """
        a, b, c = nodes

        # Edge weight class
        w_sorted = sorted(edge_weights)
        if all(w > 0.5 for w in edge_weights):
            weight_class = "strong"
        elif w_sorted[0] < 0.3:
            weight_class = "weak"
        else:
            weight_class = "medium"

        # Embedding class: how many other nodes connect to all 3 triangle nodes
        common = len(
            set(G.neighbors(a))
            & set(G.neighbors(b))
            & set(G.neighbors(c)) - {a, b, c}
        )

        if common >= 3:
            embedding_class = "core"
        elif common == 0:
            embedding_class = "isolated"
        else:
            embedding_class = "bridge"

        return f"{embedding_class}_{weight_class}"

    def _role_to_label(self, role_str: str) -> int:
        """Convert role string to integer label.

        Parameters
        ----------
        role_str : str
            Role string (e.g., "core_strong").

        Returns
        -------
        int
            Label (0-6).
        """
        roles = {
            "core_strong": 0,
            "core_medium": 1,
            "bridge_strong": 2,
            "bridge_medium": 3,
            "isolated_strong": 4,
            "isolated_medium": 5,
            "isolated_weak": 6,
        }
        return roles.get(role_str, 6)


class A123CortexMDataset(InMemoryDataset):
    """A1 and A2/3 mouse auditory cortex dataset.

    Loads neural correlation data from mouse auditory cortex regions. Supports
    multiple benchmark tasks:

    1. Graph Classification: Predict frequency bin (0-8) from graph structure
    2. Triangle Classification: Classify topological role of triangles (motifs)

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset including corr_threshold,
        n_bins, min_neurons, and optional triangle_task settings.

    Attributes
    ----------
    URLS : dict
        Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT : dict
        Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES : dict
        Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {
        "Auditory cortex data": "https://gcell.umd.edu/data/Auditory_cortex_data.zip",
    }

    FILE_FORMAT: ClassVar = {
        "Auditory cortex data": "zip",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters
        # defensive parameter access with sensible defaults
        try:
            self.corr_threshold = float(parameters.get("corr_threshold", 0.2))
        except Exception:
            self.corr_threshold = float(
                getattr(parameters, "corr_threshold", 0.2)
            )

        try:
            self.n_bins = int(parameters.get("n_bins", 9))
        except Exception:
            self.n_bins = int(getattr(parameters, "n_bins", 9))

        try:
            self.min_neurons = int(parameters.get("min_neurons", 8))
        except Exception:
            self.min_neurons = int(getattr(parameters, "min_neurons", 8))

        # Triangle classification task settings
        try:
            self.triangle_task_enabled = bool(
                parameters.get("triangle_task", {}).get("enabled", False)
            )
        except Exception:
            self.triangle_task_enabled = False

        self.session_map = {}
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

        # For this dataset we don't assume the internal _data is a torch_geometric Data
        # (this dataset exposes helper methods to construct subgraphs on demand).

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return ["Auditory cortex data/"]

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
        """Download the dataset from a URL and extract to the raw directory."""
        # Download data from the source
        dataset_key = "Auditory cortex data"
        self.url = self.URLS[dataset_key]
        self.file_format = self.FILE_FORMAT[dataset_key]

        # Use self.name as the downloadable dataset name
        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format=self.file_format,
            verify=False,
            timeout=60,  # 60 seconds per chunk read timeout
            retries=3,  # Retry up to 3 times
        )

        # Extract zip file
        folder = self.raw_dir
        filename = f"{self.name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        # Delete zip file
        os.unlink(path)

        # Move files from extracted "Auditory cortex data/" directory to raw_dir
        downloaded_dir = osp.join(folder, self.name)
        if osp.exists(downloaded_dir):
            for file in os.listdir(downloaded_dir):
                src = osp.join(downloaded_dir, file)
                dst = osp.join(folder, file)
                if osp.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.move(src, dst)
            # Delete the extracted top-level directory
            shutil.rmtree(downloaded_dir)
        self.data_dir = folder

    @staticmethod
    def extract_samples(data_dir: str, n_bins: int, min_neurons: int = 8):
        """Extract subgraph samples from raw .mat files.

        Parameters
        ----------
        data_dir : str
            Directory containing the raw .mat files.
        n_bins : int
            Number of frequency bins to use for binning.
        min_neurons : int, optional
            Minimum number of neurons required per sample. Defaults to 8.

        Returns
        -------
        pd.DataFrame
            DataFrame containing extracted samples with columns for
            session_file, session_id, layer, bf_bin, neuron_indices,
            corr, and noise_corr.
        """
        mat_files = collect_mat_files(data_dir)

        samples = []
        session_id = 0
        for f in mat_files:
            print(f"Processing session {session_id}: {os.path.basename(f)}")
            mt = process_mat(scipy.io.loadmat(f))
            for layer in range(1, 6):
                scorrs = np.array(mt["selectZCorrInfo"]["SigCorrs"])
                ncorrs = np.array(mt["selectZCorrInfo"]["NoiseCorrsTrial"])
                bfvals = np.array(mt["BFInfo"][layer]["BFval"]).ravel()
                if scorrs.size == 0 or bfvals.size == 0:
                    continue

                bin_ids = bfvals.astype(int)

                for bin_idx in range(n_bins):
                    sel = np.where(bin_ids == bin_idx)[0]
                    if len(sel) < min_neurons:
                        continue
                    subcorr = scorrs[np.ix_(sel, sel)]
                    samples.append(
                        {
                            "session_file": f,
                            "session_id": session_id,
                            "layer": layer,
                            "bf_bin": int(bin_idx),
                            "neuron_indices": sel.tolist(),
                            "corr": subcorr.astype(float),
                            "noise_corr": ncorrs[np.ix_(sel, sel)].astype(
                                float
                            ),
                        }
                    )
            session_id += 1

        samples = pd.DataFrame(samples)
        return samples

    def _sample_to_pyg_data(
        self, sample: dict, threshold: float = 0.2
    ) -> Data:
        """Convert a sample dictionary to a PyTorch Geometric Data object.

        Converts correlation matrices to graph representation with node features
        and edges for graph-level classification tasks.

        Parameters
        ----------
        sample : dict
            Sample dictionary containing 'corr', 'noise_corr', 'session_id',
            'layer', and 'bf_bin' keys.
        threshold : float, optional
            Correlation threshold for creating edges. Defaults to 0.2.

        Returns
        -------
        torch_geometric.data.Data
            Data object with node features [mean_corr, std_corr, noise_diag],
            edges from thresholded correlation, and label y as integer bf_bin.
        """
        corr = np.asarray(sample.get("corr"))
        if corr.ndim != 2 or corr.size == 0:
            # empty placeholder graph
            x = torch.zeros((0, 3), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            n = corr.shape[0]
            # sanitize
            corr = np.nan_to_num(corr)

            mean_corr = corr.mean(axis=1)
            std_corr = corr.std(axis=1)
            noise_diag = np.zeros(n)
            if "noise_corr" in sample and sample["noise_corr"] is not None:
                nc = np.asarray(sample["noise_corr"])
                if nc.shape == corr.shape:
                    noise_diag = np.diag(nc)

            x_np = np.vstack([mean_corr, std_corr, noise_diag]).T
            x = torch.tensor(x_np, dtype=torch.float)

            # build edges from thresholded correlation (upper triangle)
            adj = (corr >= threshold).astype(int)
            iu = np.triu_indices(n, k=1)
            sel = np.where(adj[iu] == 1)[0]
            if sel.size == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float)
            else:
                rows = iu[0][sel]
                cols = iu[1][sel]
                edge_index_np = np.vstack([rows, cols])
                edge_index = torch.tensor(edge_index_np, dtype=torch.long)
                # make undirected
                edge_index = to_undirected(edge_index)
                # edge_attr: corresponding corr weights (for both directions, if made undirected)
                weights = corr[rows, cols]
                weights = (
                    np.repeat(weights, 2)
                    if edge_index.size(1) == weights.size * 2
                    else weights
                )
                edge_attr = torch.tensor(
                    weights.reshape(-1, 1), dtype=torch.float
                )

        y = torch.tensor([int(sample.get("bf_bin", -1))], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        # attach metadata
        data.session_id = int(sample.get("session_id", -1))
        data.layer = int(sample.get("layer", -1))
        return data

    def _extract_triangles_from_graphs(self) -> list:
        """Extract raw triangle data from all graphs with NetworkX representations.

        Returns a list of dicts, each containing graph metadata and triangle info.

        Returns
        -------
        list of dict
            Each dict has keys:
            - 'graph_idx': index of source graph
            - 'tri': triangle dict from classifier (with nodes, edge_weights, role, label)
            - 'G': NetworkX graph object (for structural queries)
            - 'num_nodes': number of nodes in graph
        """
        classifier = TriangleClassifier(min_weight=self.corr_threshold)
        raw_triangles = []

        print("[A123] Starting triangle extraction from graphs...")

        for graph_idx, data in enumerate(self.data):
            if graph_idx % 100 == 0:
                print(
                    f"[A123] Processing graph {graph_idx}/{len(self.data)} "
                    f"for triangle extraction..."
                )

            # Skip graphs with no edges
            if data.edge_index.shape[1] == 0:
                continue

            # Build NetworkX graph for structural queries
            num_nodes = (
                data.x.shape[0]
                if hasattr(data, "x") and data.x is not None
                else 0
            )
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            for i in range(data.edge_index.shape[1]):
                u = int(data.edge_index[0, i].item())
                v = int(data.edge_index[1, i].item())
                G.add_edge(u, v)

            # Extract triangles
            try:
                triangles = classifier.extract_triangles(
                    data.edge_index,
                    data.edge_attr.squeeze()
                    if data.edge_attr is not None
                    else torch.ones(data.edge_index.shape[1]),
                    num_nodes,
                )
            except Exception as e:
                print(
                    f"[A123] Warning: Could not extract triangles for graph {graph_idx}: {e}"
                )
                continue

            # Store raw triangle data with graph context
            for tri in triangles:
                raw_triangles.extend(
                    {
                        "graph_idx": graph_idx,
                        "tri": tri,
                        "G": G,
                        "num_nodes": num_nodes,
                    }
                )

        return raw_triangles

    def create_triangle_classification_task(self) -> list:
        """Create triangle-level classification dataset from graph-level data.

        Extracts all triangles from each graph and creates a new dataset where
        each sample is a triangle classified by its topological role. Features
        are purely topological (edge weights only) - independent of original
        node properties or frequency information.

        Returns
        -------
        list of torch_geometric.data.Data
            Triangle-level samples with 3D edge weight features and role labels.
        """
        raw_triangles = self._extract_triangles_from_graphs()
        triangle_data_list = []

        print("[A123] Creating triangle classification task...")

        for item in raw_triangles:
            tri = item["tri"]
            graph_idx = item["graph_idx"]

            # Topological features only: edge weights
            tri_edge_weights = torch.tensor(
                tri["edge_weights"], dtype=torch.float32
            )  # (3,)

            # Create data object for this triangle
            tri_data = Data(
                x=tri_edge_weights.unsqueeze(0),  # (1, 3) - edge weights only
                y=torch.tensor(tri["label"], dtype=torch.long),
                nodes=torch.tensor(tri["nodes"], dtype=torch.long),
                role=tri["role"],
                graph_idx=graph_idx,
            )

            triangle_data_list.append(tri_data)

        print(f"[A123] Created {len(triangle_data_list)} triangle samples")
        return triangle_data_list

    def create_triangle_common_neighbors_task(self) -> list:
        """Create triangle-level dataset where label is the number of common neighbours.

        For each triangle (a,b,c) we compute:
          - feature: the degrees of the three nodes (structural, no weights)
          - label: number of nodes that are neighbours to all three (common neighbours)

        Returns
        -------
        list of torch_geometric.data.Data
            Each Data contains x (1,3) degrees, y (scalar) common-neighbour count,
            nodes (3,), role (str) optionally, and graph_idx metadata.
        """
        raw_triangles = self._extract_triangles_from_graphs()
        triangle_data_list = []

        print("[A123] Creating triangle common-neighbors task...")

        for item in raw_triangles:
            tri = item["tri"]
            G = item["G"]
            graph_idx = item["graph_idx"]

            a, b, c = tri["nodes"]

            # Compute common neighbours (exclude triangle nodes)
            common = (
                set(G.neighbors(a)) & set(G.neighbors(b)) & set(G.neighbors(c))
            ) - {a, b, c}
            num_common = len(common)

            # Node degree features (structural)
            deg_a = G.degree(a)
            deg_b = G.degree(b)
            deg_c = G.degree(c)
            tri_feats = torch.tensor(
                [deg_a, deg_b, deg_c], dtype=torch.float32
            )

            tri_data = Data(
                x=tri_feats.unsqueeze(0),  # (1,3)
                y=torch.tensor([int(num_common)], dtype=torch.long),
                nodes=torch.tensor(tri["nodes"], dtype=torch.long),
                role=tri.get("role", ""),
                graph_idx=graph_idx,
            )

            triangle_data_list.append(tri_data)

        print(f"[A123] Created {len(triangle_data_list)} triangle CN samples")
        return triangle_data_list

    def process(self) -> None:
        """Generate raw files into collated PyG dataset and save to disk.

        This implementation mirrors other datasets in the repo: it calls the
        static helper `extract_samples()` to enumerate subgraphs, converts each
        to a `torch_geometric.data.Data` object via `_sample_to_pyg_data()`,
        optionally computes/attaches topology vectors, collates and saves.

        If triangle_task is enabled, also creates and saves triangle-level dataset.
        """
        data_dir = self.raw_dir

        print(f"[A123] Processing dataset from: {data_dir}")
        print(f"[A123] Files in raw_dir: {os.listdir(data_dir)}")

        # extract sample descriptions
        print("[A123] Starting extract_samples()...")
        samples = A123CortexMDataset.extract_samples(
            data_dir, self.n_bins, self.min_neurons
        )

        print(f"[A123] Extracted {len(samples)} samples")

        data_list = []
        for idx, (_, s) in enumerate(samples.iterrows()):
            if idx % 100 == 0:
                print(
                    f"[A123] Converting sample {idx}/{len(samples)} to PyG Data..."
                )
            d = self._sample_to_pyg_data(s, threshold=self.corr_threshold)
            data_list.append(d)

        # collate and save processed dataset
        print(f"[A123] Collating {len(data_list)} samples...")
        self.data, self.slices = self.collate(data_list)
        self._data_list = None
        print(f"[A123] Saving processed data to {self.processed_paths[0]}...")
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

        # If triangle task is enabled, create and save triangle classification dataset
        if self.triangle_task_enabled:
            print(
                "[A123] Triangle task enabled. Creating triangle classification dataset..."
            )
            triangle_data = self.create_triangle_classification_task()

            # Save triangle dataset to separate file
            triangle_processed_path = self.processed_paths[0].replace(
                "data.pt", "data_triangles.pt"
            )
            print(f"[A123] Collating {len(triangle_data)} triangle samples...")
            triangle_collated, triangle_slices = self.collate(triangle_data)
            print(
                f"[A123] Saving triangle dataset to {triangle_processed_path}..."
            )
            fs.torch_save(
                (
                    triangle_collated.to_dict(),
                    triangle_slices,
                    {},
                    triangle_collated.__class__,
                ),
                triangle_processed_path,
            )
            print("[A123] Triangle task dataset saved!")

        # If triangle common-neighbours task is enabled, create and save it
        if self.triangle_common_task_enabled:
            print(
                "[A123] Triangle common-neighbours task enabled. Creating dataset..."
            )
            triangle_cn_data = self.create_triangle_common_neighbors_task()

            triangle_cn_processed_path = self.processed_paths[0].replace(
                "data.pt", "data_triangles_common_neighbors.pt"
            )
            print(
                f"[A123] Collating {len(triangle_cn_data)} triangle CN samples..."
            )
            triangle_cn_collated, triangle_cn_slices = self.collate(
                triangle_cn_data
            )
            print(
                f"[A123] Saving triangle CN dataset to {triangle_cn_processed_path}..."
            )
            fs.torch_save(
                (
                    triangle_cn_collated.to_dict(),
                    triangle_cn_slices,
                    {},
                    triangle_cn_collated.__class__,
                ),
                triangle_cn_processed_path,
            )
            print("[A123] Triangle CN dataset saved!")

        print("[A123] Processing complete!")

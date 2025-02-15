import hashlib
import os.path as osp
import pickle
from collections.abc import Callable

import networkx as nx
import numpy as np
import omegaconf
import rootutils
import toponetx.datasets.graph as graph
import torch
import torch_geometric
from gudhi.datasets.generators import points
from gudhi.datasets.remote import (
    fetch_bunny,
    fetch_daily_activities,
    fetch_spiral_2d,
)
from topomodelx.utils.sparse import from_sparse
from torch_geometric.data import Data
from torch_sparse import coalesce

rootutils.setup_root("./", indicator=".project-root", pythonpath=True)


def get_complex_connectivity(complex, max_rank, signed=False):
    r"""Gets the connectivity matrices for the complex.

    Parameters
    ----------
    complex : topnetx.CellComplex, topnetx.SimplicialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    signed : bool
        If True, returns signed connectivity matrices.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "hodge_laplacian",
        ]:
            try:
                connectivity[f"{connectivity_info}_{rank_idx}"] = from_sparse(
                    getattr(complex, f"{connectivity_info}_matrix")(
                        rank=rank_idx, signed=signed
                    )
                )
            except ValueError:  # noqa: PERF203
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    connectivity["shape"] = practical_shape
    return connectivity


def generate_zero_sparse_connectivity(m, n):
    r"""Generates a zero sparse connectivity matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    torch.sparse_coo_tensor
        Zero sparse connectivity matrix.
    """
    return torch.sparse_coo_tensor((m, n)).coalesce()


def load_cell_complex_dataset(cfg):
    r"""Loads cell complex datasets."""


def load_simplicial_dataset(cfg):
    r"""Loads simplicial datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Simplicial dataset.
    """
    if cfg["data_name"] != "KarateClub":
        return NotImplementedError
    data = graph.karate_club(complex_type="simplicial", feat_dim=2)
    max_rank = data.dim
    features = {}
    dict_feat_equivalence = {
        0: "node_feat",
        1: "edge_feat",
        2: "face_feat",
        3: "tetrahedron_feat",
    }
    for rank_idx in range(max_rank + 1):
        try:
            features[f"x_{rank_idx}"] = torch.tensor(
                np.stack(
                    list(
                        data.get_simplex_attributes(
                            dict_feat_equivalence[rank_idx]
                        ).values()
                    )
                )
            )
        except ValueError:  # noqa: PERF203
            features[f"x_{rank_idx}"] = torch.tensor(
                np.zeros((data.shape[rank_idx], 0))
            )
    features["y"] = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    # features['num_nodes'] = data.shape[0]
    features["x"] = features["x_0"]
    connectivity = get_complex_connectivity(data, max_rank)
    data = torch_geometric.data.Data(**connectivity, **features)

    # Project node-level features to edge-level (WHY DO WE NEED IT, data already has x_1)
    data.x_1 = data.x_1 + torch.mm(data.incidence_1.to_dense().T, data.x_0)

    return torch_geometric.transforms.random_node_split.RandomNodeSplit(
        num_val=4, num_test=4
    )(data)


def load_hypergraph_pickle_dataset(cfg):
    r"""Loads hypergraph datasets from pickle files.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Hypergraph dataset.
    """
    data_dir = cfg["data_dir"]
    print(f"Loading {cfg['data_domain']} dataset name: {cfg['data_name']}")

    # Load node features:

    with open(osp.join(data_dir, "features.pickle"), "rb") as f:
        features = pickle.load(f)
        features = features.todense()

    # Load node labels:
    with open(osp.join(data_dir, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # Load hypergraph.
    with open(osp.join(data_dir, "hypergraph.pickle"), "rb") as f:
        # Hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f"number of hyperedges: {len(hypergraph)}")

    edge_idx = 0  # num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph:
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    # check that every node is in some hyperedge
    if len(np.unique(node_list)) != num_nodes:
        # add self hyperedges to isolated nodes
        isolated_nodes = np.setdiff1d(
            np.arange(num_nodes), np.unique(node_list)
        )

        for node in isolated_nodes:
            node_list += [node]
            edge_list += [edge_idx]
            edge_idx += 1
            hypergraph[f"Unique_additonal_he_{edge_idx}"] = [node]

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(
        x=features,
        x_0=features,
        edge_index=edge_index,
        incidence_hyperedges=edge_index,
        y=labels,
    )

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    num_class = len(np.unique(labels.numpy()))

    # Add parameters to attribute
    data.n_x = n_x
    data.num_hyperedges = len(hypergraph)
    data.num_class = num_class

    data.incidence_hyperedges = torch.sparse_coo_tensor(
        data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(data.num_nodes, data.num_hyperedges),
    )

    # Print some info
    print("Final num_hyperedges", data.num_hyperedges)
    print("Final num_nodes", data.num_nodes)
    print("Final num_class", data.num_class)

    return data


def load_point_cloud(
    num_classes: int = 2, num_samples: int = 18, seed: int = 42
):
    """Create a toy point cloud dataset"""
    rng = np.random.default_rng(seed)

    points = torch.tensor(rng.random((num_samples, 2)), dtype=torch.float)
    classes = torch.tensor(
        rng.integers(num_classes, size=num_samples), dtype=torch.long
    )
    features = torch.tensor(
        rng.integers(3, size=(num_samples, 1)), dtype=torch.float
    )

    return torch_geometric.data.Data(x=features, y=classes, pos=points)


def load_manual_graph():
    """Create a manual graph for testing purposes."""
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the edges
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [5, 7],
        [2, 7],
        [0, 7],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )


def load_manual_mol():
    """Create a manual graph for testing the ring implementation.
    Actually is the 471 molecule of QM9 dataset."""
    # Define the vertices
    vertices = [i for i in range(12)]
    y = torch.tensor(
        [
            [
                2.2569e00,
                4.5920e01,
                -6.3076e00,
                1.9211e00,
                8.2287e00,
                4.6414e02,
                2.6121e00,
                -8.3351e03,
                -8.3349e03,
                -8.3349e03,
                -8.3359e03,
                2.0187e01,
                -4.8740e01,
                -4.9057e01,
                -4.9339e01,
                -4.5375e01,
                6.5000e00,
                3.8560e00,
                3.0122e00,
            ]
        ]
    )

    # Define the edges
    edges = [
        [0, 1],
        [0, 6],
        [1, 0],
        [1, 2],
        [1, 3],
        [1, 5],
        [2, 1],
        [2, 3],
        [2, 7],
        [2, 8],
        [3, 1],
        [3, 2],
        [3, 4],
        [3, 9],
        [4, 3],
        [4, 5],
        [5, 1],
        [5, 4],
        [5, 10],
        [5, 11],
        [6, 0],
        [7, 2],
        [8, 2],
        [9, 3],
        [10, 5],
        [11, 5],
    ]

    # Add smile
    smiles = "[H]O[C@@]12C([H])([H])O[C@]1([H])C2([H])[H]"

    # # Create a graph
    # G = nx.Graph()

    # # Add vertices
    # G.add_nodes_from(vertices)

    # # Add edges
    # G.add_edges_from(edges)

    # G.to_undirected()
    # edge_list = torch.Tensor(list(G.edges())).T.long()

    x = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    pos = torch.tensor(
        [
            [-0.0520, 1.4421, 0.0438],
            [-0.0146, 0.0641, 0.0278],
            [-0.2878, -0.7834, -1.1968],
            [-1.1365, -0.9394, 0.0399],
            [-0.4768, -1.7722, 0.9962],
            [0.6009, -0.8025, 1.1266],
            [0.6168, 1.7721, -0.5660],
            [-0.7693, -0.2348, -2.0014],
            [0.3816, -1.5834, -1.5029],
            [-2.2159, -0.8594, 0.0798],
            [1.5885, -1.2463, 0.9538],
            [0.5680, -0.3171, 2.1084],
        ]
    )

    assert len(x) == len(vertices)
    assert len(pos) == len(vertices)

    return torch_geometric.data.Data(
        x=torch.tensor(x).float(),
        edge_index=torch.tensor(edges).T.long(),
        num_nodes=len(vertices),
        y=torch.tensor(y),
        smiles=smiles,
        pos=pos,
    )


def get_Planetoid_pyg(cfg):
    r"""Loads Planetoid graph datasets from torch_geometric.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Graph dataset.
    """
    data_dir, data_name = cfg["data_dir"], cfg["data_name"]
    dataset = torch_geometric.datasets.Planetoid(data_dir, data_name)
    data = dataset.data
    data.num_nodes = data.x.shape[0]
    return data


def get_TUDataset_pyg(cfg):
    r"""Loads TU graph datasets from torch_geometric.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    list
        List containing the graph dataset.
    """
    data_dir, data_name = cfg["data_dir"], cfg["data_name"]
    dataset = torch_geometric.datasets.TUDataset(root=data_dir, name=data_name)
    return [data for data in dataset]


def ensure_serializable(obj):
    r"""Ensures that the object is serializable.

    Parameters
    ----------
    obj : object
        Object to ensure serializability.

    Returns
    -------
    object
        Object that is serializable.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = ensure_serializable(value)
        return obj
    elif isinstance(obj, list | tuple):  # noqa: RET505
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, str | int | float | bool | type(None)):
        return obj
    elif isinstance(obj, omegaconf.dictconfig.DictConfig):
        return dict(obj)
    else:
        return None


def make_hash(o):
    r"""Makes a hash from a dictionary, list, tuple or set to any level, that
    contains only other hashable types (including any lists, tuples, sets, and
    dictionaries).

    Parameters
    ----------
    o : dict, list, tuple, set
        Object to hash.

    Returns
    -------
    int
        Hash of the object.
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    # Convert the hex back to int and restrict it to the relevant int range
    return int(hash_as_hex, 16) % 4294967295


def load_gudhi_dataset(
    cfg: omegaconf.DictConfig,
    feature_generator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    target_generator: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch_geometric.data.Data:
    """Load a dataset from the gudhi.datasets module."""
    if not cfg.data_name.startswith("gudhi_"):
        raise ValueError(
            "This function should only be used with gudhi datasets"
        )

    gudhi_dataset_name = cfg.data_name.removeprefix("gudhi_")

    if gudhi_dataset_name == "sphere":
        points_data = points.sphere(
            n_samples=cfg["n_samples"],
            ambient_dim=cfg["ambient_dim"],
            sample=cfg["sample"],
        )
    elif gudhi_dataset_name == "torus":
        points_data = points.torus(
            n_samples=cfg["n_samples"], dim=cfg["dim"], sample=cfg["sample"]
        )
    elif gudhi_dataset_name == "bunny":
        file_path = osp.join(
            rootutils.find_root(), cfg["data_dir"], "bunny", "bunny.npy"
        )
        points_data = fetch_bunny(
            file_path=file_path,
            accept_license=cfg.get("accept_license", False),
        )
    elif gudhi_dataset_name == "spiral_2d":
        file_path = osp.join(
            rootutils.find_root(),
            cfg["data_dir"],
            "spiral_2d",
            "spiral_2d.npy",
        )
        points_data = fetch_spiral_2d(file_path=file_path)
    elif gudhi_dataset_name == "daily_activities":
        file_path = osp.join(
            rootutils.find_root(),
            cfg["data_dir"],
            "activities",
            "activities.npy",
        )
        data = fetch_daily_activities(file_path=file_path)
        points_data = data[:, :3]
    else:
        raise ValueError(f"Gudhi dataset {gudhi_dataset_name} not recognized.")

    pos = torch.tensor(points_data, dtype=torch.float)
    if feature_generator:
        x = feature_generator(pos)
        if x.shape[0] != pos.shape[0]:
            raise ValueError(
                "feature_generator must not change first dimension of points data."
            )
    else:
        x = None

    if target_generator:
        y = target_generator(pos)
        if y.shape[0] != pos.shape[0]:
            raise ValueError(
                "target_generator must not change first dimension of points data."
            )
    elif gudhi_dataset_name == "daily_activities":
        # Target is the activity type
        # 14. for 'cross_training', 18. for 'jumping', 13. for 'stepper', or 9. for 'walking'
        y = torch.tensor(data[:, 3:], dtype=torch.float)
    else:
        y = None

    return torch_geometric.data.Data(x=x, y=y, pos=pos, complex_dim=0)


def load_random_points(
    dim: int, num_classes: int, num_samples: int, seed: int = 42
) -> torch_geometric.data.Data:
    """Create a random point cloud dataset."""
    rng = np.random.default_rng(seed)

    points = torch.tensor(rng.random((num_samples, dim)), dtype=torch.float)
    classes = torch.tensor(
        rng.integers(num_classes, size=num_samples), dtype=torch.long
    )
    features = torch.tensor(
        rng.integers(2, size=(num_samples, 1)), dtype=torch.float
    )

    return torch_geometric.data.Data(
        x=features, y=classes, pos=points, complex_dim=0
    )


def load_manual_points():
    pos = torch.tensor(
        [
            [1.0, 1.0],
            [7.0, 0.0],
            [4.0, 6.0],
            [9.0, 6.0],
            [0.0, 14.0],
            [2.0, 19.0],
            [9.0, 17.0],
        ],
        dtype=torch.float,
    )
    x = torch.ones_like(pos, dtype=torch.float)
    y = torch.randint(0, 2, (pos.shape[0],), dtype=torch.float)
    return torch_geometric.data.Data(x=x, y=y, pos=pos, complex_dim=0)

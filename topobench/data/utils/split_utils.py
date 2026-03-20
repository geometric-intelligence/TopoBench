"""Split utilities."""

import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from topobench.dataloader import DataloadDataset


# Generate splits in different fasions
def k_fold_split(labels, parameters, root=None):
    """Return train and valid indices as in K-Fold Cross-Validation.

    If the split already exists it loads it automatically, otherwise it creates the
    split file for the subsequent runs.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameters.
    root : str, optional
        Root directory for data splits. Overwrite the default directory.

    Returns
    -------
    dict
        Dictionary containing the train, validation and test indices, with keys "train", "valid", and "test".
    """

    data_dir = (
        parameters["data_split_dir"]
        if root is None
        else os.path.join(root, "data_splits")
    )
    k = parameters.k
    fold = parameters.data_seed
    assert fold < k, "data_seed needs to be less than k"

    torch.manual_seed(0)
    np.random.seed(0)

    split_dir = os.path.join(data_dir, f"{k}-fold")

    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    if not os.path.isfile(split_path):
        n = len(labels)
        x_idx = np.arange(n)
        x_idx = np.random.permutation(x_idx)
        labels = labels[x_idx]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for fold_n, (train_idx, valid_idx) in enumerate(
            skf.split(x_idx, labels)
        ):
            split_idx = {
                "train": train_idx,
                "valid": valid_idx,
                "test": valid_idx,
            }

            # Check that all nodes/graph have been assigned to some split
            assert np.all(
                np.sort(
                    np.array(
                        split_idx["train"].tolist()
                        + split_idx["valid"].tolist()
                    )
                )
                == np.sort(np.arange(len(labels)))
            ), "Not every sample has been loaded."
            split_path = os.path.join(split_dir, f"{fold_n}.npz")

            np.savez(split_path, **split_idx)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert np.unique(
        np.array(
            split_idx["train"].tolist()
            + split_idx["valid"].tolist()
            + split_idx["test"].tolist()
        )
    ).shape[0] == len(labels), "Not all nodes within splits"

    return split_idx


def random_splitting(labels, parameters, root=None, global_data_seed=42):
    r"""Randomly splits label into train/valid/test splits.

    Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameter.
    root : str, optional
        Root directory for data splits. Overwrite the default directory.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict:
        Dictionary containing the train, validation and test indices with keys "train", "valid", and "test".
    """
    fold = (
        parameters["data_seed"] % 10
    )  # Ensure fold is between 0 and 9, TODO: Modify hardcoded 10 split number
    data_dir = (
        parameters["data_split_dir"]
        if root is None
        else os.path.join(root, "data_splits")
    )
    train_prop = parameters["train_prop"]
    valid_prop = (1 - train_prop) / 2

    # Create split directory if it does not exist
    split_dir = os.path.join(
        data_dir, f"train_prop={train_prop}_global_seed={global_data_seed}"
    )
    generate_splits = False
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
        generate_splits = True

    # Generate splits if they do not exist
    if generate_splits:
        # Set initial seed
        torch.manual_seed(global_data_seed)
        np.random.seed(global_data_seed)
        # Generate a split
        n = len(labels)
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        # Generate 10 splits
        for fold_n in range(10):
            # Permute indices
            perm = torch.as_tensor(np.random.permutation(n))

            train_indices = perm[:train_num]
            val_indices = perm[train_num : train_num + valid_num]
            test_indices = perm[train_num + valid_num :]
            split_idx = {
                "train": train_indices,
                "valid": val_indices,
                "test": test_indices,
            }

            # Save generated split
            split_path = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path, **split_idx)

    # Load the split
    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert np.unique(
        np.array(
            split_idx["train"].tolist()
            + split_idx["valid"].tolist()
            + split_idx["test"].tolist()
        )
    ).shape[0] == len(labels), "Not all nodes within splits"

    return split_idx


def assign_train_val_test_mask_to_graphs(dataset, split_idx):
    """Split the graph dataset into train, validation, and test datasets.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Considered dataset.
    split_idx : dict
        Dictionary containing the train, validation, and test indices.

    Returns
    -------
    tuple:
        Tuple containing the train, validation, and test datasets.
    """

    data_train_lst, data_val_lst, data_test_lst = [], [], []

    # Assign masks directly by iterating over pre-split indices
    for i in split_idx["train"]:
        graph = dataset[i]
        graph.train_mask = torch.tensor([1], dtype=torch.long)
        graph.val_mask = torch.tensor([0], dtype=torch.long)
        graph.test_mask = torch.tensor([0], dtype=torch.long)
        data_train_lst.append(graph)

    for i in split_idx["valid"]:
        graph = dataset[i]
        graph.train_mask = torch.tensor([0], dtype=torch.long)
        graph.val_mask = torch.tensor([1], dtype=torch.long)
        graph.test_mask = torch.tensor([0], dtype=torch.long)
        data_val_lst.append(graph)

    for i in split_idx["test"]:
        graph = dataset[i]
        graph.train_mask = torch.tensor([0], dtype=torch.long)
        graph.val_mask = torch.tensor([0], dtype=torch.long)
        graph.test_mask = torch.tensor([1], dtype=torch.long)
        data_test_lst.append(graph)

    return (
        DataloadDataset(data_train_lst),
        DataloadDataset(data_val_lst),
        DataloadDataset(data_test_lst),
    )


def load_transductive_splits(dataset, parameters):
    r"""Load the graph dataset with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert len(dataset) == 1, (
        "Dataset should have only one graph in a transductive setting."
    )

    data = dataset.data_list[0]
    labels = data.y.numpy()

    # Ensure labels are one dimensional array
    assert len(labels.shape) == 1, "Labels should be one dimensional array"

    root = (
        dataset.dataset.get_data_dir()
        if hasattr(dataset.dataset, "get_data_dir")
        else None
    )

    if parameters.split_type == "random":
        splits = random_splitting(labels, parameters, root=root)

    elif parameters.split_type == "k-fold":
        splits = k_fold_split(labels, parameters, root=root)

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random' or 'k-fold'"
        )

    # Assign train val test masks to the graph
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    if parameters.get("standardize", False):
        # Standardize the node features respecting train mask
        data.x = (data.x - data.x[data.train_mask].mean(0)) / data.x[
            data.train_mask
        ].std(0)
        data.y = (data.y - data.y[data.train_mask].mean(0)) / data.y[
            data.train_mask
        ].std(0)

    return DataloadDataset([data]), None, None


def load_inductive_splits(dataset, parameters):
    r"""Load multiple-graph datasets with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert len(dataset) > 1, (
        "Datasets should have more than one graph in an inductive setting."
    )
    # Check if labels are ragged (different sizes across graphs)
    label_list = [data.y.squeeze(0).numpy() for data in dataset]
    label_shapes = [label.shape for label in label_list]
    # Use dtype=object only if labels have different shapes (ragged)
    labels = (
        np.array(label_list, dtype=object)
        if len(set(label_shapes)) > 1
        else np.array(label_list)
    )

    root = (
        dataset.dataset.get_data_dir()
        if hasattr(dataset.dataset, "get_data_dir")
        else None
    )

    if parameters.split_type == "random":
        split_idx = random_splitting(labels, parameters, root=root)

    elif parameters.split_type == "k-fold":
        assert type(labels) is not object, (
            "K-Fold splitting not supported for ragged labels."
        )
        split_idx = k_fold_split(labels, parameters, root=root)

    elif parameters.split_type == "fixed" and hasattr(dataset, "split_idx"):
        split_idx = dataset.split_idx

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random', 'k-fold' or 'fixed'.\
            If 'fixed' is chosen, the dataset should have the attribute split_idx"
        )

    train_dataset, val_dataset, test_dataset = (
        assign_train_val_test_mask_to_graphs(dataset, split_idx)
    )

    return train_dataset, val_dataset, test_dataset


def load_coauthorship_hypergraph_splits(data, parameters, train_prop=0.5):
    r"""Load the split generated by rand_train_test_idx function.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.
    train_prop : float
        Proportion of training data.

    Returns
    -------
    torch_geometric.data.Data:
        Graph dataset with the specified split.
    """

    data_dir = os.path.join(
        parameters["data_split_dir"], f"train_prop={train_prop}"
    )
    load_path = f"{data_dir}/split_{parameters['data_seed']}.npz"
    splits = np.load(load_path, allow_pickle=True)

    # Upload masks
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    # Check that all nodes assigned to splits
    assert (
        torch.unique(
            torch.concat([data.train_mask, data.val_mask, data.test_mask])
        ).shape[0]
        == data.num_nodes
    ), "Not all nodes within splits"
    return DataloadDataset([data]), None, None

def load_edge_transductive_splits(preprocessor, split_params) -> tuple[
    DataloadDataset, DataloadDataset | None, DataloadDataset | None
]:
    """
    Load edge-level dataset splits for transductive link prediction.

    Parameters
    ----------
    preprocessor : PreProcessor
        Object containing the full graph dataset.
    split_params : dict or omegaconf.DictConfig
        Split configuration including ``val_prop`` (default 0.1), ``test_prop``
        (default 0.1), and ``is_undirected`` (default True).

    Returns
    -------
    tuple of DataloadDataset
        A tuple containing the (train, val, test) datasets.
    """
    from topobench.transforms.data_manipulations.negative_links_sampling import (
        NegativeSamplingTransform,
    )
    # Get the full (single) graph from the underlying PyG dataset.
    full_data = preprocessor.dataset[0]

    # Read split hyperparameters with sensible defaults.
    val_ratio = float(split_params.get("val_prop", 0.10))
    test_ratio = float(split_params.get("test_prop", 0.10))
    is_undirected = bool(split_params.get("is_undirected", True))
    neg_sampling_ratio = float(split_params.get("neg_sampling_ratio", 1.0))

    # Instantiate the RandomLinkSplit transform.
    splitter = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=is_undirected,
        add_negative_train_samples=False,
        neg_sampling_ratio=neg_sampling_ratio,
    )

    # Apply the splitter to obtain train/val/test Data objects.
    train_data, val_data, test_data = splitter(full_data)
    
    # Negative transform:
    neg_transform = NegativeSamplingTransform(neg_pos_ratio=split_params.get("neg_pos_ratio", 1.0), method=split_params.get("neg_sampling_method", "sparse"))

    # Wrap each split into a DataloadDataset (expected by TBDataloader).
    dataset_train = DataloadDataset([train_data], _dynamic_transform=neg_transform)
    dataset_val = DataloadDataset([val_data])
    dataset_test = DataloadDataset([test_data])

    return dataset_train, dataset_val, dataset_test
    
def load_edge_inductive_splits(preprocessor, split_params) -> tuple[
    DataloadDataset, DataloadDataset | None, DataloadDataset | None
]:
    r"""Load inductive edge-level splits for link prediction.

    Parameters
    ----------
    preprocessor : PreProcessor
        Preprocessor containing the underlying PyG dataset.
    split_params : dict or omegaconf.DictConfig
        Split configuration, including split type and negative sampling
        parameters.

    Returns
    -------
    tuple of DataloadDataset
        Train, validation and test datasets for edge-level link prediction.
    """
    from topobench.transforms.data_manipulations.negative_links_sampling import (
        NegativeSamplingTransform,
    )
    
    # Underlying PyG dataset with multiple graphs (MUTAG-style).
    dataset = preprocessor.dataset
    assert len(dataset) > 1, (
        "Inductive edge-level split requires a multi-graph dataset."
    )

    # Build per-graph labels as in `load_inductive_splits`.
    label_list = [data.y.squeeze(0).numpy() for data in dataset]
    label_shapes = [label.shape for label in label_list]
    labels = (
        np.array(label_list, dtype=object)
        if len(set(label_shapes)) > 1
        else np.array(label_list)
    )

    # Root is only used to adjust path for saved splits; we can rely on
    # split_params["data_split_dir"] directly, so set root=None.
    root = None

    if split_params.split_type == "random":
        split_idx = random_splitting(labels, split_params, root=root)

    elif split_params.split_type == "k-fold":
        assert type(labels) is not object, (
            "K-Fold splitting not supported for ragged labels."
        )
        split_idx = k_fold_split(labels, split_params, root=root)

    elif split_params.split_type == "fixed" and hasattr(dataset, "split_idx"):
        split_idx = dataset.split_idx

    else:
        raise NotImplementedError(
            f"split_type {split_params.split_type} not valid. Choose either "
            "'random', 'k-fold' or 'fixed'. If 'fixed' is chosen, the dataset "
            "should have the attribute split_idx"
        )

    neg_pos_ratio = float(split_params.get("neg_pos_ratio", 1.0))
    neg_method = split_params.get("neg_sampling_method", "sparse")
    data_seed = int(split_params.get("data_seed", 0))
    neg_sampling_ratio = float(split_params.get("neg_sampling_ratio", 1.0))

    # Helpers to attach edge labels
    def build_pos_edge_labels(data):
        """Attach positive edge labels: all edges are positives."""
        data = data.clone()
        pos_edge_index = data.edge_index
        num_pos = pos_edge_index.size(1)

        data.edge_label_index = pos_edge_index
        data.edge_label = pos_edge_index.new_ones(num_pos)
        return data

    def add_static_negatives(data):
        """Perform one-off negative sampling (used for val/test)."""
        data = build_pos_edge_labels(data)
        device = data.edge_index.device

        pos_edge_index = data.edge_label_index.to(device)
        num_pos = pos_edge_index.size(1)
        if num_pos == 0:
            raise ValueError("Graph has no positive edges for link prediction.")

        num_neg = max(1, int(neg_sampling_ratio * num_pos))

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=num_neg,
            method=neg_method,
        ).to(device)

        pos_label = torch.ones(num_pos, device=device)
        neg_label = torch.zeros(num_neg, device=device)

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat([pos_label, neg_label], dim=0).long()

        data.edge_label_index = edge_label_index
        data.edge_label = edge_label
        return data

    # Build Data lists for each split
    train_indices = split_idx["train"]
    val_indices = split_idx["valid"]
    test_indices = split_idx["test"]

    train_data_list = [
        build_pos_edge_labels(dataset[int(i)]) for i in train_indices
    ]
    val_data_list = [
        add_static_negatives(dataset[int(i)]) for i in val_indices
    ]
    test_data_list = [
        add_static_negatives(dataset[int(i)]) for i in test_indices
    ]

    # Train: dynamic negatives each epoch.
    neg_transform = NegativeSamplingTransform(
        neg_pos_ratio=neg_pos_ratio,
        method=neg_method,
        seed=data_seed,
    )

    dataset_train = DataloadDataset(
        train_data_list, _dynamic_transform=neg_transform
    )
    dataset_val = DataloadDataset(val_data_list)
    dataset_test = DataloadDataset(test_data_list)

    return dataset_train, dataset_val, dataset_test

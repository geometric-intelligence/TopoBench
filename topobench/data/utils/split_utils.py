"""Split utilities."""

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

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
                "train": train_indices.numpy()
                if hasattr(train_indices, "numpy")
                else np.array(train_indices),
                "valid": val_indices.numpy()
                if hasattr(val_indices, "numpy")
                else np.array(val_indices),
                "test": test_indices.numpy()
                if hasattr(test_indices, "numpy")
                else np.array(test_indices),
            }

            # Save generated split
            split_path = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path, **split_idx)

    # Load the split
    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    train_arr = split_idx["train"]
    val_arr = split_idx["valid"]
    test_arr = split_idx["test"]

    all_indices = np.concatenate([train_arr, val_arr, test_arr])
    unique_indices = np.unique(all_indices)

    assert unique_indices.shape[0] == len(labels), (
        f"Not all nodes within splits: {unique_indices.shape[0]} != {len(labels)}"
    )

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

    # Check if this is multi-rank cell prediction
    target_ranks = getattr(dataset, "target_ranks", None)
    if target_ranks is not None and len(target_ranks) > 1:
        return load_multirank_transductive_splits(dataset, parameters)

    # Single rank or node/graph prediction
    labels = data.y.numpy()

    # Check for rank-specific mask (e.g., mask_1 for edges)
    # If present, split only on filtered/valid entities for honest ratios
    rank_mask = None
    valid_indices = None
    target_ranks = getattr(
        dataset, "target_ranks", [1]
    )  # Default to rank 1 for edges

    if target_ranks:
        rank = target_ranks[0]  # Single rank case
        mask_attr = f"mask_{rank}"
        if hasattr(data, mask_attr):
            rank_mask = getattr(data, mask_attr)  # Boolean mask
            valid_indices = torch.where(rank_mask)[
                0
            ]  # Original indices of valid entities
            labels = labels[rank_mask.numpy()]  # Filter to valid entities only

    # Handle multi-dimensional labels (e.g., multi-label classification)
    if len(labels.shape) > 1:
        # Use first column for stratification (common practice)
        stratify_labels = (
            labels[:, 0] if labels.shape[1] > 0 else labels.flatten()
        )
    else:
        stratify_labels = labels

    root = dataset.get_data_dir() if hasattr(dataset, "get_data_dir") else None

    if parameters.split_type == "random":
        splits = random_splitting(stratify_labels, parameters, root=root)

    elif parameters.split_type == "k-fold":
        splits = k_fold_split(stratify_labels, parameters, root=root)

    elif parameters.split_type == "fixed" and hasattr(dataset, "split_idx"):
        splits = dataset.split_idx
        if splits is None:
            raise ValueError(
                "Dataset has split_type='fixed' but split_idx property returned None. "
                "Either the dataset doesn't support fixed splits or they failed to load."
            )

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose 'random', 'k-fold', or 'fixed'.\n"
            f"If 'fixed' is chosen, the dataset must have a split_idx property."
        )

    # Assign train val test masks to the graph
    # If we filtered by rank_mask, map indices back to original positions
    if valid_indices is not None:
        # Splits are indices into filtered data, map back to original
        train_mask = valid_indices[torch.from_numpy(splits["train"])]
        val_mask = valid_indices[torch.from_numpy(splits["valid"])]
        test_mask = valid_indices[torch.from_numpy(splits["test"])]
    else:
        # No filtering: use indices directly
        train_mask = torch.from_numpy(splits["train"])
        val_mask = torch.from_numpy(splits["valid"])
        test_mask = torch.from_numpy(splits["test"])

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    if parameters.get("standardize", False):
        # Standardize the node features respecting train mask
        data.x = (data.x - data.x[data.train_mask].mean(0)) / data.x[
            data.train_mask
        ].std(0)
        data.y = (data.y - data.y[data.train_mask].mean(0)) / data.y[
            data.train_mask
        ].std(0)

    return DataloadDataset([data]), None, None


def get_multilabel_stratification_targets(
    labels: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Generate a single stratification target vector for multi-label data.

    For multi-label classification, uses the index of the most frequent label
    per sample (argmax). This is simpler and more robust than Label Powerset,
    avoiding issues with rare label combinations.

    Parameters
    ----------
    labels : np.ndarray or pd.DataFrame
        The multi-label target array (2D) or vector (1D).
        Can be a NumPy array or Pandas DataFrame.

    Returns
    -------
    np.ndarray
        A 1D array suitable for the 'stratify' parameter in sklearn.
        For 1D input: returns as-is.
        For 2D input: returns argmax (most frequent label index).
    """
    # Standardize input to NumPy array
    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    # Handle 1D arrays (standard classification)
    if labels.ndim == 1:
        return labels

    # Handle 2D arrays (multi-label classification)
    if labels.ndim == 2 and labels.shape[1] > 1:
        # Use argmax: index of most frequent label (or first '1' for binary)
        # This ensures stratification works even with rare label combinations
        return labels.argmax(axis=1)

    # Fallback for 2D arrays with single column
    return labels.flatten()


def load_multirank_transductive_splits(
    dataset, parameters
) -> tuple[list[Any], None, None]:
    r"""Load dataset with multi-rank cell-level splits.

    For datasets with cell-level predictions across multiple ranks (e.g., edges,
    triangles, tetrahedra simultaneously), this function creates independent
    train/val/test splits for each rank on valid entities (filtered by masks)
    using multi-label stratification.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Dataset with multi-rank cell labels.
    parameters : DictConfig
        Configuration parameters containing split_type and train_prop.

    Returns
    -------
    list:
        List containing the train dataset (validation and test are None for transductive).

    Notes
    -----
    Expects dataset to have:
    - target_ranks: list of ranks to split
    - data.cell_labels_{rank}: labels for each rank

    Creates per-rank masks:
    - data.train_mask_{rank}: training indices for rank
    - data.val_mask_{rank}: validation indices for rank
    - data.test_mask_{rank}: test indices for rank
    """
    assert len(dataset) == 1, (
        "Dataset should have only one graph/complex in a transductive setting."
    )

    data = dataset.data_list[0]
    target_ranks = dataset.target_ranks

    root = dataset.get_data_dir() if hasattr(dataset, "get_data_dir") else None

    # Split each rank independently
    for rank in target_ranks:
        label_attr = f"cell_labels_{rank}"

        if not hasattr(data, label_attr):
            raise ValueError(
                f"Data object missing {label_attr} for rank {rank}. "
                f"Available attributes: {list(data.keys())}"
            )

        labels = getattr(data, label_attr).numpy()

        # Check for rank-specific mask
        # If present, split only on filtered entities for honest ratios
        rank_mask = None
        valid_indices = None
        mask_attr = f"mask_{rank}"

        if hasattr(data, mask_attr):
            rank_mask = getattr(data, mask_attr)  # Boolean mask
            valid_indices = torch.where(rank_mask)[
                0
            ]  # Original indices of valid entities
            labels = labels[rank_mask.numpy()]  # Filter to valid entities only

        stratify_labels = get_multilabel_stratification_targets(labels)

        # Create rank-specific root directory for splits
        # This ensures each rank gets independent splits
        rank_root = os.path.join(root, f"rank_{rank}") if root else None

        # Perform splitting
        if parameters.split_type == "random":
            splits = random_splitting(
                stratify_labels, parameters, root=rank_root
            )
        elif parameters.split_type == "k-fold":
            splits = k_fold_split(stratify_labels, parameters, root=root)
        elif parameters.split_type == "fixed" and hasattr(
            dataset, "split_idx"
        ):
            splits = dataset.split_idx
            if splits is None:
                raise ValueError(
                    "Dataset has split_type='fixed' but split_idx property returned None. "
                    "Either the dataset doesn't support fixed splits or they failed to load."
                )
        else:
            raise NotImplementedError(
                f"split_type {parameters.split_type} not valid. "
                f"Choose 'random', 'k-fold', or 'fixed'.\n"
                f"If 'fixed' is chosen, the dataset must have a split_idx property."
            )

        # Store per-rank masks
        # If we filtered by rank_mask, map indices back to original positions
        if valid_indices is not None:
            # Splits are indices into filtered data, map back to original
            train_mask = valid_indices[torch.from_numpy(splits["train"])]
            val_mask = valid_indices[torch.from_numpy(splits["valid"])]
            test_mask = valid_indices[torch.from_numpy(splits["test"])]
        else:
            # No filtering: use indices directly
            train_mask = torch.from_numpy(splits["train"])
            val_mask = torch.from_numpy(splits["valid"])
            test_mask = torch.from_numpy(splits["test"])

        setattr(data, f"train_mask_{rank}", train_mask)
        setattr(data, f"val_mask_{rank}", val_mask)
        setattr(data, f"test_mask_{rank}", test_mask)

    # Assumes DataloadDataset is available in scope
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

    root = dataset.get_data_dir() if hasattr(dataset, "get_data_dir") else None

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

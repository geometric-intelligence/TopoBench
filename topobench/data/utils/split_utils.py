"""Split utilities."""

import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from topobench.dataloader import DataloadDataset


# Generate splits in different fasions


def k_fold_split(labels, parameters, root=None):
    """Return train and valid indices as in K-Fold Cross-Validation.

    If the split already exists it loads it automatically, otherwise it creates the
    split files for subsequent runs.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor. May contain NaN for unlabeled nodes.
    parameters : DictConfig
        Configuration parameters. Must contain:
            - data_split_dir
            - k  (number of folds)
            - data_seed (which fold to use)

    Returns
    -------
    dict
        Dictionary containing the train, validation and test indices,
        with keys "train", "valid", and "test".
        Indices refer to the original node indices (only labeled nodes).
    """

    data_dir = (
        parameters["data_split_dir"]
        if root is None
        else os.path.join(root, "data_splits")
    )
    k = parameters.k
    fold = parameters.data_seed
    assert fold < k, "data_seed needs to be less than k"

    # --- ASSERT: regression (float labels) not supported for stratified splitting ---
    if np.issubdtype(labels.dtype, np.floating):
        raise AssertionError(
            "Regression tasks are not compatible with stratified splitting. "
            "StratifiedKFold requires discrete integer class labels."
        )

    # Mask for labeled nodes (assumes NaN indicates unlabeled)
    mask_labeled = (
        labels >= 0
    )  # Mask of labeled nodes (missing values are huge negative numbers when converted to int)

    # Original indices of labeled nodes
    labeled_indices = np.where(mask_labeled)[0]
    labels_labeled = labels[mask_labeled]
    n_labeled = labeled_indices.shape[0]

    # Directory where all K splits will be stored
    split_dir = os.path.join(data_dir, f"{k}-fold")

    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    # Path for the specific fold we want to use
    split_path = os.path.join(split_dir, f"{fold}.npz")

    # If the current fold file does not exist, (re)generate all K folds
    if not os.path.isfile(split_path):
        # Set seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)

        # Indices in the labeled set (0..n_labeled-1)
        idx_labeled = np.arange(n_labeled)

        # Stratified K-Fold on labeled nodes
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for fold_n, (train_local, valid_local) in enumerate(
            skf.split(idx_labeled, labels_labeled)
        ):
            # Map local indices back to original node indices
            train_idx = labeled_indices[train_local]
            valid_idx = labeled_indices[valid_local]

            # Here we set "test" equal to "valid" for convenience.
            # If you have a separate test set, you can change this logic.
            split_idx = {
                "train": train_idx.astype(np.int64),
                "valid": valid_idx.astype(np.int64),
                "test": valid_idx.astype(np.int64),
            }

            # Sanity check: train + valid should cover all labeled nodes
            all_assigned = np.unique(np.concatenate([train_idx, valid_idx]))
            assert all_assigned.shape[0] == n_labeled, (
                "Not every labeled sample has been assigned in this fold."
            )

            # Save this fold
            split_path_fold = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path_fold, **split_idx)

    # Load the requested fold
    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_file = np.load(split_path)
    split_idx = {
        "train": split_file["train"],
        "valid": split_file["valid"],
        "test": split_file["test"],
    }

    # Final sanity check: all indices refer only to labeled nodes
    all_assigned = np.unique(
        np.concatenate(
            [split_idx["train"], split_idx["valid"], split_idx["test"]]
        )
    )
    assert all_assigned.shape[0] == n_labeled, (
        "Not all labeled nodes are within splits."
    )
    assert np.all(mask_labeled[all_assigned]), (
        "Some unlabeled nodes appear in splits."
    )

    return split_idx


def stratified_splitting(labels, parameters, global_data_seed=42):
    r"""Randomly splits nodes into train/valid/test splits in a stratified fashion.
    Making sure that each split has the same proportion of classes.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor. Può contenere NaN per nodi non etichettati.
    parameters : DictConfig
        Configuration parameter.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict:
        Dictionary containing the train, validation and test indices with keys
        "train", "valid", and "test". Gli indici sono riferiti ai nodi originali.
    """
    fold = parameters["data_seed"]
    data_dir = parameters["data_split_dir"]
    train_prop = parameters["train_prop"]
    val_prop = parameters.get("val_prop", (1 - train_prop) / 2)

    # --- ASSERT: regression (float labels) not supported for stratified splitting ---
    if np.issubdtype(labels.dtype, np.floating):
        raise AssertionError(
            "Regression tasks are not compatible with stratified splitting. "
            "StratifiedKFold requires discrete integer class labels."
        )

    split_dir = os.path.join(
        data_dir,
        f"train_prop={train_prop}_val_prop={val_prop}_global_seed={global_data_seed}",
    )

    # Converting array to numpy to easily handle NaN values
    mask_labeled = (
        labels >= 0
    )  # Mask of labeled nodes (missing values are huge negative numbers when converted to int)
    labeled_indices = np.where(mask_labeled)[0]  # indexes of labeled nodes
    labels_labeled = labels[mask_labeled]  # labels of labeled nodes
    n_labeled = labeled_indices.shape[0]

    generate_splits = False
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
        generate_splits = True

    if generate_splits:
        # generate 10 splits
        for fold_n in range(10):
            fold_seed = global_data_seed + fold_n

            # Train vs (Valid+Test), stratified
            train_idx, val_test_idx, y_train, y_val_test = train_test_split(
                labeled_indices,
                labels_labeled,
                test_size=(1.0 - train_prop),
                stratify=labels_labeled,
                random_state=fold_seed,
                shuffle=True,
            )

            test_size_2 = (1.0 - train_prop - val_prop) / (1.0 - train_prop)

            # Split (Valid+Test) stratified
            val_idx, test_idx, _, _ = train_test_split(
                val_test_idx,
                y_val_test,
                test_size=test_size_2,
                stratify=y_val_test,
                random_state=fold_seed,
                shuffle=True,
            )

            split_idx = {
                "train": train_idx.astype(np.int64),
                "valid": val_idx.astype(np.int64),
                "test": test_idx.astype(np.int64),
            }

            # Save the split
            split_path = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path, **split_idx)

    # Loading the split
    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_file = np.load(split_path)
    split_idx = {
        "train": split_file["train"],
        "valid": split_file["valid"],
        "test": split_file["test"],
    }

    # checking that all labeled nodes are within splits
    all_assigned = np.concatenate(
        [split_idx["train"], split_idx["valid"], split_idx["test"]]
    )

    # All labeled nodes must appear in splits
    assert np.unique(all_assigned).shape[0] == n_labeled, (
        "Not all labeled nodes are within splits"
    )

    # No unlabeled nodes in splits
    assert np.all(mask_labeled[all_assigned]), (
        "Some unlabeled nodes appear in splits"
    )

    return split_idx


def random_splitting(labels, parameters, root=None, global_data_seed=42):
    r"""Randomly splits label into train/valid/test splits.

    Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor. Può contenere NaN per i nodi non etichettati.
    parameters : DictConfig
        Configuration parameter.
    root : str, optional
        Root directory for data splits. Overwrite the default directory.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict:
        Dictionary containing the train, validation and test indices with
        keys "train", "valid", and "test". Gli indici sono riferiti ai nodi
        originali (solo quelli etichettati).
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
    val_prop = parameters.get("val_prop", (1 - train_prop) / 2)

    # Directory where to save the splits
    split_dir = os.path.join(
        data_dir,
        f"train_prop={train_prop}_val_prop={val_prop}_global_seed={global_data_seed}",
    )

    mask_labeled = (
        labels >= 0
    )  # Mask of labeled nodes (missing values are huge negative numbers when converted to int)
    labeled_indices = np.where(mask_labeled)[0]  # index of labeled nodes
    n_labeled = labeled_indices.shape[0]

    generate_splits = False
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
        generate_splits = True

    # Genereting splits
    if generate_splits:
        # Seed globali
        torch.manual_seed(global_data_seed)
        np.random.seed(global_data_seed)

        # Number of sample in train and valid sets
        train_num = int(n_labeled * train_prop)
        valid_num = int(n_labeled * val_prop)

        # Genereate 10 splits
        for fold_n in range(10):
            # Permutation of labeled indices
            perm = np.random.permutation(labeled_indices)

            train_indices = perm[:train_num]
            val_indices = perm[train_num : train_num + valid_num]
            test_indices = perm[train_num + valid_num :]

            split_idx = {
                "train": train_indices.astype(np.int64),
                "valid": val_indices.astype(np.int64),
                "test": test_indices.astype(np.int64),
            }

            # Save the split
            split_path = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path, **split_idx)

    # load the split
    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_file = np.load(split_path)
    split_idx = {
        "train": split_file["train"],
        "valid": split_file["valid"],
        "test": split_file["test"],
    }

    # Check the correctness of the split
    all_assigned = np.concatenate(
        [split_idx["train"], split_idx["valid"], split_idx["test"]]
    )
    assert np.unique(all_assigned).shape[0] == n_labeled, (
        "Not all labeled nodes are within splits"
    )

    # check that no unlabeled nodes are in splits
    assert np.all(mask_labeled[all_assigned]), (
        "Some unlabeled nodes appear in splits"
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

    elif parameters.split_type == "stratified":
        splits = stratified_splitting(labels, parameters)

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

    elif parameters.split_type == "stratified":
        split_idx = stratified_splitting(labels, parameters)

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

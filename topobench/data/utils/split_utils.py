"""Split utilities."""

import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from topobench.data.splits import (
    apply_transductive_split,
    inductive_split_views,
    validate_transductive_masks,
)
from topobench.dataloader import DataloadDataset


def k_fold_split_fixed(labels, parameters, split_idx_list):
    """Return train and valid indices as in K-Fold Cross-Validation.

    If the split already exists it loads it automatically, otherwise it creates the
    split file for the subsequent runs.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameters.
    split_idx_list : dict
        Pre-computed split indices keyed by "train", "valid", and "test",
        each containing one entry per fold.

    Returns
    -------
    dict
        Dictionary containing the train, validation and test indices, with keys "train", "valid", and "test".
    """

    data_dir = parameters.data_split_dir
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
        n = labels.shape[0]
        x_idx = np.arange(n)
        x_idx = np.random.permutation(x_idx)
        labels = labels[x_idx]

        for fold_n in range(len(split_idx_list["train"])):
            split_idx = {
                "train": split_idx_list["train"][fold_n],
                "valid": split_idx_list["valid"][fold_n],
                "test": split_idx_list["test"][fold_n],
            }

            # Check that all nodes/graph have been assigned to some split
            # assert np.all(
            #     np.sort(
            #         np.array(
            #             split_idx["train"]
            #             + split_idx["valid"]
            #         )
            #     )
            #     == np.sort(np.arange(len(labels)))
            # ), "Not every sample has been loaded."
            split_path = os.path.join(split_dir, f"{fold_n}.npz")

            np.savez(split_path, **split_idx)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    # assert (
    #     np.unique(
    #         np.array(
    #             split_idx["train"].tolist()
    #             + split_idx["valid"].tolist()
    #             + split_idx["test"].tolist()
    #         )
    #     ).shape[0]
    #     == labels.shape[0]
    # ), "Not all nodes within splits"

    return split_idx


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


def stratified_splitting(labels, parameters, global_data_seed=42):
    r"""Stratified splits label into train/valid/test splits.

    Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameter.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict:
        Dictionary containing the train, validation and test indices with keys "train", "valid", and "test".
    """
    fold = parameters["data_seed"]
    data_dir = parameters["data_split_dir"]
    train_prop = parameters["train_prop"]
    valid_prop = (1 - train_prop) / 2
    test_prop = valid_prop

    # Create split directory if it does not exist
    split_dir = os.path.join(
        data_dir,
        f"train_prop={train_prop}_global_seed={global_data_seed}_stratified",
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
        n = labels.shape[0]

        indices = np.arange(n)
        # Generate 10 splits
        for fold_n in range(10):
            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=test_prop,
                shuffle=True,
                stratify=labels,
                random_state=fold_n,
            )

            adjusted_valid_prop = valid_prop / (1 - test_prop)

            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=adjusted_valid_prop,
                shuffle=True,
                stratify=labels[train_val_indices],
                random_state=fold_n,
            )

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
    assert (
        np.unique(
            np.array(
                split_idx["train"].tolist()
                + split_idx["valid"].tolist()
                + split_idx["test"].tolist()
            )
        ).shape[0]
        == labels.shape[0]
    ), "Not all nodes within splits"

    return split_idx




def load_transductive_splits(dataset, parameters):
    r"""Load one graph with canonical transductive masks.

    Split algorithms continue to return indices. This boundary converts those
    indices once to full-length boolean masks, validates the node partition,
    and returns one native source graph without materialized phase copies.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset | torch.utils.data.Dataset
        Dataset containing exactly one homogeneous graph.
    parameters : DictConfig
        Split configuration.

    Returns
    -------
    tuple[list, None, None]
        Native singleton source dataset and absent phase-specific datasets.
    """
    if len(dataset) != 1:
        raise ValueError(
            "transductive splitting requires exactly one source graph"
        )

    data = dataset[0]
    labels = data.y.detach().cpu().numpy()
    if labels.ndim != 1:
        raise ValueError("Labels should be one dimensional array")

    wrapped_dataset = getattr(dataset, "dataset", None)
    get_data_dir = getattr(wrapped_dataset, "get_data_dir", None)
    root = get_data_dir() if callable(get_data_dir) else None

    if parameters.split_type == "random":
        splits = random_splitting(labels, parameters, root=root)
        apply_transductive_split(
            data,
            train=splits["train"],
            val=splits["valid"],
            test=splits["test"],
        )
    elif parameters.split_type == "stratified":
        splits = stratified_splitting(labels, parameters)
        apply_transductive_split(
            data,
            train=splits["train"],
            val=splits["valid"],
            test=splits["test"],
        )
    elif parameters.split_type == "k-fold":
        splits = k_fold_split(labels, parameters, root=root)
        holdout = np.asarray(splits["valid"])
        split_point = (len(holdout) + 1) // 2
        apply_transductive_split(
            data,
            train=splits["train"],
            val=holdout[:split_point],
            test=holdout[split_point:],
        )
    elif parameters.split_type == "fixed":
        fixed_masks = (
            data.train_mask,
            data.val_mask,
            data.test_mask,
        )
        if all(
            isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
            for mask in fixed_masks
        ):
            validate_transductive_masks(data)
        else:
            apply_transductive_split(
                data,
                train=fixed_masks[0],
                val=fixed_masks[1],
                test=fixed_masks[2],
            )
    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either "
            "'random', 'stratified', 'k-fold', or 'fixed'"
        )

    if data.x.shape[0] == 0 or not torch.any(data.train_mask):
        raise ValueError("transductive training data must not be empty")

    if parameters.get("standardize", False):
        data.x = (data.x - data.x[data.train_mask].mean(0)) / data.x[
            data.train_mask
        ].std(0)
        data.y = (data.y - data.y[data.train_mask].mean(0)) / data.y[
            data.train_mask
        ].std(0)

    return [data], None, None


def load_inductive_splits(dataset, parameters):
    r"""Load multiple-graph splits as lazy views over one source dataset.

    Fixed splits do not read graph items at split construction time. Split
    algorithms that derive indices from labels inspect labels as required, but
    the returned phases always remain index-backed ``Subset`` views and never
    add redundant graph-level masks.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset | torch.utils.data.Dataset
        Source graph dataset.
    parameters : DictConfig
        Split configuration.

    Returns
    -------
    tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]
        Non-empty train, validation, and test views over ``dataset``.
    """
    if len(dataset) <= 1:
        raise ValueError(
            "inductive splitting requires more than one graph"
        )

    if parameters.split_type == "fixed" and hasattr(dataset, "split_idx"):
        return inductive_split_views(dataset, dataset.split_idx)

    label_list = [data.y.squeeze(0).numpy() for data in dataset]
    label_shapes = [label.shape for label in label_list]
    labels = (
        np.array(label_list, dtype=object)
        if len(set(label_shapes)) > 1
        else np.array(label_list)
    )

    wrapped_dataset = getattr(dataset, "dataset", None)
    get_data_dir = getattr(wrapped_dataset, "get_data_dir", None)
    root = get_data_dir() if callable(get_data_dir) else None

    if parameters.split_type == "random":
        split_idx = random_splitting(labels, parameters, root=root)
    elif parameters.split_type == "stratified":
        split_idx = stratified_splitting(labels, parameters)
    elif parameters.split_type == "k-fold":
        if labels.dtype == object:
            raise ValueError(
                "K-Fold splitting not supported for ragged labels."
            )
        split_idx = k_fold_split(labels, parameters, root=root)
    elif parameters.split_type == "k-fold-fixed":
        split_idx = k_fold_split_fixed(
            labels,
            parameters,
            dataset.split_idx_list,
        )
    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either "
            "'random', 'stratified', 'k-fold' or 'fixed'. If 'fixed' is "
            "chosen, the dataset should have the attribute split_idx"
        )

    return inductive_split_views(dataset, split_idx)


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

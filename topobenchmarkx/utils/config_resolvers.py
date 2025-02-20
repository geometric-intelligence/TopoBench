"""Configuration resolvers for the topobenchmarkx package."""

import os
from collections import defaultdict

import numpy as np


def get_routes_from_neighborhoods(neighborhoods):
    """Get the routes from the neighborhoods.

    Combination of src_rank, dst_rank. ex: [[0, 0], [1, 0], [1, 1], [1, 1], [2, 1]].

    Parameters
    ----------
    neighborhoods : list
        List of neighborhoods of interest.

    Returns
    -------
    list
        List of routes.
    """
    routes = []
    for neighborhood in neighborhoods:
        split = neighborhood.split("-")
        src_rank = int(split[-1])
        r = int(split[0]) if len(split) == 3 else 1
        if "incidence" in neighborhood:
            route = (
                [src_rank, src_rank - r]
                if "down" in neighborhood
                else [src_rank, src_rank + r]
            )
        elif "adjacency" in neighborhood:
            route = [src_rank, src_rank]
        else:
            raise Exception(f"Invalid neighborhood {neighborhood}")

        routes.append(route)
    return routes


def get_default_transform(dataset, model):
    r"""Get default transform for a given data domain and model.

    Parameters
    ----------
    dataset : str
        Dataset name. Should be in the format "data_domain/name".
    model : str
        Model name. Should be in the format "model_domain/name".

    Returns
    -------
    str
        Default transform.
    """
    data_domain, dataset = dataset.split("/")
    model_domain, model = model.split("/")
    # Check if there is a default transform for the dataset at ./configs/transforms/dataset_defaults/
    # If not, use the default lifting transform for the dataset to be compatible with the model
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    model_configs_dir = os.path.join(
        base_dir, "configs", "transforms", "model_defaults"
    )
    dataset_configs_dir = os.path.join(
        base_dir, "configs", "transforms", "dataset_defaults"
    )
    model_defaults = [f.split(".")[0] for f in os.listdir(model_configs_dir)]
    datasets_with_defaults = [
        f.split(".")[0] for f in os.listdir(dataset_configs_dir)
    ]
    if model in model_defaults:
        return f"model_defaults/{model}"
    elif dataset in datasets_with_defaults:
        return f"dataset_defaults/{dataset}"
    else:
        if data_domain == model_domain:
            return "no_transform"
        else:
            return f"liftings/{data_domain}2{model_domain}_default"


def get_required_lifting(data_domain, model):
    r"""Get required transform for a given data domain and model.

    Parameters
    ----------
    data_domain : str
        Dataset domain.
    model : str
        Model name. Should be in the format "model_domain/name".

    Returns
    -------
    str
        Required transform.
    """
    data_domain = data_domain
    model_domain = model.split("/")[0]
    if data_domain == model_domain:
        return "no_lifting"
    else:
        return f"{data_domain}2{model_domain}_default"


def get_monitor_metric(task, metric):
    r"""Get monitor metric for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".
    metric : str
        Name of the metric function.

    Returns
    -------
    str
        Monitor metric.

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if task == "classification" or task == "regression":
        return f"val/{metric}"
    else:
        raise ValueError(f"Invalid task {task}")


def get_monitor_mode(task):
    r"""Get monitor mode for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".

    Returns
    -------
    str
        Monitor mode, either "max" or "min".

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if task == "classification":
        return "max"
    elif task == "regression":
        return "min"
    else:
        raise ValueError(f"Invalid task {task}")


def infer_in_channels(dataset, transforms):
    r"""Infer the number of input channels for a given dataset.

    Parameters
    ----------
    dataset : DictConfig
        Configuration parameters for the dataset.
    transforms : DictConfig
        Configuration parameters for the transforms.

    Returns
    -------
    list
        List with dimensions of the input channels.
    """

    def find_complex_lifting(transforms):
        r"""Find if there is a complex lifting in the complex_transforms.

        Parameters
        ----------
        transforms : List[str]
            List of transforms.

        Returns
        -------
        bool
            True if there is a complex lifting, False otherwise.
        str
            Name of the complex lifting, if it exists.
        """
        if transforms is None:
            return False, None
        complex_transforms = [
            "graph2cell_lifting",
            "graph2simplicial_lifting",
            "graph2combinatorial_lifting",
            "graph2hypergraph_lifting",
            "pointcloud2graph_lifting",
            "pointcloud2simplicial_lifting",
            "pointcloud2combinatorial_lifting",
            "pointcloud2hypergraph_lifting",
            "pointcloud2cell_lifting",
        ]
        for t in complex_transforms:
            if t in transforms:
                return True, t
        return False, None

    def check_for_type_feature_lifting(transforms, lifting):
        r"""Check the type of feature lifting in the dataset.

        Parameters
        ----------
        transforms : DictConfig
            Configuration parameters for the transforms.
        lifting : str
            Name of the complex lifting.

        Returns
        -------
        str
            Type of feature lifting.
        """
        lifting_params_keys = transforms[lifting].keys()
        if "feature_lifting" in lifting_params_keys:
            feature_lifting = transforms[lifting]["feature_lifting"]
        else:
            feature_lifting = "ProjectionSum"

        return feature_lifting

    there_is_complex_lifting, lifting = find_complex_lifting(transforms)
    if there_is_complex_lifting:
        # Get type of feature lifting
        feature_lifting = check_for_type_feature_lifting(transforms, lifting)

        if isinstance(dataset.parameters.num_features, int):
            # Case when the dataset has no edge attributes
            if feature_lifting == "Concatenation":
                return_value = [dataset.parameters.num_features]
                for i in range(2, transforms[lifting].complex_dim + 1):
                    return_value += [int(return_value[-1]) * i]

                return return_value

            else:
                # ProjectionSum feature lifting by default
                return [dataset.parameters.num_features] * transforms[
                    lifting
                ].complex_dim
        else:
            # Case when the dataset has edge attributes
            if not transforms[lifting].preserve_edge_attr:
                if feature_lifting == "Concatenation":
                    return_value = [dataset.parameters.num_features[0]]
                    for i in range(2, transforms[lifting].complex_dim + 1):
                        return_value += [int(return_value[-1]) * i]

                    return return_value

                else:
                    # ProjectionSum feature lifting by default
                    return [dataset.parameters.num_features[0]] * transforms[
                        lifting
                    ].complex_dim

            else:
                return list(dataset.parameters.num_features) + [
                    dataset.parameters.num_features[1]
                ] * (
                    transforms[lifting].complex_dim
                    - len(dataset.parameters.num_features)
                )
    else:
        if isinstance(dataset.parameters.num_features, int):
            return [dataset.parameters.num_features]
        else:
            return [dataset.parameters.num_features[0]]


def infere_num_cell_dimensions(selected_dimensions, in_channels):
    r"""Infer the length of a list.

    Parameters
    ----------
    selected_dimensions : list
        List of selected dimensions. If not None it will be used to infer the length.
    in_channels : list
        List of input channels. If selected_dimensions is None, this list will be used to infer the length.

    Returns
    -------
    int
        Length of the input list.
    """
    if selected_dimensions is not None:
        return len(selected_dimensions)
    else:
        return len(in_channels)


def get_default_metrics(task):
    r"""Get default metrics for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".

    Returns
    -------
    list
        List of default metrics.

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if "classification" in task:
        return ["accuracy", "precision", "recall", "auroc"]
    elif "regression" in task:
        return ["mse", "mae"]
    else:
        raise ValueError(f"Invalid task {task}")


def get_list_element(list, index):
    r"""Get element of a list.

    Parameters
    ----------
    list : list
        List of elements.
    index : int
        Index of the element to get.

    Returns
    -------
    any
        Element of the list.
    """
    return list[index]


def infer_in_khop_feature_dim(dataset_in_channels, max_hop):
    r"""Infer the dimension of the feature vector in the SANN k-hop model.

    Parameters
    ----------
    dataset_in_channels : np.ndarray
        1D array of input channels for the dataset.
    max_hop : int
        Maximum hop distance.

    Returns
    -------
    int :
        Dimension of the feature vector in the SANN k-hop model.
    """

    def compute_recursive_sequence(initial_values, time_steps):
        """Compute the sequence D_k^(t) based on the given recursive formula.

        D_k^(t) = 2 * D_k^(t-1) + D_(k-1)^(t-1) + D_(k+1)^(t-1)

        Parameters
        ----------
        initial_values : np.ndarray
            1D array of initial values for D_k^(0), where k = 0, 1, ..., N-1.
        time_steps : int
            Number of time steps to compute the sequence.

        Returns
        -------
        np.ndarray
            2D array where each row corresponds to D_k^(t) for a specific time step.
        """
        # Initialize the result array
        N = len(initial_values)
        results = np.zeros((time_steps + 1, N))
        results[0] = initial_values  # Set the initial values

        # Iterate over time steps
        for t in range(1, time_steps + 1):
            for k in range(N):
                # Use modular arithmetic to handle boundary conditions (e.g., cyclic boundaries)
                D_k = 2 * results[t - 1][k] if k > 0 else results[t - 1][k]
                D_k_minus_1 = results[t - 1][k - 1] if k - 1 >= 0 else 0
                D_k_plus_1 = results[t - 1][k + 1] if k + 1 < N else 0

                results[t][k] = D_k + D_k_minus_1 + D_k_plus_1

        return results

    result = np.transpose(
        compute_recursive_sequence(dataset_in_channels, max_hop)
    )

    return result.astype(np.int32).tolist()


def infer_in_hasse_graph_agg_dim(
    neighborhoods,
    complex_dim,
    dim_in,
    dim_hidden_graph,
    dim_hidden_node,
    copy_initial,
):
    """Compute which input dimensions need to changed based on if they are the output of a neighborhood.

    Set the list of dimensions as outputs to the hasse graph as a GNN

    Parameters
    ----------
    neighborhoods : List[str]
        List of strings representing the neighborhood.
    complex_dim : int
        Maximum dimension of the complex.
    dim_in : int
        The dataset feature input dimension.
    dim_hidden_graph : int
        The output hidden dimension of the GNN over the Hasse Graph aggregation.
    dim_hidden_node : int
        The output hidden dimension of the GNN over the Hasse Graph for each node.
    copy_initial : bool
        If the initial features should be copied as the 0-th hop.

    Returns
    -------
    np.ndarray
        A 2D array where.
    """
    # TODO, to my understanding this should never change

    neighbor_targets = defaultdict(int)
    routes = get_routes_from_neighborhoods(neighborhoods)

    for _s, t in routes:
        neighbor_targets[t] += 1

    dim_hidden = dim_hidden_graph + dim_hidden_node
    hop_num = (
        int(copy_initial) + 1
    )  # If copy_intial the there are two hops, else just 1
    results = np.zeros(shape=(complex_dim + 1, hop_num))
    if copy_initial:
        # First dimension is always the input dimension
        if isinstance(dim_in, int):
            results.fill(dim_in)
        else:
            results.fill(dim_in[0])

    else:
        results.fill(dim_hidden)

    for i in range(complex_dim + 1):
        results[i][hop_num - 1] = max(1, neighbor_targets[i]) * dim_hidden

    return results.astype(np.int32).tolist()


def set_preserve_edge_attr(model_name, default=True):
    r"""Set the preserve_edge_attr parameter of datasets depending on the model.

    Parameters
    ----------
    model_name : str
        Model name.
    default : bool, optional
        Default value for the parameter. Defaults to True.

    Returns
    -------
    bool
        Default if the model can preserve edge attributes, False otherwise.
    """
    if model_name == "sann":
        return False
    else:
        return default

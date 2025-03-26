"""Configuration resolvers for the topobench package."""

import os

import omegaconf
import torch


def get_default_trainer():
    r"""Get default trainer configuration.

    Returns
    -------
    str
        Default trainer configuration file name.
    """
    return "gpu" if torch.cuda.is_available() else "cpu"


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
    model_domain = model.split("/")[0]
    # Check if there is a default transform for the dataset at ./configs/transforms/dataset_defaults/
    # If not, use the default lifting transform for the dataset to be compatible with the model
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    configs_dir = os.path.join(
        base_dir, "configs", "transforms", "dataset_defaults"
    )
    datasets_with_defaults = [f.split(".")[0] for f in os.listdir(configs_dir)]
    if dataset in datasets_with_defaults:
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
    if (
        task == "classification"
        or task == "regression"
        or task == "multilabel classification"
    ):
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
    if task == "classification" or task == "multilabel classification":
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

    # Make it possible to pass lifting configuration as file path
    if transforms is not None and transforms.keys() == {"liftings"}:
        transforms = transforms.liftings

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
            # Default liftig configurations
            "graph2cell_lifting",
            "graph2simplicial_lifting",
            "graph2combinatorial_lifting",
            "graph2hypergraph_lifting",
            "pointcloud2graph_lifting",
            "pointcloud2simplicial_lifting",
            "pointcloud2combinatorial_lifting",
            "pointcloud2hypergraph_lifting",
            "pointcloud2cell_lifting",
            "hypergraph2combinatorial_lifting",
            # Make it possible to run directly from the folder
            "graph2cell",
            "graph2simplicial",
            "graph2combinatorial",
            "graph2hypergraph",
            "pointcloud2graph",
            "pointcloud2simplicial",
            "pointcloud2combinatorial",
            "pointcloud2hypergraph",
            "pointcloud2cell",
            "hypergraph2combinatorial",
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

        # Check if the dataset.parameters.num_features defines a single value or a list
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
        # Case when the dataset has edge attributes (cells attributes)
        else:
            assert (
                type(dataset.parameters.num_features)
                is omegaconf.listconfig.ListConfig
            ), (
                f"num_features should be a list of integers, not {type(dataset.parameters.num_features)}"
            )
            # If preserve_edge_attr == False
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
            # If preserve_edge_attr == True
            else:
                return list(dataset.parameters.num_features) + [
                    dataset.parameters.num_features[1]
                ] * (
                    transforms[lifting].complex_dim
                    - len(dataset.parameters.num_features)
                )

    # Case when there is no lifting
    elif not there_is_complex_lifting:
        # Check if dataset and model are from the same domain and data_domain is higher-order

        # TODO: Does this if statement ever execute? model_domain == data_domain and data_domain in ["simplicial", "cell", "combinatorial", "hypergraph"]
        # BUT get_default_transform() returns "no_transform" when model_domain == data_domain
        if (
            dataset.loader.parameters.get("model_domain", "graph")
            == dataset.loader.parameters.data_domain
            and dataset.loader.parameters.data_domain
            in ["simplicial", "cell", "combinatorial", "hypergraph"]
        ):
            if isinstance(
                dataset.parameters.num_features,
                omegaconf.listconfig.ListConfig,
            ):
                return list(dataset.parameters.num_features)
            else:
                raise ValueError(
                    "The dataset and model are from the same domain but the data_domain is not higher-order."
                )

        elif isinstance(dataset.parameters.num_features, int):
            return [dataset.parameters.num_features]

        else:
            return [dataset.parameters.num_features[0]]

    # This else is never executed
    else:
        raise ValueError(
            "There is a problem with the complex lifting. Please check the configuration file."
        )


def infer_num_cell_dimensions(selected_dimensions, in_channels):
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


def get_default_metrics(task, metrics=None):
    r"""Get default metrics for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".
    metrics : list, optional
        List of metrics to be used. If None, the default metrics will be used.

    Returns
    -------
    list
        List of default metrics.

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if metrics is not None:
        return metrics
    else:
        if "classification" in task:
            return ["accuracy", "precision", "recall", "auroc"]
        elif "regression" in task:
            return ["mse", "mae"]
        else:
            raise ValueError(f"Invalid task {task}")

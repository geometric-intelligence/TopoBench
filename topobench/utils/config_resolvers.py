"""Configuration resolvers for the topobench package."""

import os

import omegaconf
import torch
from omegaconf import OmegaConf

from topobench.nn.capabilities import (
    validated_edge_attr_mode,
    validated_edge_weight_mode,
    validated_graph_feature_width,
)


def register_all_resolvers():
    """Register every supported custom OmegaConf resolver exactly once."""
    resolvers = {
        "define_task_level": define_task_level,
        "get_default_metrics": get_default_metrics,
        "get_default_trainer": get_default_trainer,
        "get_default_transform": get_default_transform,
        "get_flattened_channels": get_flattened_channels,
        "get_list_element": get_list_element,
        "get_monitor_metric": get_monitor_metric,
        "get_monitor_mode": get_monitor_mode,
        "get_non_relational_out_channels": get_non_relational_out_channels,
        "get_pse_dimensions": get_pse_dimensions,
        "get_fes_dimensions": get_fes_dimensions,
        "get_all_encoding_dimensions": get_all_encoding_dimensions,
        "infer_in_channels": infer_in_channels,
        "infer_list_length": infer_list_length,
        "infer_list_length_plus_one": infer_list_length_plus_one,
        "set_preserve_edge_attr": set_preserve_edge_attr,
        "validated_edge_attr_mode": validated_edge_attr_mode,
        "validated_edge_weight_mode": validated_edge_weight_mode,
    }
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)
    OmegaConf.register_new_resolver(
        "parameter_multiplication",
        lambda x, y: int(int(x) * int(y)),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "get_hop_num_gpse",
        lambda x: int(x) + 1,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "get_hop_num_pses",
        lambda x, y: len(x) + int(y),
        replace=True,
    )
    OmegaConf.register_new_resolver("pid", lambda: os.getpid(), replace=True)


def define_task_level(dataset_task_level, learning_setting):
    r"""Define the task level for a given dataset task level and learning setting.

    Parameters
    ----------
    dataset_task_level : str
        Task level defined in the dataset configuration file.
    learning_setting : str
        Learning setting defined in the dataset split parameters.

    Returns
    -------
    str
        Task level for the model.

    Raises
    ------
    ValueError
        If the dataset task level or learning setting is invalid.
    """
    if dataset_task_level == "node" and learning_setting == "inductive":
        return "node_inductive"
    else:
        return dataset_task_level


def get_flattened_channels(num_nodes, channels):
    r"""Get the output dimension of flattening a feature matrix.

    Parameters
    ----------
    num_nodes : int
        Hidden dimension for the first layer.
    channels : int
        Channel dimension.

    Returns
    -------
    int
        Flatenned cchannels dimension.
    """
    return num_nodes * channels


def get_non_relational_out_channels(num_nodes, channels, task_level):
    r"""Get the output dimension for a non-relational model.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the input graph.
    channels : int
        Channel dimension.
    task_level : int
        Task level for the model.

    Returns
    -------
    int
        Output dimension.
    """
    if task_level == "node":  # node-level task
        return num_nodes * channels
    elif task_level == "graph":  # graph-level task
        return channels
    else:
        raise ValueError(f"Invalid task level {task_level}")


def get_default_trainer():
    r"""Get default trainer configuration.

    Returns
    -------
    str
        Default trainer configuration file name.
    """
    return "gpu" if torch.cuda.is_available() else "cpu"


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
    """Return a same-domain dataset/model default or ``no_transform``."""
    data_domain, dataset_name = dataset.split("/", maxsplit=1)
    model_domain, model_name = model.split("/", maxsplit=1)
    if data_domain != model_domain:
        raise ValueError(
            "Cross-domain lifting is unsupported: "
            f"dataset={data_domain!r}, model={model_domain!r}"
        )

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    default_roots = (
        (
            "model_dataset_defaults",
            f"{model_name}_{dataset_name}",
        ),
        ("dataset_defaults", dataset_name),
        ("model_defaults", model_name),
    )
    for group, selector in default_roots:
        config_path = os.path.join(
            base_dir,
            "configs",
            "transforms",
            group,
            f"{selector}.yaml",
        )
        if os.path.isfile(config_path):
            return f"{group}/{selector}"
    return "no_transform"


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
    if task in ("classification", "regression"):
        return f"val/{metric}"
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
    if task == "regression":
        return "min"
    raise ValueError(f"Invalid task {task}")


def get_pse_dimensions(encodings, parameters):
    r"""Get dimensions of positional or structural encodings.

    Parameters
    ----------
    encodings : list
        List of positional or structural encodings.
    parameters : dict
        Dictionary of parameters for the positional or structural encodings, which should
        contain the key "parameters" with the parameters for each encoding.

    Returns
    -------
    list
        List with dimensions of the positional or structural encodings.
    """
    dimensions = []
    for pse in encodings:
        if pse == "LapPE":
            if parameters[pse].get("include_eigenvalues"):
                dimensions.append(parameters[pse].get("max_pe_dim") * 2)
            else:
                dimensions.append(parameters[pse].get("max_pe_dim"))
        elif pse == "RWSE":
            dimensions.append(parameters[pse].get("max_pe_dim"))
        elif pse == "ElectrostaticPE":
            dimensions.append(7)
        elif pse == "HKdiagSE":
            kernel_param = parameters[pse].get("kernel_param_HKdiagSE")
            # Handle both OmegaConf ListConfig and regular lists/tuples
            if (
                isinstance(kernel_param, (list, tuple))
                or type(kernel_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(kernel_param[1] - kernel_param[0])
            else:
                dimensions.append(kernel_param)
    return dimensions


def get_fes_dimensions(encodings, parameters):
    r"""Get dimensions of feature encodings.

    Parameters
    ----------
    encodings : list
        List of feature encodings.
    parameters : dict
        Dictionary of parameters for the feature encodings.

    Returns
    -------
    list
        List with dimensions of the feature encodings.
    """
    dimensions = []
    for fe in encodings:
        if fe == "HKFE":
            kernel_param = parameters[fe].get("kernel_param_HKFE")
            # Handle both OmegaConf ListConfig and regular lists/tuples
            if (
                isinstance(kernel_param, (list, tuple))
                or type(kernel_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(kernel_param[1] - kernel_param[0])
            else:
                dimensions.append(kernel_param)
        elif fe == "KHopFE":
            # max_hop - 1 because the 0th hop is the features themselves
            dimensions.append(parameters[fe].get("max_hop") - 1)
        elif fe == "PPRFE":
            fe_params = parameters.get(fe, {})
            alpha_param = fe_params.get("alpha_param_PPRFE", [0.1, 10])

            if (
                isinstance(alpha_param, (list, tuple))
                or type(alpha_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(alpha_param[1])
            else:
                dimensions.append(alpha_param)
        elif fe == "SheafConnLapPE":
            dimensions.append(parameters[fe].get("max_pe_dim"))
    return dimensions


def get_all_encoding_dimensions(encodings, parameters):
    r"""Get dimensions of all encodings (PSEs and FEs) in order.

    Parameters
    ----------
    encodings : list
        List of all encodings (both PSEs and FEs).
    parameters : dict
        Dictionary of parameters for all encodings.

    Returns
    -------
    list
        List with dimensions of all encodings in the same order as input.
    """
    dimensions = []
    for enc in encodings:
        # PSE encodings
        if enc == "LapPE":
            if parameters[enc].get("include_eigenvalues"):
                dimensions.append(parameters[enc].get("max_pe_dim") * 2)
            else:
                dimensions.append(parameters[enc].get("max_pe_dim"))
        elif enc == "RWSE":
            dimensions.append(parameters[enc].get("max_pe_dim"))
        elif enc == "ElectrostaticPE":
            dimensions.append(7)
        elif enc == "HKdiagSE":
            kernel_param = parameters[enc].get("kernel_param_HKdiagSE")
            # Handle both OmegaConf ListConfig and regular lists/tuples
            if (
                isinstance(kernel_param, (list, tuple))
                or type(kernel_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(kernel_param[1] - kernel_param[0])
            else:
                dimensions.append(kernel_param)
        # FE encodings
        elif enc == "HKFE":
            kernel_param = parameters[enc].get("kernel_param_HKFE")
            # Handle both OmegaConf ListConfig and regular lists/tuples
            if (
                isinstance(kernel_param, (list, tuple))
                or type(kernel_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(kernel_param[1] - kernel_param[0])
            else:
                dimensions.append(kernel_param)
        elif enc == "KHopFE":
            # max_hop - 1 because the 0th hop is the features themselves
            dimensions.append(parameters[enc].get("max_hop") - 1)
        elif enc == "PPRFE":
            # Safely get parameters, defaulting to empty dict if missing
            enc_params = parameters.get(enc, {})
            # Safely get alpha_param, defaulting to [0.1, 10]
            alpha_param = enc_params.get("alpha_param_PPRFE", [0.1, 10])

            if (
                isinstance(alpha_param, (list, tuple))
                or type(alpha_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(alpha_param[1])
            else:
                dimensions.append(alpha_param)
        elif enc == "SheafConnLapPE":
            dimensions.append(parameters[enc].get("max_pe_dim"))
    return dimensions


def check_pses_in_transforms(transforms):
    r"""Check if there are positional or structural encodings in the transforms.

    Parameters
    ----------
    transforms : DictConfig
        Configuration parameters for the transforms.

    Returns
    -------
    int
       Count of the number of features added by the encodings.
    """
    added_features = 0
    # Single transform
    transform = transforms.get("transform_name", None)
    if transform is not None:
        if transform == "LapPE":
            if transforms.get("include_eigenvalues"):
                added_features += transforms.get("max_pe_dim") * 2
            else:
                added_features += transforms.get("max_pe_dim")
        elif transform == "RWSE" or transform == "SheafConnLapPE":
            added_features += transforms.get("max_pe_dim")
    # Potentially multiple transforms
    for key in transforms:
        if "CombinedPSEs" in key or "encodings" in key:
            for pse in transforms[key].get("encodings", []):
                if pse == "LapPE":
                    if (
                        transforms[key]
                        .get("parameters")
                        .get(pse)
                        .get("include_eigenvalues")
                    ):
                        added_features += (
                            transforms[key]
                            .get("parameters")
                            .get(pse)
                            .get("max_pe_dim")
                            * 2
                        )
                    else:
                        added_features += (
                            transforms[key]
                            .get("parameters")
                            .get(pse)
                            .get("max_pe_dim")
                        )
                elif pse == "RWSE":
                    added_features += (
                        transforms[key]
                        .get("parameters")
                        .get(pse)
                        .get("max_pe_dim")
                    )
                elif pse == "ElectrostaticPE":
                    added_features += 7
                elif pse == "HKdiagSE":
                    kernel_param = (
                        transforms[key]
                        .get("parameters")
                        .get(pse)
                        .get("kernel_param_HKdiagSE")
                    )
                    added_features += (
                        (kernel_param[1] - kernel_param[0])
                        if type(kernel_param)
                        is omegaconf.listconfig.ListConfig
                        else kernel_param
                    )
        elif "LapPE" in key and omegaconf.OmegaConf.is_dict(transforms[key]):
            if transforms[key].get("include_eigenvalues"):
                added_features += transforms[key].get("max_pe_dim") * 2
            else:
                added_features += transforms[key].get("max_pe_dim")
        elif (
            "RWSE" in key or "SheafConnLapPE" in key
        ) and omegaconf.OmegaConf.is_dict(transforms[key]):
            added_features += transforms[key].get("max_pe_dim")
        elif "ElectrostaticPE" in key and omegaconf.OmegaConf.is_dict(
            transforms[key]
        ):
            added_features += 7
        elif "HKdiagSE" in key and omegaconf.OmegaConf.is_dict(
            transforms[key]
        ):
            kernel_param = transforms[key].get("kernel_param_HKdiagSE")
            added_features += (
                (kernel_param[1] - kernel_param[0])
                if type(kernel_param) is omegaconf.listconfig.ListConfig
                else kernel_param
            )

    return added_features


def check_fes_in_transforms(transforms):
    r"""Check if there are feature encodings in the transforms.

    Parameters
    ----------
    transforms : DictConfig
        Configuration parameters for the transforms.

    Returns
    -------
    int
        Count of the number of features added by the encodings.
    """
    added_features = 0
    # Single transform
    transform = transforms.get("transform_name", None)
    if transform is not None:
        if transform == "HKFE":
            kernel_param = transforms.get("kernel_param_HKFE")
            added_features += (
                (kernel_param[1] - kernel_param[0])
                if type(kernel_param) is omegaconf.listconfig.ListConfig
                else kernel_param
            )
        elif transform == "KHopFE":
            # max_hop - 1 because the 0th hop is the features themselves
            added_features += transforms.get("max_hop") - 1
        elif transform == "PPRFE":
            alpha_param = transforms.get("alpha_param_PPRFE")
            if (
                isinstance(alpha_param, (list, tuple))
                or type(alpha_param) is omegaconf.listconfig.ListConfig
            ):
                added_features += alpha_param[1]
            else:
                added_features += alpha_param
        elif transform == "SheafConnLapPE":
            added_features += transforms.get("max_pe_dim")
    # Potentially multiple transforms
    for key in transforms:
        if "CombinedFEs" in key:
            for fe in transforms[key].get("encodings", []):
                if fe == "HKFE":
                    kernel_param = (
                        transforms[key]
                        .get("parameters")
                        .get(fe)
                        .get("kernel_param_HKFE")
                    )
                    added_features += (
                        (kernel_param[1] - kernel_param[0])
                        if type(kernel_param)
                        is omegaconf.listconfig.ListConfig
                        else kernel_param
                    )
                elif fe == "KHopFE":
                    # max_hop - 1 because the 0th hop is the features themselves
                    added_features += (
                        transforms[key]
                        .get("parameters")
                        .get(fe)
                        .get("max_hop")
                        - 1
                    )
                elif fe == "PPRFE":
                    # Safely chain the gets so it never throws an error
                    fe_params = (
                        transforms[key].get("parameters", {}).get(fe, {})
                    )
                    alpha_param = fe_params.get("alpha_param_PPRFE", [0.1, 10])

                    if (
                        isinstance(alpha_param, (list, tuple))
                        or type(alpha_param) is omegaconf.listconfig.ListConfig
                    ):
                        added_features += alpha_param[1]
                    else:
                        added_features += alpha_param
                elif fe == "SheafConnLapPE":
                    added_features += (
                        transforms[key]
                        .get("parameters")
                        .get(fe)
                        .get("max_pe_dim")
                    )
        elif "HKFE" in key and omegaconf.OmegaConf.is_dict(transforms[key]):
            kernel_param = transforms[key].get("kernel_param_HKFE")
            added_features += (
                (kernel_param[1] - kernel_param[0])
                if type(kernel_param) is omegaconf.listconfig.ListConfig
                else kernel_param
            )
        elif "KHopFE" in key and omegaconf.OmegaConf.is_dict(transforms[key]):
            # max_hop - 1 because the 0th hop is the features themselves
            added_features += transforms[key].get("max_hop") - 1
        elif "PPRFE" in key and omegaconf.OmegaConf.is_dict(transforms[key]):
            alpha_param = transforms[key].get("alpha_param_PPRFE")
            if (
                isinstance(alpha_param, (list, tuple))
                or type(alpha_param) is omegaconf.listconfig.ListConfig
            ):
                added_features += alpha_param[1]
            else:
                added_features += alpha_param
        elif "SheafConnLapPE" in key and omegaconf.OmegaConf.is_dict(
            transforms[key]
        ):
            added_features += transforms[key].get("max_pe_dim")
    return added_features


def infer_in_channels(dataset, transforms):
    """Infer and validate one native node-feature width."""
    added_features = 0
    if transforms is not None:
        added_features = check_pses_in_transforms(
            transforms
        ) + check_fes_in_transforms(transforms)

    data_domain = dataset.loader.parameters.data_domain
    if data_domain == "graph":
        return int(
            validated_graph_feature_width(dataset, transforms) + added_features
        )

    num_features = dataset.parameters.num_features
    if isinstance(num_features, int):
        return int(num_features + added_features)
    if isinstance(
        num_features,
        (list, tuple, omegaconf.listconfig.ListConfig),
    ):
        if not num_features:
            raise ValueError("dataset.parameters.num_features cannot be empty")
        return int(num_features[0] + added_features)
    raise TypeError(
        "dataset.parameters.num_features must be an integer or sequence"
    )


def infer_list_length(list):
    r"""Infer the length of a list.

    Parameters
    ----------
    list : list
        List.

    Returns
    -------
    int
        Length of the input list.
    """
    return len(list)


def infer_list_length_plus_one(list):
    r"""Infer the length of a list plus one.

    Parameters
    ----------
    list : list
        List.

    Returns
    -------
    int
        Length of the input list plus one.
    """
    return len(list) + 1


def get_default_metrics(task, num_classes, metrics=None):
    """Return metrics validated against the authoritative evaluator registry."""
    from topobench.evaluator.registry import BUILTIN_METRIC_SPECS

    supported_tasks = {"classification", "regression"}
    if task not in supported_tasks:
        raise ValueError(
            "Supported tasks are exactly: classification, regression"
        )
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer, not a boolean")
    if task == "classification" and num_classes < 2:
        raise ValueError("classification num_classes must be at least 2")
    if task == "regression" and num_classes != 1:
        raise ValueError("regression num_classes must be 1")

    if metrics is None:
        return [
            name
            for name, spec in BUILTIN_METRIC_SPECS.items()
            if task in spec.tasks
            and (not spec.binary_only or num_classes == 2)
        ]

    selected = list(metrics)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate metric names are not allowed")
    for metric in selected:
        if metric not in BUILTIN_METRIC_SPECS:
            raise ValueError(f"Unsupported metric {metric!r} for {task}")
        spec = BUILTIN_METRIC_SPECS[metric]
        if task not in spec.tasks:
            registry_task = "/".join(sorted(spec.tasks))
            raise ValueError(
                f"{metric} is a {registry_task} metric, not a {task} metric"
            )
        if spec.binary_only and num_classes != 2:
            raise ValueError(
                f"metric {metric!r} is available only for binary "
                "classification"
            )
    return selected


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
    if model_name in ["hopse_m", "hopse_g"]:
        return True
    elif model_name in ["sann"]:
        return False
    else:
        return default

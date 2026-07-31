"""Data IO utilities."""

import json

import numpy as np
import pandas as pd
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import SimplicialComplex
from torch_geometric.data import Data

from topobench.data.utils.utils import get_complex_connectivity


def read_ndim_manifolds(
    path,
    dim,
    y_val="betti_numbers",
    neighborhoods=None,
    signed=True,
    slice=None,
):
    """Load MANTRA dataset.

    Parameters
    ----------
    path : str
        Path to the dataset.
    dim : int
        Dimension of the manifolds to load, required to make sanity checks.
    y_val : str, optional
        The triangulation information to use as label. Can be one of ['betti_numbers', 'torsion_coefficients',
        'name', 'genus', 'orientable'] (default: "orientable").
    neighborhoods : list of str, optional
        The connectivity to consider when building the simplicial complex (default: None, which means all).
    signed : bool, optional
        Whether to consider signed incidence matrices (default: True).
    slice : int, optional
        Slice of the dataset to load. If None, load the entire dataset (default: None). Used for testing.

    Returns
    -------
    torch_geometric.data.Data
        Data object of the manifold for the MANTRA dataset.
    """
    # Assert that y_val is one of the valid options
    # for each surface
    if dim == 2:
        assert y_val in [
            "betti_numbers",
            "torsion_coefficients",
            "name",
            "genus",
            "orientable",
        ]
    elif dim == 3:
        assert y_val in ["betti_numbers", "torsion_coefficients", "name"]
    else:
        raise ValueError("Invalid dimension. Only 2 and 3 are supported.")

    TORSION_COEF_NAMES = ["", "Z_2"]
    HOMEO_NAMES = [
        "",
        "Klein bottle",
        "RP^2",
        "S^2",
        "T^2",
        "S^2 twist S^1",
        "S^2 x S^1",
        "S^3",
    ]

    TORSION_COEF_NAME_TO_IDX = {
        name: i for i, name in enumerate(TORSION_COEF_NAMES)
    }
    HOMEO_NAME_TO_IDX = {name: i for i, name in enumerate(HOMEO_NAMES)}

    # Load file
    with open(path) as f:
        manifold_list = json.load(f)

    data_list = []
    # For each manifold
    for manifold in manifold_list[:slice]:
        n_vertices = manifold["n_vertices"]
        x = torch.ones(n_vertices, 1)
        y_value = manifold[y_val]

        if y_val == "betti_numbers":
            y = torch.tensor(y_value, dtype=torch.long).unsqueeze(dim=0)
        elif y_val == "genus":
            y = torch.tensor([y_value], dtype=torch.long).squeeze()
        elif y_val == "torsion_coefficients":
            y = torch.tensor(
                [TORSION_COEF_NAME_TO_IDX[coef] for coef in y_value],
                dtype=torch.long,
            ).unsqueeze(dim=0)
        elif y_val == "name":
            y = torch.tensor(
                [HOMEO_NAME_TO_IDX[y_value]], dtype=torch.long
            ).squeeze(0)
        elif y_val == "orientable":
            y = torch.tensor([y_value], dtype=torch.long).squeeze()
        else:
            raise ValueError(f"Invalid y_val: {y_val}")

        sc = SimplicialComplex()

        # Insert all simplices
        sc.add_simplices_from(manifold["triangulation"])

        # Build the simplex tensors for features, having only a one
        x_i = {
            f"x_{i}": torch.ones(len(sc.skeleton(i)), 1)
            for i in range(dim + 1)
        }

        # Construct the connectivity matrices
        if dim == 2:
            inc_dict = get_complex_connectivity(
                sc, dim + 1, neighborhoods=neighborhoods, signed=signed
            )
            assert inc_dict["incidence_3"].size(1) == 0, (
                "For 2-dim manifolds there shouldn't be any tetrahedrons."
            )
        else:
            inc_dict = get_complex_connectivity(
                sc, dim, neighborhoods=neighborhoods, signed=signed
            )

        inc_dict["edge_index"] = torch.Tensor(
            from_sparse(sc.adjacency_matrix(rank=0)).indices()
        )
        data = Data(x=x, y=y, **x_i, **inc_dict)

        data_list.append(data)
    return data_list


def read_us_county_demos(path, year=2012, y_col="Election"):
    """Load US County Demos dataset.

    Parameters
    ----------
    path : str
        Path to the dataset.
    year : int, optional
        Year to load the features (default: 2012).
    y_col : str, optional
        Column to use as label. Can be one of ['Election', 'MedianIncome',
        'MigraRate', 'BirthRate', 'DeathRate', 'BachelorRate', 'UnemploymentRate'] (default: "Election").

    Returns
    -------
    torch_geometric.data.Data
        Data object of the graph for the US County Demos dataset.
    """
    edges_df = pd.read_csv(f"{path}/county_graph.csv")
    stat = pd.read_csv(
        f"{path}/county_stats_{year}.csv", encoding="ISO-8859-1"
    )

    keep_cols = [
        "FIPS",
        "DEM",
        "GOP",
        "MedianIncome",
        "MigraRate",
        "BirthRate",
        "DeathRate",
        "BachelorRate",
        "UnemploymentRate",
    ]

    # Select columns, replace ',' with '.' and convert to numeric
    stat = stat.loc[:, keep_cols]
    stat["MedianIncome"] = stat["MedianIncome"].replace(",", ".", regex=True)
    stat = stat.apply(pd.to_numeric, errors="coerce")

    # Step 2: Substitute NaN values with column mean
    for column in stat.columns:
        if column != "FIPS":
            mean_value = stat[column].mean()
            stat[column] = stat[column].fillna(mean_value)
    stat = stat[keep_cols].dropna()

    # Delete edges that are not present in stat df
    unique_fips = stat["FIPS"].unique()

    src_ = edges_df["SRC"].apply(lambda x: x in unique_fips)
    dst_ = edges_df["DST"].apply(lambda x: x in unique_fips)

    edges_df = edges_df[src_ & dst_]

    # Remove rows from stat df where edges_df['SRC'] or edges_df['DST'] are not present
    stat = stat[
        stat["FIPS"].isin(edges_df["SRC"]) & stat["FIPS"].isin(edges_df["DST"])
    ]
    stat = stat.reset_index(drop=True)

    # Remove rows where SRC == DST
    edges_df = edges_df[edges_df["SRC"] != edges_df["DST"]]

    # Get torch_geometric edge_index format
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )

    # Make edge_index undirected
    edge_index = torch_geometric.utils.to_undirected(edge_index)

    # Convert edge_index back to pandas DataFrame
    edges_df = pd.DataFrame(edge_index.numpy().T, columns=["SRC", "DST"])

    del edge_index

    # Map stat['FIPS'].unique() to [0, ..., num_nodes]
    fips_map = {fips: i for i, fips in enumerate(stat["FIPS"].unique())}
    stat["FIPS"] = stat["FIPS"].map(fips_map)

    # Map edges_df['SRC'] and edges_df['DST'] to [0, ..., num_nodes]
    edges_df["SRC"] = edges_df["SRC"].map(fips_map)
    edges_df["DST"] = edges_df["DST"].map(fips_map)

    # Get torch_geometric edge_index format
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )

    # Remove isolated nodes (Note: this function maps the nodes to [0, ..., num_nodes] automatically)
    edge_index, _, mask = torch_geometric.utils.remove_isolated_nodes(
        edge_index
    )

    # Convert mask to index
    index = np.arange(mask.size(0))[mask]
    stat = stat.iloc[index]
    stat = stat.reset_index(drop=True)

    # Get new values for FIPS from current index
    # To understand why please print stat.iloc[[516, 517, 518, 519, 520]] for 2012 year
    # Basically the FIPS values have been shifted
    stat["FIPS"] = stat.reset_index()["index"]

    # Create Election variable
    stat["Election"] = (stat["DEM"] - stat["GOP"]) / (
        stat["DEM"] + stat["GOP"]
    )

    # Drop DEM and GOP columns and FIPS
    stat = stat.drop(columns=["DEM", "GOP", "FIPS"])

    # Prediction col
    x_col = list(stat.columns)
    x_col.remove(y_col)

    x = torch.tensor(stat[x_col].to_numpy(), dtype=torch.float32)
    y = torch.tensor(stat[y_col].to_numpy(), dtype=torch.float32)

    data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)

    return data

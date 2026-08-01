"""Configuration file for pytest."""
import os
from pathlib import Path
from omegaconf import OmegaConf

# 1. Register the 'env' resolver for OmegaConf
if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver("env", lambda key, default=None: os.getenv(key, default))

# 2. Set a fallback PROJECT_ROOT so tests don't crash if it's not set in the shell
if "PROJECT_ROOT" not in os.environ:
    # Set PROJECT_ROOT to the directory containing the 'test' folder
    # Assuming: project_root/test/conftest.py
    os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent.resolve())

import pytest
import torch
import torch_geometric
from topobench.data import HypergraphData
from topobench.data.datasets.synthetic_hypergraph_dataset import (
    make_synthetic_hypergraph_data,
)


@pytest.fixture
def mocker_fixture(mocker):
    """Return pytest mocker, used when one want to use mocker in setup_method.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        A pytest mocker.

    Returns
    -------
    pytest_mock.plugin.MockerFixture
        A pytest mocker.
    """
    return mocker


@pytest.fixture
def synthetic_hypergraph() -> HypergraphData:
    """Return a fresh clone of the deterministic production hypergraph."""
    return make_synthetic_hypergraph_data().clone()


@pytest.fixture
def simple_graph_0():
    """Create a manual graph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        A simple graph data object.
    """
    num_nodes = 8
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    edge_list = torch.tensor(
        [
            [0, 0, 0, 2, 2, 2, 3, 5],
            [1, 2, 4, 3, 5, 7, 6, 6],
        ],
        dtype=torch.long,
    )

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=num_nodes,
        y=torch.tensor(y),
    )
    return data

@pytest.fixture
def simple_graph_1():
    """Create a manual graph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        A simple graph data object.
    """
    num_nodes = 8
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    edge_list = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 5, 5],
            [1, 2, 4, 7, 2, 4, 3, 5, 7, 4, 6, 6, 7],
        ],
        dtype=torch.long,
    )

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=num_nodes,
        y=torch.tensor(y),
    )
    return data


@pytest.fixture
def simple_graph_2():
    """Create a manual graph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        A simple graph data object.
    """
    num_nodes = 9
    y = [0, 1, 1, 1, 0, 0, 0, 0, 0]
    edge_list = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 5, 5],
            [1, 2, 3, 4, 8, 2, 3, 4, 3, 5, 6, 8, 4, 6, 6, 7],
        ],
        dtype=torch.long,
    )

    # Generate feature from 0 to 9
    x = (
        torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000, 10000])
        .unsqueeze(1)
        .float()
    )

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=num_nodes,
        y=torch.tensor(y),
    )
    return data

"""Test for the NeighborCellsLoader class."""

import os
import shutil
import pytest
import torch
from hydra import compose

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from topobench.data.preprocessor import PreProcessor
from topobench.data.utils.utils import load_manual_graph
from topobench.data.batching import NeighborCellsLoader
from topobench.run import initialize_hydra

# Define base paths for temporary directories
SIMPLICIAL_PATH = "./graph2simplicial_lifting_test/"
HYPERGRAPH_PATH = "./graph2hypergraph_lifting_test/"


@pytest.fixture(scope="session", autouse=True)
def initialize_hydra_once():
    """Initialize Hydra for the entire test session."""
    initialize_hydra()

@pytest.fixture
def setup_teardown_path():
    """
    Fixture to create and clean up temporary directories for tests.
    
    Ensures a clean state for each test that uses it.
    
    Yields
    -------
    None
        Control is yielded to the test function, allowing it to run.
    """
    # Clean up any existing directories before a test run
    if os.path.isdir(SIMPLICIAL_PATH):
        shutil.rmtree(SIMPLICIAL_PATH)
    if os.path.isdir(HYPERGRAPH_PATH):
        shutil.rmtree(HYPERGRAPH_PATH)

    # Yield control to the test function
    yield

    # Clean up after the test function has run
    if os.path.isdir(SIMPLICIAL_PATH):
        shutil.rmtree(SIMPLICIAL_PATH)
    if os.path.isdir(HYPERGRAPH_PATH):
        shutil.rmtree(HYPERGRAPH_PATH)


@pytest.fixture
def load_preprocessed_data_simplicial(setup_teardown_path):
    """Load and preprocess data for simplicial lifting.
    
    Parameters
    ----------
    setup_teardown_path : fixture
        Fixture to set up and tear down temporary directories.
        
    Returns
    -------
    Data
        The preprocessed simplicial data.
    """
    cfg = compose(config_name="run.yaml",
                  overrides=["dataset=graph/manual_dataset", "model=simplicial/san"],
                  return_hydra_config=True)
    data = load_manual_graph()
    preprocessed_dataset = PreProcessor(data, SIMPLICIAL_PATH, cfg['transforms'])
    return preprocessed_dataset[0]


@pytest.fixture
def load_preprocessed_data_hypergraph(setup_teardown_path):
    """Load and preprocess data for hypergraph lifting.
    
    Parameters
    ----------
    setup_teardown_path : fixture
        Fixture to set up and tear down temporary directories.
    
    Returns
    -------
    Data
        The preprocessed hypergraph data.
    """
    cfg = compose(config_name="run.yaml",
                  overrides=["dataset=graph/manual_dataset", "model=hypergraph/allsettransformer"],
                  return_hydra_config=True)
    data = load_manual_graph()
    preprocessed_dataset = PreProcessor(data, HYPERGRAPH_PATH, cfg['transforms'])
    return preprocessed_dataset[0]


def run_loader_test(data, rank_key, rank, num_neighbors, batch_size):
    """Helper function to run the NeighborCellsLoader test logic.
    
    Parameters
    ----------
    data : Data
        The preprocessed data to be loaded.
    rank_key : str
        The key in the data that corresponds to the rank.
    rank : int
        The rank of the cell type to be tested.
    num_neighbors : list[int]
        The number of neighbors to consider for the loader.
    batch_size : int
        The batch size for the loader.
    """
    n_cells = data[rank_key].shape[0]
    train_prop = 0.5
    n_train = int(train_prop * n_cells)
    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    train_mask[:n_train] = 1

    # Assign a dummy 'y' attribute as it might be expected by the loader or downstream
    data.y = torch.zeros(n_cells, dtype=torch.long)

    loader = NeighborCellsLoader(data,
                                 rank=rank,
                                 num_neighbors=num_neighbors,
                                 input_nodes=train_mask,
                                 batch_size=batch_size,
                                 shuffle=False)

    train_nodes = []
    for batch in loader:
        train_nodes.extend(batch.n_id.tolist())

    for i in range(n_train):
        assert i in train_nodes, f"Node {i} from training mask not found in loaded batches for rank {rank}"


def test_neighbor_cells_loader_simplicial(load_preprocessed_data_simplicial):
    """
    Test NeighborCellsLoader for simplicial data structures.
    
    Parameters
    ----------
    load_preprocessed_data_simplicial : fixture
        Preprocessed simplicial data.
    """
    data = load_preprocessed_data_simplicial
    batch_size = 2

    run_loader_test(data, f'x_{0}', 0, [-1], batch_size)

    run_loader_test(data, f'x_{1}', 1, [-1, -1], batch_size)


def test_neighbor_cells_loader_hypergraph(load_preprocessed_data_hypergraph):
    """
    Test NeighborCellsLoader for hypergraph data structures.
    
    Parameters
    ----------
    load_preprocessed_data_hypergraph : fixture
        Preprocessed hypergraph data.
    """
    data = load_preprocessed_data_hypergraph
    batch_size = 2

    run_loader_test(data, f'x_{0}', 0, [-1], batch_size)

    run_loader_test(data, 'x_hyperedges', 1, [-1, -1], batch_size)
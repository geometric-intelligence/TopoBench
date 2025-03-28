"""Test the Mapper lifting."""
import pytest
import torch
import torch_geometric
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    SVDFeatureReduction,
    ToUndirected,
)

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2hypergraph.mapper_lifting import MapperLifting

expected_edge_incidence = torch.tensor([
    [1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
    [1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1.,
        1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0.]
])


def enriched_manual_graph():
    """Enrich the `load_manual_graph` graph with additional information.

    This function enriches the graph loaded from `load_manual_graph` with
    undirected edges, new node features, and node positions, to facilitate
    testing of filter functions.

    Returns
    -------
    torch_geometric.data.Data
        The enriched graph data object.
    """
    data = load_manual_graph()
    undirected_edges = torch_geometric.utils.to_undirected(data.edge_index)
    new_x = torch.t(
        torch.tensor(
            [
                [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0],
                [-0.5, -2.5, -5.0, -25.0, -50.0, -250.0, -500.0, -2500.0],
            ]
        )
    )
    data.edge_index = undirected_edges
    data.x = new_x
    new_pos = torch.t(
        torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]])
    ).float()
    data.pos = new_pos
    return data


def naive_filter(data, filter):
    """Filter the data using the specified filter function.

    This function applies a filter to the input graph data and returns the
    filtered data. It supports Laplacian eigenvector positional encoding,
    SVD feature reduction, feature sum, position sum, feature PCA, and
    position PCA.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input graph data.
    filter : str
        The filter function to use. Can be "laplacian", "svd",
        "feature_sum", "position_sum", "feature_pca", or "position_pca".

    Returns
    -------
    torch.Tensor
        The filtered data.

    Raises
    ------
    ValueError
        If the specified filter is not supported.
    """
    n_samples = data.x.shape[0]
    if filter == "laplacian":
        transform1 = ToUndirected()
        transform2 = AddLaplacianEigenvectorPE(k=1, is_undirected=True)
        filtered_data = transform2(transform1(data))
        filtered_data = filtered_data["laplacian_eigenvector_pe"]
    elif filter == "svd":
        svd = SVDFeatureReduction(out_channels=1)
        filtered_data = svd(data).x
    elif filter == "feature_sum":
        filtered_data = torch.zeros([n_samples, 1])
        for i in range(n_samples):
            for j in range(data.x.shape[1]):
                filtered_data[i] += data.x[i, j]
    elif filter == "position_sum":
        filtered_data = torch.zeros([n_samples, 1])
        for i in range(n_samples):
            for j in range(data.pos.shape[1]):
                filtered_data[i] += data.pos[i, j]
    elif filter == "feature_pca":
        U, S, V = torch.pca_lowrank(data.x, q=1)
        filtered_data = torch.matmul(data.x, V[:, :1])
    elif filter == "position_pca":
        U, S, V = torch.pca_lowrank(data.pos, q=1)
        filtered_data = torch.matmul(data.pos, V[:, :1])
    else:
        raise ValueError(f"Unsupported filter: {filter}")
    return filtered_data


def naive_cover(filtered_data):
    """Construct a naive cover_mask from filtered data and default lift parameters.

    This function constructs a boolean cover mask based on the filtered data,
    using a set of intervals defined by the range of the data.  It serves
    as a baseline for testing the `cover` method in the `MapperLifting` class.

    Parameters
    ----------
    filtered_data : torch.Tensor
        The data to use to construct the cover.

    Returns
    -------
    torch.Tensor
        The boolean cover mask.
    """
    cover_mask = torch.full((filtered_data.shape[0], 10), False, dtype=torch.bool)
    data_min = torch.min(filtered_data) - 1e-3
    data_max = torch.max(filtered_data) + 1e-3
    data_range = data_max - data_min
    # width of each interval in the cover
    cover_width = data_range / (10 - (10 - 1) * 0.3)
    lows = torch.zeros(10)
    for i in range(10):
        lows[i] = (data_min) + (i) * (1 - 0.3) * cover_width
    highs = lows + cover_width
    # construct boolean cover
    for j, pt in enumerate(filtered_data):
        for i in range(10):
            if (pt > lows[i] or torch.isclose(pt, lows[i])) and (
                pt < highs[i] or torch.isclose(pt, highs[i])
            ):
                cover_mask[j, i] = True
    # delete empty covers
    keep = torch.full([10], True, dtype=torch.bool)
    count_falses = 0
    for i in range(10):
        for j in range(filtered_data.shape[0]):
            if not cover_mask[j, i]:
                count_falses += 1
        if count_falses == filtered_data.shape[0]:
            keep[i] = False
        count_falses = 0
    return torch.t(torch.t(cover_mask)[keep])


class TestMapperLifting:
    """Test the MapperLifting class."""

    def setup(self, filter):
        """Set up the test environment.

        This method sets up the test environment by loading the enriched
        manual graph and initializing the `MapperLifting` class with the
        specified filter.

        Parameters
        ----------
        filter : str
            The filter function to use.
        """
        self.data = enriched_manual_graph()
        self.filter_name = filter
        self.mapper_lift = MapperLifting(filter_attr=filter)

    @pytest.mark.parametrize(
        "filter",
        [
            "laplacian",
            "svd",
            "feature_pca",
            "position_pca",
            "feature_sum",
            "position_sum",
        ],
    )
    def test_filter(self, filter):
        """Test the filter method.

        This method tests the `_filter` method of the `MapperLifting` class
        by comparing its output with the output of the `naive_filter` function.

        Parameters
        ----------
        filter : str
            The filter function to use.
        """
        self.setup(filter)
        lift_filter_data = self.mapper_lift._filter(self.data)
        naive_filter_data = naive_filter(self.data, filter)
        if filter != "laplacian":
            assert torch.all(
                torch.isclose(lift_filter_data, naive_filter_data)
            ), f"Something is wrong with filtered values using {self.filter_name}. The lifted filter data is {lift_filter_data} and the naive filter data is {naive_filter_data}."
        if filter == "laplacian":
            # laplacian filter produces an eigenvector up to a unit multiple.
            # instead we check their absolute values.
            assert torch.all(
                torch.isclose(torch.abs(lift_filter_data), torch.abs(naive_filter_data))
            ), f"Something is wrong with filtered values using {self.filter_name}. The lifted filter data is {lift_filter_data} and the naive filter data is {naive_filter_data}."

    @pytest.mark.parametrize(
        "filter",
        [
            "laplacian",
            "svd",
            "feature_pca",
            "position_pca",
            "feature_sum",
            "position_sum",
        ],
    )
    def test_cover(self, filter):
        """Test the cover method.

        This method tests the `cover` method of the `MapperLifting` class by
        comparing its output with the output of the `naive_cover` function.

        Parameters
        ----------
        filter : str
            The filter function to use.
        """
        self.setup(filter)
        self.mapper_lift.forward(self.data.clone())
        lift_cover_mask = self.mapper_lift.cover
        naive_cover_mask = naive_cover(self.mapper_lift.filtered_data[filter])
        assert torch.all(
            naive_cover_mask == lift_cover_mask
        ), f"Something is wrong with the cover mask using {self.filter_name}. Lifted cover mask is {lift_cover_mask} and naive cover mask {naive_cover_mask}."

    @pytest.mark.parametrize(
        "filter",
        [
            "laplacian",
            "svd",
            "feature_pca",
            "position_pca",
            "feature_sum",
            "position_sum",
        ],
    )
    def test_cluster(self, filter):
        """Test the cluster method.

        This method tests the clustering performed by the `MapperLifting` class by
        comparing the resulting clusters with a set of expected clusters.  It checks
        both the number of clusters and the node subsets within each cluster.

        Parameters
        ----------
        filter : str
            The filter function to use for the Mapper lifting.
        """
        expected_clusters = {
            "laplacian": {
                0: (0, torch.tensor([6.0])),
                1: (1, torch.tensor([3.0])),
                2: (1, torch.tensor([5.0])),
                3: (2, torch.tensor([5.0])),
                4: (3, torch.tensor([7.0])),
                5: (4, torch.tensor([2.0, 7.0])),
                6: (5, torch.tensor([0.0, 1.0, 4.0])),
            },
            "svd": {
                0: (0, torch.tensor([7.0])),
                1: (1, torch.tensor([6.0])),
                2: (2, torch.tensor([5.0, 6.0])),
                3: (3, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
            },
            "feature_pca": {
                0: (0, torch.tensor([7.0])),
                1: (1, torch.tensor([6.0])),
                2: (2, torch.tensor([5.0, 6.0])),
                3: (3, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
            },
            "position_pca": {
                0: (0, torch.tensor([7.0])),
                1: (1, torch.tensor([6.0])),
                2: (2, torch.tensor([5.0])),
                3: (3, torch.tensor([4.0])),
                4: (4, torch.tensor([3.0])),
                5: (5, torch.tensor([2.0])),
                6: (6, torch.tensor([1.0])),
                7: (7, torch.tensor([0.0])),
            },
            "feature_sum": {
                0: (0, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
                1: (1, torch.tensor([5.0, 6.0])),
                2: (2, torch.tensor([6.0])),
                3: (3, torch.tensor([7.0])),
            },
            "position_sum": {
                0: (0, torch.tensor([0.0])),
                1: (1, torch.tensor([1.0])),
                2: (2, torch.tensor([2.0])),
                3: (3, torch.tensor([3.0])),
                4: (4, torch.tensor([4.0])),
                5: (5, torch.tensor([5.0])),
                6: (6, torch.tensor([6.0])),
                7: (7, torch.tensor([7.0])),
            },
        }
        self.setup(filter)
        self.mapper_lift.forward(self.data.clone())
        lift_clusters = self.mapper_lift.clusters
        if filter != "laplacian":
            assert (
                expected_clusters[self.filter_name].keys() == lift_clusters.keys()
            ), f"Different number of clusters using {filter}. Expected {list(expected_clusters[filter])} but got {list(lift_clusters)}."
            for cluster in lift_clusters:
                assert (
                    expected_clusters[self.filter_name][cluster][0]
                    == lift_clusters[cluster][0]
                )
                assert torch.equal(
                    expected_clusters[self.filter_name][cluster][1],
                    lift_clusters[cluster][1],
                ), f"Something is wrong with the clustering using {self.filter_name}. Expected node subset {expected_clusters[self.filter_name][cluster][1]} but got {lift_clusters[cluster][1]} for cluster {cluster}."
        # Laplacian function projects up to a unit. This causes clusters to not be identical by index
        # instead we check if the node subsets of the lifted set are somewhere in the expected set.
        if filter == "laplacian":
            assert len(lift_clusters) == len(
                expected_clusters["laplacian"]
            ), f"Different number of clusters using {filter}. Expected {len(expected_clusters[filter])} clusters but got {len(lift_clusters)}."
            lift_cluster_nodes = [value[1].tolist() for value in lift_clusters.values()]
            expected_cluster_nodes = [
                value[1].tolist() for value in expected_clusters[filter].values()
            ]
            for node_subset in lift_cluster_nodes:
                assert (
                    node_subset in expected_cluster_nodes
                ), f"{node_subset} is a cluster not in {expected_cluster_nodes} but in {lift_cluster_nodes}."
                expected_cluster_nodes.remove(node_subset)
            assert (
                expected_cluster_nodes == []
            ), "Expected clusters contain more clusters than in the lifted cluster."

    @pytest.mark.parametrize(
        "filter",
        [
            "laplacian",
            "svd",
            "feature_pca",
            "position_pca",
            "feature_sum",
            "position_sum",
        ],
    )
    def test_lift_topology(self, filter):
        """Test the lift topology method.

        This method tests the `lift_topology` method by comparing the number of
        hyperedges and the hyperedge incidence matrix with expected values.

        Parameters
        ----------
        filter : str
            The filter function to use.
        """
        expected_lift = {
            "laplacian1": {
                "num_hyperedges": 33,
                "incidence_hyperedges": torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
            "laplacian2": {
                "num_hyperedges": 33,
                "incidence_hyperedges": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
            "svd": {
                "num_hyperedges": 30,
                "incidence_hyperedges": expected_edge_incidence,
            },
            "feature_pca": {
                "num_hyperedges": 30,
                "incidence_hyperedges": expected_edge_incidence,
            },
            "position_pca": {
                "num_hyperedges": 34,
                "incidence_hyperedges": torch.tensor(
                    [[1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                    0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                    [1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1.,
                    1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
                    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                    1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                    0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                    0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]])
            },
            "feature_sum": {
                "num_hyperedges": 30,
                "incidence_hyperedges": torch.tensor(
                    [[1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                    0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1.,
                    1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
                    0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1.,
                    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                    1., 1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                    0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                    0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1.]]),
            },
            "position_sum": {
                "num_hyperedges": 34,
                "incidence_hyperedges": torch.tensor(
                    [[1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                    0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1.,
                    1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
                    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                    1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                    0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                    0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.]]),
            },
        }
        self.setup(filter)
        lifted_topology = self.mapper_lift.forward(self.data.clone())
        if filter != "laplacian":
            assert (
                lifted_topology["num_hyperedges"]
                == expected_lift[self.filter_name]["num_hyperedges"]
            ), f"Different number of hyperedges using {self.filter_name}. Expected {expected_lift[self.filter_name]['num_hyperedges']} but got {lifted_topology['num_hyperedges']}."
            assert torch.all(
                torch.isclose(
                    lifted_topology["incidence_hyperedges"].to_dense(),
                    expected_lift[self.filter_name]["incidence_hyperedges"],
                )
            ), f"Different hyperedge incidence using {self.filter_name}."
        if filter == "position_pca":
            assert (
                lifted_topology["num_hyperedges"]
                == expected_lift["position_pca"]["num_hyperedges"]
            ), f"Different number of hyperedges using {self.filter_name}. Expected {expected_lift['position_pca']['num_hyperedges']} but got {lifted_topology['num_hyperedges']}."
            assert torch.all(
                torch.isclose(
                    torch.abs(lifted_topology["incidence_hyperedges"].to_dense()),
                    torch.abs(expected_lift["position_pca"]["incidence_hyperedges"]),
                )
            ), f"Different hyperedge incidence using {self.filter_name}."
        if filter == "laplacian":
            # The first laplacian transform is "laplacian1" and the second is "laplacian2"
            # due to an eigenvector having two possible projections up to unit scaling.
            # rather than test with 2 filters per each case, simply compute the forward function twice.
            lifted_topology2 = self.mapper_lift.forward(self.data.clone())
            num_hyperedges = (
                lifted_topology["num_hyperedges"] + lifted_topology2["num_hyperedges"]
            )
            assert (
                num_hyperedges == 66
            ), f"Different number of hyperedges using {self.filter_name}. Expected {66} but got {num_hyperedges}."

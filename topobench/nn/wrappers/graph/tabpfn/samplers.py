from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence, Optional, Any
import networkx as nx
from sklearn.neighbors import NearestNeighbors

class BaseSampler(ABC):
    """
    Abstract base class for sampling neighbor indices from training data.
    """

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: np.ndarray, graph: Optional[nx.Graph] = None
    ) -> None:
        """
        Fit the sampler on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training labels.
        graph : networkx.Graph, optional
            Optional graph structure for graph-based samplers.
        """
        pass

    @abstractmethod
    def sample(self, x: np.ndarray, idx: int) -> Sequence[int]:
        """
        Return indices of neighbors for the sample at index `idx` or features `x`.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Feature vector of the sample.
        idx : int
            Index of the sample in the training set (used by graph samplers).

        Returns
        -------
        neighbors : Sequence[int]
            Indices of sampled neighbors.
        """
        pass


class KNNSampler(BaseSampler):
    """
    Sampler that returns k nearest neighbors using sklearn's NearestNeighbors.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._nn: Optional[NearestNeighbors] = None

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        **kwargs: Any
    ) -> None:
        self._nn = NearestNeighbors(n_neighbors=self.k)
        self._nn.fit(X)

    def sample(self, x: np.ndarray, idx: int = -1) -> Sequence[int]:
        if self._nn is None:
            raise RuntimeError("KNNSampler must be fitted before sampling.")
        # Query the k nearest neighbors for the feature vector x
        return self._nn.kneighbors(x.reshape(1, -1), return_distance=False)[0]


class GraphHopSampler(BaseSampler):
    def __init__(self, n_hops: int = 2) -> None:
        self.n_hops = n_hops
        self.graph: Optional[nx.Graph] = None

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        **kwargs: Any
    ) -> None:
        edge_index = kwargs.pop("edge_index", None)
        train_mask = kwargs.pop("train_mask", None)
        if edge_index is None:
            raise ValueError(
                "GraphHopSampler requires an edge_index to be provided in fit()."
            )
        self.graph = nx.Graph()
        self.graph.add_edges_from(edge_index)

        self.train_mask = train_mask
        if self.train_mask is None:
            # Train_mask must be provided
            raise RuntimeError(
                "Train_mask must be provided."
            )

    def sample(self, x: np.ndarray, idx: int, **kwargs: Any) -> Sequence[int]:
        if self.graph is None:
            raise RuntimeError(
                "GraphHopSampler must be fitted with edge_index before sampling."
            )
        if self.graph.has_node(idx) is False:
            # Node index idx does not exist in the graph (has no neighbors in the training graph)
            return []
        close_nodes = nx.single_source_shortest_path_length(
            self.graph, idx, cutoff=self.n_hops
        )
        # Deleting the node itself from the close nodes
        close_nodes.pop(idx, None)
        close_nodes = list(close_nodes.values())

        # Taking only the nodes that are in the training set (we cannot use others label)
        close_nodes = set(close_nodes).intersection(self.train_mask)
        return close_nodes


class CompositeSampler(BaseSampler):
    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop("k", 2)
        self.knn_sampler = KNNSampler(k=self.k)
        self.n_hops = kwargs.pop("n_hops", 2)
        self.graph_hop_sampler = GraphHopSampler(n_hops=self.n_hops)

    def fit(
        self, X: np.ndarray, y: np.ndarray, **kwargs
    ) -> None:
        self.knn_sampler.fit(X, y, **kwargs)
        self.graph_hop_sampler.fit(X, y, **kwargs)

    def sample(self, x: np.ndarray, idx: int, **kwargs: Any) -> Sequence[int]:
        neighbors = []
        # Finding neighbors using the comparing the node features with the knn sampler
        knn_neighbors = self.knn_sampler.sample(x, idx)
        # Adding them to the neighbors list
        neighbors.extend(knn_neighbors)
        for knn_neighbor in knn_neighbors:
            # Finding neighbors using the graph hop sampler sampling from the knn neighbors
            graph_neighbors = self.graph_hop_sampler.sample(
                x, knn_neighbor, **kwargs
            )
            neighbors.extend(graph_neighbors)

        # Adding the direct neighbors of the test node
        # Finding neighbors using the graph hop sampler sampling from the knn neighbors
        direct_graph_neighbors = self.graph_hop_sampler.sample(
            x, idx, **kwargs
        )
        neighbors.extend(direct_graph_neighbors)
        seen = set()
        unique_neighbors = [
            n for n in neighbors if n not in seen and not seen.add(n)
        ]
        return unique_neighbors
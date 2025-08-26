import numpy as np
from typing import Sequence, Any
from topobench.nn.wrappers.graph.tabpfn.samplers.base_sampler import (
    BaseSampler,
)
from topobench.nn.wrappers.graph.tabpfn.samplers.knn_sampler import KNNSampler
from topobench.nn.wrappers.graph.tabpfn.samplers.graph_sampler import (
    GraphHopSampler,
)


class CompositeSampler(BaseSampler):
    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop("k", 2)
        self.knn_sampler = KNNSampler(k=self.k)
        self.n_hops = kwargs.pop("n_hops", 2)
        self.graph_hop_sampler = GraphHopSampler(n_hops=self.n_hops)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
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
                knn_neighbor, **kwargs
            )
            neighbors.extend(graph_neighbors)

        # Adding the direct neighbors of the test node
        # Finding neighbors using the graph hop sampler sampling from the knn neighbors
        direct_graph_neighbors = self.graph_hop_sampler.sample(idx, **kwargs)
        neighbors.extend(direct_graph_neighbors)
        seen = set()
        unique_neighbors = [
            n for n in neighbors if n not in seen and not seen.add(n)
        ]
        return unique_neighbors

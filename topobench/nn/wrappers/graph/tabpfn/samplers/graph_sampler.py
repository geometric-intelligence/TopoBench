import numpy as np
from typing import Optional, Any, Sequence
import networkx as nx
from topobench.nn.wrappers.graph.tabpfn.samplers.base_sampler import (
    BaseSampler,
)


class GraphHopSampler(BaseSampler):
    def __init__(self, n_hops: int = 2, **kwargs) -> None:
        self.n_hops = n_hops
        self.graph: Optional[nx.Graph] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        edge_index = kwargs.pop("edge_index", None)
        train_mask = kwargs.pop("train_mask", None)
        if edge_index is None:
            raise ValueError(
                "GraphHopSampler requires an edge_index to be provided in fit()."
            )
        self.graph = nx.Graph()

        heads = edge_index[0]
        tails = edge_index[1]

        self.graph.add_edges_from(zip(heads, tails))

        self.train_mask = train_mask
        if self.train_mask is None:
            # Train_mask must be provided
            raise RuntimeError("Train_mask must be provided.")

    def sample(self, idx: int, **kwargs: Any) -> Sequence[int]:
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
        close_nodes = list(close_nodes.keys())

        # Taking only the nodes that are in the training set (we cannot use others label)
        close_nodes = set(close_nodes).intersection(list(self.train_mask))
        return list(close_nodes)

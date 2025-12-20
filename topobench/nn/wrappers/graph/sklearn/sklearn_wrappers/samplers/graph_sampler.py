import numpy as np
from typing import Optional, Any, Sequence
import networkx as nx
from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.samplers.base_sampler import (
    BaseSampler,
)


class GraphHopSampler(BaseSampler):
    def __init__(
        self,
        n_hops: int = 2,
        drop: Optional[str] = None,  # {"even", "odd", None}
        **kwargs,
    ) -> None:
        self.n_hops = n_hops
        self.drop_parity = drop
        self.graph: Optional[nx.Graph] = None

        if self.drop_parity not in {None, "even", "odd"}:
            raise ValueError(
                "drop_parity must be one of {None, 'even', 'odd'}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        edge_index = kwargs.pop("edge_index", None)
        train_mask = kwargs.pop("train_mask", None)

        if edge_index is None:
            raise ValueError(
                "GraphHopSampler requires an edge_index to be provided in fit()."
            )

        self.graph = nx.Graph()
        heads, tails = edge_index
        self.graph.add_edges_from(zip(heads, tails))

        if train_mask is None:
            raise RuntimeError("train_mask must be provided.")

        self.train_mask = set(train_mask)

    def _sample(
        self, x: np.ndarray, ids: list, **kwargs: Any
    ) -> Sequence[int]:

        if self.graph is None:
            raise RuntimeError(
                "GraphHopSampler must be fitted before sampling."
            )

        selected_nodes = set()

        for idx in ids:
            if not self.graph.has_node(idx):
                continue

            # dict: node -> hop_distance
            hop_dict = nx.single_source_shortest_path_length(
                self.graph, idx, cutoff=self.n_hops
            )

            for node, hop in hop_dict.items():
                if self.drop_parity == "even" and hop % 2 == 0:
                    continue
                if self.drop_parity == "odd" and hop % 2 == 1:
                    continue

                selected_nodes.add(node)

        # Keep only training nodes
        selected_nodes &= self.train_mask

        return list(selected_nodes)
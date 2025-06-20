from abc import ABC, abstractmethod
from typing import Sequence, Optional, Any, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from copy import deepcopy
import torch


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
        self, X: np.ndarray, y: np.ndarray, edge_index: list = None
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
        self, X: np.ndarray, y: np.ndarray, edge_index: list = None
    ) -> None:
        if edge_index is None:
            raise ValueError(
                "GraphHopSampler requires an edge_index to be provided in fit()."
            )
        self.graph = nx.Graph()
        self.graph.add_edges_from(edge_index)

    def sample(self, x: np.ndarray, idx: int, **kwargs: Any) -> Sequence[int]:
        if self.graph is None:
            raise RuntimeError(
                "GraphHopSampler must be fitted with edge_index before sampling."
            )
        edges_idx_to_train = kwargs.pop(
            "edges_idx_to_train", None
        )  # Ignore edges_idx_to_train if provided
        temp_graph = nx.Graph(
            self.graph
        )  # Ensure we work with a copy of the graph

        if edges_idx_to_train is not None and len(edges_idx_to_train) > 0:
            temp_graph.add_edges_from(edges_idx_to_train)

        if temp_graph.has_node(idx) is False:
            # Node index idx does not exist in the graph (has no neighbors in the training graph)
            return []
        close_nodes = nx.single_source_shortest_path_length(
            temp_graph, idx, cutoff=self.n_hops
        )
        close_nodes.pop(idx, None)

        return list(close_nodes.values())


class CompositeSampler(BaseSampler):
    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop("k", 2)
        self.knn_sampler = KNNSampler(k=self.k)
        self.n_hops = kwargs.pop("n_hops", 2)
        self.graph_hop_sampler = GraphHopSampler(n_hops=self.n_hops)

    def fit(
        self, X: np.ndarray, y: np.ndarray, edge_index: list = None
    ) -> None:
        self.knn_sampler.fit(X, y, edge_index=edge_index)
        self.graph_hop_sampler.fit(X, y, edge_index=edge_index)

    def sample(self, x: np.ndarray, idx: int, **kwargs: Any) -> Sequence[int]:
        neighbors = []
        test_idx_to_train = kwargs.pop(
            "edges_idx_to_train", None
        )  # Ignore edges_idx_to_train if provided
        # Finding neighbors using the comparing the node features with the knn sampler
        knn_neighbors = self.knn_sampler.sample(x, idx)
        # Adding them to the neighbors list
        neighbors.extend(knn_neighbors)
        for knn_neighbor in knn_neighbors:
            links_to_add = []
            # For the knn neighbors, all the edges should be already in the training set
            # Adding only the edges that connect the knn neighbors to the test data (if any)
            for head, tail in test_idx_to_train:
                if head == knn_neighbor or tail == knn_neighbor:
                    links_to_add.append((head, tail))
            # Finding neighbors using the graph hop sampler sampling from the knn neighbors
            graph_neighbors = self.graph_hop_sampler.sample(
                x, knn_neighbor, edges_idx_to_train=links_to_add, **kwargs
            )
            neighbors.extend(graph_neighbors)

        # Adding the direct neighbors of the test node
        # Finding neighbors using the graph hop sampler sampling from the knn neighbors
        direct_graph_neighbors = self.graph_hop_sampler.sample(
            x, idx, test_idx_to_train=test_idx_to_train, **kwargs
        )
        neighbors.extend(direct_graph_neighbors)
        seen = set()
        unique_neighbors = [
            n for n in neighbors if n not in seen and not seen.add(n)
        ]
        return unique_neighbors


class TabPFNWrapper(torch.nn.Module):
    """
    Wrapper for TabPFN that delegates neighbor sampling to a BaseSampler.
    """

    def __init__(
        self,
        backbone: Any,
        # sampler: Any,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.sampler = kwargs.get("sampler", None)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

        # Counters
        self.n_no_neighbors = 0
        self.n_features_constant = 0
        self.n_model_trained = 0

    def fit(
        self, x: np.ndarray, y: np.ndarray, graph: Optional[nx.Graph] = None
    ) -> "TabPFNWrapper":
        # Store training data
        self.X_train = x.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(self.y_train)
        # Fit sampler
        self.sampler.fit(self.X_train, self.y_train, edge_index=graph)
        return self

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # if self.X_train is None or self.y_train is None:
        #    raise RuntimeError("Model has not been fitted. Call `fit` first.")
        train_mask = batch.get("train_mask", None)
        test_mask = batch.get("test_mask", None)

        X_train = batch["x_0"][train_mask].cpu().numpy().copy()
        y_train = batch["y"][train_mask].cpu().numpy().copy()
        # Record unique class labels for probability vectors
        classes_ = np.unique(y_train)
        n_classes = len(classes_)

        heads = batch["edge_index"].cpu().numpy()[0]
        tails = batch["edge_index"].cpu().numpy()[1]

        train_edges = []

        for head, tail in zip(heads, tails):
            if head in train_mask and tail in train_mask:
                train_edges.append((head, tail))

        self.classes_ = np.unique(y_train)
        node_features = batch["x_0"].cpu().numpy()
        # node_pe = [batch[f"x0_i"].cpu().numpy() for i in range(1, 5)]
        labels = batch["y"].cpu().numpy()

        if self.sampler is None:
            # If sample is None training the network on the whole dataset
            self.backbone.fit(X_train, y_train)
            # Predict probabilities for the ehole dataset (to allow compatibility with the rest of the code)
            prob = self.backbone.predict_proba(node_features)
            # Convert probabilities to float tensor to the right device
            prob_tensor = (
                torch.from_numpy(prob).float().to(batch["x_0"].device)
            )
            return {
                "labels": batch["y"],
                "batch_0": batch["batch_0"],
                "x_0": prob_tensor,
            }

        # Fit sampler
        self.sampler.fit(X_train, self.y_train, edge_index=train_edges)

        probs = []
        for idx, x_np in enumerate(node_features):
            if idx in train_mask:
                # If the sample is in the training set, return one-hot encoded probabilities
                one_hot = np.zeros(n_classes, dtype=float)
                one_hot[np.where(classes_ == batch.y[idx].item())[0][0]] = 1.0
                probs.append(torch.tensor(one_hot, dtype=torch.float32))
                continue

            # getting the connection to training nodes
            edges_idx_to_train = []
            for head, tail in zip(heads, tails):
                if (head in train_mask and tail == idx) or (
                    tail in train_mask and head == idx
                ):
                    edges_idx_to_train.append((head, tail))

            # Sample neighbor indices
            nbr_idx = self.sampler.sample(
                x_np, idx, edges_idx_to_train=edges_idx_to_train
            )
            if len(nbr_idx) == 0:
                self.n_no_neighbors += 1
                uniform = np.full(
                    len(self.classes_), 1.0 / len(self.classes_), dtype=float
                )
                probs.append(torch.from_numpy(uniform).float())
                continue

            X_nb = node_features[nbr_idx]
            y_nb = labels[nbr_idx]

            # Remove constant features
            const_cols = np.ptp(X_nb, axis=0) == 0
            if np.all(const_cols):
                self.n_features_constant += 1
                counts = np.array(
                    [np.sum(y_nb == c) for c in self.classes_], dtype=float
                )
                dist = counts / counts.sum()
                probs.append(torch.from_numpy(dist).float())
                continue

            if np.any(const_cols):
                X_nb = X_nb[:, ~const_cols]
                x_filtered = x_np[~const_cols]
            else:
                x_filtered = x_np

            # Fit backbone on neighbors
            model = deepcopy(self.backbone)
            model.fit(X_nb, y_nb)
            self.n_model_trained += 1

            raw_proba = model.predict_proba(x_filtered.reshape(1, -1))[0]
            full_proba = np.zeros(len(self.classes_), dtype=float)
            for i, cls in enumerate(model.classes_):
                global_idx = np.where(self.classes_ == cls)[0][0]
                full_proba[global_idx] = raw_proba[i]
            probs.append(torch.from_numpy(full_proba).float())

        prob_tensor = torch.stack(probs).to(batch["x_0"].device)
        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": prob_tensor,
        }

    def __call__(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.forward(batch)

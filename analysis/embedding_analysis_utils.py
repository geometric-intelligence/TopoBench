"""Core metrics for post-hoc analysis of learned graph embeddings."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _as_numpy(values: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Convert values to a finite CPU NumPy array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return values


def evaluate_kmeans_nmi(
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    seeds: Iterable[int],
) -> pd.DataFrame:
    """Run raw-embedding k-means and return NMI for every seed."""
    values = _as_numpy(embeddings, "embeddings")
    target = _as_numpy(labels, "labels").astype(int)
    n_clusters = len(np.unique(target))
    rows = []
    for seed in seeds:
        prediction = KMeans(
            n_clusters=n_clusters, n_init=1, random_state=seed
        ).fit_predict(values)
        rows.append(
            {
                "seed": seed,
                "nmi": normalized_mutual_info_score(target, prediction),
            }
        )
    return pd.DataFrame(rows)


def compute_knn_purity(
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    k: int = 10,
) -> float:
    """Compute same-class purity among cosine-nearest neighbors."""
    values = _as_numpy(embeddings, "embeddings")
    target = _as_numpy(labels, "labels").astype(int)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(
        values
    )
    indices = neighbors.kneighbors(values, return_distance=False)[:, 1:]
    return float((target[indices] == target[:, None]).mean())


def compute_cross_graph_knn_purity(
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    graph_ids: np.ndarray | torch.Tensor,
    k: int = 10,
    candidate_neighbors: int = 500,
) -> dict[str, float]:
    """Measure purity after excluding neighbors from the query graph."""
    values = _as_numpy(embeddings, "embeddings")
    target = _as_numpy(labels, "labels").astype(int)
    graphs = _as_numpy(graph_ids, "graph_ids").astype(int)
    candidate_neighbors = min(candidate_neighbors + 1, len(values))
    neighbors = NearestNeighbors(
        n_neighbors=candidate_neighbors, metric="cosine"
    ).fit(values)
    indices = neighbors.kneighbors(values, return_distance=False)[:, 1:]
    same_graph_fraction = (graphs[indices[:, :k]] == graphs[:, None]).mean()
    purities = []
    for node, candidates in enumerate(indices):
        cross_graph = candidates[graphs[candidates] != graphs[node]][:k]
        if len(cross_graph) < k:
            raise ValueError(
                f"Node {node} has fewer than {k} cross-graph candidates."
            )
        purities.append((target[cross_graph] == target[node]).mean())
    return {
        "cross_graph_purity": float(np.mean(purities)),
        "same_graph_fraction": float(same_graph_fraction),
    }


def compute_edge_cosine_gap(
    embeddings: np.ndarray | torch.Tensor,
    edge_index: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Compare same-community and boundary-edge cosine similarity."""
    values = _as_numpy(embeddings, "embeddings")
    edges = _as_numpy(edge_index, "edge_index").astype(int)
    target = _as_numpy(labels, "labels").astype(int)
    unit = normalize(values)
    source, destination = edges
    edge_cosine = (unit[source] * unit[destination]).sum(axis=1)
    same_class = target[source] == target[destination]
    if not same_class.any() or same_class.all():
        raise ValueError(
            "Both same- and different-community edges are required."
        )
    same_edge = edge_cosine[same_class].mean()
    different_edge = edge_cosine[~same_class].mean()
    return {
        "same_edge_cosine": float(same_edge),
        "different_edge_cosine": float(different_edge),
        "edge_cosine_gap": float(same_edge - different_edge),
    }

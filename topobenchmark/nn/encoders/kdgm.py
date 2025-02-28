"""KDGM module."""

import torch
from torch import nn


def pairwise_euclidean_distances(x: torch.Tensor, dim: int = -1) -> tuple:
    r"""Compute pairwise Euclidean distances between points in a tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of points. Each row represents a point in a multidimensional space.
    dim : int, optional
        Dimension along which to compute the squared distances.
        Defaults to -1 (last dimension).

    Returns
    -------
    tuple
        A tuple containing two elements:
        - dist (torch.Tensor): Squared pairwise Euclidean distances matrix
        - x (torch.Tensor): The original input tensor
    """
    dist = torch.cdist(x, x) ** 2
    return dist, x


def pairwise_poincare_distances(x: torch.Tensor, dim: int = -1) -> tuple:
    r"""Compute pairwise distances in the Poincarè disk model (Hyperbolic space).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of points. Each row represents a point in a multidimensional space.
    dim : int, optional
        Dimension along which to compute the squared distances.
        Defaults to -1 (last dimension).

    Returns
    -------
    tuple
        A tuple containing two elements:
        - dist (torch.Tensor): Squared pairwise hyperbolic distances matrix
        - x (torch.Tensor): Normalized input tensor in the Poincarè disk
    """
    x_norm = (x**2).sum(dim, keepdim=True)
    x_norm = (x_norm.sqrt() - 1).relu() + 1
    x = x / (x_norm * (1 + 1e-2))
    x_norm = (x**2).sum(dim, keepdim=True)

    pq = torch.cdist(x, x) ** 2
    dist = (
        torch.arccosh(
            1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))
        )
        ** 2
    )
    return dist, x


class DGM_d(nn.Module):
    r"""Distance Graph Matching (DGM) neural network module.

    This class implements a graph matching technique that learns to sample
    edges based on distance metrics in either Euclidean or Hyperbolic space.

    Parameters
    ----------
    base_enc : nn.Module
        Base encoder for transforming input features.
    embed_f : nn.Module
        Embedding function for further feature transformation.
    k : int, optional
        Number of edges to sample in each graph. Defaults to 5.
    distance : str, optional
        Distance metric to use for edge sampling.
        Choices are 'euclidean' or 'hyperbolic'.
        Defaults to 'euclidean'.
    sparse : bool, optional
        Flag to indicate sparse sampling strategy.
        Defaults to True.
    """

    def __init__(
        self, base_enc, embed_f, k=5, distance="euclidean", sparse=True
    ):
        super().__init__()

        self.sparse = sparse
        self.temperature = nn.Parameter(
            torch.tensor(1.0 if distance == "hyperbolic" else 4.0).float()
        )
        self.base_enc = base_enc
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k

        self.debug = False
        if distance == "euclidean":
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor, fixedges=None
    ) -> tuple:
        r"""Forward pass of the Distance Graph Matching module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing node features.
        batch : torch.Tensor
            Batch information for graph-level processing.
        fixedges : torch.Tensor, optional
            Predefined edges to use instead of sampling.
            Defaults to None.

        Returns
        -------
        tuple
            A tuple containing four elements:
            - base_encoded_features (torch.Tensor)
            - embedded_features (torch.Tensor)
            - sampled_edges (torch.Tensor)
            - edge_sampling_log_probabilities (torch.Tensor)
        """
        # Input embedding
        x_ = self.base_enc(x, batch)
        x = self.embed_f(x, batch)

        if self.training:
            if fixedges is not None:
                return (
                    x,
                    fixedges,
                    torch.zeros(
                        fixedges.shape[0],
                        fixedges.shape[-1] // self.k,
                        self.k,
                        dtype=torch.float,
                        device=x.device,
                    ),
                )

            D, _x = self.distance(x)

            # sampling here
            edges_hat, logprobs = self.sample_without_replacement(D)
        else:
            with torch.no_grad():
                if fixedges is not None:
                    return (
                        x,
                        fixedges,
                        torch.zeros(
                            fixedges.shape[0],
                            fixedges.shape[-1] // self.k,
                            self.k,
                            dtype=torch.float,
                            device=x.device,
                        ),
                    )
                D, _x = self.distance(x)

                # sampling here
                edges_hat, logprobs = self.sample_without_replacement(D)

        if self.debug:
            self.D = D
            self.edges_hat = edges_hat
            self.logprobs = logprobs
            self.x = x

        return x_, x, edges_hat, logprobs

    def sample_without_replacement(self, logits: torch.Tensor) -> tuple:
        r"""Sample edges without replacement using a temperature-scaled Gumbel-top-k method.

        Parameters
        ----------
        logits : torch.Tensor
            Input logits representing edge weights or distances.
            Shape should be (n, n) where n is the number of nodes.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - edges (torch.Tensor): Sampled edges without replacement
            - logprobs (torch.Tensor): Log probabilities of the sampled edges
        """
        b = 1
        n, _ = logits.shape
        logits = logits * torch.exp(torch.clamp(self.temperature, -5, 5))

        q = torch.rand_like(logits) + 1e-8
        lq = logits - torch.log(-torch.log(q))
        logprobs, indices = torch.topk(-lq, self.k)

        rows = (
            torch.arange(n)
            .view(1, n, 1)
            .to(logits.device)
            .repeat(b, 1, self.k)
        )
        edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)

        if b == 1:
            edges.squeeze(0)

        return edges, logprobs

"""Diagnostic metrics for (Identity) Sheaf Networks.

This module implements two metrics from the Identity Sheaf Network paper [1]:

* ``class_gain`` / ``heterophily_category`` -- the neighborhood-distribution
  heterophily measure of Wang et al. (2024), used in [1] to explain when an
  Identity Sheaf Network matches learnable Sheaf Neural Networks
  (Definitions 4.1 and 4.2).
* ``rayleigh_quotient`` -- the normalized Dirichlet energy used as an
  oversmoothing measure (Definition 5.2).

These are analysis tools (for the evaluation notebook); they are not part of
the model's forward pass.

[1] Hernandez Caralt et al. "On the Necessity of Learnable Sheaf Laplacians."
GRaM Workshop, ICLR 2026. https://arxiv.org/abs/2603.05395
"""

import torch
import torch_sparse
from torch_geometric.utils import to_undirected


def neighbor_class_matrix(edge_index, y, num_classes=None):
    """
    Aggregate, per class, the class distribution of neighbors.

    For every directed pair ``(u, v)`` the neighbor ``v`` contributes to the
    row of ``u``'s class. The graph is symmetrized first so undirected
    neighborhoods are counted consistently.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    y : torch.Tensor
        Node labels of shape [num_nodes], integer-valued in {0, ..., c-1}.
    num_classes : int, optional
        Number of classes c. Inferred from ``y`` if None. Default is None.

    Returns
    -------
    counts : torch.Tensor
        Matrix of shape [c, c] where ``counts[k, i]`` is the number of
        class-i neighbors aggregated over all class-k nodes.
    class_sizes : torch.Tensor
        Vector of shape [c] with the number of nodes per class.
    """
    y = y.view(-1).long()
    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    edge_index = to_undirected(edge_index)
    row, col = edge_index

    counts = torch.zeros(num_classes, num_classes, device=y.device)
    # counts[class(u), class(v)] += 1 for every edge (u, v)
    idx = y[row] * num_classes + y[col]
    counts.view(-1).index_add_(
        0, idx, torch.ones(idx.size(0), device=y.device)
    )

    class_sizes = torch.bincount(y, minlength=num_classes).float()
    return counts, class_sizes


def class_gain(edge_index, y, num_classes=None):
    """
    Compute the pairwise class-gain matrix (Definition 4.1 of [1]).

    For each class ``k`` let ``m_k`` be the normalized neighbor-class
    distribution and ``d_k`` the average degree of class-k nodes. The gain
    between classes k and t is
    ``gain(k, t) = || sqrt(d_k) * m_k - sqrt(d_t) * m_t ||_1``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    y : torch.Tensor
        Node labels of shape [num_nodes].
    num_classes : int, optional
        Number of classes c. Inferred from ``y`` if None. Default is None.

    Returns
    -------
    torch.Tensor
        Matrix of shape [c, c] with ``gain(k, t)``; the diagonal is zero.

    References
    ----------
    [1] Hernandez Caralt et al. "On the Necessity of Learnable Sheaf
        Laplacians." GRaM Workshop, ICLR 2026.
        https://arxiv.org/abs/2603.05395
    """
    counts, class_sizes = neighbor_class_matrix(edge_index, y, num_classes)

    # Per-class average degree d_k = (total neighbors of class-k nodes) / |k|.
    total_neighbors = counts.sum(dim=1)
    safe_sizes = class_sizes.clamp(min=1)
    d = total_neighbors / safe_sizes

    # Normalized neighbor-class distribution m_k (rows sum to 1).
    safe_totals = total_neighbors.clamp(min=1).unsqueeze(1)
    m = counts / safe_totals

    # scaled_k = sqrt(d_k) * m_k
    scaled = torch.sqrt(d).unsqueeze(1) * m

    # gain(k, t) = L1 distance between scaled rows k and t.
    gain = torch.cdist(scaled, scaled, p=1)
    return gain


def heterophily_category(gain_matrix, threshold=0.2):
    """
    Classify heterophily as good / bad / mixed (Definition 4.2 of [1]).

    Parameters
    ----------
    gain_matrix : torch.Tensor
        Pairwise class-gain matrix of shape [c, c] from :func:`class_gain`.
    threshold : float, optional
        The threshold ``s`` used in the definition. Default is 0.2.

    Returns
    -------
    str
        ``"good"`` if the minimum off-diagonal gain exceeds ``threshold``,
        ``"bad"`` if the maximum off-diagonal gain is below ``threshold``,
        otherwise ``"mixed"``.

    References
    ----------
    [1] Hernandez Caralt et al. "On the Necessity of Learnable Sheaf
        Laplacians." GRaM Workshop, ICLR 2026.
        https://arxiv.org/abs/2603.05395
    """
    c = gain_matrix.size(0)
    if c < 2:
        return "mixed"

    off_diag = gain_matrix[~torch.eye(c, dtype=torch.bool)]
    if off_diag.min().item() > threshold:
        return "good"
    if off_diag.max().item() < threshold:
        return "bad"
    return "mixed"


def rayleigh_quotient(x, laplacian, num_nodes=None, eps=1e-12):
    """
    Compute the Rayleigh quotient / normalized Dirichlet energy (Def 5.2).

    ``R(x) = (x^T L x) / (x^T x)`` for a positive semi-definite operator
    ``L``. For sheaf representations ``x`` is the flattened stalk tensor of
    shape [num_nodes * d, channels] and ``L`` the sheaf Laplacian; for the
    graph case ``L`` is the graph Laplacian.

    Parameters
    ----------
    x : torch.Tensor
        Representation of shape [n, f] (or [n]).
    laplacian : tuple or torch.Tensor
        The operator L, either a sparse ``(indices, values)`` tuple (as
        returned by the Laplacian builders) or a dense [n, n] tensor.
    num_nodes : int, optional
        Size n of the operator. Required when ``laplacian`` is a sparse
        tuple; inferred from a dense tensor otherwise. Default is None.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-12.

    Returns
    -------
    torch.Tensor
        Scalar Rayleigh quotient.

    References
    ----------
    [1] Hernandez Caralt et al. "On the Necessity of Learnable Sheaf
        Laplacians." GRaM Workshop, ICLR 2026.
        https://arxiv.org/abs/2603.05395
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)

    if isinstance(laplacian, tuple):
        indices, values = laplacian
        if num_nodes is None:
            num_nodes = x.size(0)
        lx = torch_sparse.spmm(indices, values, num_nodes, num_nodes, x)
    else:
        lx = laplacian @ x

    numerator = (x * lx).sum()
    denominator = (x * x).sum().clamp(min=eps)
    return numerator / denominator

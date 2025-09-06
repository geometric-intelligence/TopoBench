# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

from scipy import sparse as sp
from torch_geometric.data import Data


def permute_graph(graph: Data, P: np.ndarray) -> Data:
    assert graph.edge_attr is None

    # Check validity of permutation matrix
    n = graph.x.size(0)
    if not is_valid_permutation_matrix(P, n):
        raise AssertionError

    # Apply permutation to features
    x = graph.x.numpy()
    x_perm = torch.FloatTensor(P @ x)

    # Apply perm to labels, if per-node
    if graph.y is None:
        y_perm = None
    elif graph.y.size(0) == n:
        y = graph.y.numpy()
        y_perm = torch.tensor(P @ y)
    else:
        y_perm = graph.y.clone().detach()

    # Apply permutation to adjacencies, if any
    if graph.edge_index.size(1) > 0:
        inps = (np.ones(graph.edge_index.size(1)), (graph.edge_index[0].numpy(), graph.edge_index[1].numpy()))
        A = sp.csr_matrix(inps, shape=(n, n))
        P = sp.csr_matrix(P)
        A_perm = P.dot(A).dot(P.transpose()).tocoo()
        edge_index_perm = torch.LongTensor(np.vstack((A_perm.row, A_perm.col)))
    else:
        edge_index_perm = graph.edge_index.clone().detach()

    # Instantiate new graph
    graph_perm = Data(x=x_perm, edge_index=edge_index_perm, y=y_perm)

    return graph_perm


def is_valid_permutation_matrix(P: np.ndarray, n: int):
    valid = True
    valid &= P.ndim == 2
    valid &= P.shape[0] == n
    valid &= np.all(P.sum(0) == np.ones(n))
    valid &= np.all(P.sum(1) == np.ones(n))
    valid &= np.all(P.max(0) == np.ones(n))
    valid &= np.all(P.max(1) == np.ones(n))
    if n > 1:
        valid &= np.all(P.min(0) == np.zeros(n))
        valid &= np.all(P.min(1) == np.zeros(n))
        valid &= not np.array_equal(P, np.eye(n))
    return valid


def generate_permutation_matrices(size, amount=10):
    Ps = list()
    random_state = np.random.RandomState()
    count = 0
    while count < amount:
        I = np.eye(size)
        perm = random_state.permutation(size)
        P = I[perm]
        if is_valid_permutation_matrix(P, size):
            Ps.append(P)
            count += 1

    return Ps
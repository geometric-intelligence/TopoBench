"""A transform that allows to derive PE and Structural Encodings of the input graph with predefined neighbourhood."""
from copy import deepcopy
from typing import List

import torch_geometric
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import networkx as nx

#from graphgym.transform.cycle_counts import count_cycles


EPS = 1e-6  # values below which we consider as zeros

PE_TYPES = [
    "ElstaticPE",
    "EquivStableLapPE",
    "HKdiagSE",
    "HKfullPE",
    "LapPE",
    "RWSE",
    "SignNet",
    "GPSE",
    "GraphLog",
    "CombinedPSE",
    "BernoulliRE",
    "NormalRE",
    "NormalFixedRE",
    "UniformRE",
]
RANDSE_TYPES = [
    "NormalSE",
    "UniformSE",
    "BernoulliSE",
]
GRAPH_ENC_TYPES = [
    "EigVals",
    "CycleGE",
    "RWGE",
]
ALL_ENC_TYPES = PE_TYPES + RANDSE_TYPES + GRAPH_ENC_TYPES


class DerivePS(torch_geometric.transforms.BaseTransform):
    r"""A transform that converts the node features of the input graph to float.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "derive_positional_structural_encodings"
        self.kwargs = kwargs
        self.device = (
            "cpu" if kwargs["device"] == "cpu" else f"cuda:{kwargs['cuda'][0]}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        # Working RWSE, ElstaticPE, 'LapPE', HKdiagSE
        # Graph level working: CycleGE
        # kwargs = {'posenc_LapPE_eigen_max_freqs':4,
        #           "posenc_LapPE_eigen_eigvec_norm": "L2",
        #           "posenc_LapPE_eigen_skip_zero_freq": True,
        #           "posenc_LapPE_eigen_eigvec_abs": True}
        
        output_pe = compute_posenc_stats(data=data, **self.kwargs)
        pes = [output_pe[key].to(self.device) for key in self.kwargs['pe_types']]
        pes = [self.pad_sequence(seq, target_dim=self.kwargs['target_pe_dim']) for seq in pes]
        pes = torch.stack(pes, dim=0)
        return pes
    def pad_sequence(self, sequence, target_dim, pad_value=0):
        """
        Pads a sequence of vectors along the second dimension to the desired size.
        
        Args:
            sequence (torch.Tensor): The input tensor of shape (17, current_dim).
            target_dim (int): The desired size of the second dimension.
            pad_value (float, optional): The value to use for padding. Default is 0.

        Returns:
            torch.Tensor: The padded tensor.
        """
        current_dim = sequence.size(1)
        if current_dim >= target_dim:
            return sequence[:target_dim]
        
        pad_size = [sequence.size(0), target_dim - current_dim]
        pad_tensor = torch.full(pad_size, pad_value, dtype=sequence.dtype, device=self.device)
        
        return torch.cat([sequence, pad_tensor], dim=1)


def compute_posenc_stats(data, pe_types, **kwargs):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticPE': Kernel based on the electrostatic interaction between nodes.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # _check_all_types(pe_types)
    output_pe = {}
    # Basic preprocessing of the input graph.
    
    N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    assert N > 0, "Graph must have at least one node."

    laplacian_norm_type = kwargs['laplacian_norm_type'] 
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    
    # PE works with undirected graphs
    if data.is_undirected():
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        # if cfg.dataset.name.startswith("ogbn"):
        if (kwargs.get("posenc_LapPE_eigen_max_freqs", None) is not None) and (kwargs["posenc_LapPE_eigen_max_freqs"] > L.shape[0]):
            evals, evects = scipy.sparse.linalg.eigsh(L, k=kwargs["posenc_LapPE_eigen_max_freqs"], which='SM')
        else:
            evals, evects = np.linalg.eigh(L.toarray())
            

        if 'LapPE' in pe_types:
            max_freqs = kwargs.get("posenc_LapPE_eigen_max_freqs", None)
            eigvec_norm = kwargs.get("posenc_LapPE_eigen_eigvec_norm", None) 
            skip_zero_freq = kwargs.get("posenc_LapPE_eigen_skip_zero_freq", True) 
            eigvec_abs = kwargs.get("posenc_LapPE_eigen_eigvec_abs", True) 
        
        EigVals, EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm,
            skip_zero_freq=skip_zero_freq,
            eigvec_abs=eigvec_abs)
        # hstack
        output_pe["LapPE"] = torch.hstack((EigVals.squeeze(-1), EigVecs))

    # Random Walks. 
    if 'RWSE' in pe_types:
        kernel_param_times = range(kwargs["kernel_param_RWSE"][0],kwargs["kernel_param_RWSE"][1])  # if no self-loop, then RWSE1 will be all zeros #cfg.posenc_RWSE.kernel
        if len(kernel_param_times) == 0:
            raise ValueError("List of kernel times required for RW")
        #TODO: Add self-loops to data.edge_index to avoid zeros in rw_landing (issue for interrank case!)
        edge_index = torch_geometric.utils.add_remaining_self_loops(data.edge_index)[0]

        rw_landing = get_rw_landing_probs(ksteps=kernel_param_times,
                                          edge_index=edge_index,#data.edge_index,
                                          num_nodes=N)
        # check that obtained pe are not all zeros
        if torch.all(rw_landing==0) == True:
            raise ValueError("RWSE is all zeros")
        
        if torch.any(torch.isnan(rw_landing)) == True:
            raise ValueError("RWSE contains NaNs")
        
        output_pe["RWSE"] = rw_landing
        
    # Electrostatic interaction inspired kernel.
    if 'ElstaticPE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        
        if torch.all(elstatic==0) == True:
            raise ValueError("ElstaticPE is all zeros")
        
        if torch.any(torch.isnan(elstatic)) == True: 
            raise ValueError("ElstaticPE contains NaNs")
        output_pe["ElstaticPE"] = elstatic
    
    if 'CycleGE' in pe_types:
        start, end = kwargs['kernel_param_CycleGE'][0], kwargs['kernel_param_CycleGE'][1]
        kernel_param_times = list(range(start, end)) #cfg.graphenc_CycleGE.kernel
        cycle_se = count_cycles(data.edge_index, data.num_nodes,kernel_param_times)
        try:
            cycle_se_original = count_cycles_original(kernel_param_times, data)
            assert cycle_se_original==cycle_se
        except:
            pass
    
        if torch.all(cycle_se==0) == True:
            raise ValueError("CycleGE is all zeros")
        
        if torch.any(torch.isnan(cycle_se)) == True:
            raise ValueError("CycleGE contains NaNs")
        output_pe["CycleGE"] = cycle_se

    # Heat Kernels.
    if 'HKdiagSE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        if laplacian_norm_type is not None or evals is None or evects is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = evals, evects
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        if 'HKdiagSE' in pe_types:
            start, end = kwargs['kernel_param_HKdiagSE'][0], kwargs['kernel_param_HKdiagSE'][1]
            kernel_param_times = range(start, end)
            
            if len(kernel_param_times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param_times,
                                            space_dim=0)
            
            if torch.all(hk_diag==0) == True:
                raise ValueError("HKdiagSE is all zeros")
            
            if torch.any(torch.isnan(hk_diag)) == True:
                raise ValueError("HKdiagSE contains NaNs")

            output_pe["HKdiagSE"] = hk_diag

    return output_pe


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2',
                         skip_zero_freq: bool = True, eigvec_abs: bool = False):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
        skip_zero_freq: Start with first non-zero frequency eigenpairs if
            set to True. Otherwise, use first max_freqs eigenpairs.
        eigvec_abs: Use the absolute value of the eigenvectors if set to True.
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = evects.shape[0]  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    offset = (abs(evals) < EPS).sum().clip(0, N) if skip_zero_freq else 0
    idx = evals.argsort()[offset:max_freqs + offset]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs + offset:
        EigVecs = F.pad(evects, (0, max_freqs + offset - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs + offset:
        EigVals = F.pad(evals, (0, max_freqs + offset - N),
                        value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)
    EigVecs = EigVecs.abs() if eigvec_abs else EigVecs

    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * ((L.diag()+1e-6) ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    evals, evecs = torch.linalg.eigh(L)
    offset = (evals < EPS).sum().item()
    if offset == num_nodes:
        return torch.zeros(num_nodes, 7, dtype=torch.float32)

    electrostatic = evecs[:, offset:] / evals[offset:] @ evecs[:, offset:].T
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization in ["L1", "L2", "abs-max", "min-max"]:
        return normalizer(EigVecs, normalization, eps)

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def normalizer(x: torch.Tensor, normalization: str = "L2", eps: float = 1e-12):
    if normalization == "none":
        return x

    elif normalization == "L1":
        # L1 normalization: vec / sum(abs(vec))
        denom = x.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: vec / sqrt(sum(vec^2))
        denom = x.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: vec / max|vec|
        denom = torch.max(x.abs(), dim=0, keepdim=True).values

    elif normalization == "min-max":
        # MinMax normalization: (vec - min(vec)) / (max(vec) - min(vec))
        x = x - x.min(dim=0, keepdim=True).values
        denom = x.max(dim=0, keepdim=True).values

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    return x / denom.clamp_min(eps).expand_as(x)



## Count cycles from graphgym
import logging
from typing import List, Optional, Set, Tuple

import networkx as nx
import torch
from torch import Tensor


class HamiltonianCycle:
    """Hamiltonian cycle object for that considers invariances of cycles.

    The main goal of this object is to reduce a set of Hamiltonian cycles, each
    in the form of a list of unique node indices, into a unique set of
    Hamiltonian cycles. In particular, there are two types of invariances to be
    considered:

        1. Shift invariance. For example, [0, 1, 2, 3] is considered the same
           as [1, 2, 3, 0].
        2. Reflection invariance. For example, [0, 1, 2, 3] is considered the
           the same as [0, 3, 2, 1].

    To efficiently deal with these two invariances, the :obj:`HamiltonianCycle`
    object stores the path list in a reduced format that follows the following
    two conventions:

        1. The first node index must be the smallest among all indices in the
           path list. If not, we apply a shift operation so that the path list
           start with the smallest node index.
        2. The second node index must be no smaller than the last node index.
           If not, we apply a reflection operation so that the second node
           index in the path list is no smaller than the last node index.

    Example:

        >>> path_list = [10, 4, 1, 6, 2]
        >>> print(HamiltonianCycle(path_list))
        (1, 4, 10, 2, 6)

    """

    def __init__(self, path: List[int]):
        self.data = path

    @property
    def reduced_repr(self) -> Tuple[int]:
        return self._reduced_repr

    @property
    def data(self) -> List[int]:
        return self._data

    @data.setter
    def data(self, val: List[int]):
        if not isinstance(val, list):
            raise TypeError(f"path must be a list of integers, got {type(val)}")
        elif len(set(val)) != len(val):
            raise ValueError(f"path must contain unique elements, got {val}")
        elif len(val) < 2:
            raise ValueError(f"path must be at least of size two, got {val}")
        else:
            self._data = val
            self._reduced_repr = self._get_reduced_repr(val)

    @staticmethod
    def _get_reduced_repr(path: List[int]) -> Tuple[int]:
        min_val = min(path)
        min_val_idx = path.index(min_val)

        path = path[min_val_idx:] + path[:min_val_idx]

        if path[-1] < path[1]:
            path = path[:1] + path[-1:0:-1]

        return tuple(path)

    def __len__(self) -> int:
        return len(self._reduced_repr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.__str__()}"

    def __str__(self) -> str:
        return str(self._reduced_repr)

    def __hash__(self) -> int:
        return hash(self._reduced_repr)

    def __eq__(self, other) -> bool:
        if not isinstance(other, HamiltonianCycle):
            raise TypeError("A HamiltonianCycle object can only be compared "
                            "against another HamiltonianCycle object, got "
                            f"{type(other)}")
        return self._reduced_repr == other._reduced_repr

    def __ne__(self, other) -> bool:
        return not self._reduced_repr.__eq__(other._reduced_repr)


def get_all_k_hamcycles(
    edge_index: Tensor,
    num_nodes: int,
    k: int,
    exact: bool = True,
) -> Set[HamiltonianCycle]:
    """Get unique length k Hamiltonian cycles in the graph.

    Args:
        edge_index: COO representation of the adjacency matrix.
        num_nodes: Total number of nodes in the graph.
        k: Target length of the Hamitonian cycles.
        exact: If set to :obj:`True`, then only return Hamiltonian cycles
            *exactly* of length :obj:`k`. Otherwise, return all Hamiltonian
            cycles round the seed node up to, and including, length :obj:`k`.
            Note that computaiton complexities are exactly the same regardless
            of whether :obj:`exact` is set to :obj:`True` or :obj:`False`.

    """
    # NOTE: force graph to be undirected.
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.detach().clone().cpu().T.numpy())

    all_k_hamcycles = set()
    for vi in range(num_nodes):
        k_hamcycles_around_vi = dfs_k_hamcycles(g, k, vi, exact=exact)
        all_k_hamcycles.update(k_hamcycles_around_vi)

    return all_k_hamcycles


def dfs_k_hamcycles(
    g: nx.Graph,
    k: int,
    seed: int,
    *,
    exact: bool = True,
    _cur_depth: int = 0,
    _cur_path: Optional[List[int]] = None,
    _paths: Optional[Set[HamiltonianCycle]] = None,
) -> Set[HamiltonianCycle]:
    """DFS all Hamiltonian cycles up to length k starting from the seed node.

    Args:
        g: Input graph (node id are assumed to be of type integer, representing
            their corresponding node indices).
        k: Target length of the Hamitonian cycles.
        seed: Seed node id.
        exact: If set to :obj:`True`, then only return Hamiltonian cycles
            *exactly* of length :obj:`k`. Otherwise, return all Hamiltonian
            cycles round the seed node up to, and including, length :obj:`k`.
            Note that computaiton complexities are exactly the same regardless
            of whether :obj:`exact` is set to :obj:`True` or :obj:`False`.

    Returns:
        Set[HamiltonianCycle]: A set of unique Hamiltonian cycles of length k
            around the seed node.

    """
    if (not isinstance(k, int)) or (k < 2):
        raise ValueError(f"k (target depth) must be an int > 1, got {k=!r}")

    _cur_path = _cur_path if _cur_path is not None else []
    _paths = _paths if _paths is not None else set()
    logging.debug(f"{_paths=}, {_cur_depth=}, {_cur_path=}, {seed=}")

    if _cur_path and (seed == _cur_path[0]):
        if (_cur_depth == k) or (not exact):
            _paths.add(HamiltonianCycle(_cur_path))

    elif (_cur_depth < k) and (seed not in _cur_path):
        next_depth = _cur_depth + 1
        next_path = _cur_path + [seed]

        for next_seed in g[seed]:
            dfs_k_hamcycles(g, k, next_seed, exact=exact, _cur_depth=next_depth,
                            _cur_path=next_path, _paths=_paths)

    return _paths


def count_cycles_original(k_list: List[int], data):
    """Count all cycles of length exactly k for k provided in the list."""
    if not isinstance(k_list, list) or not k_list:
        raise ValueError("k_list must be a non-empty list of integers, "
                         f"got {k_list=!r}")

    hamcycles = get_all_k_hamcycles(data.edge_index, data.num_nodes,
                                    max(k_list), False)

    count_dict = {k: 0 for k in k_list}
    for hc in hamcycles:
        if (size := len(hc)) in count_dict:
            count_dict[size] += 1
    cycle_counts = [count_dict[k] for k in k_list]

    return torch.FloatTensor(cycle_counts).unsqueeze(0)



def count_cycles(edge_index, num_nodes, cycle_lengths):

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.detach().clone().cpu().T.numpy())
    def normalize_cycle(cycle):
        """Normalize cycle to account for shift and reflection invariance."""
        n = len(cycle)
        cycle_variants = [
            tuple(cycle[i:] + cycle[:i]) for i in range(n)
        ] + [
            tuple(cycle[i:] + cycle[:i])[::-1] for i in range(n)
        ]
        return min(cycle_variants)

    def find_cycles(v, visited, path, start, length):
        if length == 0:
            if start in G[v]:
                cycle = normalize_cycle(path + [start])
                unique_cycles.add(cycle)
            return

        visited.add(v)
        path.append(v)
        for neighbor in G[v]:
            if neighbor not in visited or (neighbor == start and length == 1):
                find_cycles(neighbor, visited.copy(), path.copy(), start, length - 1)

    cycle_counts = {length: 0 for length in cycle_lengths}

    for length in cycle_lengths:
        unique_cycles = set()
        for node in G:
            find_cycles(node, set(), [], node, length)
        cycle_counts[length] = len(unique_cycles)

    return torch.tensor([cycle_counts[length] for length in cycle_lengths]).float().unsqueeze(0)

# def _combine_encs(
#     data,
#     in_node_encs: List[str],
#     out_node_encs: List[str],
#     out_graph_encs: List[str],
#     cfg,
# ):
#     combined_stats = []

#     for name, pes in zip(["x", "y"], [in_node_encs, out_node_encs]):
#         if pes == "none":
#             continue

#         _check_all_types(pe_types := pes.split("+"))
#         pe_list: List[torch.Tensor] = []

#         for pe_type in pe_types:
#             if pe_type in ["LapPE", "EquivStableLapPE"]:
#                 if cfg[f"posenc_{pe_type}"].eigen.stack_eigval:
#                     pe = torch.hstack((data.EigVecs, data.EigVals.squeeze(-1)))
#                 else:
#                     pe = data.EigVecs
#             elif pe_type == "SignNet":
#                 pe = torch.hstack((data.eigvecs_sn,
#                                    data.eigvals_sn.squeeze(-1)))
#             elif pe_type in RANDSE_TYPES:
#                 pe = getattr(data, name)
#             else:
#                 pe = getattr(data, f"pestat_{pe_type}")
#             pe_list.append(pe)

#         combined_node_pe = torch.nan_to_num(torch.hstack(pe_list))

#         if (name == "y") and cfg.dataset.combine_output_pestat:
#             combined_stats.append(combined_node_pe)
#         else:
#             setattr(data, name, combined_node_pe)

#     # Graph level encoding targets
#     if out_graph_encs != "none":
#         enc_list: List[torch.Tensor] = []
#         _check_all_types(enc_types := out_graph_encs.split("+"))
#         for enc_type in enc_types:
#             if enc_type == "EigVals":
#                 enc = data.EigVals[0].T
#             elif enc_type == "RWGE":
#                 enc = data.pestat_RWSE.mean(0, keepdim=True)
#             else:
#                 enc = getattr(data, f"gestat_{enc_type}")
#             enc_list.append(enc)

#         combined_graph_pe = torch.nan_to_num(torch.hstack(enc_list))

#         if cfg.dataset.combine_output_pestat:
#             combined_stats.append(combined_graph_pe.repeat(data.x.shape[0], 1))
#         else:
#             data.y_graph = combined_graph_pe

#     # Combined pestat
#     if cfg.dataset.combine_output_pestat:
#         data.pestat_CombinedPSE = torch.hstack(combined_stats)
#         cfg.posenc_CombinedPSE._raw_dim = data.pestat_CombinedPSE.shape[1]
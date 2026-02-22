"""Sheaf Connection Laplacian Positional Encoding (ConnLap) Transform."""

import warnings

import numpy as np
import torch
from scipy.linalg import svd
from scipy.sparse import diags as sp_diags
from scipy.sparse import lil_matrix
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class SheafConnLapPE(BaseTransform):
    r"""Sheaf Connection Laplacian Positional Encoding (SheafConnLapPE) transform.

    Based on "Sheaf-based Positional Encodings for Graph Neural Networks"
    by He, Bodnar & Liò (NeurIPS 2023 Workshop / PMLR 2024).
    https://openreview.net/pdf?id=ZtAabWUPu3

    The Connection Laplacian generalises the standard graph Laplacian by
    replacing each scalar off-diagonal entry (-1) with a d×d orthogonal
    *restriction map* — a rotation encoding the geometric alignment between
    the local node-feature neighbourhoods of the two endpoints.

    For each edge (v, u), the algorithm:

    1. Runs local PCA on the 1-hop feature neighbourhood of v and u separately,
       yielding orthonormal bases B_v, B_u ∈ R^{p×d} that approximate the
       local tangent spaces T_{x_v}M and T_{x_u}M under the manifold assumption.
    2. Solves the orthogonal Procrustes problem to find the rotation O_{vu} ∈ O(d)
       that best maps B_v onto B_u (closed form: SVD of B_v^T B_u).
    3. Sets the off-diagonal block L_F[v, u] = -O_{vu}.

    The resulting nd×nd block matrix L_F is symmetric positive semi-definite.
    Its k smallest non-trivial eigenvectors (each reshaped from nd to n×d) are
    concatenated column-wise to form a PE of total dimension k×d per node.

    On homophilic edges (similar features) O_{vu} ≈ I and the Connection
    Laplacian closely resembles the standard Laplacian. On heterophilic edges
    O_{vu} is a non-trivial rotation, introducing cross-dimensional coupling
    that encodes semantic disagreement — information the standard Laplacian
    cannot represent.

    .. note::
        **Feature dimension requirement (Fix 1):** ``data.x`` must be present
        and ``data.x.shape[1] >= stalk_dim``. The method assumes that node
        features lie near a ``stalk_dim``-dimensional manifold; if feature_dim
        < stalk_dim this assumption is violated and the PCA basis would contain
        zero columns, making the Procrustes rotation degenerate and breaking
        the PSD property of L_F. A ``ValueError`` is raised in this case.

        **Undirected graphs (Fix 2):** Cellular sheaves are defined on
        undirected simple graphs. If ``edge_index`` is not already
        bidirectional, it is symmetrised automatically before building the
        adjacency list (idempotent for standard PyG format). Self-loops are
        silently removed: they have no well-defined restriction map in the
        sheaf structure and would corrupt the diagonal of L_F.

        **Isolated nodes (Fix 4):** For isolated nodes (degree 0), the
        diagonal block of L_F is the zero matrix, making D^{-1/2} undefined.
        The normalisation substitutes 1.0 for these zero diagonal entries,
        which is equivalent to adding a unit self-loop for numerical purposes.
        The eigenvector components of isolated nodes are still well-defined
        (the corresponding rows of L_F remain all-zero), but their PE values
        will reflect their position in the global spectrum rather than local
        connectivity.

        **Memory (Fix 6):** The dense eigendecomposition (``np.linalg.eigh``)
        requires O((n·d)²) memory. A warning is emitted for graphs where
        n·d > 10,000 (roughly 3,300 nodes at stalk_dim=3). At n·d=15,000
        the matrix is ~1.7 GB float64.

    Parameters
    ----------
    max_pe_dim : int
        Total output PE dimension. Must be divisible by ``stalk_dim``.
        Internally, the number of eigenvectors used is
        ``k = max_pe_dim // stalk_dim``, so the output shape is always
        ``[num_nodes, max_pe_dim]`` (zero-padded if fewer eigenvectors
        are available).
    stalk_dim : int, optional
        Dimension d of each stalk / restriction map. Controls the rank of
        the local tangent-space approximation. Default is 3, as used in
        the paper experiments. Must be <= feature_dim of ``data.x``.
    include_first : bool, optional
        If False (default), discards eigenvectors whose eigenvalue is below
        ``eps`` (the trivial zero-eigenvectors / global sections of the sheaf).
    concat_to_x : bool, optional
        If True (default), concatenates the PE with ``data.x``.
        If False, stores it as ``data.SheafPE`` instead.
    eps : float, optional
        Threshold below which eigenvalues are considered trivial. Default 1e-6.
    **kwargs
        Additional keyword arguments (unused; reserved for future extensions).
    """

    def __init__(
        self,
        max_pe_dim: int,
        stalk_dim: int = 3,
        include_first: bool = False,
        concat_to_x: bool = True,
        eps: float = 1e-6,
        # Fix 5: removed `tolerance` parameter — it was dead code with no
        # effect on the dense np.linalg.eigh solver path.  If a sparse solver
        # is introduced in future, the parameter can be re-added then.
        **kwargs,
    ):
        if max_pe_dim % stalk_dim != 0:
            raise ValueError(
                f"max_pe_dim ({max_pe_dim}) must be divisible by "
                f"stalk_dim ({stalk_dim}). "
                f"The number of eigenvectors used is k = max_pe_dim // stalk_dim."
            )
        self.max_pe_dim = max_pe_dim
        self.stalk_dim = stalk_dim
        self.k = max_pe_dim // stalk_dim  # number of eigenvectors
        self.include_first = include_first
        self.concat_to_x = concat_to_x
        self.eps = eps

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, data: Data) -> Data:
        """Compute and attach the ConnLap PE to a graph data object.

        Parameters
        ----------
        data : Data
            Input graph. ``data.x`` must be set and
            ``data.x.shape[1] >= stalk_dim``.

        Returns
        -------
        Data
            Graph with PE concatenated to ``data.x`` (``concat_to_x=True``)
            or stored in ``data.SheafPE`` (``concat_to_x=False``).

        Raises
        ------
        ValueError
            If ``data.x`` is None, or if ``feature_dim < stalk_dim``.
        """
        if data.x is None:
            raise ValueError(
                "SheafConnLapPE requires node features (data.x) to compute "
                "local PCA tangent spaces. data.x is None."
            )

        # Snapshot raw features *before* any concatenation — PCA always
        # operates on the original input features, not any augmented version.
        x_np = data.x.detach().cpu().numpy().astype(np.float64)

        # Fix 1: guard against stalk_dim > feature_dim.
        # The manifold assumption (features lie near a d-dimensional subspace)
        # only makes sense when d ≤ p.  If p < d, the Gram–Schmidt padding in
        # _local_pca_basis can only produce p - n_comp ≤ p < d vectors, leaving
        # the last d - p columns of every basis all-zeros.  Zero columns in B_v
        # make M = B_v^T B_u rank-deficient, so the SVD in Procrustes returns a
        # degenerate O_{vu} that is no longer a proper orthogonal matrix and
        # breaks the PSD property of L_F.
        feature_dim = x_np.shape[1]
        if feature_dim < self.stalk_dim:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be >= stalk_dim "
                f"({self.stalk_dim}).  The Connection Laplacian assumes node "
                f"features lie near a {self.stalk_dim}-dimensional manifold; "
                f"this is impossible when the ambient feature space has fewer "
                f"than {self.stalk_dim} dimensions.  Either reduce stalk_dim "
                f"or use higher-dimensional features."
            )

        pe = self._compute_sheaf_pe(data.edge_index, data.num_nodes, x_np)

        if self.concat_to_x:
            data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.SheafPE = pe

        return data

    # ──────────────────────────────────────────────────────────────────────
    # Core pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _compute_sheaf_pe(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        x_np: np.ndarray,
    ) -> torch.Tensor:
        """Full ConnLap PE pipeline.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor of shape (2, num_edges).
        num_nodes : int
            Number of nodes in the graph.
        x_np : np.ndarray
            Node feature matrix of shape (num_nodes, feature_dim).

        Returns
        -------
        torch.Tensor
            PE matrix of shape (num_nodes, max_pe_dim).
        """
        device = edge_index.device
        d = self.stalk_dim

        # Degenerate graph: return zero PE
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        # Fix 6: warn when the dense nd×nd matrix will be very large.
        # np.linalg.eigh allocates O((n·d)²) memory; at nd=15,000 this is
        # ~1.7 GB float64.  We warn at nd > 10,000 (~3,300 nodes at d=3)
        # to surface this before the allocation happens.
        nd = num_nodes * d
        if nd > 10_000:
            warnings.warn(
                f"SheafConnLapPE: the dense eigendecomposition will allocate a "
                f"{nd}×{nd} matrix (~{nd**2 * 8 / 1e9:.1f} GB float64). "
                f"Consider reducing stalk_dim (currently {d}) or the graph size "
                f"(currently {num_nodes} nodes). For very large graphs a sparse "
                f"solver would be more appropriate.",
                ResourceWarning,
                stacklevel=3,
            )

        # Fix 2: symmetrise edge_index to guarantee an undirected adjacency
        # list.  Cellular sheaves are defined on undirected simple graphs; an
        # asymmetric edge_index (only one direction stored per edge) would
        # produce an asymmetric L_F, causing np.linalg.eigh to silently return
        # wrong eigenvectors (it only reads the lower triangle of its input).
        # torch.unique(dim=1) deduplicates, so this is idempotent for graphs
        # already stored in standard PyG format (both directions present).
        # Self-loops are also removed: they have no well-defined restriction
        # map in the sheaf structure and overwrite the diagonal degree block
        # with -I_d, causing negative diagonal entries and NaN normalisation.
        ei_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(
            dim=1
        )
        # Remove self-loops (src == dst)
        not_self_loop = ei_sym[0] != ei_sym[1]
        ei_sym = ei_sym[:, not_self_loop]
        ei = ei_sym.cpu().numpy()

        # Build adjacency list from the symmetrised edge_index
        adjacency = [[] for _ in range(num_nodes)]
        for col in range(ei.shape[1]):
            u, v = int(ei[0, col]), int(ei[1, col])
            adjacency[u].append(v)

        # ── Step 1: local PCA basis per node ──────────────────────────────
        # B_v ∈ R^{p×d}: orthonormal basis for the tangent space at node v,
        # estimated from the 1-hop feature neighbourhood.
        bases = [
            self._local_pca_basis(v, x_np, adjacency[v])
            for v in range(num_nodes)
        ]

        # ── Steps 2 & 3: build the sparse nd×nd connection Laplacian ──────
        # For each edge (v,u): O_{vu} = Procrustes(B_v, B_u), then
        # L_F[v,u] = -O_{vu}, L_F[v,v] += I_d per incident edge.
        L_F = self._build_connection_laplacian(num_nodes, adjacency, bases)

        # ── Step 4: k smallest eigenvectors of the sheaf Laplacian ────────

        # np.linalg.eigh: dense, exact, deterministic, returns eigenvalues in
        # ascending order.  This is the correct default for graphs up to ~3 k
        # nodes (nd ≤ 9 k) — well within the target scale.
        # eigsh (ARPACK) would be faster for very large graphs but introduces
        # non-deterministic sign flips; the dense path avoids this entirely.
        evals, evecs = np.linalg.eigh(L_F.toarray())
        # eigh guarantees ascending order — no sort needed.

        # Drop trivial eigenvectors (global sections with eigenvalue ≈ 0)
        if not self.include_first:
            mask = evals > self.eps
            evals, evecs = evals[mask], evecs[:, mask]  # noqa: F841

        # Take the k smallest remaining
        k_use = min(self.k, evecs.shape[1])
        evecs = evecs[:, :k_use]  # (nd, k_use)

        # Fix 3: sign canonicalisation for eigenvectors.
        # np.linalg.eigh is deterministic for fixed input, but the sign of each
        # eigenvector is arbitrary (both v and -v are valid).  This matters when
        # comparing PEs across augmented or perturbed graph versions.  We follow
        # the convention from LapPE: for each eigenvector (viewed as a full
        # nd-vector), find the entry with the largest absolute value and flip
        # the sign of the entire vector if that entry is negative.  This gives
        # a deterministic, canonical orientation for every eigenvector.
        max_abs_idx = np.argmax(np.abs(evecs), axis=0)  # (k_use,)
        signs = np.sign(evecs[max_abs_idx, np.arange(k_use)])  # (k_use,)
        signs[signs == 0] = 1.0  # guard against the (pathological) zero case
        evecs = evecs * signs[np.newaxis, :]  # broadcast

        # ── Step 5: reshape eigenvectors → PE matrix ──────────────────────
        # Each eigenvector is (nd,) = (n*d,); reshape to (n, d).
        # Pack k_use such matrices side-by-side → (n, k_use * d).
        pe_np = np.zeros((num_nodes, k_use * d), dtype=np.float64)
        for i in range(k_use):
            vec = evecs[:, i].reshape(num_nodes, d)  # (n, d)
            pe_np[:, i * d : (i + 1) * d] = vec

        # Zero-pad to max_pe_dim if fewer than k eigenvectors were available
        if k_use * d < self.max_pe_dim:
            pad = np.zeros((num_nodes, self.max_pe_dim - k_use * d))
            pe_np = np.concatenate([pe_np, pad], axis=1)

        return torch.from_numpy(pe_np).to(dtype=torch.float32, device=device)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _local_pca_basis(
        self,
        node_idx: int,
        x: np.ndarray,
        neighbors: list,
    ) -> np.ndarray:
        """Orthonormal tangent-space basis B_v ∈ R^{p×d} via local PCA.

        Gathers features of node v and its 1-hop neighbours, centres them,
        and returns the top-d right singular vectors. These span the dominant
        directions of local feature variation — an approximation of T_{x_v}M
        under the manifold assumption.

        If the neighbourhood has fewer than d linearly independent directions
        (e.g. very small degree or duplicate features), the basis is padded
        with orthogonal complement vectors from the standard basis.

        Note: the guard in ``forward`` ensures p >= d before this method is
        called, so the Gram–Schmidt padding loop can always find d - n_comp
        additional orthogonal vectors in R^p (Fix 1).

        Parameters
        ----------
        node_idx : int
            Index of the target node.
        x : np.ndarray
            Node feature matrix of shape (n, p); guaranteed p >= stalk_dim by Fix 1.
        neighbors : list[int]
            Indices of 1-hop neighbours of node_idx.

        Returns
        -------
        np.ndarray
            Orthonormal basis matrix of shape (p, stalk_dim).
        """
        p = x.shape[1]
        d = self.stalk_dim
        local_idx = [node_idx] + list(neighbors)
        local_x = x[local_idx]
        local_x = local_x - local_x.mean(axis=0)

        if len(local_idx) < 2:
            # Isolated node: fall back to the first d standard basis vectors.
            # p >= d is guaranteed by Fix 1, so np.eye(p, d) has full column
            # rank and no zero columns.
            return np.eye(p, d)

        try:
            _, _, Vt = svd(local_x, full_matrices=False)
            # Vt shape: (min(n_local, p), p). Rows = principal directions.
            n_comp = Vt.shape[0]
        except np.linalg.LinAlgError:
            return np.eye(p, d)

        if n_comp >= d:
            return Vt[:d].T  # (p, d) — top-d principal directions as columns

        # Fewer PCA components than d: pad with orthogonal extras via
        # Gram–Schmidt.  Because p >= d (Fix 1) there are always p - n_comp >= 0
        # standard basis vectors available to extend the basis to d columns.
        basis = np.zeros((p, d))
        basis[:, :n_comp] = Vt.T
        for target_col in range(n_comp, d):
            for j in range(p):
                candidate = np.eye(1, p, j).flatten()
                for filled in range(target_col):
                    candidate -= (candidate @ basis[:, filled]) * basis[
                        :, filled
                    ]
                norm = np.linalg.norm(candidate)
                if norm > 1e-10:
                    basis[:, target_col] = candidate / norm
                    break
        return basis  # (p, d)

    @staticmethod
    def _orthogonal_procrustes(B_v: np.ndarray, B_u: np.ndarray) -> np.ndarray:
        """Find O* ∈ O(d) minimising ||B_u - B_v O||_F.

        Closed-form solution: SVD of B_v^T B_u = U S V^T → O* = U V^T.

        This is the parallel transport approximation from T_{x_v}M to T_{x_u}M,
        used as the restriction map for edge (v, u).

        Parameters
        ----------
        B_v : np.ndarray
            Orthonormal basis of shape (p, d) for the source node.
        B_u : np.ndarray
            Orthonormal basis of shape (p, d) for the target node.

        Returns
        -------
        np.ndarray
            Orthogonal rotation matrix of shape (d, d).
        """
        M = B_v.T @ B_u  # (d, d)
        U, _, Vt = svd(M, full_matrices=False)
        return U @ Vt  # (d, d) orthogonal

    def _build_connection_laplacian(
        self,
        num_nodes: int,
        adjacency: list,
        bases: list,
    ):
        """Assemble the sparse normalised connection Laplacian.

        Block structure (nd × nd, where nd = num_nodes × stalk_dim):

            Diagonal block [v, v] :  deg(v) · I_d
            Off-diagonal   [v, u] : -O_{vu}   for each directed edge (v→u)

        After assembly, symmetrically normalises:
            Δ_F = D^{-1/2} L_F D^{-1/2}
        where D is the scalar diagonal of L_F (each entry = deg(v)).

        Fix 4 note — isolated nodes:
            For isolated nodes, deg(v) = 0 so the diagonal block is the zero
            matrix, making D^{-1/2} undefined.  We substitute 1.0 for these
            zero diagonal entries (``diag_safe`` below), which is equivalent to
            giving isolated nodes a unit self-loop for normalisation purposes.
            Their rows/columns in L_F remain all-zero (they have no edges), so
            their eigenvector components are unaffected by this substitution —
            it only prevents a division-by-zero during normalisation.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        adjacency : list[list[int]]
            Adjacency list where adjacency[v] is the list of neighbours of v,
            guaranteed undirected by Fix 2.
        bases : list[np.ndarray]
            PCA bases where bases[v] is B_v of shape (p, stalk_dim).

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse normalised connection Laplacian of shape (nd, nd).
        """
        d = self.stalk_dim
        nd = num_nodes * d

        L_F = lil_matrix((nd, nd), dtype=np.float64)

        for v in range(num_nodes):
            rv = v * d
            # Diagonal block: deg(v) * I_d
            deg_v = len(adjacency[v])
            for i in range(d):
                L_F[rv + i, rv + i] = float(deg_v)

            # Off-diagonal blocks: -O_{vu} for each neighbour u
            for u in adjacency[v]:
                O_vu = self._orthogonal_procrustes(bases[v], bases[u])
                ru = u * d
                for i in range(d):
                    for j in range(d):
                        L_F[rv + i, ru + j] = -O_vu[i, j]

        L_F = L_F.tocsr()

        # Symmetric normalisation: Δ_F = D^{-1/2} L_F D^{-1/2}
        diag = np.array(L_F.diagonal(), dtype=np.float64)
        # Fix 4: substitute 1.0 for isolated-node zero diagonal entries so
        # that D^{-1/2} is defined.  See docstring above for the rationale.
        diag_safe = np.where(np.abs(diag) > 1e-10, diag, 1.0)
        inv_sqrt = 1.0 / np.sqrt(diag_safe)
        D_inv_sqrt = sp_diags(inv_sqrt)
        return D_inv_sqrt @ L_F @ D_inv_sqrt

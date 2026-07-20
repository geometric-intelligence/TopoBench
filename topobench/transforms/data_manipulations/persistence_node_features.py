"""Local persistent-homology node features (degree filtration)."""

import numpy as np
import torch
import torch_geometric


class PersistenceNodeFeatures(torch_geometric.transforms.BaseTransform):
    r"""Append local persistent-homology features to node features.

    For every node :math:`v` the transform builds the 1-hop ego
    subgraph, endows its clique complex (up to dimension 2) with the
    sublevel-set degree filtration
    :math:`f(u) = \deg(u) / \max_w \deg(w)` (edges and triangles enter
    at the maximum of their vertices), and computes 0- and
    1-dimensional persistence with GUDHI. Six barcode summaries are
    concatenated to ``data.x``:

    1. number of finite :math:`H_0` bars, normalized by ego size;
    2. total finite :math:`H_0` persistence;
    3. maximum finite :math:`H_0` persistence;
    4. number of :math:`H_1` bars (incl. essential), normalized by
       the ego edge count;
    5. total finite :math:`H_1` persistence;
    6. first Betti number of the ego graph,
       :math:`\beta_1 = E - V + C`, normalized by the ego edge count.

    Features 4-6 are sensitive to triangles and short cycles through
    a node; features 1-3 summarize the local degree landscape.

    Parameters
    ----------
    **kwargs : optional
        Additional parameters for the transform (stored, unused).
    """

    NUM_ADDED_FEATURES = 6

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "persistence_node_features"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(type={self.type!r}, "
            f"parameters={self.parameters!r})"
        )

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph. Must expose ``edge_index``; ``x`` is
            extended in place (or created if absent).

        Returns
        -------
        torch_geometric.data.Data
            The data object with persistence features appended to
            ``data.x``.
        """
        import gudhi  # local import: optional heavy dependency

        n = (
            data.x.shape[0]
            if data.get("x", None) is not None
            else int(data.num_nodes)
        )
        adj: list[set[int]] = [set() for _ in range(n)]
        if data.get("edge_index", None) is not None:
            src, dst = data.edge_index
            for s, t in zip(src.tolist(), dst.tolist(), strict=True):
                if s != t:
                    adj[s].add(t)
                    adj[t].add(s)
        deg = [len(a) for a in adj]
        max_deg = max(deg, default=0) or 1

        feats = torch.zeros(n, self.NUM_ADDED_FEATURES)
        for v in range(n):
            nodes = [v, *sorted(adj[v])]
            local = {u: i for i, u in enumerate(nodes)}
            fil = [deg[u] / max_deg for u in nodes]

            st = gudhi.SimplexTree()
            for i in range(len(nodes)):
                st.insert([i], filtration=fil[i])
            n_edges = 0
            for u in nodes:
                for w in adj[u]:
                    if w in local and u < w:
                        st.insert(
                            [local[u], local[w]],
                            filtration=max(fil[local[u]], fil[local[w]]),
                        )
                        n_edges += 1
            st.expansion(2)
            st.compute_persistence(persistence_dim_max=True)
            h0 = st.persistence_intervals_in_dimension(0)
            h1 = st.persistence_intervals_in_dimension(1)

            n_comp = 0
            fin0 = np.zeros(0)
            if len(h0):
                finite_mask = np.isfinite(h0[:, 1])
                n_comp = int((~finite_mask).sum())
                fin0 = h0[finite_mask, 1] - h0[finite_mask, 0]
            fin1 = np.zeros(0)
            if len(h1):
                finite_mask1 = np.isfinite(h1[:, 1])
                fin1 = h1[finite_mask1, 1] - h1[finite_mask1, 0]

            ego_size = len(nodes)
            edge_norm = max(n_edges, 1)
            beta1 = n_edges - ego_size + n_comp
            feats[v, 0] = len(fin0) / ego_size
            feats[v, 1] = float(fin0.sum())
            feats[v, 2] = float(fin0.max()) if len(fin0) else 0.0
            feats[v, 3] = len(h1) / edge_norm
            feats[v, 4] = float(fin1.sum())
            feats[v, 5] = beta1 / edge_norm

        if data.get("x", None) is None:
            data.x = feats
        else:
            data.x = torch.cat(
                [data.x.float(), feats.to(data.x.device)], dim=1
            )
        return data

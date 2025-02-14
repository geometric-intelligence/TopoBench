import abc

import networkx as nx
import numpy as np
import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import CellComplex, SimplicialComplex
from torch_geometric.utils.convert import to_networkx

from topobenchmark.data.utils.domain import ComplexData
from topobenchmark.data.utils.utils import (
    generate_zero_sparse_connectivity,
    select_neighborhoods_of_interest,
)


class Adapter(abc.ABC):
    """Adapt between data structures representing the same domain."""

    def __call__(self, domain):
        """Adapt domain's data structure."""
        return self.adapt(domain)

    @abc.abstractmethod
    def adapt(self, domain):
        """Adapt domain's data structure."""


class IdentityAdapter(Adapter):
    """Identity adaptation.

    Retrieves same data structure for domain.
    """

    def adapt(self, domain):
        """Adapt domain."""
        return domain


class Data2Graph(Adapter):
    """Data to nx.Graph adaptation.

    Parameters
    ----------
    preserve_edge_attr : bool
        Whether to preserve edge attributes.
    to_undirected (bool or str, optional): If set to :obj:`True`, will
        return a :class:`networkx.Graph` instead of a
        :class:`networkx.DiGraph`.
        By default, will include all edges and make them undirected.
        If set to :obj:`"upper"`, the undirected graph will only correspond
        to the upper triangle of the input adjacency matrix.
        If set to :obj:`"lower"`, the undirected graph will only correspond
        to the lower triangle of the input adjacency matrix.
        Only applicable in case the :obj:`data` object holds a homogeneous
        graph. (default: :obj:`False`)
    """

    def __init__(self, preserve_edge_attr=False, to_undirected=True):
        self.preserve_edge_attr = preserve_edge_attr
        self.to_undirected = to_undirected

    def adapt(self, domain: torch_geometric.data.Data) -> nx.Graph:
        r"""Generate a NetworkX graph from the input data object.

        Parameters
        ----------
        domain : torch_geometric.data.Data
            The input data.

        Returns
        -------
        nx.Graph
            The generated NetworkX graph.
        """
        edge_attrs = (
            "edge_attr"
            if self.preserve_edge_attr and hasattr(domain, "edge_attr")
            else None
        )
        return to_networkx(
            domain,
            to_undirected=self.to_undirected,
            node_attrs="x",
            edge_attrs=edge_attrs,
        )


class Complex2ComplexData(Adapter):
    """toponetx.Complex to ComplexData adaptation.

    NB: order of features plays a crucial role, as ``ComplexData``
    simply stores them as lists (i.e. the reference to the indices
    of the simplex are lost).

    Parameters
    ----------
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.
    transfer_features : bool, optional
        Whether to transfer features.
    """

    def __init__(
        self,
        neighborhoods=None,
        signed=False,
        transfer_features=True,
    ):
        super().__init__()
        self.neighborhoods = neighborhoods
        self.signed = signed
        self.transfer_features = transfer_features
        self._features_key = "x"

    def adapt(self, domain):
        """Adapt toponetx.Complex to ComplexData.

        Parameters
        ----------
        domain : toponetx.Complex

        Returns
        -------
        ComplexData
        """
        practical_dim = (
            domain.practical_dim
            if hasattr(domain, "practical_dim")
            else domain.dim
        )
        dim = domain.dim

        signed = self.signed
        neighborhoods = self.neighborhoods

        connectivity_infos = [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]

        practical_shape = list(
            np.pad(
                list(domain.shape), (0, practical_dim + 1 - len(domain.shape))
            )
        )
        data = {
            connectivity_info: [] for connectivity_info in connectivity_infos
        }
        for rank in range(practical_dim + 1):
            for connectivity_info in connectivity_infos:
                try:
                    data[connectivity_info].append(
                        from_sparse(
                            getattr(domain, f"{connectivity_info}_matrix")(
                                rank=rank, signed=signed
                            )
                        )
                    )
                except ValueError:
                    if connectivity_info == "incidence":
                        data[connectivity_info].append(
                            generate_zero_sparse_connectivity(
                                m=practical_shape[rank - 1],
                                n=practical_shape[rank],
                            )
                        )
                    else:
                        data[connectivity_info].append(
                            generate_zero_sparse_connectivity(
                                m=practical_shape[rank],
                                n=practical_shape[rank],
                            )
                        )

        # TODO: handle this
        if neighborhoods is not None:
            data = select_neighborhoods_of_interest(data, neighborhoods)

        if self.transfer_features:
            if isinstance(domain, SimplicialComplex):
                get_features = domain.get_simplex_attributes
            elif isinstance(domain, CellComplex):
                get_features = domain.get_cell_attributes
            else:
                raise ValueError("Can't transfer features.")

            data["features"] = []
            for rank in range(dim + 1):
                rank_features_dict = get_features(self._features_key, rank)
                if rank_features_dict:
                    rank_features = torch.tensor(
                        list(rank_features_dict.values())
                    )
                else:
                    rank_features = None
                data["features"].append(rank_features)

            for _ in range(dim + 1, practical_dim + 1):
                data["features"].append(None)

        return ComplexData(**data)


class ComplexData2Dict(Adapter):
    """ComplexData to dict adaptation."""

    def adapt(self, domain):
        """Adapt ComplexData to dict.

        Parameters
        ----------
        domain : ComplexData

        Returns
        -------
        dict
        """
        data = {}
        connectivity_infos = [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]
        for connectivity_info in connectivity_infos:
            info = getattr(domain, connectivity_info)
            for rank, rank_info in enumerate(info):
                data[f"{connectivity_info}_{rank}"] = rank_info

        data["shape"] = domain.shape

        for index, values in enumerate(domain.features):
            if values is not None:
                data[f"x_{index}"] = values

        return data


class HypergraphData2Dict(Adapter):
    """HypergraphData to dict adaptation."""

    def adapt(self, domain):
        """Adapt HypergraphData to dict.

        Parameters
        ----------
        domain : HypergraphData

        Returns
        -------
        dict
        """
        hyperedges_key = domain.rank_keys()[-1]
        return {
            "incidence_hyperedges": domain.incidence[hyperedges_key],
            "num_hyperedges": domain.num_hyperedges,
            "x_0": domain.features[0],
            "x_hyperedges": domain.features[hyperedges_key],
        }


class AdapterComposition(Adapter):
    """Composed adapter."""

    def __init__(self, adapters):
        super().__init__()
        self.adapters = adapters

    def adapt(self, domain):
        """Adapt domain"""
        for adapter in self.adapters:
            domain = adapter(domain)

        return domain


class Complex2Dict(AdapterComposition):
    """toponetx.Complex to dict adaptation.

    Parameters
    ----------
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.
    transfer_features : bool, optional
        Whether to transfer features.
    """

    def __init__(
        self,
        neighborhoods=None,
        signed=False,
        transfer_features=True,
    ):
        tnxcomplex2complex = Complex2ComplexData(
            neighborhoods=neighborhoods,
            signed=signed,
            transfer_features=transfer_features,
        )
        complex2dict = ComplexData2Dict()
        super().__init__(adapters=(tnxcomplex2complex, complex2dict))

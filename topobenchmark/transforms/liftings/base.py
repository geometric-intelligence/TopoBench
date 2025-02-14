"""Abstract class for topological liftings."""

import abc

import torch_geometric

from topobenchmark.data.utils import (
    Complex2ComplexData,
    ComplexData2Dict,
    Data2Graph,
    HypergraphData2Dict,
    IdentityAdapter,
)


class LiftingTransform(torch_geometric.transforms.BaseTransform):
    """Lifting transform.

    Parameters
    ----------
    lifting : LiftingMap
        Lifting map.
    data2domain : Converter
        Conversion between ``torch_geometric.Data`` into
        domain for consumption by lifting.
    domain2dict : Converter
        Conversion between output domain of feature lifting
        and ``torch_geometric.Data``.
    domain2domain : Converter
        Conversion between output domain of lifting
        and input domain for feature lifting.
    feature_lifting : FeatureLiftingMap
        Feature lifting map.
    """

    def __init__(
        self,
        lifting,
        data2domain=None,
        domain2dict=None,
        domain2domain=None,
        feature_lifting=None,
    ):
        if data2domain is None or data2domain == "Identity":
            data2domain = IdentityAdapter()

        if domain2dict is None or data2domain == "Identity":
            domain2dict = IdentityAdapter()

        if domain2domain is None or data2domain == "Identity":
            domain2domain = IdentityAdapter()

        if isinstance(lifting, str):
            from topobenchmark.transforms import TRANSFORMS

            lifting = TRANSFORMS[lifting]()

        if feature_lifting is None:
            feature_lifting = "IdentityFeatureLifting"

        if isinstance(feature_lifting, str):
            from topobenchmark.transforms import TRANSFORMS

            feature_lifting = TRANSFORMS[feature_lifting]()

        self.data2domain = data2domain
        self.domain2domain = domain2domain
        self.domain2dict = domain2dict
        self.lifting = lifting
        self.feature_lifting = feature_lifting

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Apply the full lifting (topology + features) to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        initial_data = data.to_dict()

        domain = self.data2domain(data)
        lifted_topology = self.lifting(domain)
        lifted_topology = self.domain2domain(lifted_topology)
        lifted_topology = self.feature_lifting(lifted_topology)
        lifted_topology_dict = self.domain2dict(lifted_topology)

        return torch_geometric.data.Data(
            **initial_data, **lifted_topology_dict
        )


class Graph2ComplexLiftingTransform(LiftingTransform):
    """Graph to complex lifting transform.

    Parameters
    ----------
    lifting : LiftingMap
        Lifting map.
    feature_lifting : FeatureLiftingMap
        Feature lifting map.
    preserve_edge_attr : bool
        Whether to preserve edge attributes.
        Ignored if ``data2domain`` is not None.
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
        Ignored if ``data2domain`` is not None.
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.
    transfer_features : bool, optional
        Whether to transfer features.
    data2domain : Converter
        Conversion between ``torch_geometric.Data`` into
        domain for consumption by lifting.
    """

    def __init__(
        self,
        lifting,
        feature_lifting="ProjectionSum",
        preserve_edge_attr=False,
        to_undirected=True,
        neighborhoods=None,
        signed=False,
        transfer_features=True,
        data2domain=None,
    ):
        if data2domain is None:
            data2domain = Data2Graph(preserve_edge_attr, to_undirected)

        super().__init__(
            lifting,
            feature_lifting=feature_lifting,
            data2domain=data2domain,
            domain2domain=Complex2ComplexData(
                neighborhoods=neighborhoods,
                signed=signed,
                transfer_features=transfer_features,
            ),
            domain2dict=ComplexData2Dict(),
        )


Graph2SimplicialLiftingTransform = Graph2ComplexLiftingTransform
Graph2CellLiftingTransform = Graph2ComplexLiftingTransform


class Graph2HypergraphLiftingTransform(LiftingTransform):
    def __init__(
        self,
        lifting,
        feature_lifting="ProjectionSum",
    ):
        super().__init__(
            lifting,
            feature_lifting=feature_lifting,
            domain2dict=HypergraphData2Dict(),
        )


class LiftingMap(abc.ABC):
    """Lifting map.

    Lifts a domain into another.
    """

    def __call__(self, domain):
        """Lift domain."""
        return self.lift(domain)

    @abc.abstractmethod
    def lift(self, domain):
        """Lift domain."""

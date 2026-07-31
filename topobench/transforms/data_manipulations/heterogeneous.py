"""Data manipulations that explicitly support native heterogeneous graphs."""

from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import Constant as PyGConstant
from torch_geometric.transforms import ToUndirected as PyGToUndirected


class HeterogeneousConstantFeatures(BaseTransform):
    """Fill selected heterogeneous node stores with constant features.

    Parameters
    ----------
    node_types : str | list[str]
        Node type or node types whose features should be created or replaced.
    value : float, default=1.0
        Constant value assigned to the new one-channel feature tensor.
    cat : bool, default=False
        Whether to concatenate the constant channel to existing features.
    **_ : object
        Ignored configuration fields supplied by the transform registry.
    """

    supports_heterodata = True

    def __init__(
        self,
        node_types: str | list[str],
        value: float = 1.0,
        cat: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.transform = PyGConstant(
            value=value,
            cat=cat,
            node_types=node_types,
        )

    def forward(self, data: HeteroData) -> HeteroData:
        """Apply constant features to the selected node stores.

        Parameters
        ----------
        data : HeteroData
            Native heterogeneous graph to transform.

        Returns
        -------
        HeteroData
            The transformed heterogeneous graph.
        """
        return self.transform(data)


class HeterogeneousToUndirected(BaseTransform):
    """Add reverse typed relations to a heterogeneous graph.

    Parameters
    ----------
    reduce : str, default="add"
        Reduction used by PyG when relation stores are merged.
    merge : bool, default=False
        Whether reverse edges should be merged into compatible relation
        stores. The default creates explicit ``rev_*`` typed relations.
    **_ : object
        Ignored configuration fields supplied by the transform registry.
    """

    supports_heterodata = True

    def __init__(
        self,
        reduce: str = "add",
        merge: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.transform = PyGToUndirected(reduce=reduce, merge=merge)

    def forward(self, data: HeteroData) -> HeteroData:
        """Add reverse typed relations.

        Parameters
        ----------
        data : HeteroData
            Native heterogeneous graph to transform.

        Returns
        -------
        HeteroData
            The graph with reverse relations added by PyG.
        """
        return self.transform(data)

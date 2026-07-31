"""Data manipulations that explicitly support native heterogeneous graphs."""

from collections.abc import Sequence

from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import Constant as PyGConstant
from torch_geometric.transforms import ToUndirected as PyGToUndirected

# Accepted by the installed ``torch_geometric.utils.scatter`` implementation,
# which ``coalesce`` uses for duplicate edge attributes.
_SUPPORTED_REDUCTIONS = (
    "add",
    "sum",
    "mean",
    "min",
    "max",
    "amin",
    "amax",
    "mul",
    "any",
)


def _normalize_node_types(
    node_types: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Normalize and validate an optional node-type selection."""
    if node_types is None:
        return None
    if isinstance(node_types, str):
        normalized = (node_types,)
    elif isinstance(node_types, Sequence):
        if not node_types:
            raise ValueError("node_types must select at least one node type")
        if not all(isinstance(node_type, str) for node_type in node_types):
            raise TypeError("node_types entries must be strings")
        normalized = tuple(node_types)
    else:
        raise TypeError(
            "node_types must be a string, a sequence of strings, or None"
        )
    if any(not node_type.strip() for node_type in normalized):
        raise ValueError("node_types entries must be non-empty strings")
    return normalized


def _require_heterodata(
    data: object,
    *,
    wrapper_name: str,
) -> HeteroData:
    """Return native heterogeneous data or raise a readable contract error."""
    if not isinstance(data, HeteroData):
        raise TypeError(
            f"{wrapper_name} expects HeteroData; "
            f"received {type(data).__name__}"
        )
    return data


class HeterogeneousConstantFeatures(BaseTransform):
    """Fill selected heterogeneous node stores with constant features.

    Parameters
    ----------
    node_types : str | Sequence[str] | None
        Node type or node types whose features should be created or replaced.
        ``None`` selects every node store, matching PyG's ``Constant``
        semantics.
    value : float, default=1.0
        Constant value assigned to the new one-channel feature tensor.
    cat : bool, default=False
        Whether to concatenate the constant channel to existing features.
    transform_name : str | None, optional
        Transform name supplied by the TopoBench registry.
    transform_type : str | None, optional
        Transform category supplied by the TopoBench registry.
    """

    supports_heterodata = True

    def __init__(
        self,
        node_types: str | Sequence[str] | None,
        value: float = 1.0,
        cat: bool = False,
        transform_name: str | None = None,
        transform_type: str | None = None,
    ) -> None:
        super().__init__()
        if type(cat) is not bool:
            raise TypeError(f"cat must be bool; received {type(cat).__name__}")
        self.node_types = _normalize_node_types(node_types)
        self.cat = cat
        self.transform_name = transform_name
        self.transform_type = transform_type
        self.transform = PyGConstant(
            value=value,
            cat=cat,
            node_types=(
                None if self.node_types is None else list(self.node_types)
            ),
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

        Raises
        ------
        TypeError
            If ``data`` is not native ``HeteroData``.
        ValueError
            If a selected node type does not exist.
        """
        data = _require_heterodata(
            data,
            wrapper_name=type(self).__name__,
        )
        if self.node_types is not None:
            unknown = tuple(
                node_type
                for node_type in self.node_types
                if node_type not in data.node_types
            )
            if unknown:
                raise ValueError(
                    f"{type(self).__name__} received unknown node types "
                    f"{unknown}; available={tuple(data.node_types)}"
                )
        return self.transform(data)


class HeterogeneousToUndirected(BaseTransform):
    """Add reverse typed relations to a heterogeneous graph.

    Parameters
    ----------
    reduce : str, default="add"
        Reduction supported by PyG's coalesce/scatter implementation.
    merge : bool, default=False
        Whether reverse edges should be merged into compatible relation
        stores. The default creates explicit ``rev_*`` typed relations.
    transform_name : str | None, optional
        Transform name supplied by the TopoBench registry.
    transform_type : str | None, optional
        Transform category supplied by the TopoBench registry.
    """

    supports_heterodata = True

    def __init__(
        self,
        reduce: str = "add",
        merge: bool = False,
        transform_name: str | None = None,
        transform_type: str | None = None,
    ) -> None:
        super().__init__()
        if type(merge) is not bool:
            raise TypeError(
                f"merge must be bool; received {type(merge).__name__}"
            )
        if not isinstance(reduce, str):
            raise TypeError(
                f"reduce must be str; received {type(reduce).__name__}"
            )
        if reduce not in _SUPPORTED_REDUCTIONS:
            supported = ", ".join(_SUPPORTED_REDUCTIONS)
            raise ValueError(
                f"Unsupported reduce {reduce!r}; supported={supported}"
            )
        self.reduce = reduce
        self.merge = merge
        self.transform_name = transform_name
        self.transform_type = transform_type
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

        Raises
        ------
        TypeError
            If ``data`` is not native ``HeteroData``.
        ValueError
            If PyG would write a generated reverse relation into an existing
            edge store.
        """
        data = _require_heterodata(
            data,
            wrapper_name=type(self).__name__,
        )
        existing_edge_types = set(data.edge_types)
        for source_type in data.edge_types:
            store = data[source_type]
            if "edge_index" not in store:
                continue
            if store.is_bipartite() or not self.merge:
                source_node_type, relation, target_node_type = source_type
                reverse_type = (
                    target_node_type,
                    f"rev_{relation}",
                    source_node_type,
                )
                if reverse_type in existing_edge_types:
                    raise ValueError(
                        f"{type(self).__name__} cannot reverse source edge "
                        f"type {source_type!r}: target edge type "
                        f"{reverse_type!r} already exists"
                    )
        return self.transform(data)

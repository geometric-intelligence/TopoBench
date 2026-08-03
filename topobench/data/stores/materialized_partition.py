"""Exact in-memory materialization over a native PyG partition."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any, Literal

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.storage import AttrType, select
from torch_geometric.loader import ClusterData
from torch_geometric.utils import subgraph as pyg_subgraph
from torch_geometric.utils import to_undirected

AttributeRole = Literal["node", "edge", "graph"]


class MaterializedHomogeneousPartition:
    """Partition one homogeneous graph and materialize exact cluster unions."""

    _GENERATED_FIELDS = frozenset(
        {
            "global_nid",
            "selected_partition_ids",
            "num_selected_partitions",
            "supervised_mask",
        }
    )
    _ROLE_RESERVED_FIELDS = _GENERATED_FIELDS | {"edge_index", "num_nodes"}
    _NODE_FIELDS = frozenset(
        {"x", "y", "pos", "train_mask", "val_mask", "test_mask"}
    )
    _EDGE_FIELDS = frozenset({"edge_attr", "edge_weight", "edge_id"})

    def __init__(
        self,
        data: Data,
        num_parts: int,
        recursive: bool = False,
        attribute_roles: Mapping[str, AttributeRole] | None = None,
    ) -> None:
        num_nodes = self._validate_graph(data)
        self._validate_num_parts(num_parts, num_nodes)
        if not isinstance(recursive, bool):
            raise TypeError(
                "MaterializedHomogeneousPartition recursive must be a bool"
            )
        self._validate_known_phase_masks(data, num_nodes)
        self._attribute_roles = self._resolve_attribute_roles(
            data,
            num_nodes,
            attribute_roles,
        )

        # METIS scores topology only; authoritative directed data stays untouched.
        scoring_data = Data(
            edge_index=to_undirected(data.edge_index, num_nodes=num_nodes),
            num_nodes=num_nodes,
        )
        cluster_data = ClusterData(
            scoring_data,
            num_parts=int(num_parts),
            recursive=recursive,
            save_dir=None,
            log=False,
        )
        self._data = data
        self._num_nodes = num_nodes
        self.num_parts = int(num_parts)
        self.recursive = recursive
        self.partition = cluster_data.partition
        self._validate_partition(num_nodes)
        # PyG defines node_perm[row] as the canonical original node identity.
        self.perm_to_global = self.partition.node_perm

    @staticmethod
    def _validate_graph(data: Data) -> int:
        if not isinstance(data, Data):
            raise TypeError(
                "MaterializedHomogeneousPartition requires homogeneous PyG Data"
            )
        edge_index = getattr(data, "edge_index", None)
        if not isinstance(edge_index, Tensor):
            raise ValueError(
                "MaterializedHomogeneousPartition graph requires tensor edge_index"
            )
        if edge_index.dtype != torch.long:
            raise TypeError(
                "MaterializedHomogeneousPartition edge_index must use torch.long"
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "MaterializedHomogeneousPartition edge_index must have shape [2, E]"
            )
        if edge_index.device.type != "cpu":
            raise ValueError(
                "MaterializedHomogeneousPartition edge_index must remain on CPU"
            )

        num_nodes = data.num_nodes
        if (
            isinstance(num_nodes, bool)
            or not isinstance(num_nodes, Integral)
            or num_nodes <= 0
        ):
            raise ValueError(
                "MaterializedHomogeneousPartition graph must contain at least one node"
            )
        num_nodes = int(num_nodes)
        if edge_index.numel() > 0:
            minimum = int(edge_index.min())
            maximum = int(edge_index.max())
            if minimum < 0 or maximum >= num_nodes:
                raise ValueError(
                    "MaterializedHomogeneousPartition edge_index values are outside "
                    f"the valid node range [0, {num_nodes})"
                )

        for key, value in data.items():
            if isinstance(value, Tensor) and value.device.type != "cpu":
                raise ValueError(
                    "MaterializedHomogeneousPartition tensor field "
                    f"'{key}' must remain on CPU"
                )
        return num_nodes

    @staticmethod
    def _validate_num_parts(num_parts: int, num_nodes: int) -> None:
        if isinstance(num_parts, bool) or not isinstance(num_parts, Integral):
            raise TypeError(
                "MaterializedHomogeneousPartition num_parts must be an integer"
            )
        if num_parts <= 0 or num_parts > num_nodes:
            raise ValueError(
                "MaterializedHomogeneousPartition num_parts must be in "
                f"[1, {num_nodes}]"
            )

    @classmethod
    def _validate_known_phase_masks(cls, data: Data, num_nodes: int) -> None:
        for name in ("train_mask", "val_mask", "test_mask"):
            if name in data:
                cls._validate_phase_mask(data[name], name, num_nodes)

    @staticmethod
    def _validate_phase_mask(mask: Any, name: str, num_nodes: int) -> Tensor:
        if not isinstance(mask, Tensor):
            raise TypeError(
                f"MaterializedHomogeneousPartition {name} must be a tensor"
            )
        if mask.dtype != torch.bool:
            raise TypeError(
                f"MaterializedHomogeneousPartition {name} must have dtype torch.bool"
            )
        if tuple(mask.shape) != (num_nodes,):
            raise ValueError(
                f"MaterializedHomogeneousPartition {name} must have shape "
                f"({num_nodes},)"
            )
        if mask.device.type != "cpu":
            raise ValueError(
                f"MaterializedHomogeneousPartition {name} must remain on CPU"
            )
        return mask

    @classmethod
    def _intrinsic_role(cls, key: str) -> AttributeRole | None:
        if key in cls._EDGE_FIELDS:
            return "edge"
        if key in cls._NODE_FIELDS:
            return "node"
        return None

    @staticmethod
    def _attribute_extent(data: Data, key: str, value: Tensor) -> int | None:
        if value.dim() == 0:
            return None
        cat_dim = data.__cat_dim__(key, value)
        if not isinstance(cat_dim, int):
            return None
        return int(value.size(cat_dim))

    @classmethod
    def _validate_role_shape(
        cls,
        data: Data,
        key: str,
        value: Any,
        role: AttributeRole,
        num_nodes: int,
        num_edges: int,
        *,
        explicit: bool,
    ) -> None:
        if role == "graph":
            return
        prefix = "attribute_roles" if explicit else "attribute"
        if not isinstance(value, Tensor):
            raise TypeError(
                f"MaterializedHomogeneousPartition {prefix} field '{key}' "
                f"with {role} role must be a tensor"
            )
        expected = num_nodes if role == "node" else num_edges
        extent = cls._attribute_extent(data, key, value)
        if extent != expected:
            raise ValueError(
                f"MaterializedHomogeneousPartition {prefix} field '{key}' "
                f"{role} role has invalid shape; expected cat dimension {expected}"
            )

    @classmethod
    def _resolve_attribute_roles(
        cls,
        data: Data,
        num_nodes: int,
        attribute_roles: Mapping[str, AttributeRole] | None,
    ) -> dict[str, AttributeRole]:
        if attribute_roles is None:
            explicit_roles: dict[str, AttributeRole] = {}
        elif not isinstance(attribute_roles, Mapping):
            raise TypeError(
                "MaterializedHomogeneousPartition attribute_roles must be a mapping"
            )
        else:
            explicit_roles = {}
            for key, role in attribute_roles.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "MaterializedHomogeneousPartition attribute_roles keys "
                        "must be strings"
                    )
                if key in cls._ROLE_RESERVED_FIELDS:
                    raise ValueError(
                        "MaterializedHomogeneousPartition attribute_roles reserved "
                        f"field '{key}' cannot be declared"
                    )
                if key not in data:
                    raise ValueError(
                        "MaterializedHomogeneousPartition attribute_roles references "
                        f"missing field '{key}'"
                    )
                if role not in ("node", "edge", "graph"):
                    raise ValueError(
                        "MaterializedHomogeneousPartition attribute_roles field "
                        f"'{key}' has invalid role {role!r}"
                    )
                intrinsic_role = cls._intrinsic_role(key)
                if intrinsic_role is not None and role != intrinsic_role:
                    raise ValueError(
                        "MaterializedHomogeneousPartition attribute_roles field "
                        f"'{key}' cannot override intrinsic {intrinsic_role} role "
                        f"with {role!r}"
                    )
                explicit_roles[key] = role

        for key in cls._GENERATED_FIELDS:
            if key in data:
                raise ValueError(
                    f"MaterializedHomogeneousPartition source field '{key}' is reserved"
                )

        num_edges = int(data.edge_index.size(1))
        roles: dict[str, AttributeRole] = {}
        for key, value in data.items():
            if key in ("edge_index", "num_nodes"):
                continue
            explicit_role = explicit_roles.get(key)
            if explicit_role is not None:
                cls._validate_role_shape(
                    data,
                    key,
                    value,
                    explicit_role,
                    num_nodes,
                    num_edges,
                    explicit=True,
                )
                roles[key] = explicit_role
                continue
            if not isinstance(value, Tensor):
                roles[key] = "graph"
                continue

            intrinsic_role = cls._intrinsic_role(key)
            if intrinsic_role is not None:
                cls._validate_role_shape(
                    data,
                    key,
                    value,
                    intrinsic_role,
                    num_nodes,
                    num_edges,
                    explicit=False,
                )
                roles[key] = intrinsic_role
                continue

            extent = cls._attribute_extent(data, key, value)
            node_match = extent == num_nodes
            edge_match = extent == num_edges
            if node_match and edge_match:
                raise ValueError(
                    f"MaterializedHomogeneousPartition attribute '{key}' is "
                    "ambiguous between node and edge domains; declare attribute_roles"
                )
            if node_match:
                roles[key] = "node"
            elif edge_match:
                roles[key] = "edge"
            else:
                roles[key] = "graph"
        return roles

    def _validate_partition(self, num_nodes: int) -> None:
        partptr = getattr(self.partition, "partptr", None)
        node_perm = getattr(self.partition, "node_perm", None)
        if not isinstance(partptr, Tensor) or not isinstance(
            node_perm, Tensor
        ):
            raise ValueError(
                "PyG ClusterData returned a partition without partptr/node_perm"
            )
        if partptr.dtype != torch.long or tuple(partptr.shape) != (
            self.num_parts + 1,
        ):
            raise ValueError(
                "PyG ClusterData returned an invalid partition partptr"
            )
        if node_perm.dtype != torch.long or tuple(node_perm.shape) != (
            num_nodes,
        ):
            raise ValueError(
                "PyG ClusterData returned an invalid partition node_perm"
            )
        if (
            int(partptr[0]) != 0
            or int(partptr[-1]) != num_nodes
            or bool((partptr[1:] < partptr[:-1]).any())
        ):
            raise ValueError(
                "PyG ClusterData returned invalid partition membership boundaries"
            )
        if not torch.equal(
            torch.sort(node_perm).values,
            torch.arange(num_nodes, dtype=torch.long),
        ):
            raise ValueError(
                "PyG ClusterData returned node_perm without every canonical node"
            )

    def _mask_for_phase(self, phase: str) -> tuple[str, Tensor]:
        if not isinstance(phase, str):
            raise TypeError(
                "MaterializedHomogeneousPartition phase must be a non-empty string"
            )
        if not phase:
            raise ValueError(
                "MaterializedHomogeneousPartition phase must be a non-empty string"
            )
        name = f"{phase}_mask"
        if name not in self._data:
            raise ValueError(
                "MaterializedHomogeneousPartition phase "
                f"'{phase}' requires source field '{name}'"
            )
        return name, self._validate_phase_mask(
            self._data[name],
            name,
            self._num_nodes,
        )

    def partition_ids_for_phase(self, phase: str) -> list[int]:
        """Return sorted partitions containing at least one phase node."""
        _, mask = self._mask_for_phase(phase)
        selected: list[int] = []
        for part_id in range(self.num_parts):
            start = int(self.partition.partptr[part_id])
            end = int(self.partition.partptr[part_id + 1])
            members = self.perm_to_global[start:end]
            if bool(mask[members].any()):
                selected.append(part_id)
        return selected

    def _normalize_partition_ids(
        self,
        partition_ids: Iterable[int],
    ) -> tuple[int, ...]:
        if isinstance(partition_ids, (str, bytes)):
            raise TypeError(
                "MaterializedHomogeneousPartition partition_ids must be an iterable "
                "of integers"
            )
        try:
            iterator = iter(partition_ids)
        except TypeError as error:
            raise TypeError(
                "MaterializedHomogeneousPartition partition_ids must be an iterable "
                "of integers"
            ) from error

        normalized: set[int] = set()
        for value in iterator:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    "MaterializedHomogeneousPartition partition_ids must contain "
                    "only non-boolean integers"
                )
            part_id = int(value)
            if part_id < 0 or part_id >= self.num_parts:
                raise ValueError(
                    "MaterializedHomogeneousPartition partition_ids contain "
                    f"out-of-range ID {part_id}; expected [0, {self.num_parts})"
                )
            normalized.add(part_id)
        if not normalized:
            raise ValueError(
                "MaterializedHomogeneousPartition partition_ids must not be empty"
            )
        return tuple(sorted(normalized))

    def materialize(
        self,
        partition_ids: Iterable[int],
        phase: str | None = None,
    ) -> Data:
        """Return the exact directed induced union of selected partitions."""
        normalized = self._normalize_partition_ids(partition_ids)
        phase_name: str | None = None
        if phase is not None:
            phase_name, _ = self._mask_for_phase(phase)

        perm_rows = torch.cat(
            [
                torch.arange(
                    int(self.partition.partptr[part_id]),
                    int(self.partition.partptr[part_id + 1]),
                    dtype=torch.long,
                )
                for part_id in normalized
            ]
        )
        global_nid = self.perm_to_global.index_select(0, perm_rows)
        edge_index, _, edge_mask = pyg_subgraph(
            global_nid,
            self._data.edge_index,
            relabel_nodes=True,
            num_nodes=self._num_nodes,
            return_edge_mask=True,
        )
        output = Data(edge_index=edge_index)
        cache = output._store.__dict__.setdefault(
            "_cached_attr", defaultdict(set)
        )

        for key, source_value in self._data.items():
            if key in ("edge_index", "num_nodes"):
                continue
            role = self._attribute_roles[key]
            if role == "node":
                cat_dim = self._data.__cat_dim__(key, source_value)
                output[key] = select(source_value, global_nid, dim=cat_dim)
                cache[AttrType.NODE].add(key)
            elif role == "edge":
                cat_dim = self._data.__cat_dim__(key, source_value)
                output[key] = select(source_value, edge_mask, dim=cat_dim)
                cache[AttrType.EDGE].add(key)
            else:
                output[key] = (
                    source_value.clone()
                    if isinstance(source_value, Tensor)
                    else source_value
                )
                cache[AttrType.OTHER].add(key)

        output.num_nodes = int(global_nid.numel())
        output.global_nid = global_nid
        cache[AttrType.NODE].add("global_nid")
        output.selected_partition_ids = torch.tensor(
            normalized, dtype=torch.long
        )
        output.num_selected_partitions = len(normalized)
        cache[AttrType.OTHER].update(
            {"selected_partition_ids", "num_selected_partitions"}
        )
        if phase_name is not None:
            output.supervised_mask = output[phase_name].clone()
            cache[AttrType.NODE].add("supervised_mask")
        return output

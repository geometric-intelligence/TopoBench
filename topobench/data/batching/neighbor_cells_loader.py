"""NeighborCellsLoader class to batch in the transductive setting when working with topological domains."""

from collections.abc import Callable

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor

from topobenchmark.data.batching.cell_loader import CellLoader
from topobenchmark.data.batching.utils import get_sampled_neighborhood
from topobenchmark.dataloader import DataloadDataset


class NeighborCellsLoader(CellLoader):
    r"""A data loader that samples neighbors for each cell. Cells are considered neighbors if they are upper or lower neighbors.

    Parameters
    ----------
        data : Any
            A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        rank : int
            The rank of the cells to consider.
        num_neighbors : List[int] or Dict[Tuple[str, str, str], List[int]]
            The number of neighbors to sample for each node in each iteration.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_nodes : torch.Tensor or str or Tuple[str, torch.Tensor]
            The indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
        input_time : torch.Tensor, optional
            Optional values to override the timestamp for the input nodes given in :obj:`input_nodes`. If not
            set, will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`).
        replace : bool, optional
            If set to :obj:`True`, will sample with replacement. (default: :obj:`False`).
        subgraph_type : SubgraphType or str, optional
            The type of the returned subgraph.
            If set to :obj:`"directional"`, the returned subgraph only holds
            the sampled (directed) edges which are necessary to compute
            representations for the sampled seed nodes.
            If set to :obj:`"bidirectional"`, sampled edges are converted to
            bidirectional edges.
            If set to :obj:`"induced"`, the returned subgraph contains the
            induced subgraph of all sampled nodes.
            (default: :obj:`"directional"`).
        disjoint : bool, optional
            If set to :obj: `True`, each seed node will create its own disjoint subgraph.
            If set to :obj:`True`, mini-batch outputs will have a :obj:`batch`
            vector holding the mapping of nodes to their respective subgraph.
            Will get automatically set to :obj:`True` in case of temporal
            sampling. (default: :obj:`False`).
        temporal_strategy : str, optional
            The sampling strategy when using temporal sampling (:obj:`"uniform"`, :obj:`"last"`).
            If set to :obj:`"uniform"`, will sample uniformly across neighbors
            that fulfill temporal constraints.
            If set to :obj:`"last"`, will sample the last `num_neighbors` that
            fulfill temporal constraints.
            (default: :obj:`"uniform"`).
        time_attr : str, optional
            The name of the attribute that denotes timestamps for either the nodes or edges in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier or equal timestamp than the center node.
            (default: :obj:`None`).
        weight_attr : str, optional
            The name of the attribute that denotes edge weights in the graph.
            If set, weighted/biased sampling will be used such that neighbors
            are more likely to get sampled the higher their edge weights are.
            Edge weights do not need to sum to one, but must be non-negative,
            finite and have a non-zero sum within local neighborhoods.
            (default: :obj:`None`).
        transform : callable, optional
            A function/transform that takes in a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`).
        transform_sampler_output : callable, optional
            A function/transform that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
            returns a transformed version. (default: :obj:`None`).
        is_sorted : bool, optional
            If set to :obj:`True`, assumes that :obj:`edge_index` is sorted by column.
            If :obj:`time_attr` is set, additionally requires that rows are
            sorted according to time within individual neighborhoods.
            This avoids internal re-sorting of the data and can improve
            runtime and memory efficiency. (default: :obj:`False`).
        filter_per_worker : bool, optional
            If set to :obj:`True`, will filter the returned data in each worker's subprocess.
            If set to :obj:`False`, will filter the returned data in the main process.
            If set to :obj:`None`, will automatically infer the decision based
            on whether data partially lives on the GPU
            (:obj:`filter_per_worker=True`) or entirely on the CPU
            (:obj:`filter_per_worker=False`).
            There exists different trade-offs for setting this option.
            Specifically, setting this option to :obj:`True` for in-memory
            datasets will move all features to shared memory, which may result
            in too many open file handles. (default: :obj:`None`).
        neighbor_sampler : NeighborSampler, optional
            The neighbor sampler implementation to be used with this loader.
            If not set, a new :class:`torch_geometric.sampler.NeighborSampler`
            instance will be created. (default: :obj:`None`).
        directed : bool, optional
            If set to :obj:`True`, will consider the graph as directed.
            If set to :obj:`False`, will consider the graph as undirected.
            (default: :obj:`True`).
        **kwargs : optional
            Additional arguments of :class:`torch.utils.data.DataLoader`, such as
            :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """

    def __init__(
        self,
        data: Data | HeteroData | tuple[FeatureStore, GraphStore],
        rank: int,
        num_neighbors: list[int] | dict[EdgeType, list[int]],
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        replace: bool = False,
        subgraph_type: SubgraphType | str = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: str | None = None,
        weight_attr: str | None = None,
        transform: Callable | None = None,
        transform_sampler_output: Callable | None = None,
        is_sorted: bool = False,
        filter_per_worker: bool | None = None,
        neighbor_sampler: NeighborSampler | None = None,
        directed: bool = True,
        **kwargs,
    ):
        if input_time is not None and time_attr is None:
            raise ValueError(
                "Received conflicting 'input_time' and "
                "'time_attr' arguments: 'input_time' is set "
                "while 'time_attr' is not set."
            )

        data_obj = Data()
        if isinstance(data, DataloadDataset):
            for tensor, name in zip(data[0][0], data[0][1], strict=False):
                setattr(data_obj, name, tensor)
        else:
            data_obj = data
        is_hypergraph = hasattr(data_obj, "incidence_hyperedges")
        n_hops = len(num_neighbors)
        data_obj = get_sampled_neighborhood(
            data_obj, rank, n_hops, is_hypergraph
        )
        self.rank = rank
        if self.rank != 0:
            # When rank is different than 0 get_sampled_neighborhood connects cells that are up to n_hops away, meaning that the NeighborhoodSampler needs to consider only one hop.
            num_neighbors = [num_neighbors[0]]
        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data_obj,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get("num_workers", 0) > 0,
                directed=directed,
            )

        super().__init__(
            data=data_obj,
            cell_sampler=neighbor_sampler,
            input_cells=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )

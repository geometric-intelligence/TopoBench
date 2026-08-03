"""Clean-process verifier for fixed sampled evaluation with workers."""

from __future__ import annotations

import gc
import hashlib
import io
import multiprocessing
from contextlib import redirect_stdout

import torch
from torch_geometric.data import HeteroData

with redirect_stdout(io.StringIO()):
    from topobench.data.datasets import make_synthetic_heterogeneous_data
    from topobench.data.heterogeneous import validate_heterogeneous_node_data
    from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule
    from topobench.transforms.data_manipulations.heterogeneous import (
        HeterogeneousConstantFeatures,
        HeterogeneousToUndirected,
    )

_CHOICE_RICH_FANOUT = [2, 3, 1]


def _make_every_relation_choice_rich(data: HeteroData) -> HeteroData:
    """Give every relation more candidates than the largest test fanout."""
    candidate_count = max(_CHOICE_RICH_FANOUT) + 1
    author_count = data["author"].num_nodes
    paper_count = data["paper"].num_nodes
    venue_count = data["venue"].num_nodes
    if author_count is None or paper_count is None or venue_count is None:
        raise SystemExit("sampling fixture node counts must be explicit")

    author_ids = torch.arange(author_count).repeat_interleave(candidate_count)
    author_slots = torch.arange(candidate_count).repeat(author_count)
    paper_ids = (author_ids + author_slots) % paper_count
    data["author", "writes", "paper"].edge_index = torch.stack(
        [author_ids, paper_ids]
    )

    paper_ids = torch.arange(paper_count).repeat_interleave(candidate_count)
    paper_slots = torch.arange(candidate_count).repeat(paper_count)
    venue_ids = (paper_ids + paper_slots) % venue_count
    data["paper", "published_in", "venue"].edge_index = torch.stack(
        [paper_ids, venue_ids]
    )
    return data


def _assert_every_relation_exceeds_fanout(
    data: HeteroData,
    fanout: list[int],
) -> None:
    """Require a genuine neighbor choice for every relation at every hop."""
    required_degree = max(fanout)
    for edge_type in data.edge_types:
        destination_type = edge_type[2]
        destination_count = data[destination_type].num_nodes
        if destination_count is None:
            raise SystemExit(f"{destination_type!r} has no node count")
        in_degree = torch.bincount(
            data[edge_type].edge_index[1],
            minlength=destination_count,
        )
        if not torch.all(in_degree > required_degree):
            raise SystemExit(
                f"{edge_type!r} minimum in-degree {int(in_degree.min())} "
                f"does not exceed every fanout in {fanout!r}"
            )


def _digest(loader: object) -> str:
    """Hash every tensor in traversal order with stable metadata."""
    digest = hashlib.sha256()
    for batch in loader:
        digest.update(repr(batch.metadata()).encode())
        for store_type in (*batch.node_types, *batch.edge_types):
            digest.update(repr(store_type).encode())
            store = batch[store_type]
            for key in sorted(store.keys()):
                value = store[key]
                digest.update(key.encode())
                if torch.is_tensor(value):
                    value = value.detach().cpu().contiguous()
                    digest.update(str(value.dtype).encode())
                    digest.update(repr(tuple(value.shape)).encode())
                    digest.update(value.numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
    return digest.hexdigest()


def main() -> None:
    """Compare worker traversals and prove nonpersistent worker release."""
    data = _make_every_relation_choice_rich(
        make_synthetic_heterogeneous_data(seed=7)
    )
    data = HeterogeneousConstantFeatures(node_types="venue")(data)
    data = HeterogeneousToUndirected(merge=False)(data)
    _assert_every_relation_exceeds_fanout(data, _CHOICE_RICH_FANOUT)
    spec = validate_heterogeneous_node_data(
        data,
        target_node_type="author",
        num_classes=2,
    )
    datamodule = HeterogeneousNodeDataModule(
        data,
        spec,
        mode="neighbor",
        num_neighbors=_CHOICE_RICH_FANOUT,
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        evaluation_seed=59,
    )
    loader = datamodule.val_dataloader()
    if loader.persistent_workers:
        raise SystemExit("fixed evaluation must force nonpersistent workers")
    train_loader = datamodule.train_dataloader()
    if not train_loader.persistent_workers:
        raise SystemExit("training must retain configured persistent workers")
    first = _digest(loader)
    gc.collect()
    if multiprocessing.active_children():
        raise SystemExit("evaluation workers remained after first exhaustion")
    retrieved_loader = datamodule.val_dataloader()
    if retrieved_loader is not loader:
        raise SystemExit("validation loader was reconstructed")
    replay = _digest(retrieved_loader)
    gc.collect()
    if multiprocessing.active_children():
        raise SystemExit("evaluation workers remained after replay exhaustion")
    if first != replay:
        raise SystemExit("fixed sampled evaluation worker replay diverged")
    early_iterator = iter(loader)
    next(early_iterator)
    early_iterator.close()  # type: ignore[attr-defined]
    gc.collect()
    if multiprocessing.active_children():
        raise SystemExit("evaluation workers remained after early close")
    print("worker-replay-ok")


if __name__ == "__main__":
    main()

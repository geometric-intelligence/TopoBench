"""Clean-process verifier for fixed sampled evaluation with workers."""

from __future__ import annotations

import hashlib

import torch

from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.data.heterogeneous import validate_heterogeneous_node_data
from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule
from topobench.transforms.data_manipulations.heterogeneous import (
    HeterogeneousConstantFeatures,
    HeterogeneousToUndirected,
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
    """Execute and compare same-loader and fresh-loader worker traversals."""
    data = make_synthetic_heterogeneous_data(seed=7)
    data = HeterogeneousConstantFeatures(node_types="venue")(data)
    data = HeterogeneousToUndirected(merge=False)(data)
    spec = validate_heterogeneous_node_data(
        data,
        target_node_type="author",
        num_classes=2,
    )
    datamodule = HeterogeneousNodeDataModule(
        data,
        spec,
        mode="neighbor",
        num_neighbors=[1, 1],
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        evaluation_seed=59,
    )
    loader = datamodule.val_dataloader()
    first = _digest(loader)
    replay = _digest(loader)
    fresh_loader_replay = _digest(datamodule.val_dataloader())
    if first != replay or first != fresh_loader_replay:
        raise SystemExit("fixed sampled evaluation worker replay diverged")
    print("worker-replay-ok")


if __name__ == "__main__":
    main()

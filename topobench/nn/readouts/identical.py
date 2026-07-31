"""Native homogeneous readout with no feature transformation."""

from typing import Any

from torch_geometric.data import Data

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class NoReadOut(AbstractZeroCellReadOut):
    """Compute logits directly from native node embeddings."""

    def forward(
        self, model_out: dict[str, Any], batch: Data
    ) -> dict[str, Any]:
        """Return model output unchanged before the base logits head."""
        del batch
        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


__all__ = ["NoReadOut"]

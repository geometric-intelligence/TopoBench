"""Types and validation for sklearn graph wrappers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Required keys for a valid batch (one graph per forward call)
REQUIRED_BATCH_KEYS = frozenset({
    "x_0",
    "y",
    "edge_index",
    "train_mask",
    "val_mask",
    "test_mask",
    "batch_0",
})


class BatchValidationError(ValueError):
    """Raised when a batch fails validation."""

    pass


def validate_batch(batch: Dict[str, Any]) -> None:
    """
    Validate that a batch dictionary has all required keys and valid types.

    The batch is expected to represent a single graph. Masks must be boolean
    tensors or numpy arrays. All tensors should have the same first dimension (num_nodes).

    Parameters
    ----------
    batch : Dict[str, Any]
        The batch dictionary to validate.

    Raises
    ------
    BatchValidationError
        If any required key is missing or validation fails.
    """
    missing = REQUIRED_BATCH_KEYS - set(batch.keys())
    if missing:
        raise BatchValidationError(
            f"Batch is missing required keys: {sorted(missing)}. "
            f"Required keys: {sorted(REQUIRED_BATCH_KEYS)}."
        )

    # Require tensor-like for core fields
    for key in ("x_0", "y", "edge_index", "batch_0"):
        val = batch[key]
        if not isinstance(val, torch.Tensor):
            raise BatchValidationError(
                f"Batch key '{key}' must be a torch.Tensor, got {type(val).__name__}."
            )

    num_nodes = batch["x_0"].shape[0]
    if batch["y"].shape[0] != num_nodes:
        raise BatchValidationError(
            f"Batch 'y' first dimension ({batch['y'].shape[0]}) must match 'x_0' ({num_nodes})."
        )
    if batch["batch_0"].shape[0] != num_nodes:
        raise BatchValidationError(
            f"Batch 'batch_0' first dimension ({batch['batch_0'].shape[0]}) must match 'x_0' ({num_nodes})."
        )

    # Masks: allow (1) boolean or 0/1 per-node mask of length num_nodes,
    # or (2) 1D integer tensor of node indices (any length)
    for key in ("train_mask", "val_mask", "test_mask"):
        val = batch[key]
        if not isinstance(val, torch.Tensor) and not (
            hasattr(val, "__len__") and hasattr(val, "dtype")
        ):
            raise BatchValidationError(
                f"Batch key '{key}' must be a tensor or array, got {type(val).__name__}."
            )
        flat = val.flatten() if hasattr(val, "flatten") else np.asarray(val).ravel()
        n_elems = len(flat)
        if n_elems == num_nodes:
            # Per-node mask: must be boolean or 0/1
            if isinstance(val, torch.Tensor):
                if val.dtype == torch.bool:
                    continue
                if val.dtype in (torch.int64, torch.long, torch.int32, torch.uint8):
                    unique = val.unique().tolist()
                    if not all(u in (0, 1) for u in unique):
                        raise BatchValidationError(
                            f"Batch key '{key}' (mask format) must be boolean or 0/1, got values {unique}."
                        )
                    continue
            else:
                arr = np.asarray(val).ravel()
                if arr.dtype == bool:
                    continue
                if np.issubdtype(arr.dtype, np.integer):
                    unique = np.unique(arr).tolist()
                    if not all(u in (0, 1) for u in unique):
                        raise BatchValidationError(
                            f"Batch key '{key}' (mask format) must be boolean or 0/1."
                        )
                    continue
            raise BatchValidationError(
                f"Batch key '{key}' (mask format) must be boolean or 0/1, got dtype {getattr(val, 'dtype', type(val))}."
            )
        # Index format: 1D integer indices in [0, num_nodes - 1]
        if isinstance(val, torch.Tensor):
            if val.dtype not in (torch.int64, torch.long, torch.int32, torch.uint8):
                raise BatchValidationError(
                    f"Batch key '{key}' (index format) must be integer tensor, got {val.dtype}."
                )
            idx = val.cpu().numpy().ravel()
        else:
            arr = np.asarray(val).ravel()
            if not np.issubdtype(arr.dtype, np.integer):
                raise BatchValidationError(
                    f"Batch key '{key}' (index format) must be integer, got {arr.dtype}."
                )
            idx = arr
        if len(idx) > 0 and (int(idx.min()) < 0 or int(idx.max()) >= num_nodes):
            raise BatchValidationError(
                f"Batch key '{key}' indices must be in [0, {num_nodes - 1}], got min={idx.min()}, max={idx.max()}."
            )


def masks_to_bool(
    batch: Dict[str, Any], num_nodes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert train/val/test masks from batch to boolean per-node arrays.

    Accepts either per-node masks (length num_nodes, bool or 0/1) or index
    arrays (1D integer tensor of node indices). Returns (train_mask, val_mask, test_mask)
    as numpy boolean arrays of shape (num_nodes,).
    """
    out = []
    for key in ("train_mask", "val_mask", "test_mask"):
        val = batch[key]
        a = val.cpu().numpy().ravel() if isinstance(val, torch.Tensor) else np.asarray(val).ravel()
        if a.size == num_nodes:
            mask = a.astype(bool).copy() if a.dtype != bool else np.asarray(a, dtype=bool).copy()
        else:
            mask = np.zeros(num_nodes, dtype=bool)
            if a.size > 0:
                mask[a.astype(np.intp)] = True
        out.append(mask)
    return tuple(out)

from __future__ import annotations

import logging
from typing import Union, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Names used when a check fails
FAIL_NO_NEIGHBORS = "no_neighbors"
FAIL_ALL_CONST = "all_features_constant"
FAIL_ONE_NEIGHBOR = "one_neighbor"
FAIL_SAME_Y = "all_the_neighborns_have_same_y"


def _to_indices(mask_or_indices: Union[np.ndarray, Iterable[int]]) -> np.ndarray:
    """Convert boolean mask to indices, or return indices as array."""
    arr = np.asarray(mask_or_indices)
    if arr.dtype == bool:
        return np.where(arr)[0]
    return arr.reshape(-1)


class CheckerError(Exception):
    """Base class for all Checker errors."""


class NoNeighborsError(CheckerError):
    pass


class AllFeaturesConstantError(CheckerError):
    pass


class OneNeighborError(CheckerError):
    pass


class AllNeighborSameY(CheckerError):
    pass


def _get_constant_columns(X: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array indicating which columns are constant in X.
    If X has only one row, all columns are considered constant.

    Parameters:
        - X (np.ndarray): 2D input array (n_samples, n_features)
    Returns:
        - np.ndarray: Boolean array of shape (n_features,), True for constant columns
    """
    if X.shape[0] == 1:
        return np.ones(X.shape[1], dtype=bool) == 0  # Returns a list of False
    return np.ptp(X, axis=0) == 0  # range == 0 ⇒ constant


class Checker:
    """
    Run neighborhood sanity checks and (if they pass) return the non-constant columns.

    Return:
        - np.ndarray (dtype=bool): mask of non-constant columns (len = n_features)
        - str: name of the failed check
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        *,
        test_ids: Sequence[int],
        neighbors_id: Sequence[int],
        val_mask: Union[np.ndarray, Iterable[int]],
        test_mask: Union[np.ndarray, Iterable[int]],
        node_features: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute the checks and return (X_nb, x_test) or raise a typed error.

        val_mask and test_mask can be boolean arrays (True = included) or index arrays.
        """
        val_ids = _to_indices(val_mask)
        test_ids_arr = _to_indices(test_mask)
        neighbors_arr = np.asarray(neighbors_id, dtype=np.intp)
        test_ids_arr_for_index = np.asarray(test_ids, dtype=np.intp)

        # Overlap checks: neighbors must not include any val or test node index
        assert set(neighbors_arr.flat).isdisjoint(set(val_ids.flat)), (
            "neighbors_id contains indices in val_mask"
        )
        assert set(neighbors_arr.flat).isdisjoint(set(test_ids_arr.flat)), (
            "neighbors_id contains indices in test_mask"
        )

        y_neighborns = labels[neighbors_arr]

        # No neighbors
        if len(neighbors_arr) == 0:
            raise NoNeighborsError(FAIL_NO_NEIGHBORS)

        # Build neighbor feature matrix
        X_nb = node_features[neighbors_arr]  # shape: (k, d)
        x_np = node_features[test_ids_arr_for_index]  # shape: (num_test_points, d)
        X = node_features[np.concatenate([neighbors_arr, test_ids_arr_for_index])]

        # Create mask of "valid" columns
        mask = np.all(np.isfinite(X), axis=0)

        # Keep only valid columns
        X_nb = X_nb[:, mask]
        x_np = x_np[:, mask]

        nan_columns = np.invert(mask)
        if nan_columns.any():
            logger.warning(
                "%s columns contained NaN values; those columns were dropped.",
                int(nan_columns.sum()),
            )

        # One neighbor
        if X_nb.shape[0] == 1:
            raise OneNeighborError(FAIL_ONE_NEIGHBOR)

        # All the neigborns have the same labels
        if len(np.unique(y_neighborns)) == 1:
            raise AllNeighborSameY(FAIL_SAME_Y)

        # Constant columns among neighbors
        const_cols = _get_constant_columns(
            X_nb
        )  # bool mask, True where constant

        if np.all(const_cols):  # all features constant
            raise AllFeaturesConstantError(FAIL_ALL_CONST)

        # Normal case - fit backbone on neighbors
        if np.any(const_cols):  # Remove constant columns for train and test
            X_nb = X_nb[:, ~const_cols]
            x_np = x_np[:, ~const_cols]
        else:
            x_np = x_np

        return X_nb, x_np

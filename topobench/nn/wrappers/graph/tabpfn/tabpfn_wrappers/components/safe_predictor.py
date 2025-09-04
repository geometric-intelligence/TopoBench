# safe_predictor.py
from typing import Tuple, Sequence, Callable, Any
import numpy as np
import torch
import copy

from .checker import (
    CheckerError,
    NoNeighborsError,
    AllNeighborSameY,
    AllFeaturesConstantError,
    OneNeighborError,
)
from .logger import Case
from .checker import Checker


class SafePredictor:
    """
    Encapsulates: run Checker, train a fresh model if needed, handle typed exceptions,
    and return (probs, preds, Case) for a batch of test nodes.
    """

    def __init__(
        self,
        *,
        checker: Checker,
        backbone_factory: Callable[[], Any],
        fallback_no_neighbors: Callable[[], Tuple[np.ndarray, np.ndarray]],
        fallback_one_neighbor: Callable[
            [np.ndarray], Tuple[np.ndarray, np.ndarray]
        ],
        fallback_all_const: Callable[
            [np.ndarray], Tuple[np.ndarray, np.ndarray]
        ],
        get_predictions: Callable[
            [Any, np.ndarray], Tuple[np.ndarray, np.ndarray]
        ],
    ):
        self.checker = checker
        self.backbone_factory = backbone_factory
        self.f_no = fallback_no_neighbors
        self.f_one = fallback_one_neighbor
        self.f_const = fallback_all_const
        self.get_preds = get_predictions

    def predict_batch(
        self,
        *,
        neighbors_id: Sequence[int],
        test_ids: Sequence[int],  # test node ids in this batch (B,)
        node_features: np.ndarray,  # (N, F)
        labels: np.ndarray,  # (N,)
        val_mask: np.ndarray,  # (N,) bool
        test_mask: np.ndarray,  # (N,) bool
    ) -> Tuple[np.ndarray, np.ndarray, Case]:
        """
        Returns:
            probs: (B, C)
            preds: (B,)
            case:  Case enum for stats
        Notes:
            - This assumes Checker(... ) raises typed errors OR returns the
              slice(s) already validated. If your Checker returns different
              things, adapt the two lines marked with 'ADAPT'.
        """
        try:
            # ADAPT: if your Checker returns something else, adjust here.
            X_nb, X_test = self.checker(
                test_ids=test_ids,
                neighbors_id=neighbors_id,
                val_mask=val_mask,
                test_mask=test_mask,
                node_features=node_features,
                labels=labels,
                # optionally return keep_mask, then you build X_nb/X_test using it
            )
            y_nb = labels[np.asarray(neighbors_id)]

            model = self.backbone_factory()  # fresh model per batch
            model.fit(X_nb, y_nb)
            probs, preds = self.get_preds(model, X_test)
            return probs, preds, Case.TRAINED

        except NoNeighborsError:
            probs, preds = self.f_no()
            return probs, preds, Case.NO_NEIGHBORS

        except OneNeighborError:
            y_nb = (
                labels[np.asarray(neighbors_id)]
                if len(neighbors_id)
                else np.empty((0,), dtype=labels.dtype)
            )
            probs, preds = self.f_one(y_nb)
            return probs, preds, Case.ONE_NEIGHBOR

        except AllNeighborSameY:
            y_nb = labels[neighbors_id]
            probs, preds = self.f_one(y_nb)  #  handle it as one neighborn
            return probs, preds, Case.SAME_Y

        except AllFeaturesConstantError:
            y_nb = (
                labels[np.asarray(neighbors_id)]
                if len(neighbors_id)
                else np.empty((0,), dtype=labels.dtype)
            )
            probs, preds = self.f_const(y_nb)
            return probs, preds, Case.ALL_CONST

        except CheckerError as e:
            # surface unexpected checker errors: this is a dev-time failure
            raise

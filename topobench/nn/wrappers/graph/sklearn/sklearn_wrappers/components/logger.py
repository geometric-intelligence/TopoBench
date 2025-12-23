# helpers.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Tuple, Sequence, Any
import numpy as np
import torch
import math


class Case(Enum):
    NO_NEIGHBORS = auto()
    ONE_NEIGHBOR = auto()
    SAME_Y = auto()
    ALL_CONST = auto()
    TRAINED = auto()


@dataclass
class LoggerStats:
    no_neighbors: int = 0
    one_neighbor: int = 0
    same_y: int = 0
    all_const: int = 0
    trained: int = 0
    all_same_label: int = 0  # keep if you use it elsewhere

    def inc(self, case: Case, n: int) -> None:
        if case is Case.NO_NEIGHBORS:
            self.no_neighbors += n
        elif case is Case.ONE_NEIGHBOR:
            self.one_neighbor += n
        elif case is Case.SAME_Y:
            self.same_y += n
        elif case is Case.ALL_CONST:
            self.all_const += n
        elif case is Case.TRAINED:
            self.trained += n
        else:
            raise ValueError(f"Unhandled case {case}")

    def check_invariants(self, num_test_points: int) -> None:
        total = (
            self.no_neighbors
            + self.one_neighbor
            + self.same_y
            + self.all_const
            + self.trained
            + self.all_same_label
        )
        if not math.isclose(
            total, num_test_points, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise AssertionError(
                f"Counters must sum to num_test_points. got={total}, expected={num_test_points}"
            )

    def log(
        self, num_test_points: int, logger: Callable[..., Any] | None
    ) -> None:
        if logger is None:
            return
        pct = lambda x: float(np.round(100.0 * x / max(1, num_test_points), 2))
        logger(
            "test/no_neighbors",
            pct(self.no_neighbors),
            prog_bar=True,
            on_step=False,
        )
        logger(
            "test/one_neighbor",
            pct(self.one_neighbor),
            prog_bar=True,
            on_step=False,
        )
        logger(
            "test/all_neighbors_same_y",
            pct(self.same_y),
            prog_bar=True,
            on_step=False,
        )
        logger(
            "test/all_features_constant",
            pct(self.all_const),
            prog_bar=True,
            on_step=False,
        )
        logger(
            "test/num_all_same_label",
            pct(self.all_same_label),
            prog_bar=True,
            on_step=False,
        )
        logger(
            "test/model_trained",
            pct(self.trained),
            prog_bar=True,
            on_step=False,
        )

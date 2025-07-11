from typing import Optional, Any
import numpy as np
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import torch
from topobench.nn.wrappers.graph.tabpfn.tabpfn_wrappers.base import BaseWrapper


class TabPFNRegressorWrapper(BaseWrapper):
    """
    Regression version of TabPFNWrapper:
    - always returns a scalar prediction per node
    - falls back to the global mean if no neighbours
    - trains backbone per‐node on its sampled neighbours
    """

    def __init__(self, backbone: Any, sampler: Optional[Any] = None, **kwargs):
        super().__init__(backbone, sampler, **kwargs)
        self.global_mean_: float = 0.0

    def _init_targets(self, y_train: np.ndarray) -> None:
        self.global_mean_ = torch.tensor(float(np.mean(y_train)))

    def _handle_no_neighbors(self) -> torch.Tensor:
        return self.global_mean_ , self.global_mean_ 
    
    def _get_prediction(self, model, X) -> torch.Tensor:
        pred = torch.tensor(model.predict(X))
        return pred, pred
    
    def _get_train_prediction(self, y) -> torch.Tensor:
        return y 
    
    def _calculate_metric_pbar(
        self, preds: list, trues: list
    )-> tuple[str, float]:
        mse = mean_squared_error(trues, preds)
        return "MSE", mse


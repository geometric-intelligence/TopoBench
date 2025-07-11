from topobench.nn.wrappers.graph.tabpfn.tabpfn_wrappers.base import BaseWrapper
import torch
import numpy as np
from sklearn.metrics import accuracy_score

class TabPFNClassifierWrapper(BaseWrapper):
    def _init_targets(self, y_train):
        self.classes_ = np.unique(y_train)
        self.num_classes_ = len(self.classes_)
        self.uniform_ = np.ones(len(self.classes_))/len(self.classes_)

    def _handle_no_neighbors(self):
        # return a uniform probability vector
        return torch.from_numpy(self.uniform_).float()

    def _get_prediction(self, model, X) -> torch.Tensor:
        #reshaping to fit the model's expected input shape
        X_reshaped = X.reshape(1, -1)
        raw_proba = model.predict_proba(X_reshaped)
        # reshaping raw_proba to 1D
        raw_proba = raw_proba[0]
        # map local class order → global class order
        full_proba = np.zeros(self.num_classes_, dtype=float)
        for i, cls in enumerate(model.classes_):
            full_proba[cls] = raw_proba[i]

        prob = torch.from_numpy(full_proba).float()
        pred = torch.tensor(self.classes_[full_proba.argmax()])
        return prob, pred
    
    def _get_train_prediction(self, y_train_point) -> torch.Tensor:
        one_hot = np.zeros(self.num_classes_, dtype=float)
        one_hot[np.where(self.classes_ == y_train_point.item())[0][0]] = 1.0
        return one_hot
    
    def _calculate_metric_pbar(
        self, preds: list, trues: list
    )-> tuple[str, float]:
        acc = accuracy_score(trues, preds)
        return "accuracy", acc
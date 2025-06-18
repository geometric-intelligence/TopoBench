
from typing import Optional, Any, Dict
import numpy as np
import networkx as nx
from copy import deepcopy
import torch
from tqdm import tqdm

class TabPFNWrapper(torch.nn.Module):
    """
    Wrapper for TabPFN that delegates neighbor sampling to a BaseSampler.
    """

    def __init__(
        self,
        backbone: Any,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.sampler = kwargs.get("sampler", None)
        self.use_embeddings = kwargs.get("use_embeddings", True)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

        # Counters
        self.n_no_neighbors = 0
        self.n_features_constant = 0
        self.n_model_trained = 0

    # Methods NOT USED
    def fit(
        self, x: np.ndarray, y: np.ndarray, graph: Optional[nx.Graph] = None
    ) -> "TabPFNWrapper":
        pass

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # if self.X_train is None or self.y_train is None:
        #    raise RuntimeError("Model has not been fitted. Call `fit` first.")
        train_mask = batch.get("train_mask", None).cpu().numpy()

        # encoded node features
        rank0_features = [ "x_0" ]

        if self.use_embeddings:
            all_keys = batch.keys() 
            # adding all the rank0 features 
            rank0_features.extend([s for s in all_keys if s.startswith('x0')])

            #deleting raw node features (we have alredy the encoded ones)
            rank0_features.remove('x0_0')

        # Concatenate tensors along the column dimension (dim=1)
        tensors_to_concat = [batch[k] for k in rank0_features]
        tensor_features = torch.cat(tensors_to_concat, dim=1)

        node_features = tensor_features.cpu().numpy().copy()
        labels = batch["y"].cpu().numpy().copy()

        X_train = node_features[train_mask]
        y_train = labels[train_mask]

        # Record unique class labels for probability vectors
        classes_ = np.unique(y_train)
        n_classes = len(classes_)

        # If sample is None training the network on the whole dataset
        if self.sampler is None:
            self.backbone.fit(X_train, y_train)
            # Predict probabilities for the ehole dataset (to allow compatibility with the rest of the code)
            prob = self.backbone.predict_proba(node_features)
            # Convert probabilities to float tensor to the right device
            prob_tensor = (
                torch.from_numpy(prob).float().to(batch["x_0"].device)
            )
            return {
                "labels": batch["y"],
                "batch_0": batch["batch_0"],
                "x_0": prob_tensor,
            }
        
        # extracting the graph information
        tail, head = batch["edge_index"].cpu().numpy()[0], batch["edge_index"].cpu().numpy()[1]

        # Fit sampler
        self.sampler.fit(
            X_train, 
            self.y_train, 
            edge_index=zip(tail, head),
            train_mask=train_mask.tolist()
        )

        probs = []
        for idx, x_np in tqdm(enumerate(node_features), total=len(node_features), desc="Sampling and predicting"):
            if idx in train_mask:
                # If the sample is in the training set, return one-hot encoded probabilities
                one_hot = np.zeros(n_classes, dtype=float)
                one_hot[np.where(classes_ == batch.y[idx].item())[0][0]] = 1.0
                probs.append(torch.tensor(one_hot, dtype=torch.float32))
                continue


            # Sample neighbor indices
            nbr_idx = self.sampler.sample(x_np, idx)
            if len(nbr_idx) == 0:
                self.n_no_neighbors += 1
                uniform = np.full(
                    n_classes, 1.0 / n_classes, dtype=float
                )
                probs.append(torch.from_numpy(uniform).float())
                continue

            X_nb = node_features[nbr_idx]
            y_nb = labels[nbr_idx]

            # Remove constant features
            const_cols = np.ptp(X_nb, axis=0) == 0
            if np.all(const_cols):
                self.n_features_constant += 1
                counts = np.array(
                    [np.sum(y_nb == c) for c in classes_], dtype=float
                )
                dist = counts / counts.sum()
                probs.append(torch.from_numpy(dist).float())
                continue

            if np.any(const_cols):
                X_nb = X_nb[:, ~const_cols]
                x_filtered = x_np[~const_cols]
            else:
                x_filtered = x_np

            # Fit backbone on neighbors
            model = deepcopy(self.backbone)
            model.fit(X_nb, y_nb)
            self.n_model_trained += 1

            raw_proba = model.predict_proba(x_filtered.reshape(1, -1))[0]
            full_proba = np.zeros(n_classes, dtype=float)
            for i, cls in enumerate(model.classes_):
                global_idx = np.where(classes_ == cls)[0][0]
                full_proba[global_idx] = raw_proba[i]
            probs.append(torch.from_numpy(full_proba).float())

        prob_tensor = torch.stack(probs).to(batch["x_0"].device)
        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": prob_tensor,
        }

    def __call__(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.forward(batch)

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
        # sampler: Any,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_embeddings = kwargs.get("use_embeddings", True)
        self.sampler = kwargs.get("sampler", None)

        # Counters
        self.n_no_neighbors = 0
        self.n_features_constant = 0
        self.n_model_trained = 0

    def fit(
        self, x: np.ndarray, y: np.ndarray, graph: Optional[nx.Graph] = None
    ) -> "TabPFNWrapper":
        pass

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        train_mask = batch.get("train_mask", None)

        # encoded node features
        rank0_features = []

        if self.use_embeddings:
            all_keys = batch.keys()
            # adding all the rank0 features
            rank0_features.extend([s for s in all_keys if s.startswith("x0")])
        else:
            rank0_features = ["x0_0"]

        # Concatenate tensors along the column dimension (dim=1)
        tensors_to_concat = [batch[k] for k in rank0_features]
        tensor_features = torch.cat(tensors_to_concat, dim=1)

        node_features = tensor_features.cpu().numpy().copy()
        labels = batch["y"].cpu().numpy()

        X_train = node_features[train_mask].copy()
        y_train = batch["y"][train_mask].cpu().numpy().copy()

        # Record unique class labels for probability vectors
        classes_ = np.unique(y_train)
        n_classes = len(classes_)

        edge_index = batch["edge_index"].cpu().numpy()

        self.classes_ = np.unique(y_train)

        if self.sampler is None:
            # If sample is None training the network on the whole dataset
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

        # Fit sampler
        self.sampler.fit(
            node_features, labels, edge_index=edge_index, train_mask=train_mask
        )

        probs = []
        correct = 0  # # of correct predictions so far
        total_pred = 0  # # of *evaluated* points so far (excl. train_mask)

        # create the progress bar once
        pbar = tqdm(
            total=len(node_features),
            desc="Sampling and predicting",
            dynamic_ncols=True,
        )

        for idx, x_np in enumerate(node_features):
            # ─────────────────────────────────────────
            # Training-set items → one-hot + skip
            # ─────────────────────────────────────────
            if idx in train_mask:
                one_hot = np.zeros(n_classes, dtype=float)
                one_hot[np.where(classes_ == batch.y[idx].item())[0][0]] = 1.0
                probs.append(torch.tensor(one_hot, dtype=torch.float32))
                pbar.update(1)  # advance bar
                continue  # DO NOT count toward accuracy

            # ─────────────────────────────────────────
            # Sample neighbours
            # ─────────────────────────────────────────
            nbr_idx = self.sampler.sample(x_np, idx)

            if len(nbr_idx) == 0:
                # no neighbours → uniform distribution
                self.n_no_neighbors += 1
                uniform = np.full(
                    len(self.classes_), 1.0 / len(self.classes_), dtype=float
                )
                probs.append(torch.from_numpy(uniform).float())

                pred_label = self.classes_[
                    uniform.argmax()
                ]  # class with max prob
                true_label = batch.y[idx].item()
                correct += pred_label == true_label
                total_pred += 1

                # update bar
                pbar.update(1)
                pbar.set_postfix(acc=f"{correct / total_pred:.2%}")
                continue

            X_nb = node_features[nbr_idx]
            y_nb = labels[nbr_idx]

            # ─────────────────────────────────────────
            # Handle constant columns
            # ─────────────────────────────────────────
            const_cols = np.ptp(X_nb, axis=0) == 0  # range = 0 ⇒ constant
            if np.all(const_cols):
                self.n_features_constant += 1
                counts = np.array(
                    [np.sum(y_nb == c) for c in self.classes_], dtype=float
                )
                dist = counts / counts.sum()  # neighbour label frequencies
                probs.append(torch.from_numpy(dist).float())

                pred_label = self.classes_[dist.argmax()]
                true_label = batch.y[idx].item()
                correct += pred_label == true_label
                total_pred += 1

                pbar.update(1)
                pbar.set_postfix(acc=f"{correct / total_pred:.2%}")
                continue

            # ─────────────────────────────────────────
            # Fit backbone on neighbours
            # ─────────────────────────────────────────
            if np.any(const_cols):
                X_nb = X_nb[:, ~const_cols]
                x_filtered = x_np[~const_cols]
            else:
                x_filtered = x_np

            model = deepcopy(self.backbone)
            model.fit(X_nb, y_nb)
            self.n_model_trained += 1

            raw_proba = model.predict_proba(x_filtered.reshape(1, -1))[0]

            # map local class order → global class order
            full_proba = np.zeros(len(self.classes_), dtype=float)
            for i, cls in enumerate(model.classes_):
                full_proba[np.where(self.classes_ == cls)[0][0]] = raw_proba[i]

            probs.append(torch.from_numpy(full_proba).float())

            pred_label = self.classes_[full_proba.argmax()]
            true_label = batch.y[idx].item()
            correct += pred_label == true_label
            total_pred += 1

            # ─────────────────────────────────────────
            # Update progress bar
            # ─────────────────────────────────────────
            pbar.update(1)
            pbar.set_postfix(acc=f"{correct / total_pred:.2%}")

        # close bar (optional when loop ends)
        pbar.close()

        # stack & return exactly like before
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

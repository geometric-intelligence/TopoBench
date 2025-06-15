import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy


class TabPFNWrapper(torch.nn.Module):
    r"""Wrapper for the TabPFN model that returns prediction probabilities as a tensor."""

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.k = kwargs.get("k", 5)
        self._nn = None
        self.X_train = None
        self.y_train = None
        self.classes_ = None

        # Counters
        self.n_no_neighbors = 0
        self.n_features_constant = 0
        self.n_model_trained = 0

    def fit(self, x, y):
        """
        Fit the NearestNeighbors on training data and store class labels.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        self.X_train = np.asarray(x)
        self.y_train = np.asarray(y)
        # Record unique class labels for probability vectors
        self.classes_ = np.unique(self.y_train)
        self._nn = NearestNeighbors(n_neighbors=self.k)
        self._nn.fit(self.X_train)
        return self

    def forward(self, batch):
        """
        Generate prediction probabilities for the batch.

        Parameters
        ----------
        batch : dict
            Must contain:
                - 'x_0': tensor or array of features, shape (N, F)
                - 'test_mask': boolean mask selecting test samples

        Returns
        -------
        dict
            Same structure as input batch with 'x_0' replaced by a tensor of shape
            (n_test, n_classes) containing predicted probabilities.
        """
        # Extract test samples
        test_mask = batch["test_mask"]

        # Ensure model is fitted
        # if self.X_train is None or self.y_train is None or self._nn is None:
        # raise ValueError("Model has not been fitted. Call `fit` first.")

        train_mask = batch["train_mask"]
        self.X_train = np.asarray(batch.x_0[train_mask].cpu().numpy())
        self.y_train = np.asarray(batch.y[train_mask].cpu().numpy())
        # Record unique class labels for probability vectors
        self.classes_ = np.unique(self.y_train)
        self._nn = NearestNeighbors(n_neighbors=self.k)
        self._nn.fit(self.X_train)

        n_classes = len(self.classes_)
        probs = torch.zeros(batch.x_0.shape[0], torch.unique(batch.y).shape[0])

        for idx in test_mask:
            # Get the features of testing sample
            x_0 = batch.x_0[idx].cpu().numpy()

            # Some logic below: TODO
            if idx in train_mask:
                # If the sample is in the training set, return one-hot encoded probabilities
                one_hot = np.zeros(n_classes, dtype=float)
                one_hot[
                    np.where(self.classes_ == batch.y[mask][idx].item())[0][0]
                ] = 1.0
                probs.append(torch.tensor(one_hot, dtype=torch.float32))
                continue

            # Find k nearest neighbors
            nbr_idx = self._nn.kneighbors(
                x_np.reshape(1, -1), return_distance=False
            )[0]

            # No neighbors -> uniform probability
            if nbr_idx.size == 0:
                self.n_no_neighbors += 1
                probs.append(torch.full((n_classes,), 1.0 / n_classes))
                continue

            X_nb = self.X_train[nbr_idx]
            y_nb = self.y_train[nbr_idx]

            # Remove constant columns
            const_cols = np.ptp(X_nb, axis=0) == 0
            if np.any(const_cols):
                X_nb = X_nb[:, ~const_cols]
                x_filtered = x_np[~const_cols]
            else:
                x_filtered = x_np

            # all the features of the neighbors are constant (but labels may differ)
            # This also works as a shortcut if all neighbor labels identical -> one-hot
            if X_nb.shape[1] == 0:
                self.n_features_constant += 1
                # count how many neighbors belong to each class
                counts = np.array(
                    [np.sum(y_nb == c) for c in self.classes_], dtype=float
                )
                # normalize to get a probability distribution
                dist = counts / counts.sum()
                probs.append(torch.tensor(dist, dtype=torch.float32))
                continue

            # Else, fit a fresh backbone and compute probabilities
            model = deepcopy(self.backbone)
            model.fit(X_nb, y_nb)
            self.n_model_trained += 1
            raw_proba = model.predict_proba(x_filtered.reshape(1, -1))[0]
            # Align to full class set
            full_proba = np.zeros(n_classes, dtype=float)
            for idx_local, cls in enumerate(model.classes_):
                global_idx = np.where(self.classes_ == cls)[0][0]
                full_proba[global_idx] = raw_proba[idx_local]

            # Right assigment of output
            # Assign the probabilities to the output tensor at right position (idx)
            probs[idx] = torch.tensor(full_proba, dtype=torch.float32)

        # Stack into (n_test, n_classes)
        prob_tensor = torch.stack(probs)

        # Return updated output dict
        model_out = {
            "labels": batch.y,
            "batch_0": batch.batch_0,
            "x_0": prob_tensor.to(
                batch.x_0.device
            ),  # Ensure same device as input
        }
        return model_out

    def __call__(self, batch):
        return self.forward(batch)

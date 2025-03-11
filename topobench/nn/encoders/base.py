"""Abstract class for feature encoders."""

from abc import abstractmethod

import torch
import torch_geometric


class AbstractFeatureEncoder(torch.nn.Module):
    r"""Abstract class to define a custom feature encoder."""

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass of the feature encoder model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x features.
        """

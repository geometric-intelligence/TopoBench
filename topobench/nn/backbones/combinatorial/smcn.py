"""Pipeline-safe SMCN placeholder backbone."""

import torch


class SMCN(torch.nn.Module):
    """Skeleton Scalable Multi-Cellular Network backbone.

    This placeholder only applies rank-wise linear updates so the TopoBench
    pipeline can validate combinatorial model integration.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        neighborhoods=None,
        layers=1,
        activation="relu",
    ):
        super().__init__()
        self.neighborhoods = neighborhoods or []
        self.layers = layers

        activation_layer = self._get_activation(activation)
        self.rank_updates = torch.nn.ModuleDict(
            {
                str(rank): self._make_rank_update(
                    in_channels, hidden_channels, layers, activation_layer
                )
                for rank in range(3)
            }
        )

    @staticmethod
    def _get_activation(name):
        activations = {
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "tanh": torch.nn.Tanh,
            "identity": torch.nn.Identity,
            None: torch.nn.Identity,
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        return activations[name]

    @staticmethod
    def _make_rank_update(
        in_channels, hidden_channels, layers, activation_layer
    ):
        modules = []
        current_channels = in_channels
        for layer_idx in range(max(layers, 1)):
            modules.append(torch.nn.Linear(current_channels, hidden_channels))
            current_channels = hidden_channels
            if layer_idx < max(layers, 1) - 1:
                modules.append(activation_layer())
        return torch.nn.Sequential(*modules)

    def forward(self, batch):
        """Apply placeholder rank-wise updates to available cell features."""
        outputs = {}
        for rank in range(3):
            x = getattr(batch, f"x_{rank}", None)
            if x is not None:
                outputs[rank] = self.rank_updates[str(rank)](x)
        return outputs

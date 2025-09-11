import torch
import torch.nn as nn
from torch.nn import Module
from torch_geometric.nn import GraphNorm

from topobench.nn.backbones.graph.sheaf_model_utils.inductive_discrete_models import (
    InductiveDiscreteBundleSheafDiffusion,
    InductiveDiscreteDiagSheafDiffusion,
    InductiveDiscreteGeneralSheafDiffusion,
)


class NSD(Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        sheaf_type="diag",
        d=2,
        num_layers=2,
        dropout=0.1,
        input_dropout=0.1,
        device="cpu",
        normalised=False,
        deg_normalised=False,
        linear=False,
        left_weights=True,
        right_weights=True,
        sparse_learner=False,
        use_act=True,
        sheaf_act="tanh",
        second_linear=False,
        orth="cayley", # SWEEP: [euler, cayley, householder, matrix_exp]
        edge_weights=False,
        max_t=1.0,
        add_lp=False,
        add_hp=False,
        pe_type=None,
        pe_dim=16,
        pe_norm=False,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sheaf_type = sheaf_type
        self.d = d
        self.num_layers = num_layers
        self.device = device

        # PE configuration
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        self.pe_norm = pe_norm

        # Input projection setup similar to GPS
        if pe_type in ["laplacian", "degree", "rwse"]:
            # If using PE, split channels between node features and PE
            self.node_proj = nn.Linear(input_dim, hidden_dim - pe_dim)
            self.pe_proj = nn.Linear(pe_dim, pe_dim)
            if self.pe_norm:
                print("Using GraphNorm for PE normalization")
                self.pe_norm_layer = GraphNorm(pe_dim)
            else:
                self.pe_norm_layer = None
            self.actual_input_dim = (
                hidden_dim  # Full hidden_dim after concatenation
            )
        elif pe_type is None or pe_type == "None":
            # No PE, use full hidden_dim for node features
            self.node_proj = nn.Identity()  # nn.Linear(input_dim, hidden_dim)
            self.pe_proj = None
            self.pe_norm_layer = None
            self.actual_input_dim = hidden_dim
        else:
            raise ValueError(
                f"Invalid PE type: {pe_type}. Supported: 'laplacian', 'degree', 'rwse', 'None"
            )

        if sheaf_type == "diag":
            assert d >= 1
            self.sheaf_class = InductiveDiscreteDiagSheafDiffusion
        elif sheaf_type == "bundle":
            assert d > 1
            self.sheaf_class = InductiveDiscreteBundleSheafDiffusion
        elif sheaf_type == "general":
            assert d > 1
            self.sheaf_class = InductiveDiscreteGeneralSheafDiffusion
        else:
            raise ValueError(f"Unknown sheaf type: {sheaf_type}")

        self.sheaf_config = {
            "d": d,
            "layers": num_layers,
            "hidden_channels": hidden_dim // d,
            "input_dim": self.actual_input_dim,  # Use adjusted input dimension
            "output_dim": hidden_dim,
            "device": device,
            "normalised": normalised,
            "deg_normalised": deg_normalised,
            "linear": linear,
            "input_dropout": input_dropout,
            "dropout": dropout,
            "left_weights": left_weights,
            "right_weights": right_weights,
            "sparse_learner": sparse_learner,
            "use_act": use_act,
            "sheaf_act": sheaf_act,
            "second_linear": second_linear,
            "orth": orth,
            "edge_weights": edge_weights,
            "max_t": max_t,
            "add_lp": add_lp,
            "add_hp": add_hp,
            "graph_size": None,
        }

        # Create the sheaf model immediately (no lazy initialization)
        self.sheaf_model = self.sheaf_class(self.sheaf_config)

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        edge_weight=None,
        batch=None,
        pe: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Forward pass with PE handling

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes (optional)
            edge_weight: Edge weights (optional)
            batch: Batch indices [num_nodes] (optional)
            pe: Positional encodings [num_nodes, pe_dim] (optional)

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Process node features similar to GPS
        h = self.node_proj(x)

        # Add positional encoding if available
        if self.pe_proj is not None:
            if pe is not None:
                # Normalize PE
                if self.pe_norm_layer is not None and pe.size(0) > 1:
                    pe_normalized = self.pe_norm_layer(pe, batch=batch)
                else:
                    pe_normalized = pe

                # Project PE
                pe_proj = self.pe_proj(pe_normalized)

                # Concatenate node features with PE
                h = torch.cat([h, pe_proj], dim=-1)
            else:
                raise ValueError(
                    "If PE type is not None, PE must be provided in forward pass"
                )

        # Pass processed features to sheaf model
        return self.sheaf_model(h, edge_index)

    def get_sheaf_model(self):
        return self.sheaf_model

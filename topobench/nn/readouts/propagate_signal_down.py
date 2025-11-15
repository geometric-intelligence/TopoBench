"""Readout layer that propagates the signal from cells of a certain order to the cells of the lower order."""

import topomodelx
import torch
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class PropagateSignalDown(AbstractZeroCellReadOut):
    r"""Propagate signal down readout layer.

    This readout layer propagates the signal from cells of a certain order to the cells of the lower order.
    It supports two modes:
    - Hierarchical (default): signals propagate sequentially from rank i → i-1 → ... → 0
    - Direct: signals from each rank propagate independently and directly to rank 0 (nodes)

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments. It should contain the following keys:
        - num_cell_dimensions (int): Highest order of cells considered by the model.
        - hidden_dim (int): Dimension of the cells representations.
        - readout_name (str): Readout name.
        - hierarchical_propagation (bool, optional): If True (default), propagate hierarchically.
          If False, propagate each rank directly to nodes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs["readout_name"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.hierarchical_propagation = kwargs.get("hierarchical_propagation", True)
        self.ranks_to_propagate = kwargs.get("ranks_to_propagate")
        self.dimensions = self.ranks_to_propagate if (self.ranks_to_propagate is not None and not self.hierarchical_propagation) else range(kwargs["num_cell_dimensions"] - 1, 0, -1) #

        # For direct propagation, we need a final projection layer
        if not self.hierarchical_propagation and len(self.dimensions) > 0:
            # Input: original nodes + contributions from all ranks
            # Total: hidden_dim * (1 + len(dimensions))
            final_input_dim = self.hidden_dim * (1 + len(self.dimensions))
            self.final_projector = torch.nn.Linear(final_input_dim, self.hidden_dim)

        for i in self.dimensions:
            setattr(
                self,
                f"agg_conv_{i}",
                topomodelx.base.conv.Conv(
                    self.hidden_dim, self.hidden_dim, aggr_norm=False
                ),
            )

            setattr(self, f"ln_{i}", torch.nn.LayerNorm(self.hidden_dim))

            if self.hierarchical_propagation:
                # Hierarchical: project from rank i to rank i-1
                setattr(
                    self,
                    f"projector_{i}",
                    torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                )
            else:
                # Direct: all ranks combine with existing node features
                setattr(
                    self,
                    f"projector_{i}",
                    torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                )

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the propagate signal down readout layer.

        In hierarchical mode (default), the layer takes the embeddings of the cells of a certain order
        and applies a convolutional layer to them. Layer normalization is then applied to the features.
        The output is concatenated with the initial embeddings of the cells and the result is projected
        with the use of a linear layer to the dimensions of the cells of lower rank. The process is
        repeated until the nodes embeddings, which are the cells of rank 0, are reached.

        In direct mode (hierarchical_propagation=False), signals from each rank are independently
        propagated directly to nodes (rank 0) and then aggregated.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        if self.hierarchical_propagation:
            # Hierarchical propagation: rank i → rank i-1 → ... → rank 0
            # ATTENTION! THIS ONLY WORKS FOR COLORED HYPERGRAPHS FOR NOW, INCIDENCES RELATE TO RANK 0
            for i in self.dimensions:
                x_i = getattr(self, f"agg_conv_{i}")(
                    model_out[f"x_{i}"], batch[f"incidence_{i}"]
                )
                x_i = getattr(self, f"ln_{i}")(x_i)
                model_out[f"x_{i - 1}"] = getattr(self, f"projector_{i}")(
                    torch.cat([x_i, model_out[f"x_{i - 1}"]], dim=1)
                )
        else:
            # Direct propagation: each rank i → rank 0 independently
            node_contributions = [model_out["x_0"]]  # Start with original node features
            
            for i in self.dimensions:
                # Apply convolution and normalization
                x_i = getattr(self, f"agg_conv_{i}")(
                    model_out[f"x_{i}"], batch[f"incidence_{i}"]
                )
                x_i = getattr(self, f"ln_{i}")(x_i)
                
                # All ranks combine with existing node features
                x_i_projected = getattr(self, f"projector_{i}")(
                    torch.cat([x_i, model_out["x_0"]], dim=1)
                )
                node_contributions.append(x_i_projected)
            
            # Aggregate all contributions (original nodes + all rank contributions)
            # Concatenate all and project to hidden_dim
            aggregated = torch.cat(node_contributions, dim=1)
            model_out["x_0"] = self.final_projector(aggregated) if len(self.dimensions) > 0 else model_out["x_0"]

        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_cell_dimensions={len(self.dimensions)}, hidden_dim={self.hidden_dim}, readout_name={self.name}, hierarchical_propagation={self.hierarchical_propagation})"

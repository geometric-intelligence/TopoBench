"""Readout layer that propagates the signal from cells of a certain order to the cells of the lower order."""

import topomodelx
import torch
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class PropagateSignalDown(AbstractZeroCellReadOut):
    r"""Propagate signal down readout layer.

    This readout layer propagates the signal from cells of a certain order to the cells of the lower order.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments. It should contain the following keys:
        - num_cell_dimensions (int): Highest order of cells considered by the model.
        - self.hidden_dim (int): Dimension of the cells representations.
        - readout_name (str): Readout name.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs["readout_name"]
        self.dimensions = range(kwargs["num_cell_dimensions"] - 1, 0, -1)
        self.hidden_dim = kwargs["hidden_dim"]

        for i in self.dimensions:
            setattr(
                self,
                f"agg_conv_{i}",
                topomodelx.base.conv.Conv(
                    self.hidden_dim, self.hidden_dim, aggr_norm=False
                ),
            )

            setattr(self, f"ln_{i}", torch.nn.LayerNorm(self.hidden_dim))

            setattr(
                self,
                f"projector_{i}",
                torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the propagate signal down readout layer.

        The layer takes the embeddings of the cells of a certain order and applies a convolutional layer to them. Layer normalization is then applied to the features. The output is concatenated with the initial embeddings of the cells and the result is projected with the use of a linear layer to the dimensions of the cells of lower rank. The process is repeated until the nodes embeddings, which are the cells of rank 0, are reached.

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
        for i in self.dimensions:
            x_i = getattr(self, f"agg_conv_{i}")(
                model_out[f"x_{i}"], batch[f"incidence_{i}"]
            )
            x_i = getattr(self, f"ln_{i}")(x_i)
            model_out[f"x_{i - 1}"] = getattr(self, f"projector_{i}")(
                torch.cat([x_i, model_out[f"x_{i - 1}"]], dim=1)
            )

        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_cell_dimensions={len(self.dimensions)}, self.hidden_dim={self.hidden_dim}, readout_name={self.name}"

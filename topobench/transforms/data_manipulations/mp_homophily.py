"""A transform that canculates message passing homophily of the input hypergraph."""

import torch
import torch_geometric


class MessagePassingHomophily(torch_geometric.transforms.BaseTransform):
    r"""Calculates message passing homophily of the input data.

    This transformation implements the methodology from the paper:
    "Hypergraph Neural Networks through the Lens of Message Passing: A Common Perspective to Homophily and Architecture Design".
    It computes homophily metrics for hypergraphs by analyzing the relationship between
    node labels within hyperedges.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional parameters for the transform.
        - top_k : int, default=3
            Number of top hyperedge cardinalities to analyze.

    Attributes
    ----------
    type : str
        Identifier for the transform type.
    top_k : int
        Number of top hyperedge cardinalities to analyze.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "calcualte_message_passing_homophily"
        self.num_steps = kwargs.get("num_steps", 3)
        self.incidence_field = kwargs.get(
            "incidence_field", "incidence_hyperedges"
        )

        assert self.incidence_field in [
            "incidence_hyperedges",
            "incidence_0",
            "incidence_1",
            "incidence_2",
        ], (
            f"Incidence field must be one of ['incidence_hyperedges', 'incidence_0', 'incidence_1', 'incidence_2'], but got {self.incidence_field}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """

        H = data[f"{self.incidence_field}"].to_dense()

        unique_labels = torch.unique(data.y)

        # Ep - (num_edges, num_classes) matrix representing the class probability distribution within every edge
        Ep = torch.zeros((self.num_steps, H.shape[1], unique_labels.shape[0]))

        # Np - (num_nodes, num_classes) matrix representing the class probability distribution within every node
        Np = torch.zeros((self.num_steps, H.shape[0], unique_labels.shape[0]))
        for step in range(self.num_steps):
            if step == 0:
                for i in range(H.shape[1]):
                    row = H.T[i]

                    node_indices = torch.where(row == 1)[0]
                    node_labels = data.y[node_indices]

                    for cl in unique_labels:
                        Ep[step, i, cl] = (
                            torch.sum(node_labels == cl) / node_labels.shape[0]
                        )
            else:
                for i in range(H.shape[1]):
                    row = H.T[i]

                    node_indices = torch.where(row == 1)[0]
                    node_probs = Np[step - 1, node_indices, :]

                    Ep[step, i, :] = node_probs.mean(dim=0)

            for i in range(H.shape[0]):
                row = H[i]

                node_indices = torch.where(row == 1)[0]
                node_probabilities = Ep[step, node_indices, :]

                Np[step, i, :] = node_probabilities.mean(dim=0)

        out = {"Ep": Ep, "Np": Np}
        data["mp_homophily"] = out

        return data

"""Node degrees transform."""

import torch_geometric


class NodeDegrees(torch_geometric.transforms.BaseTransform):
    r"""A transform that calculates the node degrees of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "node_degrees"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

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
        field_to_process = [
            key
            for key in data.to_dict()
            for field_substring in self.parameters["selected_fields"]
            if field_substring in key and key != "incidence_0"
        ]
        for field in field_to_process:
            data = self.calculate_node_degrees(data, field)

        return data

    def calculate_node_degrees(
        self, data: torch_geometric.data.Data, field: str
    ) -> torch_geometric.data.Data:
        r"""Calculate the node degrees of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        field : str
            The field to calculate the node degrees.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        if data[field].is_sparse:
            degrees = abs(data[field].to_dense()).sum(1)
        else:
            assert field == "edge_index", (
                "Following logic of finding degrees is only implemented for edge_index"
            )

            # Get number of nodes
            if data.get("num_nodes", None):
                max_num_nodes = data["num_nodes"]
            else:
                max_num_nodes = data["x"].shape[0]
            degrees = (
                torch_geometric.utils.to_dense_adj(
                    data[field],
                    max_num_nodes=max_num_nodes,
                )
                .squeeze(0)
                .sum(1)
            )

        if "incidence" in field:
            field_name = (
                str(int(field.split("_")[1]) - 1) + "_cell" + "_degrees"
            )
        else:
            field_name = "node_degrees"

        data[field_name] = degrees.unsqueeze(1)
        return data

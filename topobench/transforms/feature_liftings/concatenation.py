"""Concatenation feature lifting."""

import torch
import torch_geometric


class Concatenation(torch_geometric.transforms.BaseTransform):
    r"""Lift r-cell features to r+1-cells by concatenation.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Concatenate r-cell features to obtain r+1-cell features.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        keys = sorted(
            [
                key.split("_")[1]
                for key in data
                if "incidence" in key and "-" not in key
            ]
        )
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                incidence = data["incidence_" + elem]
                _, n = incidence.shape

                if n != 0:
                    idxs_list = []
                    for n_feature in range(n):
                        idxs_for_feature = incidence.indices()[
                            0, incidence.indices()[1, :] == n_feature
                        ]
                        idxs_list.append(torch.sort(idxs_for_feature)[0])

                    idxs = torch.stack(idxs_list, dim=0)
                    values = data[f"x_{idx_to_project}"][idxs].view(n, -1)
                else:
                    m = data[f"x_{int(elem) - 1}"].shape[1] * (int(elem) + 1)
                    values = torch.zeros([0, m])

                data["x_" + elem] = values
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Apply the lifting to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        data = self.lift_features(data)
        return data

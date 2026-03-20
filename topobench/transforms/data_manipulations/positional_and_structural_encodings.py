"""Combined Positional and Structural Encodings Transform."""

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class CombinedPSEs(BaseTransform):
    r"""
    Combined PSEs transform.

    Applies one or more pre-defined positional or structural encoding transforms
    (LapPE, RWSE) to a graph, storing their outputs and optionally
    concatenating them to `data.x`.

    Parameters
    ----------
    encodings : list of str
        List of structural encodings to apply. Supported values are
        "LapPE" for Laplacian Positional Encoding and "RWSE" for
        Random Walk Structural Encoding.
    parameters : dict, optional
        Additional parameters for the encoding transforms.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        encodings: list[str],
        parameters: dict | None = None,
        **kwargs,
    ):
        self.encodings = encodings
        self.parameters = parameters if parameters is not None else {}

    def forward(self, data: Data) -> Data:
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with added structural encodings.
        """
        from topobench.transforms.data_manipulations import (
            RWSE,
            ElectrostaticPE,
            HKdiagSE,
            LapPE,
        )

        for enc in self.encodings:
            if enc == "LapPE":
                lappe = LapPE(**self.parameters.get("LapPE", {}))
                data = lappe(data)
            elif enc == "RWSE":
                rwse = RWSE(**self.parameters.get("RWSE", {}))
                data = rwse(data)
            elif enc == "ElectrostaticPE":
                electrostatic_pe = ElectrostaticPE(
                    **self.parameters.get("ElectrostaticPE", {})
                )
                data = electrostatic_pe(data)
            elif enc == "HKdiagSE":
                hkdiag_se = HKdiagSE(**self.parameters.get("HKdiagSE", {}))
                data = hkdiag_se(data)
            else:
                raise ValueError(f"Unsupported encoding type: {enc}")

        return data


class SelectDestinationPSEs(BaseTransform):
    r"""
    Select Destination Positional and Structural Encodings (PSEs) transform.

    Selects and retains only the PSEs corresponding to the destination nodes
    of edges in `data.edge_index`.

    Parameters
    ----------
    encoding_key : str
        Key in `data` where the PSEs are stored (e.g., 'LapPE', 'RWSE').
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, encodings, **kwargs):
        self.encodings = encodings

    def forward(self, data: Data, n_dst_nodes: int) -> Data:
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        n_dst_nodes : int
            Number of destination nodes.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with selected PSEs.
        """
        new_data = {}
        new_data["x"] = data.x[:n_dst_nodes, :] if data.x is not None else None
        for encoding_key in self.encodings:
            if hasattr(data, encoding_key):
                pe = getattr(data, encoding_key)
                selected_pe = pe[:n_dst_nodes, :]
                new_data[encoding_key] = selected_pe
            else:
                raise ValueError(
                    f"Encoding key '{encoding_key}' not found in data."
                )
        return Data(**new_data)

    def __call__(self, data: Data, n_dst_nodes: int) -> Data:
        return self.forward(data, n_dst_nodes)

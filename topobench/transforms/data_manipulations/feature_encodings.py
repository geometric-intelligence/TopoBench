"""Combined Feature Encodings Transform."""

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class CombinedFEs(BaseTransform):
    r"""
    Combined FEs transform.

    Applies one or more pre-defined feature encoding transforms
    (KHopFE, HKFE, SheafConnLapPE) to a graph, storing their outputs
    and optionally concatenating them to `data.x`.

    Parameters
    ----------
    encodings : list of str
        List of feature encodings to apply. Supported values are
        "KHopFE" for K-hop Feature Encoding, "HKFE" for Heat Kernel
        Feature Encoding, and "SheafConnLapPE" for Sheaf Connection
        Laplacian Positional Encoding.
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
            The transformed data with added feature encodings.
        """
        from topobench.transforms.data_manipulations import (
            HKFE,
            KHopFE,
            SheafConnLapPE,
        )

        for enc in self.encodings:
            if enc == "HKFE":
                hkfe = HKFE(**self.parameters.get("HKFE", {}))
                data = hkfe(data)
            elif enc == "KHopFE":
                khopfe = KHopFE(**self.parameters.get("KHopFE", {}))
                data = khopfe(data)
            elif enc == "SheafConnLapPE":
                shfes = SheafConnLapPE(
                    **self.parameters.get("SheafConnLapPE", {})
                )
                data = shfes(data)
            else:
                raise ValueError(f"Unsupported encoding type: {enc}")

        return data


class SelectDestinationFEs(BaseTransform):
    r"""
    Select Destination Feature Encodings (FEs) transform.

    Selects and retains only the FEs corresponding to the destination nodes
    of edges in `data.edge_index`.

    Parameters
    ----------
    encodings : list of str
        List of encoding keys in `data` where the FEs are stored (e.g., 'HKFE', 'KHopFE').
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
            The transformed data with selected FEs.
        """
        new_data = {}
        new_data["x"] = data.x[:n_dst_nodes, :] if data.x is not None else None
        for encoding_key in self.encodings:
            if hasattr(data, encoding_key):
                fe = getattr(data, encoding_key)
                selected_fe = fe[:n_dst_nodes, :]
                new_data[encoding_key] = selected_fe
            else:
                raise ValueError(
                    f"Encoding key '{encoding_key}' not found in data."
                )
        return Data(**new_data)

    def __call__(self, data: Data, n_dst_nodes: int) -> Data:
        return self.forward(data, n_dst_nodes)

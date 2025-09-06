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
    max_pe_dim : int, optional
        Maximum dimension for positional encodings (default: 10).
    encoding_parameters : dict, optional
        Additional parameters for the encoding transforms.
    concat_to_x : bool, optional
        If True, concatenates the computed encodings to `data.x`.
        Default is True.
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
        from topobench.transforms.data_manipulations import RWSE, LapPE

        for enc in self.encodings:
            if enc == "LapPE":
                lappe = LapPE(**self.parameters.get("LapPE", {}))
                data = lappe(data)
            elif enc == "RWSE":
                rwse = RWSE(**self.parameters.get("RWSE", {}))
                data = rwse(data)
            else:
                raise ValueError(f"Unsupported encoding type: {enc}")

        return data

"""DataTransform class."""

import torch_geometric
from torch_geometric.data import Data, HeteroData

from topobench.transforms import TRANSFORMS


class DataTransform(torch_geometric.transforms.BaseTransform):
    r"""Instantiate and apply one configured TopoBench transform.

    Parameters
    ----------
    transform_name : str | None
        Exact registered transform name. ``None`` disables transformation.
    **kwargs : object
        Additional arguments passed to the registered transform.
    """

    def __init__(
        self,
        transform_name: str | None,
        **kwargs: object,
    ) -> None:
        super().__init__()

        kwargs["transform_name"] = transform_name
        self.parameters = kwargs

        self.transform = (
            None
            if transform_name is None
            else TRANSFORMS[transform_name](**kwargs)
        )

    def forward(self, data: Data | HeteroData) -> Data | HeteroData:
        r"""Apply the configured transform to supported PyG data.

        Parameters
        ----------
        data : Data | HeteroData
            The input data to transform.

        Returns
        -------
        Data | HeteroData
            The transformed data, or the unchanged input when no transform is
            configured.

        Raises
        ------
        TypeError
            If the input is unsupported, a transform has not declared
            heterogeneous-data support, or the result is unsupported.
        """
        if not isinstance(data, (Data, HeteroData)):
            raise TypeError(
                "DataTransform requires Data or HeteroData, received "
                f"{type(data).__name__}"
            )
        if self.transform is None:
            return data
        if isinstance(data, HeteroData) and not getattr(
            self.transform,
            "supports_heterodata",
            False,
        ):
            raise TypeError(
                f"{type(self.transform).__name__} does not declare "
                "HeteroData support for "
                f"metadata={data.metadata()}"
            )
        transformed = self.transform(data)
        if not isinstance(transformed, (Data, HeteroData)):
            raise TypeError(
                f"{type(self.transform).__name__} returned unsupported type "
                f"{type(transformed).__name__}"
            )
        return transformed

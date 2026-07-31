"""DataTransform class."""

import torch_geometric
from torch_geometric.data import Data, HeteroData

from topobench.transforms import TRANSFORMS


class DataTransform(torch_geometric.transforms.BaseTransform):
    r"""Instantiate and apply one configured TopoBench transform.

    Parameters
    ----------
    transform_name : str | None
        Registered transform name. ``None`` and the legacy ``"Identity"``
        spelling disable transformation.
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

        # ``Identity`` is the legacy configured spelling of a disabled
        # transform. Normalize it to the same path as ``None`` so it remains
        # representation-agnostic without opting an existing transform class
        # into heterogeneous support.
        self.transform = (
            None
            if transform_name in (None, "Identity")
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
            If a transform has not declared heterogeneous-data support or
            returns an unsupported representation.
        """
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

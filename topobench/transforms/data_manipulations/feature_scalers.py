import torch
import torch_geometric
import torch_geometric.transforms as T
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    FunctionTransformer,
)
from functools import partial
from typing import List, Union


class FeatureScaler(T.BaseTransform):
    r"""Class for scaling node features or other attributes using sklearn scalers.

    Adapts sklearn preprocessing to PyTorch Geometric data objects.
    It converts tensors to numpy, applies the specific scaler, and converts back.

    Parameters
    ----------
    scaler_type : str
        The type of scaler to use. Options include:
        'standard-scaler', 'min-max-scaler', 'robust-scaler',
        'power-transform-yeo-johnson', 'quantile-transform-normal',
        'quantile-transform-uniform', 'none'.
    fields : str or List[str]
        The field(s) in the data object to scale (e.g. 'x', 'edge_attr').
    **kwargs : optional
        Additional arguments passed to the scaler constructor.
        Note: Framework keys like 'transform_type' or 'transform_name' are automatically
        filtered out.
    """

    SCALERS = {
        "none": partial(
            FunctionTransformer, func=lambda x: x, inverse_func=lambda x: x
        ),
        "standard-scaler": partial(StandardScaler, copy=False),
        "min-max-scaler": partial(MinMaxScaler, clip=False, copy=False),
        "robust-scaler": partial(RobustScaler, copy=False),
        "power-transform-yeo-johnson": partial(
            PowerTransformer,
            method="yeo-johnson",
            standardize=True,
            copy=False,
        ),
        "quantile-transform-normal": partial(
            QuantileTransformer,
            output_distribution="normal",
            subsample=None,
            random_state=0,
            copy=False,
        ),
        "quantile-transform-uniform": partial(
            QuantileTransformer,
            output_distribution="uniform",
            subsample=None,
            random_state=0,
            copy=False,
        ),
    }

    def __init__(
        self,
        scaler_type: str,
        fields: Union[str, List[str]] = "x",
        **kwargs,
    ) -> None:
        super().__init__()

        # --- FIX START: Disentangle parameters ---
        # Explicitly remove keys that belong to the framework config
        # but are not valid arguments for sklearn scalers.
        kwargs.pop("transform_name", None)
        kwargs.pop("transform_type", None)
        # --- FIX END ---

        if scaler_type not in self.SCALERS:
            raise ValueError(
                f"Scaler '{scaler_type}' not recognized. "
                f"Valid options: {list(self.SCALERS.keys())}"
            )

        self.type = scaler_type
        self.fields = [fields] if isinstance(fields, str) else fields

        # The remaining kwargs are now safe to pass to the scaler
        self.scaler_kwargs = kwargs
        self.scaler_factory = self.SCALERS[scaler_type]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, fields={self.fields!r})"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Apply the transform to the input data."""

        for field in self.fields:
            if not hasattr(data, field) or data[field] is None:
                continue

            # 1. Get tensor and ensure shape is 2D (N, F)
            tensor = data[field]
            original_device = tensor.device
            original_dtype = tensor.dtype

            # Reshape 1D tensors to (N, 1) for sklearn compatibility
            is_1d = tensor.dim() == 1
            if is_1d:
                tensor = tensor.view(-1, 1)

            # 2. Convert to Numpy
            array = tensor.detach().cpu().numpy()

            # 3. Instantiate and apply the scaler
            scaler = self.scaler_factory(**self.scaler_kwargs)

            # Direct call: Errors (e.g. ValueError on NaNs) will crash execution
            scaled_array = scaler.fit_transform(array)

            # 4. Convert back to Tensor
            scaled_tensor = torch.from_numpy(scaled_array).to(
                device=original_device, dtype=original_dtype
            )

            # Restore 1D shape if input was 1D
            if is_1d:
                scaled_tensor = scaled_tensor.squeeze(1)

            data[field] = scaled_tensor

        return data

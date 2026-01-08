import torch
import torch_geometric
import torch_geometric.transforms as T
import numpy as np
from hydra.utils import instantiate
from typing import Optional, Any

class FeatureImputer(T.BaseTransform):
    r"""Class for imputing missing values (NaNs) in PyTorch Geometric data objects.
    
    It allows selective imputation based on feature subsets (numerical, categorical, 
    or fraction) using indices stored within the data object.

    Parameters
    ----------
    imputer : Any
        The imputation class compatible with sklearn's interface (e.g., SimpleImputer).
    feature_type : str, optional
        The type of features to impute: 'numerical', 'categorical', or 'fraction'.
        If None, the imputation is applied to all columns in the field.
    field : str
        The data object field to process (e.g., 'x', 'edge_attr').
    """

    def __init__(
            self, imputer: Any, 
            feature_type: Optional[str] = None, 
            field: str = "x",
            **kwargs
            ):
            super().__init__()
            self.imputer = instantiate(imputer)
            self.feature_type = feature_type
            self.field = field

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(strategy={self.strategy!r}, "
                f"feature_type={self.feature_type!r}, fields={self.field!r})")

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        if not hasattr(data, self.field) or data[self.field] is None:
            raise KeyError(f"Data object has no field '{self.field}' to process.")

        tensor = data[self.field]
        array = tensor.detach().cpu().numpy()
            
        # Identify columns to process
        current_indices = getattr(data, f"ids_cols_{self.feature_type}", [])
        if torch.is_tensor(current_indices):
            current_indices = current_indices.tolist()
            
        target_indices = sorted(current_indices) if current_indices else []

        # Skip if specific type requested but no columns found
        if target_indices:
            # Estraiamo, processiamo e sovrascriviamo nelle STESSE posizioni
            subset = array[:, target_indices]
            processed_subset = self.imputer.fit_transform(subset)
                    
            if hasattr(processed_subset, "toarray"):
                processed_subset = processed_subset.toarray()
                    
            array[:, target_indices] = processed_subset
            
        # Case: Process All (no feature_type specified)
        else:
            processed = self.imputer.fit_transform(array)
            array = processed.toarray() if hasattr(processed, "toarray") else processed

        # Save back to tensor
        data[self.field] = torch.from_numpy(array).to(tensor.device, tensor.dtype)

        return data
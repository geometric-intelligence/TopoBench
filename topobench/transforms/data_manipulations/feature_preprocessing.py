import torch
import torch_geometric
import torch_geometric.transforms as T
import numpy as np
from hydra.utils import instantiate
from sklearn.compose import ColumnTransformer
from typing import Optional, Any

class ColumnPreprocessor(T.BaseTransform):
    """
    Dynamically builds and applies a ColumnTransformer based on the 
    indices present in the Data object.
    """
    def __init__(
        self,
        num_scaler: Optional[Any] = None,
        cat_encoder: Optional[Any] = None,
        frac_scaler: Optional[Any] = None,
        field: str = "x",
        **kwargs
    ) -> None:
        super().__init__()
        self.engines = {
            'numerical': num_scaler,
            'categorical': cat_encoder,
            'fraction': frac_scaler
        }
        self.field = field

    def forward(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        # creating the ColumnTransformer dynamically
        transformers = []
        
        for feat_type, engine in self.engines.items():
            if engine is None:
                continue
            
            # Get indices from data (e.g., data.ids_cols_numerical)
            indices = getattr(data, f"ids_cols_{feat_type}", [])
            if torch.is_tensor(indices):
                indices = indices.tolist()
            
            # Only add to transformer if there are actually columns of this type
            if indices:
                # We name the step by the feature type for easy tracking
                transformers.append((feat_type, instantiate(engine), indices))
        
        # remainder='passthrough' ensures we don't lose columns not covered by types
        ct = ColumnTransformer(transformers, remainder='passthrough')

        if not hasattr(data, self.field) or data[self.field] is None:
            raise KeyError(f"Data object has no field '{self.field}' to process.")

        tensor = data[self.field]
        array = tensor.detach().cpu().numpy()
            
        # Apply transformations
        processed_array = ct.fit_transform(array)
        if hasattr(processed_array, "toarray"):
            processed_array = processed_array.toarray()

        # Save back to tensor
        data[self.field] = torch.from_numpy(processed_array).to(
            device=tensor.device, dtype=tensor.dtype
        )

        return data
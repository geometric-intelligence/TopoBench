"""Explicit exports for the supported native data manipulations."""

from .all_encodings import CombinedEncodings, SelectDestinationEncodings
from .combined_feature_encodings import CombinedFEs, SelectDestinationFEs
from .combined_positional_and_structural_encodings import CombinedPSEs
from .constant_node_features import ConstantNodeFeatures
from .electrostatic_encodings import ElectrostaticPE
from .heterogeneous import (
    HeterogeneousConstantFeatures,
    HeterogeneousToUndirected,
)
from .hk_feature_encodings import HKFE
from .hkdiag_encodings import HKdiagSE
from .identity_transform import IdentityTransform
from .infere_knn_connectivity import InfereKNNConnectivity
from .infere_radius_connectivity import InfereRadiusConnectivity
from .keep_only_connected_component import KeepOnlyConnectedComponent
from .keep_selected_data_fields import KeepSelectedDataFields
from .keep_selected_target_indices import KeepSelectedTargetIndices
from .khop_feature_encodings import KHopFE
from .laplacian_encodings import LapPE
from .node_degrees import NodeDegrees
from .node_features_to_float import NodeFeaturesToFloat
from .one_hot_degree_features import OneHotDegreeFeatures
from .ppr_feature_encodings import PPRFE
from .random_walk_encodings import RWSE
from .rename_fields import RenameFields
from .sheaf_connlap_encodings import SheafConnLapPE

DATA_MANIPULATIONS = dict(
    sorted(
        {
            transform_class.__name__: transform_class
            for transform_class in (
                CombinedEncodings,
                CombinedFEs,
                CombinedPSEs,
                ConstantNodeFeatures,
                ElectrostaticPE,
                HeterogeneousConstantFeatures,
                HeterogeneousToUndirected,
                HKFE,
                HKdiagSE,
                IdentityTransform,
                InfereKNNConnectivity,
                InfereRadiusConnectivity,
                KeepOnlyConnectedComponent,
                KeepSelectedDataFields,
                KeepSelectedTargetIndices,
                KHopFE,
                LapPE,
                NodeDegrees,
                NodeFeaturesToFloat,
                OneHotDegreeFeatures,
                PPRFE,
                RWSE,
                RenameFields,
                SelectDestinationEncodings,
                SelectDestinationFEs,
                SheafConnLapPE,
            )
        }.items()
    )
)

__all__ = [*DATA_MANIPULATIONS]

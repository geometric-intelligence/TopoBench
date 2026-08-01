"""Public registry for the supported native transforms."""

from .data_manipulations import (
    HKFE,
    PPRFE,
    RWSE,
    CombinedEncodings,
    CombinedFEs,
    CombinedPSEs,
    ConstantNodeFeatures,
    ElectrostaticPE,
    HeterogeneousConstantFeatures,
    HeterogeneousToUndirected,
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
    RenameFields,
    SelectDestinationEncodings,
    SelectDestinationFEs,
    SheafConnLapPE,
)

TRANSFORMS = dict(
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

__all__ = [*TRANSFORMS]

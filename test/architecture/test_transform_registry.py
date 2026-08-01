"""Architecture tests for the explicit native transform registry."""

import topobench.transforms as transforms
import topobench.transforms.data_manipulations as data_manipulations

EXPECTED_TRANSFORMS = tuple(
    sorted(
        {
            "CombinedEncodings",
            "SelectDestinationEncodings",
            "CombinedFEs",
            "SelectDestinationFEs",
            "CombinedPSEs",
            "ConstantNodeFeatures",
            "ElectrostaticPE",
            "HeterogeneousConstantFeatures",
            "HeterogeneousToUndirected",
            "HKFE",
            "HKdiagSE",
            "IdentityTransform",
            "InfereKNNConnectivity",
            "InfereRadiusConnectivity",
            "KeepOnlyConnectedComponent",
            "KeepSelectedDataFields",
            "KeepSelectedTargetIndices",
            "KHopFE",
            "LapPE",
            "NodeDegrees",
            "NodeFeaturesToFloat",
            "OneHotDegreeFeatures",
            "PPRFE",
            "RWSE",
            "RenameFields",
            "SheafConnLapPE",
        }
    )
)


def _assert_exact_registry(registry: dict[str, type]) -> None:
    assert type(registry) is dict
    assert tuple(registry) == EXPECTED_TRANSFORMS
    assert all(
        name == transform_class.__name__
        for name, transform_class in registry.items()
    )


def test_transform_registries_expose_only_native_public_transforms() -> None:
    _assert_exact_registry(data_manipulations.DATA_MANIPULATIONS)
    _assert_exact_registry(transforms.TRANSFORMS)
    assert transforms.TRANSFORMS == data_manipulations.DATA_MANIPULATIONS


def test_transform_modules_have_exact_deterministic_public_exports() -> None:
    assert data_manipulations.__all__ == [*EXPECTED_TRANSFORMS]
    assert transforms.__all__ == [*EXPECTED_TRANSFORMS]
    for name in EXPECTED_TRANSFORMS:
        transform_class = transforms.TRANSFORMS[name]
        assert getattr(transforms, name) is transform_class
        assert getattr(data_manipulations, name) is transform_class
        assert transform_class.__module__.startswith(
            "topobench.transforms.data_manipulations."
        )


def test_lifting_and_discovery_compatibility_exports_are_absent() -> None:
    for module in (transforms, data_manipulations):
        for removed_name in (
            "FEATURE_LIFTINGS",
            "LIFTINGS",
            "ModuleExportsManager",
            "manager",
        ):
            assert not hasattr(module, removed_name)

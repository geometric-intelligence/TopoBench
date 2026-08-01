"""Audit graph dataset feature policies and their required defaults."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from topobench.data.capabilities import GRAPH_DATASET_MANIFEST
from topobench.data.qualification import DATASET_QUALIFICATION_MANIFEST

GRAPH_CONFIG_DIR = Path("configs/dataset/graph")
DEFAULT_TRANSFORM_DIR = Path("configs/transforms/dataset_defaults")
ALLOWED_POLICIES = {
    "continuous",
    "categorical_one_hot",
    "degree",
    "constant",
}
MOLECULAR_CATEGORICAL_SELECTORS = frozenset(
    {
        "BBB_Martins",
        "CYP3A4_Veith",
        "Caco2_Wang",
        "Clearance_Hepatocyte_AZ",
        "ogbg-molhiv",
    }
)
EXPECTED_SPECIAL_POLICIES = {
    "AQSOL": "degree",
    "IMDB-BINARY": "degree",
    "IMDB-MULTI": "degree",
    "REDDIT-BINARY": "constant",
    "ZINC": "categorical_one_hot",
    "ZINC_OGB": "categorical_one_hot",
    "BBB_Martins": "categorical_one_hot",
    "CYP3A4_Veith": "categorical_one_hot",
    "Caco2_Wang": "categorical_one_hot",
    "Clearance_Hepatocyte_AZ": "categorical_one_hot",
    "ogbg-molhiv": "categorical_one_hot",
}
REQUIRED_DEFAULTS = {
    "categorical_one_hot": {"one_hot_node_degree_features"},
    "degree": {"node_degrees", "one_hot_node_degree_features"},
    "constant": {"constant_node_features"},
}


def _default_transform_names(config: DictConfig) -> set[str]:
    raw = OmegaConf.to_container(config, resolve=False)
    names: set[str] = set()
    for entry in raw["defaults"]:
        if isinstance(entry, str):
            names.add(entry)
            continue
        names.update(str(value) for value in entry.values())
    return names


def test_every_retained_graph_dataset_has_a_valid_feature_policy() -> None:
    for path in GRAPH_CONFIG_DIR.glob("*.yaml"):
        config = OmegaConf.load(path)

        policy = config.parameters.get("feature_policy")
        assert policy in ALLOWED_POLICIES, f"{path.stem}: {policy!r}"
        expected = EXPECTED_SPECIAL_POLICIES.get(path.stem, "continuous")
        assert policy == expected, f"{path.stem}: expected {expected!r}"


def test_transform_provided_policies_have_required_dataset_defaults() -> None:
    for selector, policy in EXPECTED_SPECIAL_POLICIES.items():
        if selector in MOLECULAR_CATEGORICAL_SELECTORS:
            continue
        default_path = DEFAULT_TRANSFORM_DIR / f"{selector}.yaml"
        assert default_path.is_file(), f"{selector} has no dataset default"
        config = OmegaConf.load(default_path)
        names = _default_transform_names(config)
        missing = REQUIRED_DEFAULTS[policy] - names
        assert not missing, (
            f"{selector} is missing defaults: {sorted(missing)}"
        )


def test_molecular_configs_and_manifests_declare_encoded_atom_features() -> (
    None
):
    for selector in MOLECULAR_CATEGORICAL_SELECTORS:
        config = OmegaConf.load(GRAPH_CONFIG_DIR / f"{selector}.yaml")

        assert config.parameters.feature_policy == "categorical_one_hot"
        assert config.parameters.num_features[0] == 174
        assert "max_dim_if_lifted" not in config.parameters
        assert "preserve_edge_attr_if_lifted" not in config.parameters
        assert (
            GRAPH_DATASET_MANIFEST[selector].feature_policy
            == "categorical_one_hot"
        )
        assert (
            DATASET_QUALIFICATION_MANIFEST[f"graph/{selector}"].feature_policy
            == "categorical_one_hot"
        )


def test_one_hot_default_uses_transform_constructor_field_names() -> None:
    config = OmegaConf.load(
        "configs/transforms/data_manipulations/"
        "one_hot_node_degree_features.yaml"
    )

    assert "degrees_field" in config
    assert "features_field" in config
    assert "degrees_fields" not in config
    assert "features_fields" not in config


def test_constant_transform_config_is_publicly_resolvable() -> None:
    config = OmegaConf.load(
        "configs/transforms/data_manipulations/constant_node_features.yaml"
    )

    assert config.transform_name == "ConstantNodeFeatures"
    unresolved = OmegaConf.to_container(config, resolve=False)
    assert unresolved["num_features"] == "${dataset.parameters.num_features}"

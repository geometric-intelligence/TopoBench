"""Tests for the post-transform homogeneous graph feature boundary."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.data.features import (
    OGB_ATOM_FEATURE_CARDINALITIES,
    encode_categorical_columns,
    prepare_graph_features,
    validate_graph_features,
)
from topobench.data.pipelines.default import DefaultDataPipeline
from topobench.dataloader import GraphDataModule
from topobench.transforms.data_manipulations import ConstantNodeFeatures


def test_continuous_features_are_accepted() -> None:
    data = Data(x=torch.randn(4, 3))

    assert validate_graph_features(data, "continuous", 3) is data


@pytest.mark.parametrize("policy", ["categorical_one_hot", "degree"])
def test_one_hot_feature_policies_are_accepted(policy: str) -> None:
    data = Data(x=torch.eye(4, dtype=torch.float))

    assert validate_graph_features(data, policy, 4) is data


def test_categorical_columns_have_deterministic_one_hot_offsets() -> None:
    categories = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [118, 4, 11, 11, 9, 5, 5, 1, 1],
        ]
    )

    encoded = encode_categorical_columns(
        categories, OGB_ATOM_FEATURE_CARDINALITIES
    )

    expected = torch.zeros(2, 174)
    expected[0, [0, 119, 124, 136, 148, 158, 164, 170, 172]] = 1
    expected[1, [118, 123, 135, 147, 157, 163, 169, 171, 173]] = 1
    torch.testing.assert_close(encoded, expected)
    assert encoded.dtype == torch.float


@pytest.mark.parametrize(
    ("categories", "message"),
    [
        (torch.zeros(9, dtype=torch.long), "rank-2"),
        (torch.zeros((2, 8), dtype=torch.long), "cardinality count"),
        (
            torch.tensor([[0.5, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float),
            "integral values",
        ),
        (
            torch.tensor([[-1, 0, 0, 0, 0, 0, 0, 0, 0]]),
            "out of range",
        ),
        (
            torch.tensor([[119, 0, 0, 0, 0, 0, 0, 0, 0]]),
            "out of range",
        ),
    ],
)
def test_categorical_column_encoder_rejects_invalid_input(
    categories: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        encode_categorical_columns(categories, OGB_ATOM_FEATURE_CARDINALITIES)


def test_categorical_policy_accepts_consistent_multi_hot_rows() -> None:
    data = Data(x=torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))

    assert validate_graph_features(data, "categorical_one_hot", 4) is data


def test_degree_policy_retains_exact_one_hot_validation() -> None:
    data = Data(x=torch.tensor([[1.0, 0.0, 1.0, 0.0]]))

    with pytest.raises(ValueError, match="degree policy requires one-hot"):
        validate_graph_features(data, "degree", 4)


def test_categorical_policy_rejects_inconsistent_active_counts() -> None:
    data = Data(x=torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="consistent multi-hot"):
        validate_graph_features(data, "categorical_one_hot", 4)


def test_constant_features_are_deterministic() -> None:
    data = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        num_nodes=4,
    )
    transform = ConstantNodeFeatures(num_features=3)

    first = transform(data.clone())
    second = transform(data.clone())

    torch.testing.assert_close(first.x, torch.ones(4, 3))
    torch.testing.assert_close(second.x, first.x)
    assert validate_graph_features(first, "constant", 3) is first


def test_featureless_graph_is_rejected_after_transforms() -> None:
    data = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        num_nodes=2,
    )

    with pytest.raises(ValueError, match="data.x is required"):
        validate_graph_features(data, "continuous", 1)


@pytest.mark.parametrize(
    ("x", "error_type", "message"),
    [
        (torch.ones(3), ValueError, "data.x must be rank-2"),
        (
            torch.ones(3, 2, dtype=torch.long),
            TypeError,
            "data.x must have a floating dtype",
        ),
    ],
)
def test_invalid_feature_shape_and_dtype_are_rejected(
    x: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        validate_graph_features(Data(x=x), "continuous", 2)


def test_policy_mismatch_is_rejected() -> None:
    data = Data(x=torch.tensor([[1.0, 0.0], [0.5, 0.5]]))

    with pytest.raises(ValueError, match="constant policy"):
        validate_graph_features(data, "constant", 2)


def test_feature_width_must_match_declared_channels() -> None:
    data = Data(x=torch.randn(4, 3))

    with pytest.raises(ValueError, match="expected 2 feature channels"):
        validate_graph_features(data, "continuous", 2)


def test_prepare_graph_features_validates_every_split() -> None:
    train = [Data(x=torch.ones(2, 1))]
    val = [Data(x=torch.ones(3, 1))]
    test = [Data(x=torch.zeros(4, 1))]

    with pytest.raises(ValueError, match="constant policy"):
        prepare_graph_features(
            train,
            val,
            test,
            feature_policy="constant",
            num_features=1,
        )


def test_default_pipeline_validates_features_before_datamodule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = Data(x=torch.ones(3, dtype=torch.float))
    preprocessor = SimpleNamespace(
        preprocessing_time=0.0,
        load_dataset_splits=lambda split_params: ([invalid], None, None),
    )
    cfg = OmegaConf.create(
        {
            "dataset": {
                "loader": {
                    "parameters": {
                        "data_domain": "graph",
                    }
                },
                "parameters": {
                    "task_level": "node",
                    "feature_policy": "continuous",
                    "num_features": 1,
                },
                "split_params": {"learning_setting": "transductive"},
                "dataloader_params": {"batch_size": 1},
            }
        }
    )
    datamodule_constructed = False

    def fail_if_constructed(**kwargs: object) -> None:
        nonlocal datamodule_constructed
        datamodule_constructed = True
        raise AssertionError("GraphDataModule must not be constructed")

    monkeypatch.setattr(
        DefaultDataPipeline,
        "preprocess",
        staticmethod(lambda cfg: preprocessor),
    )
    monkeypatch.setattr(
        "topobench.data.pipelines.default.GraphDataModule",
        fail_if_constructed,
    )

    with pytest.raises(ValueError, match="data.x must be rank-2"):
        DefaultDataPipeline().build(cfg)

    assert not datamodule_constructed


def test_default_pipeline_skips_graph_policy_for_hypergraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = Data(x=torch.ones(3, dtype=torch.float))
    preprocessor = SimpleNamespace(
        preprocessing_time=0.0,
        load_dataset_splits=lambda split_params: ([data], None, None),
    )
    cfg = OmegaConf.create(
        {
            "dataset": {
                "loader": {
                    "parameters": {
                        "data_domain": "hypergraph",
                    }
                },
                "parameters": {
                    "task_level": "node",
                },
                "split_params": {"learning_setting": "transductive"},
                "dataloader_params": {"batch_size": 1},
            }
        }
    )

    def fail_graph_validation(*args: object, **kwargs: object) -> None:
        raise AssertionError("hypergraph data must skip graph feature policy")

    monkeypatch.setattr(
        DefaultDataPipeline,
        "preprocess",
        staticmethod(lambda cfg: preprocessor),
    )
    monkeypatch.setattr(
        "topobench.data.pipelines.default.prepare_graph_features",
        fail_graph_validation,
    )

    output = DefaultDataPipeline().build(cfg)

    assert isinstance(output.datamodule, GraphDataModule)

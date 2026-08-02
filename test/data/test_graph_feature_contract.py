"""Tests for the post-transform homogeneous graph feature boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

import topobench.data.features as graph_features
from topobench.data.capabilities import GRAPH_DATASET_MANIFEST
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

    assert validate_graph_features(data, "continuous", base_num_features=3, total_num_features=3) is data


@pytest.mark.parametrize("policy", ["categorical_one_hot", "degree"])
def test_one_hot_feature_policies_are_accepted(policy: str) -> None:
    data = Data(x=torch.eye(4, dtype=torch.float))

    assert validate_graph_features(data, policy, base_num_features=4, total_num_features=4) is data


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

    assert validate_graph_features(data, "categorical_one_hot", base_num_features=4, total_num_features=4) is data


def test_degree_policy_retains_exact_one_hot_validation() -> None:
    data = Data(x=torch.tensor([[1.0, 0.0, 1.0, 0.0]]))

    with pytest.raises(ValueError, match="degree policy requires one-hot"):
        validate_graph_features(data, "degree", base_num_features=4, total_num_features=4)


def test_categorical_policy_rejects_inconsistent_active_counts() -> None:
    data = Data(x=torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="consistent multi-hot"):
        validate_graph_features(data, "categorical_one_hot", base_num_features=4, total_num_features=4)


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
    assert validate_graph_features(first, "constant", base_num_features=3, total_num_features=3) is first


def test_featureless_graph_is_rejected_after_transforms() -> None:
    data = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        num_nodes=2,
    )

    with pytest.raises(ValueError, match="data.x is required"):
        validate_graph_features(data, "continuous", base_num_features=1, total_num_features=1)


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
        validate_graph_features(Data(x=x), "continuous", base_num_features=2, total_num_features=2)


def test_policy_mismatch_is_rejected() -> None:
    data = Data(x=torch.tensor([[1.0, 0.0], [0.5, 0.5]]))

    with pytest.raises(ValueError, match="constant policy"):
        validate_graph_features(data, "constant", base_num_features=2, total_num_features=2)


def test_feature_width_must_match_declared_channels() -> None:
    data = Data(x=torch.randn(4, 3))

    with pytest.raises(ValueError, match="expected 2 feature channels"):
        validate_graph_features(data, "continuous", base_num_features=2, total_num_features=2)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            Data(x=torch.ones(4, 3)),
            "expected shape=(N, 2)",
        ),
        (
            Data(x=torch.ones(3, 2), num_nodes=4),
            "expected one row per num_nodes=4",
        ),
    ],
    ids=["width", "row-count"],
)
def test_feature_shape_contract_errors_are_fully_contextual(
    data: Data,
    expected: str,
) -> None:
    with pytest.raises(ValueError) as error:
        validate_graph_features(
            data,
            "continuous",
            base_num_features=2,
            total_num_features=2,
            selector="SyntheticGraph",
            item="graph[7]",
        )

    message = str(error.value)
    assert "SyntheticGraph" in message
    assert "graph[7]" in message
    assert "field x" in message
    assert f"shape={tuple(data.x.shape)}" in message
    assert f"dtype={data.x.dtype}" in message
    assert expected in message


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (None, "required"),
        (torch.tensor([1]), "floating dtype"),
        (torch.tensor([[1.0]]), "shape=(1,)"),
        (torch.tensor([float("nan")]), "finite"),
        (torch.tensor([float("inf")]), "finite"),
    ],
    ids=["missing", "dtype", "rank", "nan", "inf"],
)
def test_qualified_graph_regression_source_rejects_invalid_targets(
    target: torch.Tensor | None,
    expected: str,
) -> None:
    data = Data(x=torch.ones(2, 4))
    if target is not None:
        data.y = target

    with pytest.raises((TypeError, ValueError)) as error:
        graph_features.validate_qualified_graph_source(
            [data],
            capability=GRAPH_DATASET_MANIFEST[
                "SyntheticGraphRegression"
            ],
            configured_num_classes=1,
            total_num_features=4,
        )

    message = str(error.value)
    assert "SyntheticGraphRegression" in message
    assert "graph[0]" in message
    assert "field y" in message
    assert expected in message


def test_qualified_graph_regression_accepts_multiple_scalar_targets() -> None:
    source = [
        Data(x=torch.ones(2, 4), y=torch.tensor([0.25])),
        Data(x=torch.zeros(3, 4), y=torch.tensor([-1.5])),
    ]

    assert (
        graph_features.validate_qualified_graph_source(
            source,
            capability=GRAPH_DATASET_MANIFEST[
                "SyntheticGraphRegression"
            ],
            configured_num_classes=1,
            total_num_features=4,
        )
        is source
    )


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
            base_num_features=1,
            total_num_features=1,
        )


def test_default_pipeline_validates_features_before_datamodule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = Data(x=torch.ones(3, dtype=torch.float), y=torch.tensor([0, 1, 0]))

    class InvalidPreprocessor:
        preprocessing_time = 0.0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Data:
            assert index == 0
            return invalid

        def load_dataset_splits(self, split_params: object) -> object:
            return [invalid], None, None

    preprocessor = InvalidPreprocessor()
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
                    "task": "classification",
                    "feature_policy": "continuous",
                    "num_features": 1,
                    "num_classes": 2,
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
        "topobench.data.pipelines.default.qualify_graph_dataset",
        lambda dataset: SimpleNamespace(
            selector="SyntheticNodeGraph",
            feature_width=1,
            feature_policy="continuous",
            task="classification",
            task_level="node",
            num_classes=2,
            allow_incomplete_class_vocabulary=False,
        ),
    )
    monkeypatch.setattr(
        "topobench.data.pipelines.default.infer_in_channels",
        lambda dataset, transforms: 1,
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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_graph_rejects_nonfinite_native_features_with_context(
    invalid: float,
) -> None:
    data = Data(x=torch.ones(2, 4), y=torch.tensor([0]))
    data.x[0, 0] = invalid

    with pytest.raises(ValueError) as error:
        validate_graph_features(
            data,
            "continuous",
            base_num_features=4,
            total_num_features=4,
            selector="SyntheticGraph",
            item="graph[0]",
        )

    message = str(error.value)
    assert "SyntheticGraph" in message
    assert "graph[0]" in message
    assert "x" in message
    assert "finite" in message


def test_graph_rejects_zero_node_source_item_with_context() -> None:
    data = Data(x=torch.empty((0, 4)), y=torch.tensor([0]), num_nodes=0)

    with pytest.raises(ValueError) as error:
        validate_graph_features(
            data,
            "continuous",
            base_num_features=4,
            total_num_features=4,
            selector="SyntheticGraph",
            item="graph[0]",
        )

    message = str(error.value)
    assert "SyntheticGraph" in message
    assert "graph[0]" in message
    assert "x" in message
    assert "(0, 4)" in message
    assert "at least one node" in message


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (torch.tensor([0.0]), "dtype"),
        (torch.tensor([[0]]), "rank-1"),
        (torch.tensor([-1]), "range"),
        (torch.tensor([2]), "range"),
    ],
    ids=["dtype", "rank", "negative", "above-range"],
)
def test_qualified_graph_source_rejects_malformed_labels_contextually(
    labels: torch.Tensor,
    expected: str,
) -> None:
    source = [Data(x=torch.ones(2, 4), y=labels)]

    with pytest.raises((TypeError, ValueError)) as error:
        graph_features.validate_qualified_graph_source(
            source,
            capability=GRAPH_DATASET_MANIFEST["SyntheticGraph"],
            configured_num_classes=2,
            total_num_features=4,
        )

    message = str(error.value)
    assert "SyntheticGraph" in message
    assert "graph[0]" in message
    assert "y" in message
    assert expected in message


def test_qualified_graph_source_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match=r"SyntheticGraph.*source.*non-empty"):
        graph_features.validate_qualified_graph_source(
            [],
            capability=GRAPH_DATASET_MANIFEST["SyntheticGraph"],
            configured_num_classes=2,
            total_num_features=4,
        )


def test_default_pipeline_qualifies_full_source_before_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidSource:
        preprocessing_time = 0.0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Data:
            assert index == 0
            return Data(x=torch.ones(2, 4), y=torch.tensor([0]))

        def load_dataset_splits(self, split_params: object) -> object:
            raise AssertionError("split construction must not run")

    capability = GRAPH_DATASET_MANIFEST["SyntheticGraph"]
    cfg = OmegaConf.create(
        {
            "dataset": {
                "loader": {
                    "parameters": {
                        "data_domain": "graph",
                        "data_name": "SyntheticGraph",
                    }
                },
                "parameters": {
                    "task": "classification",
                    "task_level": "graph",
                    "feature_policy": "continuous",
                    "num_features": 4,
                    "num_classes": 2,
                },
                "split_params": {"learning_setting": "inductive"},
                "dataloader_params": {"batch_size": 1},
            }
        }
    )
    monkeypatch.setattr(
        DefaultDataPipeline,
        "preprocess",
        staticmethod(lambda cfg: InvalidSource()),
    )
    monkeypatch.setattr(
        "topobench.data.pipelines.default.qualify_graph_dataset",
        lambda dataset: capability,
    )
    monkeypatch.setattr(
        "topobench.data.pipelines.default.infer_in_channels",
        lambda dataset, transforms: 4,
    )

    with pytest.raises(ValueError, match=r"SyntheticGraph.*missing.*1"):
        DefaultDataPipeline().build(cfg)




def test_qualified_graph_source_rejects_missing_runtime_class() -> None:
    source = [
        Data(x=torch.ones(2, 4), y=torch.tensor([0])),
        Data(x=torch.ones(3, 4), y=torch.tensor([0])),
    ]

    with pytest.raises(ValueError, match=r"SyntheticGraph.*y.*missing.*1"):
        graph_features.validate_qualified_graph_source(
            source,
            capability=GRAPH_DATASET_MANIFEST["SyntheticGraph"],
            configured_num_classes=2,
            total_num_features=4,
        )


def test_explicit_manifest_exception_allows_incomplete_full_vocabulary() -> None:
    source = [Data(x=torch.ones(2, 4), y=torch.tensor([0]))]
    capability = replace(
        GRAPH_DATASET_MANIFEST["SyntheticGraph"],
        allow_incomplete_class_vocabulary=True,
    )

    assert (
        graph_features.validate_qualified_graph_source(
            source,
            capability=capability,
            configured_num_classes=2,
            total_num_features=4,
        )
        is source
    )


def test_qualified_graph_source_does_not_mutate_features_or_labels() -> None:
    source = [
        Data(x=torch.ones(2, 4), y=torch.tensor([0])),
        Data(x=torch.zeros(3, 4), y=torch.tensor([1])),
    ]
    snapshots = [
        (data.x.clone(), data.y.clone(), data.x.dtype, data.y.dtype)
        for data in source
    ]

    graph_features.validate_qualified_graph_source(
        source,
        capability=GRAPH_DATASET_MANIFEST["SyntheticGraph"],
        configured_num_classes=2,
        total_num_features=4,
    )

    for data, (x, y, x_dtype, y_dtype) in zip(
        source,
        snapshots,
        strict=True,
    ):
        assert data.x.dtype == x_dtype
        assert data.y.dtype == y_dtype
        assert torch.equal(data.x, x)
        assert torch.equal(data.y, y)




def test_qualified_full_graph_source_allows_phase_to_omit_class() -> None:
    class_zero = Data(x=torch.ones(2, 4), y=torch.tensor([0]))
    class_one = Data(x=torch.ones(3, 4), y=torch.tensor([1]))
    source = [class_zero, class_one]

    assert (
        graph_features.validate_qualified_graph_source(
            source,
            capability=GRAPH_DATASET_MANIFEST["SyntheticGraph"],
            configured_num_classes=2,
            total_num_features=4,
        )
        is source
    )
    assert prepare_graph_features(
        [class_zero],
        [class_one],
        None,
        feature_policy="continuous",
        base_num_features=4,
        total_num_features=4,
    ) == ([class_zero], [class_one], None)

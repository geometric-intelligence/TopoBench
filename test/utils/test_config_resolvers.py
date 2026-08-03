"""Unit tests for config resolvers."""

import pytest
from omegaconf import OmegaConf
import hydra
from topobench.utils.config_resolvers import (
    register_all_resolvers,
    define_task_level,
    infer_in_channels,
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    set_preserve_edge_attr,
    check_pses_in_transforms,
    check_fes_in_transforms,
    get_fes_dimensions,
    get_all_encoding_dimensions,
)

class TestConfigResolvers:
    """Test config resolvers."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        register_all_resolvers()
        self.dataset_config_1 = OmegaConf.load("configs/dataset/graph/MUTAG.yaml")
        self.dataset_config_2 = OmegaConf.load("configs/dataset/graph/cocitation_cora.yaml")
        hydra.initialize(version_base="1.3", config_path="../../configs", job_name="job")

    def test_define_task_level(self):
        """Test define_task_level."""
        # node + inductive -> node_inductive (the bug-fix branch)
        assert define_task_level("node", "inductive") == "node_inductive"

        # else branch: any other combination returns dataset_task_level unchanged
        assert define_task_level("node", "transductive") == "node"
        assert define_task_level("graph", "inductive") == "graph"
        assert define_task_level("graph", "transductive") == "graph"

    def test_get_default_trainer(self):
        """Test get_default_trainer."""
        out = get_default_trainer()
        assert isinstance(out, str)

    @pytest.mark.parametrize(
        ("task", "num_classes", "expected"),
        [
            (
                "classification",
                3,
                ["accuracy", "precision", "recall", "f1", "auroc"],
            ),
            (
                "classification",
                2,
                [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auroc",
                    "auprc",
                    "somers_d",
                ],
            ),
            ("regression", 1, ["mae", "mse", "rmse", "r2"]),
        ],
    )
    def test_get_default_metrics(
        self, task, num_classes, expected
    ):
        """Default metrics use the reduced task and metric vocabulary."""
        assert get_default_metrics(task, num_classes) == expected

    @pytest.mark.parametrize(
        "task",
        ["multioutput classification", "multilabel classification", "other"],
    )
    def test_get_default_metrics_rejects_removed_tasks(self, task):
        with pytest.raises(ValueError, match="Supported tasks"):
            get_default_metrics(task, 2)

    def test_get_default_transform(self):
        """Choose only same-domain dataset/model defaults."""
        assert get_default_transform("graph/MUTAG", "graph/gat") == "no_transform"
        assert (
            get_default_transform("graph/ZINC", "graph/gcn")
            == "dataset_defaults/ZINC"
        )
        assert (
            get_default_transform("graph/MUTAG", "graph/gps")
            == "model_defaults/gps"
        )
        assert (
            get_default_transform("graph/ZINC", "graph/gps")
            == "model_dataset_defaults/gps_ZINC"
        )
        with pytest.raises(
            ValueError,
            match="Cross-domain lifting is unsupported",
        ):
            get_default_transform("graph/MUTAG", "hypergraph/edgnn")
        with pytest.raises(
            ValueError,
            match="Cross-domain lifting is unsupported",
        ):
            get_default_transform("hypergraph/SyntheticHypergraph", "graph/gcn")





    def test_get_monitor_metric(self):
        """Test get_monitor_metric."""
        out = get_monitor_metric("classification", "F1")
        assert out == "val/F1"

        assert get_monitor_metric("multioutput classification", "accuracy") == "val/accuracy"
        assert get_monitor_metric("multilabel classification", "f1") == "val/f1"

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_monitor_metric("mix", "F1")

    def test_get_monitor_mode(self):
        """Test get_monitor_mode."""
        out = get_monitor_mode("regression")
        assert out == "min"

        out = get_monitor_mode("classification")
        assert out == "max"

        assert get_monitor_mode("multilabel classification") == "max"
        assert get_monitor_mode("multioutput classification") == "min"

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_monitor_mode("mix")

    def test_infer_in_channels(self):
        """Test infer_in_channels."""
        in_channels = infer_in_channels(self.dataset_config_1, None)
        assert in_channels == 7

        in_channels = infer_in_channels(self.dataset_config_2, None)
        assert in_channels == 1433


        cfg = hydra.compose(config_name="run.yaml", overrides=["model=graph/gcn", "dataset=graph/MUTAG", "transforms=combined_fe"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == 48

    @pytest.mark.parametrize(
        ("dataset_path", "transforms_path", "expected"),
        [
            ("configs/dataset/graph/cocitation_cora.yaml", None, 1433),
            (
                "configs/dataset/graph/ZINC.yaml",
                "configs/transforms/dataset_defaults/ZINC.yaml",
                21,
            ),
            (
                "configs/dataset/graph/IMDB-BINARY.yaml",
                "configs/transforms/dataset_defaults/IMDB-BINARY.yaml",
                136,
            ),
            (
                "configs/dataset/graph/REDDIT-BINARY.yaml",
                "configs/transforms/dataset_defaults/REDDIT-BINARY.yaml",
                10,
            ),
            ("configs/dataset/hypergraph/SyntheticHypergraph.yaml", None, 4),
        ],
    )
    def test_no_lifting_returns_scalar_channels_for_native_features(
        self,
        dataset_path,
        transforms_path,
        expected,
    ):
        """Graph and hypergraph native features resolve to one scalar width."""
        dataset = OmegaConf.load(dataset_path)
        transforms = (
            OmegaConf.load(transforms_path)
            if transforms_path is not None
            else None
        )

        assert infer_in_channels(dataset, transforms) == expected



    def test_get_default_metrics_with_params(self):
        """Explicit metrics are validated against the active vocabulary."""
        out = get_default_metrics(
            "classification", 10, ["accuracy", "precision"]
        )
        assert out == ["accuracy", "precision"]

        with pytest.raises(ValueError, match="binary classification"):
            get_default_metrics("classification", 3, ["auprc"])
        with pytest.raises(ValueError, match="classification metric"):
            get_default_metrics("regression", 1, ["accuracy"])
        with pytest.raises(ValueError, match="regression metric"):
            get_default_metrics("classification", 3, ["mae"])

    def test_set_preserve_edge_attr(self):
        """Surviving graph models retain the dataset's declared default."""
        assert set_preserve_edge_attr(model_name="gcn", default=True) is True
        assert set_preserve_edge_attr(model_name="gcn", default=False) is False



    def test_check_pses_in_transforms_empty(self):
        """Test check_pses_in_transforms with no encodings."""
        transforms = OmegaConf.create({})
        result = check_pses_in_transforms(transforms)
        assert result == 0

    def test_single_transform_lappe_with_eigenvalues(self):
        """Test single transform with LapPE including eigenvalues."""
        transforms = OmegaConf.create({
            "transform_name": "LapPE",
            "include_eigenvalues": True,
            "max_pe_dim": 8
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16  # 8 * 2

    def test_single_transform_lappe_without_eigenvalues(self):
        """Test single transform with LapPE without eigenvalues."""
        transforms = OmegaConf.create({
            "transform_name": "LapPE",
            "include_eigenvalues": False,
            "max_pe_dim": 8
        })
        result = check_pses_in_transforms(transforms)
        assert result == 8

    def test_single_transform_rwse(self):
        """Test single transform with RWSE."""
        transforms = OmegaConf.create({
            "transform_name": "RWSE",
            "max_pe_dim": 16
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16

    def test_check_pses_in_transforms_lappe_only(self):
        """Test check_pses_in_transforms with only LapPE encoding."""
        # LapPE without eigenvalues
        transforms = OmegaConf.create({
            "LapPE": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 8

    def test_check_pses_in_transforms_lappe_with_eigenvalues(self):
        """Test check_pses_in_transforms with LapPE including eigenvalues."""
        transforms = OmegaConf.create({
            "LapPE": {
                "max_pe_dim": 8,
                "include_eigenvalues": True,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16  # 8 * 2

    def test_check_pses_in_transforms_rwse_only(self):
        """Test check_pses_in_transforms with only RWSE encoding."""
        transforms = OmegaConf.create({
            "RWSE": {
                "max_pe_dim": 8,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 8

    def test_check_pses_in_transforms_combined_pses_lappe_rwse(self):
        """Test check_pses_in_transforms with CombinedPSEs containing both LapPE and RWSE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 12  # 8 + 4

    def test_check_pses_in_transforms_combined_pses_with_eigenvalues(self):
        """Test check_pses_in_transforms with CombinedPSEs where LapPE includes eigenvalues."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": True,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 20  # (8 * 2) + 4

    def test_check_pses_in_transforms_combined_pses_lappe_only(self):
        """Test check_pses_in_transforms with CombinedPSEs containing only LapPE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 16,
                        "include_eigenvalues": False,
                        "concat_to_x": False
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16

    def test_check_pses_in_transforms_combined_pses_rwse_only(self):
        """Test check_pses_in_transforms with CombinedPSEs containing only RWSE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["RWSE"],
                "parameters": {
                    "RWSE": {
                        "max_pe_dim": 12,
                        "concat_to_x": False
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 12

    def test_check_pses_in_transforms_multiple_separate_transforms(self):
        """Test check_pses_in_transforms with multiple separate encoding transforms."""
        transforms = OmegaConf.create({
            "LapPE_1": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            },
            "RWSE_1": {
                "max_pe_dim": 4,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 12  # 8 + 4

    def test_check_pses_in_transforms_multiple_lappe_transforms(self):
        """Test check_pses_in_transforms with multiple LapPE transforms."""
        transforms = OmegaConf.create({
            "LapPE_first": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            },
            "LapPE_second": {
                "max_pe_dim": 4,
                "include_eigenvalues": True,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16  # 8 + (4 * 2)

    def test_check_pses_in_transforms_mixed_transforms(self):
        """Test check_pses_in_transforms with mixed transform types."""
        transforms = OmegaConf.create({
            "some_other_transform": {
                "param1": "value1"
            },
            "LapPE": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            },
            "another_transform": {
                "param2": "value2"
            },
            "RWSE": {
                "max_pe_dim": 4,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 12  # 8 + 4

    def test_check_pses_in_transforms_combined_and_separate(self):
        """Test check_pses_in_transforms with both CombinedPSEs and separate encodings."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            },
            "LapPE_extra": {
                "max_pe_dim": 2,
                "include_eigenvalues": False,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 14  # (8 + 4) + 2

    def test_check_pses_in_transforms_different_dimensions(self):
        """Test check_pses_in_transforms with various dimension sizes."""
        # Test with different max_pe_dim values
        for dim in [1, 2, 4, 8, 16, 32]:
            transforms = OmegaConf.create({
                "RWSE": {
                    "max_pe_dim": dim,
                    "concat_to_x": True
                }
            })
            result = check_pses_in_transforms(transforms)
            assert result == dim

    def test_check_pses_in_transforms_combined_pses_empty_encodings(self):
        """Test check_pses_in_transforms with CombinedPSEs but empty encodings list."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": [],
                "parameters": {}
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 0

    def test_check_pses_in_transforms_complex_scenario(self):
        """Test check_pses_in_transforms with a complex scenario."""
        transforms = OmegaConf.create({
            "preprocessing": {
                "some_param": "value"
            },
            "CombinedPSEs_1": {
                "encodings": ["LapPE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 16,
                        "include_eigenvalues": True,
                        "concat_to_x": True
                    }
                }
            },
            "other_transform": {
                "param": "value"
            },
            "RWSE_custom": {
                "max_pe_dim": 8,
                "concat_to_x": False
            },
            "CombinedPSEs_2": {
                "encodings": ["RWSE"],
                "parameters": {
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 44  # (16 * 2) + 8 + 4

    @pytest.mark.parametrize("max_pe_dim,include_eigenvalues,expected", [
        (4, False, 4),
        (4, True, 8),
        (8, False, 8),
        (8, True, 16),
        (16, False, 16),
        (16, True, 32),
    ])
    def test_check_pses_in_transforms_lappe_parametrized(self, max_pe_dim, include_eigenvalues, expected):
        """Parametrized test for LapPE with different configurations.

        Parameters
        ----------
        max_pe_dim : int
            Maximum positional encoding dimension for LapPE.
        include_eigenvalues : bool
            Whether to include eigenvalues in the encoding.
        expected : int
            Expected dimension of the positional encoding.
        """
        transforms = OmegaConf.create({
            "LapPE": {
                "max_pe_dim": max_pe_dim,
                "include_eigenvalues": include_eigenvalues,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == expected

    @pytest.mark.parametrize("lappe_dim,rwse_dim,expected", [
        (4, 4, 8),
        (8, 4, 12),
        (4, 8, 12),
        (16, 8, 24),
        (8, 16, 24),
    ])
    def test_check_pses_in_transforms_combined_parametrized(self, lappe_dim, rwse_dim, expected):
        """Parametrized test for CombinedPSEs with different dimension combinations.

        Parameters
        ----------
        lappe_dim : int
            Dimension for LapPE encoding.
        rwse_dim : int
            Dimension for RWSE encoding.
        expected : int
            Expected combined dimension of both encodings.
        """
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": lappe_dim,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": rwse_dim,
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == expected

    def test_check_pses_in_transforms_electrostatic_pe_only(self):
        """Test check_pses_in_transforms with only ElectrostaticPE encoding."""
        transforms = OmegaConf.create({
            "ElectrostaticPE": {
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 7

    def test_check_pses_in_transforms_hkdiag_se_only(self):
        """Test check_pses_in_transforms with only HKdiagSE encoding."""
        transforms = OmegaConf.create({
            "HKdiagSE": {
                "kernel_param_HKdiagSE": [1, 5],
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 4  # range(1, 5) = 4

    def test_check_pses_in_transforms_hkdiag_se_different_ranges(self):
        """Test check_pses_in_transforms with HKdiagSE using different kernel param ranges."""
        transforms = OmegaConf.create({
            "HKdiagSE": {
                "kernel_param_HKdiagSE": [1, 9],
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 8  # range(1, 9) = 8

    def test_check_pses_in_transforms_combined_pses_electrostatic_pe(self):
        """Test check_pses_in_transforms with CombinedPSEs containing ElectrostaticPE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["ElectrostaticPE"],
                "parameters": {
                    "ElectrostaticPE": {
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 7

    def test_check_pses_in_transforms_combined_pses_hkdiag_se(self):
        """Test check_pses_in_transforms with CombinedPSEs containing HKdiagSE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["HKdiagSE"],
                "parameters": {
                    "HKdiagSE": {
                        "kernel_param_HKdiagSE": [1, 5],
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 4

    def test_check_pses_in_transforms_combined_all_four(self):
        """Test check_pses_in_transforms with CombinedPSEs containing all four encoding types."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE", "ElectrostaticPE", "HKdiagSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    },
                    "ElectrostaticPE": {
                        "concat_to_x": True
                    },
                    "HKdiagSE": {
                        "kernel_param_HKdiagSE": [1, 4],
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 22  # 8 + 4 + 7 + 3

    def test_check_pses_in_transforms_combined_all_four_with_eigenvalues(self):
        """Test check_pses_in_transforms with all four encodings and LapPE eigenvalues."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE", "ElectrostaticPE", "HKdiagSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": True,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    },
                    "ElectrostaticPE": {
                        "concat_to_x": True
                    },
                    "HKdiagSE": {
                        "kernel_param_HKdiagSE": [1, 4],
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 30  # (8*2) + 4 + 7 + 3

    def test_check_pses_in_transforms_separate_all_four(self):
        """Test check_pses_in_transforms with all four encodings as separate transforms."""
        transforms = OmegaConf.create({
            "LapPE": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            },
            "RWSE": {
                "max_pe_dim": 4,
                "concat_to_x": True
            },
            "ElectrostaticPE": {
                "concat_to_x": True
            },
            "HKdiagSE": {
                "kernel_param_HKdiagSE": [1, 5],
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 23  # 8 + 4 + 7 + 4

    def test_check_pses_in_transforms_mixed_combined_and_separate_with_new_encodings(self):
        """Test check_pses_in_transforms with CombinedPSEs and separate ElectrostaticPE/HKdiagSE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            },
            "ElectrostaticPE_extra": {
                "concat_to_x": True
            },
            "HKdiagSE_extra": {
                "kernel_param_HKdiagSE": [1, 6],
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 24  # (8 + 4) + 7 + 5

    def test_check_fes_in_transforms_empty(self):
        """Test check_fes_in_transforms with no encodings."""
        transforms = OmegaConf.create({})
        assert check_fes_in_transforms(transforms) == 0

    def test_check_fes_single_transform_pprfe_list(self):
        """Single flat PPRFE: alpha list uses second element (ListConfig path)."""
        transforms = OmegaConf.create(
            {
                "transform_name": "PPRFE",
                "alpha_param_PPRFE": [0.1, 5],
            }
        )
        assert check_fes_in_transforms(transforms) == 5

    def test_check_fes_single_transform_pprfe_scalar(self):
        """Single flat PPRFE: scalar alpha counts as fixed dimension."""
        transforms = OmegaConf.create(
            {
                "transform_name": "PPRFE",
                "alpha_param_PPRFE": 4,
            }
        )
        assert check_fes_in_transforms(transforms) == 4

    def test_check_fes_single_transform_sheaf(self):
        """Single flat SheafConnLapPE uses max_pe_dim."""
        transforms = OmegaConf.create(
            {"transform_name": "SheafConnLapPE", "max_pe_dim": 6}
        )
        assert check_fes_in_transforms(transforms) == 6

    def test_check_fes_keyed_pprfe_list(self):
        """Keyed transform whose name contains PPRFE."""
        transforms = OmegaConf.create(
            {"run_PPRFE_1": {"alpha_param_PPRFE": [0.1, 8]}}
        )
        assert check_fes_in_transforms(transforms) == 8

    def test_check_fes_keyed_pprfe_scalar(self):
        """Keyed PPRFE with scalar alpha_param."""
        transforms = OmegaConf.create({"extra_PPRFE": {"alpha_param_PPRFE": 3}})
        assert check_fes_in_transforms(transforms) == 3

    def test_check_fes_keyed_sheaf(self):
        """Keyed transform whose name contains SheafConnLapPE."""
        transforms = OmegaConf.create(
            {"custom_SheafConnLapPE": {"max_pe_dim": 9}}
        )
        assert check_fes_in_transforms(transforms) == 9

    def test_check_fes_combined_fes_pprfe_and_sheaf(self):
        """Test CombinedFEs inner loop with PPRFE list and SheafConnLapPE."""
        transforms = OmegaConf.create(
            {
                "CombinedFEs": {
                    "encodings": ["PPRFE", "SheafConnLapPE"],
                    "parameters": {
                        "PPRFE": {"alpha_param_PPRFE": [0.1, 7], "concat_to_x": True},
                        "SheafConnLapPE": {
                            "max_pe_dim": 4,
                            "stalk_dim": 2,
                            "concat_to_x": True,
                        },
                    },
                }
            }
        )
        assert check_fes_in_transforms(transforms) == 7 + 4

    def test_check_fes_combined_fes_pprfe_default_alpha(self):
        """Test CombinedFEs PPRFE with missing alpha_param using default [0.1, 10]."""
        transforms = OmegaConf.create(
            {
                "CombinedFEs": {
                    "encodings": ["PPRFE"],
                    "parameters": {"PPRFE": {"concat_to_x": True}},
                }
            }
        )
        assert check_fes_in_transforms(transforms) == 10

    def test_check_fes_combined_fes_pprfe_scalar_alpha(self):
        """Test CombinedFEs PPRFE with scalar alpha_param."""
        transforms = OmegaConf.create(
            {
                "CombinedFEs": {
                    "encodings": ["PPRFE"],
                    "parameters": {
                        "PPRFE": {"alpha_param_PPRFE": 11, "concat_to_x": True}
                    },
                }
            }
        )
        assert check_fes_in_transforms(transforms) == 11

    def test_check_fes_in_transforms_hkfe(self):
        """Test check_fes_in_transforms with HKFE."""
        transforms = OmegaConf.create({
            "transform_name": "HKFE",
            "kernel_param_HKFE": [1, 5]
        })
        assert check_fes_in_transforms(transforms) == 4

        transforms = OmegaConf.create({
            "transform_name": "HKFE",
            "kernel_param_HKFE": 3
        })
        assert check_fes_in_transforms(transforms) == 3

        transforms = OmegaConf.create({
            "CombinedFEs": {
                "encodings": ["HKFE"],
                "parameters": {
                    "HKFE": {"kernel_param_HKFE": [2, 7]}
                }
            }
        })
        assert check_fes_in_transforms(transforms) == 5

        transforms = OmegaConf.create({
            "HKFE_extra": {"kernel_param_HKFE": [0, 10]}
        })
        assert check_fes_in_transforms(transforms) == 10

    def test_check_fes_in_transforms_khopfe(self):
        """Test check_fes_in_transforms with KHopFE."""
        transforms = OmegaConf.create({
            "transform_name": "KHopFE",
            "max_hop": 5
        })
        assert check_fes_in_transforms(transforms) == 4

        transforms = OmegaConf.create({
            "CombinedFEs": {
                "encodings": ["KHopFE"],
                "parameters": {
                    "KHopFE": {"max_hop": 3}
                }
            }
        })
        assert check_fes_in_transforms(transforms) == 2

        transforms = OmegaConf.create({
            "KHopFE_extra": {"max_hop": 6}
        })
        assert check_fes_in_transforms(transforms) == 5

    def test_get_fes_dimensions_khopfe(self):
        """Test get_fes_dimensions with KHopFE using max_hop - 1."""
        encodings = ["KHopFE"]
        parameters = {"KHopFE": {"max_hop": 5}}
        assert get_fes_dimensions(encodings, parameters) == [4]

    def test_get_fes_dimensions_hkfe(self):
        """Test get_fes_dimensions with HKFE."""
        encodings = ["HKFE"]
        parameters = {"HKFE": {"kernel_param_HKFE": [1, 5]}}
        assert get_fes_dimensions(encodings, parameters) == [4]

        parameters = {"HKFE": {"kernel_param_HKFE": 3}}
        assert get_fes_dimensions(encodings, parameters) == [3]

    def test_get_fes_dimensions_pprfe_list_tuple(self):
        """Test get_fes_dimensions with PPRFE alpha as tuple returning second element."""
        encodings = ["PPRFE"]
        parameters = {"PPRFE": {"alpha_param_PPRFE": (0.1, 6)}}
        assert get_fes_dimensions(encodings, parameters) == [6]

    def test_get_fes_dimensions_pprfe_omegaconf_list(self):
        """Test get_fes_dimensions with PPRFE alpha as OmegaConf list."""
        parameters = OmegaConf.create(
            {"PPRFE": {"alpha_param_PPRFE": [0.1, 12]}}
        )
        assert get_fes_dimensions(["PPRFE"], parameters) == [12]

    def test_get_fes_dimensions_pprfe_scalar(self):
        """Test get_fes_dimensions with PPRFE scalar alpha."""
        encodings = ["PPRFE"]
        parameters = {"PPRFE": {"alpha_param_PPRFE": 5}}
        assert get_fes_dimensions(encodings, parameters) == [5]

    def test_get_fes_dimensions_pprfe_missing_uses_default(self):
        """Test get_fes_dimensions with missing PPRFE block using default alpha upper bound 10."""
        assert get_fes_dimensions(["PPRFE"], {}) == [10]

    def test_get_fes_dimensions_sheaf(self):
        """Test get_fes_dimensions with SheafConnLapPE."""
        encodings = ["SheafConnLapPE"]
        parameters = {"SheafConnLapPE": {"max_pe_dim": 8}}
        assert get_fes_dimensions(encodings, parameters) == [8]

    def test_get_all_encoding_dimensions_khopfe_pprfe_sheaf(self):
        """Test get_all_encoding_dimensions with KHopFE, PPRFE list, and SheafConnLapPE branches."""
        encodings = ["KHopFE", "PPRFE", "SheafConnLapPE"]
        parameters = {
            "KHopFE": {"max_hop": 4},
            "PPRFE": {"alpha_param_PPRFE": [0.1, 9]},
            "SheafConnLapPE": {"max_pe_dim": 3},
        }
        assert get_all_encoding_dimensions(encodings, parameters) == [3, 9, 3]

    def test_get_all_encoding_dimensions_pprfe_scalar(self):
        """Test get_all_encoding_dimensions with PPRFE scalar alpha."""
        assert get_all_encoding_dimensions(
            ["PPRFE"], {"PPRFE": {"alpha_param_PPRFE": 2}}
        ) == [2]

    def test_get_all_encoding_dimensions_pprfe_missing_uses_default(self):
        """Test get_all_encoding_dimensions with PPRFE absent from parameters using default 10."""
        assert get_all_encoding_dimensions(["PPRFE"], {}) == [10]

    def test_get_all_encoding_dimensions_exhaustive(self):
        """Test get_all_encoding_dimensions for all supported encodings."""
        encodings = ["LapPE", "RWSE", "ElectrostaticPE", "HKdiagSE", "HKFE", "KHopFE", "PPRFE", "SheafConnLapPE"]
        parameters = {
            "LapPE": {"max_pe_dim": 8, "include_eigenvalues": True},
            "RWSE": {"max_pe_dim": 4},
            "ElectrostaticPE": {},
            "HKdiagSE": {"kernel_param_HKdiagSE": [1, 5]},
            "HKFE": {"kernel_param_HKFE": [2, 8]},
            "KHopFE": {"max_hop": 3},
            "PPRFE": {"alpha_param_PPRFE": [0.1, 7]},
            "SheafConnLapPE": {"max_pe_dim": 5}
        }
        expected = [16, 4, 7, 4, 6, 2, 7, 5]
        assert get_all_encoding_dimensions(encodings, parameters) == expected

        # Test scalar params
        parameters["HKdiagSE"]["kernel_param_HKdiagSE"] = 10
        parameters["HKFE"]["kernel_param_HKFE"] = 12
        parameters["PPRFE"]["alpha_param_PPRFE"] = 15
        parameters["LapPE"]["include_eigenvalues"] = False
        expected = [8, 4, 7, 10, 12, 2, 15, 5]
        assert get_all_encoding_dimensions(encodings, parameters) == expected

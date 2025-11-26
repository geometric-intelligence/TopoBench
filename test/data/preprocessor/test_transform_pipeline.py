"""Tests for TransformPipeline.

This test suite validates the two-tier transform pipeline that separates
heavy (offline) and light (runtime) transforms for fast experimentation.
"""

import pytest
import torch
import torch_geometric
from torch_geometric.data import Data

from topobench.data.preprocessor._ondisk.transform_pipeline import (
    TransformPipeline,
)


# Mock transforms for testing
class MockLiftingTransform(torch_geometric.transforms.BaseTransform):
    """Mock topological lifting (heavy)."""

    __module__ = "topobench.transforms.liftings.graph2simplicial"

    def __init__(self):
        super().__init__()
        self.parameters = {"lifting_type": "simplicial"}

    def forward(self, data):
        # Add marker to track this transform was applied
        data.lifting_applied = True
        return data


class MockNormalizationTransform(torch_geometric.transforms.BaseTransform):
    """Mock normalization (light)."""

    __module__ = "topobench.transforms.data_manipulations"

    def __init__(self):
        super().__init__()
        self.parameters = {"normalization_type": "standard"}

    def forward(self, data):
        # Add marker to track this transform was applied
        data.normalization_applied = True
        return data


class MockAugmentationTransform(torch_geometric.transforms.BaseTransform):
    """Mock augmentation (light)."""

    __module__ = "topobench.transforms.data_manipulations"

    def __init__(self, angle=15):
        super().__init__()
        self.angle = angle
        self.parameters = {"angle": angle}

    def forward(self, data):
        # Add marker to track this transform was applied
        data.augmentation_applied = True
        data.augmentation_angle = self.angle
        return data


def create_test_data():
    """Create simple test data."""
    return Data(
        x=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        y=torch.tensor([0]),
    )


class TestTransformPipeline:
    """Test suite for TransformPipeline."""

    def test_tier_modes(self):
        """Test different classification modes."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        
        # Auto: separates heavy and light
        auto = TransformPipeline(transforms, transform_tier="auto")
        assert len(auto.heavy_transforms) == 1
        assert len(auto.light_transforms) == 1
        
        # All heavy: everything in heavy
        heavy = TransformPipeline(transforms, transform_tier="all_heavy")
        assert len(heavy.heavy_transforms) == 2
        assert len(heavy.light_transforms) == 0
        
        # All light: everything in light
        light = TransformPipeline(transforms, transform_tier="all_light")
        assert len(light.heavy_transforms) == 0
        assert len(light.light_transforms) == 2
    
    def test_manual_override(self):
        """Manual mode with tier overrides."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        override = {"MockLiftingTransform": "light", "MockNormalizationTransform": "heavy"}
        
        pipeline = TransformPipeline(transforms, transform_tier="manual", tier_override=override)
        
        assert isinstance(pipeline.heavy_transforms[0], MockNormalizationTransform)
        assert isinstance(pipeline.light_transforms[0], MockLiftingTransform)

    def test_apply_heavy(self):
        """Test heavy transform application."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        pipeline = TransformPipeline(transforms, transform_tier="auto")
        data = create_test_data()
        
        result = pipeline.apply_heavy(data)
        
        assert hasattr(result, "lifting_applied")  # Heavy applied
        assert not hasattr(result, "normalization_applied")  # Light not applied
    
    def test_apply_light(self):
        """Test light transform application."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        pipeline = TransformPipeline(transforms, transform_tier="auto")
        data = create_test_data()
        
        result = pipeline.apply_light(data)
        
        assert not hasattr(result, "lifting_applied")  # Heavy not applied
        assert hasattr(result, "normalization_applied")  # Light applied
    
    def test_two_tier_composition(self):
        """Heavy then light produces complete transformation."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        pipeline = TransformPipeline(transforms, transform_tier="auto")
        data = create_test_data()
        
        preprocessed = pipeline.apply_heavy(data)
        final = pipeline.apply_light(preprocessed)
        
        assert hasattr(final, "lifting_applied")
        assert hasattr(final, "normalization_applied")

    def test_cache_key_ignores_light(self):
        """Cache key based only on heavy transforms."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        pipeline = TransformPipeline(transforms, transform_tier="auto")
        
        key1 = pipeline.compute_cache_key()
        pipeline.light_transforms[0] = MockAugmentationTransform()
        key2 = pipeline.compute_cache_key()
        
        assert key1 == key2  # Unchanged
    
    def test_cache_key_changes_with_heavy(self):
        """Cache key changes when heavy transforms change."""
        class DifferentLifting(MockLiftingTransform):
            def __init__(self):
                super().__init__()
                self.parameters = {"lifting_type": "cell"}
        
        pipeline1 = TransformPipeline([MockLiftingTransform()], transform_tier="all_heavy")
        pipeline2 = TransformPipeline([DifferentLifting()], transform_tier="all_heavy")
        
        assert pipeline1.compute_cache_key() != pipeline2.compute_cache_key()

    def test_summary_and_repr(self):
        """Test summary generation and string representation."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        pipeline = TransformPipeline(transforms, transform_tier="auto")
        
        summary = pipeline.get_summary()
        assert summary["heavy_count"] == 1
        assert summary["light_count"] == 1
        
        repr_str = repr(pipeline)
        assert "TransformPipeline" in repr_str
        assert "heavy=1" in repr_str

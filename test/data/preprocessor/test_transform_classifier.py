"""Tests for TransformClassifier.

This test suite validates automatic classification of transforms into
heavy (offline) and light (runtime) categories for the two-tier system.
"""

import pytest
import torch_geometric

from topobench.data.preprocessor._ondisk.transform_classifier import (
    TransformClassifier,
)


# Mock transforms for testing
class MockLiftingTransform(torch_geometric.transforms.BaseTransform):
    """Mock topological lifting (heavy)."""

    __module__ = "topobench.transforms.liftings.graph2simplicial"

    def forward(self, data):
        return data


class MockNormalizationTransform(torch_geometric.transforms.BaseTransform):
    """Mock normalization (light)."""

    __module__ = "topobench.transforms.data_manipulations"

    def forward(self, data):
        return data


class MockRandomTransform(torch_geometric.transforms.BaseTransform):
    """Mock augmentation (light)."""

    __module__ = "topobench.transforms.data_manipulations"

    def forward(self, data):
        return data


class MockLaplacianTransform(torch_geometric.transforms.BaseTransform):
    """Mock Laplacian computation (heavy)."""

    __module__ = "topobench.transforms.data_manipulations"

    def forward(self, data):
        return data


class MockUnknownTransform(torch_geometric.transforms.BaseTransform):
    """Mock unknown transform (should default to heavy)."""

    __module__ = "topobench.transforms.unknown"

    def forward(self, data):
        return data


class TestTransformClassifier:
    """Test suite for TransformClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return TransformClassifier()

    def test_pattern_classification(self, classifier):
        """Test pattern-based classification for heavy and light."""
        # Heavy patterns
        assert classifier.classify(MockLiftingTransform()) == "heavy"
        assert classifier.classify(MockLaplacianTransform()) == "heavy"
        
        # Light patterns
        assert classifier.classify(MockNormalizationTransform()) == "light"
        assert classifier.classify(MockRandomTransform()) == "light"

    def test_module_classification(self, classifier):
        """Test module-based classification."""
        # Liftings module → heavy
        class LiftingModuleTransform(torch_geometric.transforms.BaseTransform):
            __module__ = "topobench.transforms.liftings.graph2simplicial"
            def forward(self, data): return data
        
        # Data manipulations → light
        class DataManipTransform(torch_geometric.transforms.BaseTransform):
            __module__ = "topobench.transforms.data_manipulations"
            def forward(self, data): return data
        
        # Feature liftings → light (checked before "liftings")
        class FeatureLiftingTransform(torch_geometric.transforms.BaseTransform):
            __module__ = "topobench.transforms.feature_liftings"
            def forward(self, data): return data
        
        assert classifier.classify(LiftingModuleTransform()) == "heavy"
        assert classifier.classify(DataManipTransform()) == "light"
        assert classifier.classify(FeatureLiftingTransform()) == "light"

    def test_manual_override(self, classifier):
        """Manual override takes highest priority."""
        # Override heavy → light
        assert classifier.classify(
            MockLiftingTransform(),
            {"MockLiftingTransform": "light"}
        ) == "light"
        
        # Override light → heavy
        assert classifier.classify(
            MockNormalizationTransform(),
            {"MockNormalizationTransform": "heavy"}
        ) == "heavy"
    
    def test_unknown_defaults_to_heavy(self, classifier):
        """Unknown transforms default to heavy (conservative)."""
        assert classifier.classify(MockUnknownTransform()) == "heavy"

    def test_classify_pipeline(self, classifier):
        """Test pipeline classification with mixed transforms."""
        transforms = [
            MockLiftingTransform(),
            MockNormalizationTransform(),
            MockLaplacianTransform(),
            MockRandomTransform(),
        ]
        heavy, light = classifier.classify_pipeline(transforms)
        
        assert len(heavy) == 2  # Lifting, Laplacian
        assert len(light) == 2  # Normalization, Random
        assert isinstance(heavy[0], MockLiftingTransform)
        assert isinstance(light[0], MockNormalizationTransform)
    
    def test_classify_pipeline_with_override(self, classifier):
        """Pipeline with manual overrides."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        override = {"MockLiftingTransform": "light", "MockNormalizationTransform": "heavy"}
        
        heavy, light = classifier.classify_pipeline(transforms, override)
        
        assert isinstance(heavy[0], MockNormalizationTransform)
        assert isinstance(light[0], MockLiftingTransform)

    def test_get_classification_summary(self, classifier):
        """Test summary generation."""
        transforms = [MockLiftingTransform(), MockNormalizationTransform()]
        summary = classifier.get_classification_summary(transforms)
        
        assert summary["heavy"] == ["MockLiftingTransform"]
        assert summary["light"] == ["MockNormalizationTransform"]
        assert summary["heavy_count"] == 1
        assert summary["light_count"] == 1

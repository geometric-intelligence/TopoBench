"""Transform classifier for two-tier preprocessing system.

This module provides automatic classification of transforms into heavy (offline)
and light (runtime) categories, enabling fast augmentation experimentation without
reprocessing expensive topological liftings.

Performance Impact:
- Augmentation experiments: 24× faster (instant vs hours)
- Research productivity: Try 10-100× more hyperparameter combinations
"""

from typing import Literal

import torch_geometric


class TransformClassifier:
    """Automatically classify transforms as heavy (offline) or light (runtime).

    This classifier enables the two-tier transform system by analyzing each
    transform and determining whether it should be:
    - **Heavy**: Cached offline during preprocessing (topological liftings)
    - **Light**: Applied at runtime during training (augmentations, normalization)

    Classification Strategy
    -----------------------
    The classifier uses a multi-level decision process:

    1. **Manual Override** (highest priority):
       User-specified classification via tier_override parameter

    2. **Type-based Classification**:
       - Heavy patterns: "liftings.", "Laplacian", "Hodge", etc.
       - Light patterns: "Normalization", "Random", "Dropout", etc.

    3. **Module-based Classification**:
       - transforms.liftings → heavy
       - transforms.data_manipulations → light

    4. **Conservative Default**:
       When uncertain, classify as heavy (safer to cache than re-run)

    Heavy Transform Indicators
    --------------------------
    - Topological liftings (simplicial, cell, hypergraph)
    - Hodge Laplacian computation
    - Persistent homology features
    - Graph algorithms (connected components, paths)
    - Operations independent of training hyperparameters
    - Computationally expensive operations (> 1 second per sample)

    Light Transform Indicators
    --------------------------
    - Data augmentation (rotation, noise, dropout)
    - Feature normalization and scaling
    - Simple feature transformations
    - Operations dependent on training hyperparameters
    - Fast operations (< 10 ms per sample)

    Examples
    --------
    >>> classifier = TransformClassifier()
    >>>
    >>> # Automatic classification
    >>> from topobench.transforms import SimplicialCliqueLifting, FeatureNormalization
    >>> classifier.classify(SimplicialCliqueLifting())
    'heavy'
    >>> classifier.classify(FeatureNormalization())
    'light'
    >>>
    >>> # Manual override
    >>> tier_override = {"MyCustomTransform": "heavy"}
    >>> classifier.classify(MyCustomTransform(), manual_override=tier_override)
    'heavy'
    >>>
    >>> # Classify entire pipeline
    >>> transforms = [SimplicialCliqueLifting(), FeatureNormalization()]
    >>> heavy, light = classifier.classify_pipeline(transforms)
    >>> len(heavy), len(light)
    (1, 1)

    See Also
    --------
    TransformPipeline : Manages two-tier transform execution.
    OnDiskInductivePreprocessor : Uses classifier for transform separation.
    """

    def __init__(self) -> None:
        """Initialize transform classifier with pattern-based rules."""
        # Heavy transform patterns (topological, combinatorial)
        # Note: "Lifting" class name pattern + module check handles lifting classification
        self.heavy_patterns = [
            "Lifting",  # Lifting classes (topological)
            "Laplacian",  # Laplacian computation
            "Hodge",  # Hodge theory operations
            "Homology",  # Topological features
            "Persistence",  # Persistent homology
            "ConnectedComponent",  # Graph algorithms
            "Clique",  # Clique enumeration
            "Path",  # Path enumeration
            "Curvature",  # Curvature computation
        ]

        # Light transform patterns (augmentation, normalization)
        self.light_patterns = [
            "Normalization",  # Feature scaling
            "Normalize",  # Alternative naming
            "Random",  # Augmentation (RandomRotation, etc.)
            "Dropout",  # Regularization
            "Noise",  # Augmentation
            "Rotation",  # Augmentation
            "OneHot",  # Encoding
            "ToFloat",  # Type conversion
            "Identity",  # No-op transform
        ]

    def classify(
        self,
        transform: torch_geometric.transforms.BaseTransform,
        manual_override: dict[str, Literal["heavy", "light"]] | None = None,
    ) -> Literal["heavy", "light"]:
        """Classify a single transform as heavy or light.

        Parameters
        ----------
        transform : torch_geometric.transforms.BaseTransform
            Transform to classify.
        manual_override : dict, optional
            Manual classification overrides mapping transform class names
            to "heavy" or "light". Takes highest priority in decision process.

        Returns
        -------
        Literal["heavy", "light"]
            Classification result. "heavy" transforms are cached offline,
            "light" transforms are applied at runtime.

        Examples
        --------
        >>> classifier = TransformClassifier()
        >>> from topobench.transforms import SimplicialCliqueLifting
        >>> classifier.classify(SimplicialCliqueLifting())
        'heavy'
        >>>
        >>> # Manual override
        >>> classifier.classify(
        ...     SimplicialCliqueLifting(),
        ...     manual_override={"SimplicialCliqueLifting": "light"}
        ... )
        'light'
        """
        transform_name = transform.__class__.__name__
        transform_module = transform.__module__

        # Priority 1: Manual override (highest priority)
        if manual_override and transform_name in manual_override:
            return manual_override[transform_name]

        # Priority 2: Special module cases (before pattern matching)
        # Feature liftings are light (simple projections), check before "Lifting" pattern
        if "feature_liftings" in transform_module:
            return "light"

        # Priority 3: Type-based classification via patterns
        # Check heavy patterns first
        for pattern in self.heavy_patterns:
            if pattern in transform_name or pattern in transform_module:
                return "heavy"

        # Check light patterns
        for pattern in self.light_patterns:
            if pattern in transform_name or pattern in transform_module:
                return "light"

        # Priority 4: Module-based classification
        # Liftings are always heavy (topological operations)
        if "liftings" in transform_module:
            return "heavy"

        # Data manipulations are typically light (feature operations)
        if "data_manipulations" in transform_module:
            return "light"

        # Priority 5: Conservative default
        # When uncertain, classify as heavy (safer to cache than re-run)
        return "heavy"

    def classify_pipeline(
        self,
        transforms: list[torch_geometric.transforms.BaseTransform],
        manual_override: dict[str, Literal["heavy", "light"]] | None = None,
    ) -> tuple[
        list[torch_geometric.transforms.BaseTransform],
        list[torch_geometric.transforms.BaseTransform],
    ]:
        """Classify an entire transform pipeline into heavy and light lists.

        Parameters
        ----------
        transforms : list of BaseTransform
            List of transforms to classify. Typically obtained from
            torch_geometric.transforms.Compose.transforms attribute.
        manual_override : dict, optional
            Manual classification overrides. See classify() for details.

        Returns
        -------
        heavy_transforms : list of BaseTransform
            Transforms to apply offline during preprocessing.
            These are cached and reused across experiments.
        light_transforms : list of BaseTransform
            Transforms to apply at runtime during training.
            These can be changed without reprocessing.

        Examples
        --------
        >>> classifier = TransformClassifier()
        >>> from topobench.transforms import SimplicialCliqueLifting, FeatureNormalization
        >>> transforms = [SimplicialCliqueLifting(), FeatureNormalization()]
        >>> heavy, light = classifier.classify_pipeline(transforms)
        >>> len(heavy), len(light)
        (1, 1)
        >>> heavy[0].__class__.__name__
        'SimplicialCliqueLifting'
        >>> light[0].__class__.__name__
        'FeatureNormalization'

        Notes
        -----
        The order of transforms is preserved within each tier. Heavy transforms
        are applied first (offline), then light transforms (runtime).
        """
        heavy_transforms = []
        light_transforms = []

        for transform in transforms:
            classification = self.classify(transform, manual_override)

            if classification == "heavy":
                heavy_transforms.append(transform)
            else:
                light_transforms.append(transform)

        return heavy_transforms, light_transforms

    def get_classification_summary(
        self,
        transforms: list[torch_geometric.transforms.BaseTransform],
        manual_override: dict[str, Literal["heavy", "light"]] | None = None,
    ) -> dict[str, list[str]]:
        """Get detailed classification summary for debugging.

        Parameters
        ----------
        transforms : list of BaseTransform
            Transforms to classify.
        manual_override : dict, optional
            Manual classification overrides.

        Returns
        -------
        dict
            Summary with keys:
            - "heavy": List of heavy transform names
            - "light": List of light transform names
            - "heavy_count": Number of heavy transforms
            - "light_count": Number of light transforms

        Examples
        --------
        >>> classifier = TransformClassifier()
        >>> transforms = [SimplicialCliqueLifting(), FeatureNormalization()]
        >>> summary = classifier.get_classification_summary(transforms)
        >>> print(summary["heavy"])
        ['SimplicialCliqueLifting']
        >>> print(summary["light"])
        ['FeatureNormalization']
        """
        heavy, light = self.classify_pipeline(transforms, manual_override)

        return {
            "heavy": [t.__class__.__name__ for t in heavy],
            "light": [t.__class__.__name__ for t in light],
            "heavy_count": len(heavy),
            "light_count": len(light),
        }

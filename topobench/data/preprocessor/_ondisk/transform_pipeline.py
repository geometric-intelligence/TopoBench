"""Transform pipeline management for two-tier preprocessing system.

This module manages the execution of transforms in a two-tier architecture:
- Heavy transforms: Applied offline during preprocessing, results cached
- Light transforms: Applied at runtime during training for fast experimentation

Performance Impact:
- Augmentation experiments: 24× faster (instant vs hours) TODO: actual numbers
- Cache efficiency: Only heavy transforms affect cache key
"""

from typing import Any

import torch_geometric
from torch_geometric.data import Data

from topobench.data.preprocessor._ondisk.transform_classifier import (
    TransformClassifier,
)
from topobench.data.preprocessor._ondisk.transform_dag import TransformDAG
from topobench.data.utils import ensure_serializable, make_hash


class TransformPipeline:
    """Manages two-tier transform execution for fast experimentation.

    This pipeline separates transforms into two tiers:
    - **Heavy tier**: Applied during preprocessing, results cached on disk
    - **Light tier**: Applied at runtime during training

    The key insight is that expensive topological liftings should be cached
    (heavy), while cheap augmentations can run at training time (light). This
    enables researchers to experiment with 10-100× augmentation variations TODO: really so much?
    without reprocessing expensive liftings.

    Parameters
    ----------
    transforms : list of BaseTransform
        Complete list of transforms to apply.
    transform_tier : str, optional
        Classification mode: "auto", "all_heavy", "all_light", or "manual".
        Default: "auto" (automatic classification).
    tier_override : dict, optional
        Manual classification overrides mapping transform names to "heavy"/"light".
        Takes highest priority. Default: None.

    Attributes
    ----------
    heavy_transforms : list of BaseTransform
        Transforms applied offline during preprocessing.
    light_transforms : list of BaseTransform
        Transforms applied at runtime during training.
    heavy_compose : Compose or None
        Composed heavy transforms.
    light_compose : Compose or None
        Composed light transforms.

    Examples
    --------
    >>> from topobench.transforms import SimplicialCliqueLifting, FeatureNormalization
    >>> transforms = [SimplicialCliqueLifting(), FeatureNormalization()]
    >>>
    >>> # Automatic classification
    >>> pipeline = TransformPipeline(transforms, transform_tier="auto")
    >>> len(pipeline.heavy_transforms), len(pipeline.light_transforms)
    (1, 1)
    >>>
    >>> # Manual override
    >>> tier_override = {"SimplicialCliqueLifting": "light"}
    >>> pipeline = TransformPipeline(
    ...     transforms,
    ...     transform_tier="manual",
    ...     tier_override=tier_override
    ... )
    >>>
    >>> # Apply heavy transforms during preprocessing
    >>> preprocessed_data = pipeline.apply_heavy(raw_data)
    >>>
    >>> # Apply light transforms at runtime
    >>> augmented_data = pipeline.apply_light(preprocessed_data)
    """

    def __init__(
        self,
        transforms: list[torch_geometric.transforms.BaseTransform],
        transform_tier: str = "auto",
        tier_override: dict[str, str] | None = None,
    ) -> None:
        """Initialize transform pipeline with tier classification.

        Parameters
        ----------
        transforms : list of BaseTransform
            Complete list of transforms.
        transform_tier : str, optional
            Classification mode (default: "auto").
        tier_override : dict, optional
            Manual overrides (default: None).
        """
        self.transforms = transforms
        self.transform_tier = transform_tier
        self.tier_override = tier_override

        # Classify transforms into heavy and light
        self._classify_transforms()

        # Build dependency graph for granular caching
        self._build_dag()

        # Create composed transforms for each tier
        self.heavy_compose = (
            torch_geometric.transforms.Compose(self.heavy_transforms)
            if self.heavy_transforms
            else None
        )
        self.light_compose = (
            torch_geometric.transforms.Compose(self.light_transforms)
            if self.light_transforms
            else None
        )

    def _classify_transforms(self) -> None:
        """Classify transforms into heavy and light tiers."""
        if self.transform_tier == "all_heavy":
            # All transforms are heavy (current behavior)
            self.heavy_transforms = self.transforms
            self.light_transforms = []

        elif self.transform_tier == "all_light":
            # All transforms are light (no preprocessing)
            self.heavy_transforms = []
            self.light_transforms = self.transforms

        elif self.transform_tier == "auto":
            # Automatic classification
            classifier = TransformClassifier()
            self.heavy_transforms, self.light_transforms = (
                classifier.classify_pipeline(
                    self.transforms, self.tier_override
                )
            )

        elif self.transform_tier == "manual":
            # Manual classification via tier_override
            if self.tier_override is None:
                raise ValueError(
                    "tier_override must be provided when transform_tier='manual'"
                )
            classifier = TransformClassifier()
            self.heavy_transforms, self.light_transforms = (
                classifier.classify_pipeline(
                    self.transforms, self.tier_override
                )
            )

        else:
            raise ValueError(
                f"Invalid transform_tier: {self.transform_tier}. "
                f"Must be one of: 'auto', 'all_heavy', 'all_light', 'manual'"
            )

    def _build_dag(self) -> None:
        """Build dependency graph from classified transforms.

        Creates a DAG with sequential dependencies (transform N depends on N-1).
        This enables per-transform hashing and granular cache invalidation.
        """
        self.dag = TransformDAG()

        # Add heavy transforms with sequential dependencies
        prev_id = None
        for transform in self.heavy_transforms:
            deps = [prev_id] if prev_id else None
            prev_id = self.dag.add_transform(
                transform, tier="heavy", dependencies=deps
            )

        # Add light transforms (depend on last heavy transform)
        last_heavy = prev_id
        prev_id = last_heavy
        for transform in self.light_transforms:
            deps = [prev_id] if prev_id else None
            prev_id = self.dag.add_transform(
                transform, tier="light", dependencies=deps
            )

    def apply_heavy(self, data: Data) -> Data:
        """Apply heavy transforms (offline preprocessing).

        This method is called during preprocessing to apply expensive
        topological liftings. Results are cached on disk for reuse.

        Parameters
        ----------
        data : Data
            Input data sample.

        Returns
        -------
        Data
            Transformed data with heavy transforms applied.

        Examples
        --------
        >>> pipeline = TransformPipeline(transforms, transform_tier="auto")
        >>> preprocessed = pipeline.apply_heavy(raw_data)
        >>> # Save to disk for caching
        """
        if self.heavy_compose is not None:
            return self.heavy_compose(data)
        return data

    def apply_light(self, data: Data) -> Data:
        """Apply light transforms (runtime augmentation).

        This method is called during training via __getitem__ to apply
        fast augmentations. Can be changed without reprocessing.

        Parameters
        ----------
        data : Data
            Input data sample (already has heavy transforms applied).

        Returns
        -------
        Data
            Transformed data with light transforms applied.

        Examples
        --------
        >>> pipeline = TransformPipeline(transforms, transform_tier="auto")
        >>> augmented = pipeline.apply_light(preprocessed_data)
        >>> # Returns instantly, can experiment with different augmentations
        """
        if self.light_compose is not None:
            return self.light_compose(data)
        return data

    def compute_cache_key(self) -> str:
        """Compute cache key from heavy transforms only.

        Light transforms don't affect the cache key since they're applied
        at runtime. This allows changing light transforms (augmentations)
        without invalidating the cache.

        Returns
        -------
        str
            Hash of heavy transform parameters.

        Examples
        --------
        >>> pipeline = TransformPipeline(transforms, transform_tier="auto")
        >>> cache_key = pipeline.compute_cache_key()
        >>> # Changing light transforms doesn't change cache_key
        >>> pipeline.light_transforms[0] = NewAugmentation()
        >>> assert pipeline.compute_cache_key() == cache_key
        """
        # Extract parameters from heavy transforms
        heavy_params = []
        for transform in self.heavy_transforms:
            if hasattr(transform, "parameters"):
                heavy_params.append(transform.parameters)
            else:
                # Fallback: use class name if no parameters attribute
                heavy_params.append(
                    {"__class__": transform.__class__.__name__}
                )

        # Ensure serializable and hash
        serializable_params = ensure_serializable(heavy_params)
        hash_int = make_hash(serializable_params)
        # Convert to hex string for consistency with file paths
        return hex(hash_int)[2:]  # Remove '0x' prefix

    def get_dag(self) -> TransformDAG:
        """Get dependency graph for advanced use cases.

        Returns
        -------
        TransformDAG
            Transform dependency graph with per-transform hashes and dependencies.

        Examples
        --------
        >>> pipeline = TransformPipeline(transforms, transform_tier="auto")
        >>> dag = pipeline.get_dag()
        >>>
        >>> # Find which transforms are affected by a change
        >>> affected = dag.get_affected_transforms("SimplicialLifting_0")
        >>> print(f"Changing lifting affects: {affected}")
        ['SimplicialLifting_0', 'FeatureNorm_0', 'RandomNoise_0']
        >>>
        >>> # Get per-transform hash
        >>> hash_val = dag.get_transform_hash("SimplicialLifting_0")
        >>> print(f"Lifting hash: {hash_val}")
        abc123def456...
        """
        return self.dag

    def get_summary(self) -> dict[str, Any]:
        """Get pipeline summary for debugging and logging.

        Returns
        -------
        dict
            Summary containing:
            - total_transforms: Total number of transforms
            - heavy_count: Number of heavy transforms
            - light_count: Number of light transforms
            - heavy_names: List of heavy transform names
            - light_names: List of light transform names
            - transform_tier: Classification mode used
            - cache_key: Hash of heavy transforms
            - dag_nodes: Number of DAG nodes

        Examples
        --------
        >>> pipeline = TransformPipeline(transforms, transform_tier="auto")
        >>> summary = pipeline.get_summary()
        >>> print(f"Heavy: {summary['heavy_count']}, Light: {summary['light_count']}")
        Heavy: 1, Light: 2
        """
        return {
            "total_transforms": len(self.transforms),
            "heavy_count": len(self.heavy_transforms),
            "light_count": len(self.light_transforms),
            "heavy_names": [
                t.__class__.__name__ for t in self.heavy_transforms
            ],
            "light_names": [
                t.__class__.__name__ for t in self.light_transforms
            ],
            "transform_tier": self.transform_tier,
            "cache_key": self.compute_cache_key(),
            "dag_nodes": len(self.dag.nodes),
        }

    def __repr__(self) -> str:
        """Return string representation of pipeline.

        Returns
        -------
        str
            Human-readable description of pipeline.
        """
        return (
            f"{self.__class__.__name__}("
            f"heavy={len(self.heavy_transforms)}, "
            f"light={len(self.light_transforms)}, "
            f"tier={self.transform_tier})"
        )

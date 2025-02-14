"""Identity transform that does nothing to the input data."""

from topobenchmark.transforms.feature_liftings.base import FeatureLiftingMap


class IdentityFeatureLifting(FeatureLiftingMap):
    """Identity feature lifting map."""

    def lift_features(self, domain):
        """Lift features of a domain using identity map."""
        return domain

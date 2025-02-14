import abc


class FeatureLiftingMap(abc.ABC):
    """Feature lifting map."""

    def __call__(self, domain):
        """Lift features of a domain."""
        return self.lift_features(domain)

    @abc.abstractmethod
    def lift_features(self, domain):
        """Lift features of a domain."""

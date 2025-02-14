"""ProjectionSum class."""

import torch

from topobenchmark.transforms.feature_liftings.base import FeatureLiftingMap


class ProjectionSum(FeatureLiftingMap):
    r"""Lift r-cell features to r+1-cells by projection."""

    def lift_features(self, domain):
        r"""Project r-cell features of a graph to r+1-cell structures.

        Parameters
        ----------
        data : Data
            The input data to be lifted.

        Returns
        -------
        Data
            Domain with the lifted features.
        """
        for key, next_key in zip(
            domain.rank_keys(), domain.rank_keys()[1:], strict=False
        ):
            if domain.features[next_key] is not None:
                continue

            domain.update_features(
                next_key,
                torch.matmul(
                    torch.abs(domain.incidence[next_key].t()),
                    domain.features[key],
                ),
            )

        return domain

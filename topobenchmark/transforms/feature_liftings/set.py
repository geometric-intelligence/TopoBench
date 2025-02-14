"""Set lifting for r-cell features to r+1-cell features."""

import torch

from topobenchmark.transforms.feature_liftings.base import FeatureLiftingMap


class Set(FeatureLiftingMap):
    """Lift r-cell features to r+1-cells by set operations."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(self, domain):
        r"""Concatenate r-cell features to r+1-cell structures.

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

            incidence = domain.incidence[next_key]
            _, n = incidence.shape

            if n != 0:
                idxs_list = []
                for n_feature in range(n):
                    idxs_for_feature = incidence.indices()[
                        0, incidence.indices()[1, :] == n_feature
                    ]
                    idxs_list.append(torch.sort(idxs_for_feature)[0])

                idxs = torch.stack(idxs_list, dim=0)
                if key == 0:
                    values = idxs
                else:
                    values = torch.sort(
                        torch.unique(
                            domain.features[key][idxs].view(idxs.shape[0], -1),
                            dim=1,
                        ),
                        dim=1,
                    )[0]
            else:
                values = torch.tensor([])

            domain.update_features(next_key, values)

        return domain

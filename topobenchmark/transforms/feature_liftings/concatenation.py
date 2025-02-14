"""Concatenation feature lifting."""

import torch

from topobenchmark.transforms.feature_liftings.base import FeatureLiftingMap


class Concatenation(FeatureLiftingMap):
    """Lift r-cell features to r+1-cells by concatenation."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(self, domain):
        r"""Concatenate r-cell features to obtain r+1-cell features.

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
                values = domain.features[key][idxs].view(n, -1)
            else:
                # NB: only works if key is an int representing rank
                m = domain.features[key].shape[1] * (next_key + 1)
                values = torch.zeros([0, m])

            domain.update_features(next_key, values)

        return domain

"""Wrapper for the SCNW model."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper


class SCNWrapper(AbstractWrapper):
    r"""Wrapper for the SCNW model.

    This wrapper defines the forward pass of the model. The SCNW model returns
    the embeddings of the cells of rank 0, 1, and 2.
    """

    def forward(self, batch):
        r"""Forward pass for the SCNW wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        laplacian_0 = self.normalize_matrix(batch.hodge_laplacian_0)
        laplacian_1 = self.normalize_matrix(batch.hodge_laplacian_1)
        laplacian_2 = self.normalize_matrix(batch.hodge_laplacian_2)
        x_0, x_1, x_2 = self.backbone(
            batch.x_0,
            batch.x_1,
            batch.x_2,
            laplacian_0,
            laplacian_1,
            laplacian_2,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_2"] = x_2
        model_out["x_1"] = x_1
        model_out["x_0"] = x_0

        return model_out

    def normalize_matrix(self, matrix):
        r"""Normalize the input matrix.

        The normalization is performed using the diagonal matrix of the inverse square root of the sum of the absolute values of the rows.

        Parameters
        ----------
        matrix : torch.sparse.FloatTensor
            Input matrix to be normalized.

        Returns
        -------
        torch.sparse.FloatTensor
            Normalized matrix.
        """
        matrix_ = matrix.to_dense()
        n, _ = matrix_.shape
        abs_matrix = abs(matrix_)
        diag_sum = abs_matrix.sum(axis=1)

        # Handle division by zero
        idxs = torch.where(diag_sum != 0)
        diag_sum[idxs] = 1.0 / torch.sqrt(diag_sum[idxs])

        diag_indices = torch.stack([torch.arange(n), torch.arange(n)])
        diag_matrix = torch.sparse_coo_tensor(
            diag_indices, diag_sum, matrix_.shape, device=matrix.device
        ).coalesce()
        normalized_matrix = diag_matrix @ (matrix @ diag_matrix)
        return normalized_matrix

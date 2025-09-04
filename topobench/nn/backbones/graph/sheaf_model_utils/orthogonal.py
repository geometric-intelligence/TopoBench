# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from torch import nn
# from experiments.neural_sheaf_diffusion.timing_utils import global_tracker


class Orthogonal(nn.Module):
    """Based on https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#orthogonal"""
    def __init__(self, d, orthogonal_map):
        super().__init__()
        assert orthogonal_map in ["matrix_exp", "cayley", "householder", "euler"]
        self.d = d
        self.orthogonal_map = orthogonal_map

    def get_2d_rotation(self, params):
        # global_tracker.start_timer("Orthogonal.get_2d_rotation")
        # assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 1
        sin = torch.sin(params * 2 * math.pi)
        cos = torch.cos(params * 2 * math.pi)
        result = torch.cat([cos, -sin,
                          sin, cos], dim=1).view(-1, 2, 2)
        # global_tracker.end_timer("Orthogonal.get_2d_rotation")
        return result

    def get_3d_rotation(self, params):
        # global_tracker.start_timer("Orthogonal.get_3d_rotation")
        assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 3

        alpha = params[:, 0].view(-1, 1) * 2 * math.pi
        beta = params[:, 1].view(-1, 1) * 2 * math.pi
        gamma = params[:, 2].view(-1, 1) * 2 * math.pi

        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta),  torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)

        result = torch.cat(
            [cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g,
             sin_a*cos_b, sin_a*sin_b*sin_g + cos_a*cos_g, sin_a*sin_b*cos_g - cos_a*sin_g,
             -sin_b, cos_b*sin_g, cos_b*cos_g], dim=1).view(-1, 3, 3)
        # global_tracker.end_timer("Orthogonal.get_3d_rotation")
        return result

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # global_tracker.start_timer("Orthogonal.forward")
        
        if self.orthogonal_map != "euler":
            # global_tracker.start_timer("Orthogonal.forward.tril_indices")
            # Construct a lower diagonal matrix where to place the parameters.
            offset = -1 if self.orthogonal_map == 'householder' else 0
            tril_indices = torch.tril_indices(row=self.d, col=self.d, offset=offset, device=params.device)
            new_params = torch.zeros(
                (params.size(0), self.d, self.d), dtype=params.dtype, device=params.device)
            new_params[:, tril_indices[0], tril_indices[1]] = params
            params = new_params
            # global_tracker.end_timer("Orthogonal.forward.tril_indices")

        if self.orthogonal_map == "matrix_exp" or self.orthogonal_map == "cayley":
            # global_tracker.start_timer(f"Orthogonal.forward.{self.orthogonal_map}")
            # We just need n x k - k(k-1)/2 parameters
            params = params.tril()
            A = params - params.transpose(-2, -1)
            # A is skew-symmetric (or skew-hermitian)
            if self.orthogonal_map == "matrix_exp":
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == "cayley":
                # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
                Id = torch.eye(self.d, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
            # global_tracker.end_timer(f"Orthogonal.forward.{self.orthogonal_map}")
        elif self.orthogonal_map == 'householder':
            # global_tracker.start_timer("Orthogonal.forward.householder")
            # Only import torch_householder when actually needed
            try:
                from torch_householder import torch_householder_orgqr
                eye = torch.eye(self.d, device=params.device).unsqueeze(0).repeat(params.size(0), 1, 1)
                A = params.tril(diagonal=-1) + eye
                Q = torch_householder_orgqr(A)
            except ImportError:
                # Fallback implementation using PyTorch's built-in QR
                print("Warning: torch_householder not available, using fallback QR implementation")
                eye = torch.eye(self.d, device=params.device).unsqueeze(0).repeat(params.size(0), 1, 1)
                A = params.tril(diagonal=-1) + eye
                Q_list = []
                for i in range(A.size(0)):
                    q, _ = torch.linalg.qr(A[i], mode='complete')
                    Q_list.append(q)
                Q = torch.stack(Q_list, dim=0)
            # global_tracker.end_timer("Orthogonal.forward.householder")
        elif self.orthogonal_map == 'euler':
            # global_tracker.start_timer("Orthogonal.forward.euler")
            assert 2 <= self.d <= 3
            if self.d == 2:
                Q = self.get_2d_rotation(params)
            else:
                Q = self.get_3d_rotation(params)
            # global_tracker.end_timer("Orthogonal.forward.euler")
        else:
            raise ValueError(f"Unsupported transformations {self.orthogonal_map}")
        
        # global_tracker.end_timer("Orthogonal.forward")
        return Q
"""Define the hypergraph convolution neural network layer."""

import math

import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from topobench.transforms.liftings.graph2hypergraph.hypergraph_laplacian import (
    Laplacian,
)


class SparseMM(torch.autograd.Function):
    """Provide sparse times dense matrix multiplication with autograd support."""

    @staticmethod
    def forward(ctx, M1, M2):
        """Compute the forward pass for sparse matrix multiplication.

        Parameters
        ----------
        ctx : object
            The context object.
        M1 : torch.Tensor
            The sparse matrix.
        M2 : torch.Tensor
            The dense matrix.

        Returns
        -------
        torch.Tensor
            The resulting multiplied matrix.
        """
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        """Compute the backward pass for sparse matrix multiplication.

        Parameters
        ----------
        ctx : object
            The context object.
        g : torch.Tensor
            The gradient tensor.

        Returns
        -------
        tuple
            The gradients for M1 and M2.
        """
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


class HyperGraphConvolution(Module):
    """Define a simple GCN layer.

    Parameters
    ----------
    a : int
        The input feature dimension.
    b : int
        The output feature dimension.
    reapproximate : bool, optional
        Whether to reapproximate the Laplacian, by default True.
    cuda : int or None, optional
        The CUDA device index, by default None.
    **kwargs : dict, optional
        Required for TopoBench to do evaluation.
    """

    def __init__(self, a, b, reapproximate=True, cuda=None, **kwargs):
        super().__init__()
        self.a, self.b = a, b
        self.reapproximate = reapproximate
        self.device = torch.device(
            "cuda:" + str(cuda) if cuda is not None else "cpu"
        )

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer parameters."""
        std = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, H, structure, m=True):
        """Compute the forward pass of the HyperGraph Convolution layer.

        Parameters
        ----------
        H : torch.Tensor
            The hidden node features.
        structure : torch.Tensor or dict
            The structural matrix or hyperedge dictionary.
        m : bool, optional
            Whether to use mediators, by default True.

        Returns
        -------
        tuple
            A tuple containing the updated node features and None.
        """
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else:
            A = structure

        A = A.to(self.device)
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)
        return AHW + b, None

    def __repr__(self):
        """Return the string representation of the module.

        Returns
        -------
        str
            The module string representation.
        """
        return (
            self.__class__.__name__
            + " ("
            + str(self.a)
            + " -> "
            + str(self.b)
            + ")"
        )

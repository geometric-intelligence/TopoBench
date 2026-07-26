"""Implicit Hypergraph Neural Network (IHGNN) backbone."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Function
import numpy as np
import scipy.sparse as sp

def get_spectral_rad(incidence, tol=1e-5):
    """Compute spectral radius of the induced node adjacency (H @ H^T)"""
    if incidence.is_sparse:
        H = incidence.coalesce().cpu()
        H_scipy = sp.coo_matrix((H.values().numpy(), H.indices().numpy()), shape=H.shape)
    else:
        H_scipy = sp.coo_matrix(incidence.detach().cpu().numpy())
        
    A_scipy = H_scipy @ H_scipy.T
    
    if A_scipy.shape[0] <= 3:
        # Fallback per test molto piccoli
        evals = np.linalg.eigvals(A_scipy.todense())
        return np.max(np.abs(evals)) + tol
        
    try:
        from scipy.sparse.linalg import eigs
        evals = eigs(A_scipy.astype(float), k=1, return_eigenvectors=False)
        return np.abs(evals[0]) + tol
    except:
        return 1.0

def projection_norm_inf(A, kappa=0.99):
    """Project onto ||A||_inf <= kappa. Returns updated A."""
    v = kappa
    A_np = A.clone().detach().cpu().numpy()
    x = np.abs(A_np).sum(axis=-1)
    for idx in np.where(x > v)[0]:
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a = np.sort(a_abs)

        s = np.sum(a) - v
        l = float(len(a))
        for i in range(len(a)):
            if s / l > a[i]:
                s -= a[i]
                l -= 1
            else:
                break
        alpha = s / l
        a = a_sign * np.maximum(a_abs - alpha, 0)
        A_np[idx, :] = a
    A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))
    return A

class ImplicitFunction(Function):
    """
    Implicit Gradient Solver for the Fixed-Point Equation.
    Adapted from the original IGNN implementation.
    """
    @staticmethod
    def forward(ctx, W, X_0, incidence, B, phi, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = ImplicitFunction.inn_pred(W, X_0, incidence, B, phi, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(W, X, incidence, B, D, X_0, torch.tensor(bw_mitr))
        if status not in "converged":
            print(f"IHGNN solver warning: Iterations not converging! err={err:.4f}, status={status}")
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):
        W, X, incidence, B, D, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy().item()
        grad_x = grad_outputs[0]

        dphi = lambda X_hat: torch.mul(X_hat, D)
        
        # Backward fixed point iteration
        grad_z, _, _, _ = ImplicitFunction.inn_pred(W.T, X_0, incidence, grad_x, dphi, mitr=bw_mitr, transposed_A=True)
        
        if incidence.is_sparse:
            hyperedge_grad = torch.sparse.mm(incidence.t(), X.T)
            grad_W_term = torch.sparse.mm(incidence, hyperedge_grad).T
        else:
            hyperedge_grad = incidence.t() @ X.T
            grad_W_term = (incidence @ hyperedge_grad).T
            
        grad_W = grad_z @ grad_W_term.T
        grad_B = grad_z

        return grad_W, None, None, grad_B, None, None, None

    @staticmethod
    def inn_pred(W, X, incidence, B, phi, mitr=300, tol=3e-6, transposed_A=False, compute_dphi=False):
        """Fixed point iteration solver"""
        err = 0
        status = 'max itrs reached'
        
        for i in range(mitr):
            X_ = W @ X
            
            # Node -> Hyperedge -> Node propagation
            if incidence.is_sparse:
                hyperedge_feat = torch.sparse.mm(incidence.t(), X_.T)
                support = torch.sparse.mm(incidence, hyperedge_feat).T
            else:
                hyperedge_feat = incidence.t() @ X_.T
                support = (incidence @ hyperedge_feat).T
                
            X_new = phi(support + B)
            err = torch.norm(X_new - X, np.inf).item()
            if err < tol:
                status = 'converged'
                break
            X = X_new

        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                X_ = W @ X
                if incidence.is_sparse:
                    hyperedge_feat = torch.sparse.mm(incidence.t(), X_.T)
                    support = torch.sparse.mm(incidence, hyperedge_feat).T
                else:
                    hyperedge_feat = incidence.t() @ X_.T
                    support = (incidence @ hyperedge_feat).T
                    
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, dphi


class IHGNN(nn.Module):
    """
    Implicit Hypergraph Neural Network Backbone for TopoBench.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, kappa=0.99):
        super().__init__()
        self.p = in_channels
        self.m = hidden_channels
        self.k = kappa
        
        # Parameters
        self.W = Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = Parameter(torch.FloatTensor(self.m, 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, incidence):
        """
        x: Node features (num_nodes, in_channels)
        incidence: Node-hyperedge incidence matrix (num_nodes, num_hyperedges)
        """
        # Normalizzazione del raggio spettrale per garantire convergenza
        if getattr(self, 'adj_rho', None) is None:
            self.adj_rho = get_spectral_rad(incidence)
            
        if self.k is not None and self.k > 0:
            # Proiezione del tensore W per soddisfare il limite del Banach Fixed-Point Theorem
            self.W = projection_norm_inf(self.W, kappa=self.k / self.adj_rho)

        # Node -> Hyperedge -> Node propagation per B
        # x_transformed: (num_nodes, hidden_channels)
        x_transformed = x @ self.Omega_1.T
        
        if incidence.is_sparse:
            # 1) Nodi -> Iperfacce: H^T @ X
            hyperedge_features = torch.sparse.mm(incidence.t(), x_transformed)
            # 2) Iperfacce -> Nodi: H @ (H^T @ X)
            node_features_aggregated = torch.sparse.mm(incidence, hyperedge_features)
        else:
            hyperedge_features = incidence.t() @ x_transformed
            node_features_aggregated = incidence @ hyperedge_features
            
        B = node_features_aggregated.T + self.bias
        
        # Inizializzazione X_0 (zero)
        X_0 = torch.zeros_like(B)
        
        # Soluzione dell'equazione a punto fisso implicita
        out = ImplicitFunction.apply(self.W, X_0, incidence, B, F.relu)
        return out.T

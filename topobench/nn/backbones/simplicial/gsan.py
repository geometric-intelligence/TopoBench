import torch
import torch.nn as nn
import torch.nn.functional as F

class GSANLayer(nn.Module):
    """
    Single layer of the Generalized Simplicial Attention Network (GSAN).
    """
    def __init__(self, in_size, out_size, k=1, dropout=0.5, alpha_leaky_relu=0.2):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.dropout = dropout

        self.leaky_relu = nn.LeakyReLU(alpha_leaky_relu)
        self.W = nn.Parameter(torch.randn(k, in_size, out_size))
        self.A = nn.Parameter(torch.randn(k, in_size, out_size))
    
        # Attributes Z0
        self.att_l0_1 = nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l0_2 = nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

        # Attributes Z1
        self.att_l1_1 = nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l1_2 = nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l1_3 = nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

        # Attributes Z2
        self.att_l2_1 = nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))
        self.att_l2_2 = nn.Parameter(torch.randn(size=(2*out_size*self.k, 1)))

        # Initialize weights
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.A)

    def E_f(self, X, W, K, L, attr, dropout, b=None, t=False):
        if b is not None and t:
            X = b.T @ X
        if b is not None and not t:
            X = b @ X

        X_f = torch.cat([X @ W[k,:,:] for k in range(K)], dim=1)
        L = L.to_dense() if L.is_sparse else L

        # Broadcast add
        E = self.leaky_relu((X_f @ attr[:self.out_size*K, :]) + (X_f @ attr[self.out_size*K:, :]).T) 
        
        zero_vec = -9e15 * torch.ones_like(E)
        E = torch.where(L != 0, E, zero_vec)

        L_f = F.dropout(F.softmax(E, dim=1), dropout, training=self.training)
        return L_f

    def compute_z0(self, z0, z1, b1, l0_sparse_1, l0_sparse_2):
        first_term = l0_sparse_1 @ z0 @ self.W[0,:,:]
        second_term = ((b1.T @ l0_sparse_2.T).T) @ z1 @ self.A[0,:,:]

        for j in range(1, self.k):
            l0_sparse_1_j = torch.linalg.matrix_power(l0_sparse_1, j+1)
            l0_sparse_2_j = torch.linalg.matrix_power(l0_sparse_2, j+1)
            first_term += l0_sparse_1_j @ z0 @ self.W[j,:,:]
            second_term += ((b1.T @ l0_sparse_2_j.T).T) @ z1 @ self.A[j,:,:]
            
        return torch.sigmoid(first_term + second_term)

    def compute_z1(self, z0, z1, z2, b1, b2, l1_sparse_1, l1_sparse_2, l1_sparse_3):
        first_term = l1_sparse_1 @ z1 @ self.W[0,:,:]
        second_term = ((b1 @ l1_sparse_2.T).T) @ z0 @ self.A[0,:,:]
        third_term = ((b2.T @ l1_sparse_3.T).T) @ z2 @ self.A[0,:,:]
        
        if torch.isnan(third_term).any():
            third_term = torch.zeros_like(third_term)

        for j in range(1, self.k):
            l1_sparse_1_j = torch.linalg.matrix_power(l1_sparse_1, j+1)
            l1_sparse_2_j = torch.linalg.matrix_power(l1_sparse_2, j+1)
            l1_sparse_3_j = torch.linalg.matrix_power(l1_sparse_3, j+1)
            
            first_term += l1_sparse_1_j @ z1 @ self.W[j,:,:]
            second_term += ((b1 @ l1_sparse_2_j.T).T) @ z0 @ self.A[j,:,:]
            third_term_tmp = ((b2.T @ l1_sparse_3_j.T).T) @ z2 @ self.A[j,:,:]
            if not torch.isnan(third_term_tmp).any():
                third_term += third_term_tmp

        return torch.sigmoid(first_term + second_term + third_term)

    def compute_z2(self, z1, z2, b2, l2_sparse_1, l2_sparse_2):
        first_term = l2_sparse_1 @ z2 @ self.W[0,:,:]
        if torch.isnan(first_term).any():
            first_term = torch.zeros_like(first_term)
            
        second_term = ((b2 @ l2_sparse_2.T).T) @ z1 @ self.A[0,:,:]
        if torch.isnan(second_term).any():
            second_term = torch.zeros_like(second_term)

        for j in range(1, self.k):
            l2_sparse_1_j = torch.linalg.matrix_power(l2_sparse_1, j+1)
            l2_sparse_2_j = torch.linalg.matrix_power(l2_sparse_2, j+1)
            
            ft_tmp = l2_sparse_1_j @ z2 @ self.W[j,:,:]
            if not torch.isnan(ft_tmp).any():
                first_term += ft_tmp
                
            st_tmp = ((b2 @ l2_sparse_2_j.T).T) @ z1 @ self.A[j,:,:]
            if not torch.isnan(st_tmp).any():
                second_term += st_tmp

        return torch.sigmoid(first_term + second_term)


    def forward(self, z0, z1, z2, b1_sparse, b2_sparse):
        l0_sparse = b1_sparse @ b1_sparse.t()
        l1_d_sparse = b1_sparse.t() @ b1_sparse
        l1_u_sparse = b2_sparse @ b2_sparse.t()
        l1_sparse = l1_d_sparse + l1_u_sparse
        l2_sparse = b2_sparse.t() @ b2_sparse

        # Z0 Attention
        l0_sparse_1 = self.E_f(z0, self.W, self.k, l0_sparse, self.att_l0_1, self.dropout)
        l0_sparse_2 = self.E_f(z0, self.A, self.k, l0_sparse, self.att_l0_2, self.dropout)

        # Z1 Attention
        l1_sparse_1 = self.E_f(z1, self.W, self.k, l1_sparse, self.att_l1_1, self.dropout)
        l1_sparse_2 = self.E_f(z0, self.A, self.k, l1_d_sparse, self.att_l1_2, self.dropout, b1_sparse, t=True)
        l1_sparse_3 = self.E_f(z2, self.A, self.k, l1_u_sparse, self.att_l1_3, self.dropout, b2_sparse)

        # Z2 Attention
        l2_sparse_1 = self.E_f(z2, self.W, self.k, l2_sparse, self.att_l2_1, self.dropout)
        l2_sparse_2 = self.E_f(z1, self.A, self.k, l2_sparse, self.att_l2_2, self.dropout, b2_sparse, t=True)

        z0_prime = self.compute_z0(z0, z1, b1_sparse, l0_sparse_1, l0_sparse_2)
        z1_prime = self.compute_z1(z0, z1, z2, b1_sparse, b2_sparse, l1_sparse_1, l1_sparse_2, l1_sparse_3)
        z2_prime = self.compute_z2(z1, z2, b2_sparse, l2_sparse_1, l2_sparse_2)

        return z0_prime, z1_prime, z2_prime


class GSAN(nn.Module):
    """
    Generalized Simplicial Attention Network (GSAN) Backbone.
    """
    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        k_len=1,
        dropout=0.5,
        alpha_leaky_relu=0.2,
        n_layers=2
    ):
        super().__init__()
        
        self.in_linear_0 = nn.Linear(in_channels_all[0], hidden_channels_all[0])
        self.in_linear_1 = nn.Linear(in_channels_all[1], hidden_channels_all[1])
        
        # In case we have 2-simplices (triangles) features
        if len(in_channels_all) > 2:
            self.in_linear_2 = nn.Linear(in_channels_all[2], hidden_channels_all[2])
        else:
            # Fallback if no 2-simplices features are present natively
            self.in_linear_2 = nn.Linear(in_channels_all[1], hidden_channels_all[1])

        self.layers = nn.ModuleList([
            GSANLayer(
                in_size=hidden_channels_all[0],
                out_size=hidden_channels_all[0],
                k=k_len,
                dropout=dropout,
                alpha_leaky_relu=alpha_leaky_relu
            ) for _ in range(n_layers)
        ])

    def forward(self, x_all, incidence_all):
        """
        x_all: tuple of (node_features, edge_features, face_features)
        incidence_all: tuple of (b1_matrix, b2_matrix)
        """
        x_0, x_1, x_2 = x_all
        b1, b2 = incidence_all
        
        x_0 = self.in_linear_0(x_0)
        x_1 = self.in_linear_1(x_1)
        x_2 = self.in_linear_2(x_2)

        # Make b1 and b2 dense if they are not, or keep them sparse depending on E_f
        # The layer currently expects torch.sparse_coo_tensor or dense.
        if not b1.is_sparse:
            b1 = b1.to_sparse()
        if not b2.is_sparse:
            b2 = b2.to_sparse()

        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, b1, b2)

        return x_0, x_1, x_2

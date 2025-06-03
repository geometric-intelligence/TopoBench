import torch.nn as nn
import torch
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
import pytorch_lightning as pl
from topobench.nn.backbones.graph.gatedgcn_layer import GatedGCNLayer
from topobench.nn.backbones.graph.gine_conv_layer import GINEConvESLapPE


class GraphGPS(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        local_gnn_type: str = "GINE",
        global_model_type: str = "Transformer",
        num_heads: int = 8,
        act: str = "relu",
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = True,
        equivstable_pe: bool = True,
    ):
        super().__init__()

        self.dim_h = hidden_channels
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = nn.ReLU if act == "relu" else nn.SiLU
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(1, hidden_channels)
        self.edge_dim = in_channels if local_gnn_type == "GINE" else 0

        # ---- Local GNN ----------------------------------------------------
        if local_gnn_type == "GINE":
            gin_nn = nn.Sequential(
                Linear_pyg(hidden_channels, hidden_channels),
                self.activation(),
                Linear_pyg(hidden_channels, hidden_channels),
            )
            if equivstable_pe:
                self.local_model = GINEConvESLapPE(
                    gin_nn, edge_dim=self.edge_dim
                )
            else:
                self.local_model = pygnn.GINEConv(gin_nn, edge_dim=64)
        elif local_gnn_type == "CustomGatedGCN":
            self.local_model = GatedGCNLayer(
                hidden_channels,
                hidden_channels,
                dropout=dropout,
                residual=True,
                act=act,
                equivstable_pe=equivstable_pe,
            )
        else:
            raise ValueError(
                f"Unsupported local GNN: {local_gnn_type}. "
                "Choose 'GINE' or 'CustomGatedGCN'."
            )

        self.local_gnn_type = local_gnn_type
        self.dropout_local = nn.Dropout(dropout)

        # ---- Global Transformer block ------------------------------------
        if global_model_type == "Transformer":
            self.self_attn = nn.MultiheadAttention(
                hidden_channels,
                num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
        else:
            raise ValueError(
                f"Unsupported global model: {global_model_type}. "
                "Choose 'Transformer'."
            )
        self.dropout_attn = nn.Dropout(dropout)

        # ---- Normalisation layers ----------------------------------------
        if layer_norm and batch_norm:
            raise ValueError("Cannot use both layer_norm and batch_norm")
        Norm = pygnn.norm.LayerNorm if layer_norm else nn.BatchNorm1d
        self.norm1_local = Norm(hidden_channels)
        self.norm1_attn = Norm(hidden_channels)
        self.norm2 = Norm(hidden_channels)

        # ---- Feed-forward block ------------------------------------------
        self.ff_linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.ff_linear2 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.ff_activation = self.activation()
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = self.input_proj(batch.x)  # project to hidden_channels
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # Make it [num_edges, 1]
        if edge_attr is None:
            edge_attr = torch.zeros(
                (edge_index.size(1), 1), device=h.device, dtype=h.dtype
            )
        edge_attr = self.edge_encoder(edge_attr.to(dtype=h.dtype))  # [E, 128]

        batch_idx = batch.batch
        y = batch.y if hasattr(batch, "y") else None
        pe = batch.pe_EquivStableLapPE if self.equivstable_pe else None

        x_dis = h @ h.T if self.training else None

        h_in1 = h.clone()
        outputs = []

        if self.local_gnn_type == "CustomGatedGCN":
            local_out = self.local_model(batch).x
        else:
            if edge_attr is None:
                edge_attr = torch.zeros(
                    (edge_index.size(1), self.edge_dim),
                    device=h.device,
                    dtype=h.dtype,
                )
            else:
                edge_attr = edge_attr.to(dtype=h.dtype)
            local_out = (
                self.local_model(h, edge_index, edge_attr, pe)
                if hasattr(self.local_model, "equivstable_pe")
                else self.local_model(h, edge_index, edge_attr)
            )

        local = self.dropout_local(local_out)
        local = h + local
        local = (
            self.norm1_local(local, batch_idx)
            if hasattr(self.norm1_local, "normalized_shape")
            else self.norm1_local(local)
        )
        outputs.append(local)

        h = local
        h_dense, mask = to_dense_batch(h, batch_idx)
        attn_out = self.self_attn(
            h_dense, h_dense, h_dense, key_padding_mask=~mask
        )[0]
        attn = attn_out[mask]
        attn = self.dropout_attn(attn)
        attn = h + attn
        attn = (
            self.norm1_attn(attn, batch_idx)
            if hasattr(self.norm1_attn, "normalized_shape")
            else self.norm1_attn(attn)
        )
        outputs.append(attn)

        h = sum(outputs)
        ff = self.ff_dropout1(self.ff_activation(self.ff_linear1(h)))
        ff = self.ff_dropout2(self.ff_linear2(ff))
        h = h + ff
        h = (
            self.norm2(h, batch_idx)
            if hasattr(self.norm2, "normalized_shape")
            else self.norm2(h)
        )

        return h, x_dis

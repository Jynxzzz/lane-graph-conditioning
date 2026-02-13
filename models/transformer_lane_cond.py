"""Lane-Conditioned Transformer for trajectory prediction.

Key upgrade over LSTM version: multi-head cross-attention between the FULL
ego trajectory sequence (B, 11, 128) and lane embeddings (B, 16, 128).
This allows per-timestep lane conditioning instead of single-vector querying.
"""

import torch
import torch.nn as nn

from models.lstm_baseline import MLPDecoder, NeighborEncoder
from models.transformer_baseline import TransformerEgoEncoder


class TransformerLaneCond(nn.Module):
    """Transformer trajectory prediction conditioned on local lane graph.

    Architecture:
        TransformerEgoEncoder → ego_seq (B, 11, d_model)
        LaneMLP + GNN message passing → lane_emb (B, 16, d_model)
        MultiheadCrossAttention(ego_seq, lane_emb) → ego_lane_seq (B, 11, d_model)
        mean_pool → ego_repr (B, d_model)
        NeighborEncoder + Fusion → (B, d_model)
        MLPDecoder + CV residual → pred_future (B, 30, 2)
    """

    def __init__(
        self,
        input_dim=2,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        future_len=30,
        use_neighbors=True,
        neighbor_hidden_dim=64,
        lane_feat_dim=26,
        lane_hidden_dim=128,
        n_mp_layers=2,
        **kwargs,
    ):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.d_model = d_model
        self.future_len = future_len

        # Ego trajectory encoder (shared with baseline)
        self.ego_encoder = TransformerEgoEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_len=11,
        )

        # Lane encoding MLP (project to d_model for cross-attention compatibility)
        self.lane_mlp = nn.Sequential(
            nn.Linear(lane_feat_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Graph message passing (same logic as LaneEncoder in lane_conditioned_lstm.py)
        self.mp_layers = nn.ModuleList()
        for _ in range(n_mp_layers):
            self.mp_layers.append(nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
            ))

        # Multi-head cross-attention: ego sequence queries lane embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)

        if use_neighbors:
            self.neighbor_encoder = NeighborEncoder(
                input_dim=input_dim,
                embed_dim=32,
                hidden_dim=neighbor_hidden_dim,
            )
            self.fusion = nn.Sequential(
                nn.Linear(d_model + neighbor_hidden_dim, d_model),
                nn.ReLU(),
            )

        self.decoder = MLPDecoder(
            input_dim=input_dim,
            hidden_dim=d_model,
            future_len=future_len,
        )

    def _encode(self, batch):
        """Encode ego trajectory with lane conditioning.

        Returns:
            ego_repr: (B, d_model) fused representation for decoding
        """
        sdc_history = batch["sdc_history"]  # (B, 11, 2)

        # Ego encoding → full sequence
        ego_seq = self.ego_encoder(sdc_history)  # (B, 11, d_model)

        # Lane encoding
        lane_features = batch["lane_features"]  # (B, 16, 26)
        lane_mask = batch["lane_mask"]           # (B, 16)
        lane_adj = batch.get("lane_adj")         # (B, 16, 16) or None

        lane_emb = self.lane_mlp(lane_features)  # (B, 16, d_model)
        mask_3d = lane_mask.unsqueeze(-1)         # (B, 16, 1)
        lane_emb = lane_emb * mask_3d

        # Graph message passing over lane adjacency
        if lane_adj is not None and len(self.mp_layers) > 0:
            for mp_layer in self.mp_layers:
                adj = lane_adj * lane_mask.unsqueeze(1)  # (B, 16, 16)
                adj_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1)
                adj_norm = adj / adj_sum
                neighbor_agg = torch.bmm(adj_norm, lane_emb)  # (B, 16, d_model)
                combined = torch.cat([lane_emb, neighbor_agg], dim=-1)
                lane_emb = mp_layer(combined)  # (B, 16, d_model)
                lane_emb = lane_emb * mask_3d

        # Multi-head cross-attention: ego queries lanes
        # key_padding_mask: True = ignore (inverted from lane_mask convention)
        key_padding_mask = (lane_mask < 0.5)  # (B, 16)

        cross_out, _ = self.cross_attn(
            query=ego_seq,       # (B, 11, d_model)
            key=lane_emb,        # (B, 16, d_model)
            value=lane_emb,      # (B, 16, d_model)
            key_padding_mask=key_padding_mask,
        )  # (B, 11, d_model)

        # Residual + LayerNorm
        ego_lane_seq = self.cross_norm(ego_seq + cross_out)  # (B, 11, d_model)

        # Safety: when ALL lanes are invalid, cross_out may be NaN
        all_invalid = (lane_mask.sum(dim=-1) == 0)  # (B,)
        if all_invalid.any():
            ego_lane_seq[all_invalid] = ego_seq[all_invalid]

        # Mean pool over time
        ego_repr = ego_lane_seq.mean(dim=1)  # (B, d_model)

        # Neighbor fusion
        if self.use_neighbors:
            neighbor_ctx = self.neighbor_encoder(
                batch["neighbor_history"], batch["neighbor_mask"]
            )
            ego_repr = self.fusion(
                torch.cat([ego_repr, neighbor_ctx], dim=-1)
            )

        return ego_repr

    def forward(self, batch):
        ego_repr = self._encode(batch)

        # Decode with CV residual
        sdc_history = batch["sdc_history"]
        h_n = ego_repr.unsqueeze(0)
        last_pos = sdc_history[:, -1, :]
        last_vel = sdc_history[:, -1, :] - sdc_history[:, -2, :]
        pred_future = self.decoder(h_n, None, last_pos, last_vel)

        return {"pred_future": pred_future}

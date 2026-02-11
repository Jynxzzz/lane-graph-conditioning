"""Lane-Conditioned LSTM for trajectory prediction.

Extends the LSTM baseline by conditioning on local lane graph features
extracted from the waterflow graph (3-hop BFS from ego lane).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm_baseline import NeighborEncoder, TrajectoryDecoder, TrajectoryEncoder


class LaneEncoder(nn.Module):
    """Encode lane features with per-lane MLP, optional graph message passing,
    and cross-attention pooling.

    When n_mp_layers > 0 and lane_adj is provided, connected lanes exchange
    information before cross-attention, encoding structural connectivity.
    """

    def __init__(self, lane_feat_dim=26, lane_hidden_dim=64, query_dim=128,
                 n_mp_layers=0):
        super().__init__()
        self.lane_mlp = nn.Sequential(
            nn.Linear(lane_feat_dim, lane_hidden_dim),
            nn.ReLU(),
            nn.Linear(lane_hidden_dim, lane_hidden_dim),
        )

        # Graph message passing layers (propagate connectivity structure)
        self.mp_layers = nn.ModuleList()
        for _ in range(n_mp_layers):
            self.mp_layers.append(nn.Sequential(
                nn.Linear(lane_hidden_dim * 2, lane_hidden_dim),
                nn.ReLU(),
            ))

        # Cross-attention: ego hidden state queries lane features
        self.query_proj = nn.Linear(query_dim, lane_hidden_dim)
        self.key_proj = nn.Linear(lane_hidden_dim, lane_hidden_dim)
        self.value_proj = nn.Linear(lane_hidden_dim, lane_hidden_dim)
        self.scale = lane_hidden_dim ** -0.5

    def forward(self, lane_features, lane_mask, ego_hidden, lane_adj=None):
        """
        Args:
            lane_features: (B, N_lanes, feat_dim) lane feature vectors
            lane_mask: (B, N_lanes) float, 1.0 for valid lanes
            ego_hidden: (B, query_dim) ego trajectory hidden state
            lane_adj: (B, N_lanes, N_lanes) float, adjacency matrix (optional)

        Returns:
            lane_context: (B, lane_hidden_dim) attended lane representation
        """
        # Per-lane MLP encoding
        lane_emb = self.lane_mlp(lane_features)  # (B, N, lane_hidden)

        # Mask invalid lanes
        mask = lane_mask.unsqueeze(-1)  # (B, N, 1)
        lane_emb = lane_emb * mask

        # Graph message passing using adjacency matrix
        if lane_adj is not None and len(self.mp_layers) > 0:
            for mp_layer in self.mp_layers:
                # Mask source lanes in adjacency
                adj = lane_adj * lane_mask.unsqueeze(1)  # (B, N, N)
                # Mean-pool over connected neighbors
                adj_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1)
                adj_norm = adj / adj_sum
                neighbor_agg = torch.bmm(adj_norm, lane_emb)  # (B, N, H)
                # Combine self + neighbor features
                combined = torch.cat([lane_emb, neighbor_agg], dim=-1)
                lane_emb = mp_layer(combined)  # (B, N, H)
                lane_emb = lane_emb * mask  # re-mask invalid

        # Cross-attention
        Q = self.query_proj(ego_hidden).unsqueeze(1)  # (B, 1, lane_hidden)
        K = self.key_proj(lane_emb)                     # (B, N, lane_hidden)
        V = self.value_proj(lane_emb)                   # (B, N, lane_hidden)

        attn_scores = (Q * K).sum(dim=-1) * self.scale  # (B, N)

        # Mask: set invalid lanes to -inf
        attn_scores = attn_scores.masked_fill(lane_mask < 0.5, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N)

        # Handle case where all lanes are invalid
        all_invalid = (lane_mask.sum(dim=-1, keepdim=True) == 0).float()
        attn_weights = attn_weights * (1 - all_invalid)

        lane_context = (attn_weights.unsqueeze(-1) * V).sum(dim=1)  # (B, lane_hidden)
        return lane_context


class LaneConditionedLSTM(nn.Module):
    """LSTM trajectory prediction conditioned on local lane graph.

    Architecture:
        TrajectoryEncoder → ego_hidden
        NeighborEncoder → neighbor_ctx (optional)
        LaneEncoder(ego_hidden, lane_features) → lane_ctx
        Fusion([ego_hidden, neighbor_ctx, lane_ctx]) → decoder_hidden
        TrajectoryDecoder → predicted future
    """

    def __init__(
        self,
        input_dim=2,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        future_len=30,
        use_neighbors=True,
        neighbor_hidden_dim=64,
        lane_feat_dim=26,
        lane_hidden_dim=64,
        decoder_type="mlp",
        n_mp_layers=0,
    ):
        super().__init__()
        self.use_neighbors = use_neighbors

        self.ego_encoder = TrajectoryEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        fusion_input_dim = hidden_dim + lane_hidden_dim
        if use_neighbors:
            self.neighbor_encoder = NeighborEncoder(
                input_dim=input_dim,
                embed_dim=32,
                hidden_dim=neighbor_hidden_dim,
            )
            fusion_input_dim += neighbor_hidden_dim

        self.lane_encoder = LaneEncoder(
            lane_feat_dim=lane_feat_dim,
            lane_hidden_dim=lane_hidden_dim,
            query_dim=hidden_dim,
            n_mp_layers=n_mp_layers,
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
        )

        if decoder_type == "mlp":
            from models.lstm_baseline import MLPDecoder
            self.decoder = MLPDecoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                future_len=future_len,
            )
        else:
            self.decoder = TrajectoryDecoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                future_len=future_len,
            )

    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                sdc_history: (B, T_hist, 2)
                neighbor_history: (B, N, T_hist, 2)
                neighbor_mask: (B, N)
                lane_features: (B, N_lanes, feat_dim)
                lane_mask: (B, N_lanes)

        Returns:
            dict with "pred_future": (B, future_len, 2)
        """
        sdc_history = batch["sdc_history"]
        h_n, c_n = self.ego_encoder(sdc_history)

        ego_hidden = h_n[-1]  # (B, hidden_dim) — top-layer hidden state

        # Lane encoding with optional graph message passing + cross-attention
        lane_context = self.lane_encoder(
            batch["lane_features"],
            batch["lane_mask"],
            ego_hidden,
            lane_adj=batch.get("lane_adj"),
        )  # (B, lane_hidden_dim)

        # Build fusion inputs
        fusion_parts = [ego_hidden, lane_context]

        if self.use_neighbors:
            neighbor_ctx = self.neighbor_encoder(
                batch["neighbor_history"], batch["neighbor_mask"]
            )
            fusion_parts.append(neighbor_ctx)

        fused = self.fusion(torch.cat(fusion_parts, dim=-1))  # (B, hidden_dim)

        # Replace top-layer hidden with fused representation
        h_n = h_n.clone()
        h_n[-1] = fused

        last_pos = sdc_history[:, -1, :]
        last_vel = sdc_history[:, -1, :] - sdc_history[:, -2, :]
        pred_future = self.decoder(h_n, c_n, last_pos, last_vel)

        return {"pred_future": pred_future}

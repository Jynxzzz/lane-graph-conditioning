"""Multi-modal Lane-Conditioned LSTM for trajectory prediction.

Combines lane graph conditioning with multi-modal (K=6) output heads.
Key comparison model: does lane conditioning also help multi-modal prediction?
"""

import torch
import torch.nn as nn

from .lstm_baseline import TrajectoryEncoder, NeighborEncoder
from .lane_conditioned_lstm import LaneEncoder
from .multimodal_lstm import MultiModalMLPDecoder


class MultiModalLaneCond(nn.Module):
    """Multi-modal LSTM with lane graph conditioning.

    Same architecture as LaneConditionedLSTM but with K prediction heads.
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
        n_mp_layers=0,
        num_modes=6,
        **kwargs
    ):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.num_modes = num_modes

        # Ego encoder
        self.ego_encoder = TrajectoryEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Lane encoder (reuse from single-modal)
        fusion_input_dim = hidden_dim + lane_hidden_dim
        self.lane_encoder = LaneEncoder(
            lane_feat_dim=lane_feat_dim,
            lane_hidden_dim=lane_hidden_dim,
            query_dim=hidden_dim,
            n_mp_layers=n_mp_layers,
        )

        # Neighbor encoder
        if use_neighbors:
            self.neighbor_encoder = NeighborEncoder(
                input_dim=input_dim,
                embed_dim=32,
                hidden_dim=neighbor_hidden_dim,
            )
            fusion_input_dim += neighbor_hidden_dim

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Multi-modal decoder (K heads)
        self.decoder = MultiModalMLPDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            future_len=future_len,
            num_modes=num_modes,
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
                lane_adj: (B, N_lanes, N_lanes) optional

        Returns:
            dict with:
                - "pred_future": (B, K, T_fut, 2)
                - "confidences": (B, K)
        """
        sdc_history = batch["sdc_history"]
        h_n, c_n = self.ego_encoder(sdc_history)

        ego_hidden = h_n[-1]  # (B, hidden_dim)

        # Lane encoding with cross-attention
        lane_context = self.lane_encoder(
            batch["lane_features"],
            batch["lane_mask"],
            ego_hidden,
            lane_adj=batch.get("lane_adj"),
        )

        # Build fusion inputs
        fusion_parts = [ego_hidden, lane_context]

        if self.use_neighbors:
            neighbor_ctx = self.neighbor_encoder(
                batch["neighbor_history"], batch["neighbor_mask"]
            )
            fusion_parts.append(neighbor_ctx)

        fused = self.fusion(torch.cat(fusion_parts, dim=-1))

        # Replace top-layer hidden with fused representation
        h_n = h_n.clone()
        h_n[-1] = fused

        # Get last position and velocity
        last_pos = sdc_history[:, -1, :]
        last_vel = sdc_history[:, -1, :] - sdc_history[:, -2, :]

        # Decode K predictions
        predictions, confidences = self.decoder(h_n, c_n, last_pos, last_vel)

        return {
            "pred_future": predictions,  # (B, K, T, 2)
            "confidences": confidences,  # (B, K)
        }

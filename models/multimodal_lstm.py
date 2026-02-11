"""Multi-modal LSTM models for trajectory prediction.

Extends baseline LSTM to output K diverse predictions and use minADE metric.
"""

import torch
import torch.nn as nn
from .lstm_baseline import TrajectoryEncoder, NeighborEncoder


class MultiModalMLPDecoder(nn.Module):
    """Multi-modal MLP decoder that outputs K predictions.

    Uses CV-residual formulation with K prediction heads.
    """

    def __init__(self, input_dim=2, hidden_dim=128, future_len=30, num_modes=6):
        super().__init__()
        self.future_len = future_len
        self.input_dim = input_dim
        self.num_modes = num_modes

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
        )

        # K prediction heads (one per mode)
        self.mode_heads = nn.ModuleList([
            nn.Linear(hidden_dim * 2, future_len * input_dim)
            for _ in range(num_modes)
        ])

        # Mode confidence scores (optional, for weighted minADE)
        self.mode_probs = nn.Linear(hidden_dim * 2, num_modes)

    def forward(self, h_n, c_n, last_pos, last_vel=None):
        """
        Args:
            h_n: (num_layers, B, hidden_dim)
            c_n: (num_layers, B, hidden_dim)
            last_pos: (B, 2)
            last_vel: (B, 2) optional

        Returns:
            predictions: (B, K, T, 2) - K trajectory predictions
            confidences: (B, K) - confidence scores per mode
        """
        B = last_pos.shape[0]
        hidden = h_n[-1]  # (B, hidden_dim)

        # Shared features
        shared_feat = self.shared(hidden)  # (B, hidden_dim * 2)

        # Generate K predictions
        predictions = []
        for head in self.mode_heads:
            residuals = head(shared_feat)  # (B, T*2)
            residuals = residuals.view(B, self.future_len, self.input_dim)

            # CV baseline
            if last_vel is not None:
                t = torch.arange(1, self.future_len + 1, device=last_pos.device).float()
                cv = last_pos.unsqueeze(1) + last_vel.unsqueeze(1) * t.unsqueeze(-1)
            else:
                cv = last_pos.unsqueeze(1).expand(B, self.future_len, self.input_dim)

            pred = cv + residuals  # (B, T, 2)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)  # (B, K, T, 2)

        # Mode confidences
        confidences = torch.softmax(self.mode_probs(shared_feat), dim=-1)  # (B, K)

        return predictions, confidences


class MultiModalLSTMBaseline(nn.Module):
    """Multi-modal LSTM for trajectory prediction with K diverse outputs."""

    def __init__(
        self,
        input_dim=2,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        future_len=30,
        use_neighbors=True,
        neighbor_hidden_dim=64,
        num_modes=6,
        **kwargs  # Ignore extra arguments (e.g., decoder_type)
    ):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.num_modes = num_modes

        # Encoder (same as baseline)
        self.ego_encoder = TrajectoryEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        if use_neighbors:
            self.neighbor_encoder = NeighborEncoder(
                input_dim=input_dim,
                embed_dim=32,
                hidden_dim=neighbor_hidden_dim,
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim + neighbor_hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        # Multi-modal decoder
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

        Returns:
            dict with:
                - "pred_future": (B, K, T_fut, 2)
                - "confidences": (B, K)
        """
        sdc_history = batch["sdc_history"]

        # Encode ego
        h_n, c_n = self.ego_encoder(sdc_history)

        # Encode neighbors and fuse
        if self.use_neighbors:
            neighbor_ctx = self.neighbor_encoder(
                batch["neighbor_history"], batch["neighbor_mask"]
            )
            fused = self.fusion(
                torch.cat([h_n[-1], neighbor_ctx], dim=-1)
            )
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


def create_multimodal_lstm(
    model_name="multimodal_lstm_baseline",
    input_dim=2,
    embed_dim=64,
    hidden_dim=128,
    num_layers=2,
    future_len=30,
    use_neighbors=True,
    neighbor_hidden_dim=64,
    num_modes=6,
    **kwargs
):
    """Factory function to create multi-modal LSTM models."""

    if model_name == "multimodal_lstm_baseline":
        return MultiModalLSTMBaseline(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            future_len=future_len,
            use_neighbors=use_neighbors,
            neighbor_hidden_dim=neighbor_hidden_dim,
            num_modes=num_modes,
        )
    else:
        raise ValueError(f"Unknown multi-modal model: {model_name}")

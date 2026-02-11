"""Transformer baseline for trajectory prediction.

Replaces the LSTM ego encoder with a Transformer encoder that produces
a full sequence representation (B, 11, 128) instead of a single vector.
Reuses NeighborEncoder and MLPDecoder from lstm_baseline.
"""

import torch
import torch.nn as nn

from models.lstm_baseline import MLPDecoder, NeighborEncoder


class TransformerEgoEncoder(nn.Module):
    """Encode history trajectory using Transformer self-attention.

    Unlike the LSTM encoder which compresses to a single hidden vector,
    this returns the full sequence (B, T, d_model), enabling per-timestep
    cross-attention with lane features.
    """

    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, seq_len=11):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, history):
        """
        Args:
            history: (B, T, 2) trajectory positions

        Returns:
            ego_seq: (B, T, d_model) full sequence representation
        """
        x = self.input_proj(history)  # (B, T, d_model)
        x = x + self.pos_embed[:, :history.shape[1], :]
        return self.encoder(x)  # (B, T, d_model)


class TransformerBaseline(nn.Module):
    """Transformer encoder + MLP decoder for trajectory prediction.

    Architecture:
        TransformerEgoEncoder → ego_seq (B, 11, d_model) → mean_pool → (B, d_model)
        NeighborEncoder → neighbor_ctx (B, 64)
        Fusion → (B, d_model)
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
        **kwargs,  # absorb LSTM-specific params (embed_dim, hidden_dim, decoder_type)
    ):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.d_model = d_model

        self.ego_encoder = TransformerEgoEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_len=11,
        )

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

    def forward(self, batch):
        sdc_history = batch["sdc_history"]  # (B, 11, 2)

        # Transformer encoding → full sequence
        ego_seq = self.ego_encoder(sdc_history)  # (B, 11, d_model)
        ego_repr = ego_seq.mean(dim=1)  # (B, d_model)

        # Neighbor encoding + fusion
        if self.use_neighbors:
            neighbor_ctx = self.neighbor_encoder(
                batch["neighbor_history"], batch["neighbor_mask"]
            )
            ego_repr = self.fusion(
                torch.cat([ego_repr, neighbor_ctx], dim=-1)
            )

        # Decode with CV residual
        # MLPDecoder expects h_n with shape (num_layers, B, hidden_dim)
        # and does h_n[-1] to get (B, hidden_dim)
        h_n = ego_repr.unsqueeze(0)  # (1, B, d_model)
        last_pos = sdc_history[:, -1, :]
        last_vel = sdc_history[:, -1, :] - sdc_history[:, -2, :]
        pred_future = self.decoder(h_n, None, last_pos, last_vel)

        return {"pred_future": pred_future}

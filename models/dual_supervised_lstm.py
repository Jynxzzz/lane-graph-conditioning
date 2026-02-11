"""Dual-Supervised LSTM for trajectory prediction.

Extends LaneConditionedLSTM with a structure prediction head
that predicts which lane each future timestep belongs to.
Trained with dual loss: trajectory regression + structure classification.
"""

import torch
import torch.nn as nn

from models.lane_conditioned_lstm import LaneConditionedLSTM
from models.lstm_baseline import MLPDecoder


class DualSupervisedLSTM(LaneConditionedLSTM):
    """Lane-conditioned LSTM with structure prediction head for dual supervision.

    Adds a classification head that predicts which lane segment the ego
    vehicle is on at each future timestep.

    With MLP decoder: structure head maps fused hidden → per-step lane logits.
    With LSTM decoder: structure head maps per-step LSTM hidden states → lane logits.
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
        max_lanes=16,
        decoder_type="mlp",
        n_mp_layers=0,
    ):
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            future_len=future_len,
            use_neighbors=use_neighbors,
            neighbor_hidden_dim=neighbor_hidden_dim,
            lane_feat_dim=lane_feat_dim,
            lane_hidden_dim=lane_hidden_dim,
            decoder_type=decoder_type,
            n_mp_layers=n_mp_layers,
        )
        self.max_lanes = max_lanes
        self.future_len = future_len
        self._decoder_type = decoder_type

        if decoder_type == "mlp":
            # MLP structure head: fused hidden → per-step lane logits
            self.structure_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, future_len * max_lanes),
            )
        else:
            # LSTM structure head: per-step hidden → lane logits
            self.structure_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, max_lanes),
            )

    def forward(self, batch):
        """
        Returns dict with:
            pred_future: (B, future_len, 2)
            lane_logits: (B, future_len, max_lanes)
        """
        sdc_history = batch["sdc_history"]
        h_n, c_n = self.ego_encoder(sdc_history)
        ego_hidden = h_n[-1]

        lane_context = self.lane_encoder(
            batch["lane_features"],
            batch["lane_mask"],
            ego_hidden,
            lane_adj=batch.get("lane_adj"),
        )

        fusion_parts = [ego_hidden, lane_context]
        if self.use_neighbors:
            neighbor_ctx = self.neighbor_encoder(
                batch["neighbor_history"], batch["neighbor_mask"]
            )
            fusion_parts.append(neighbor_ctx)

        fused = self.fusion(torch.cat(fusion_parts, dim=-1))
        h_n = h_n.clone()
        h_n[-1] = fused

        last_pos = sdc_history[:, -1, :]
        last_vel = sdc_history[:, -1, :] - sdc_history[:, -2, :]

        if self._decoder_type == "mlp":
            # MLP decoder: predict trajectory directly
            pred_future = self.decoder(h_n, c_n, last_pos, last_vel)

            # Structure head: fused hidden → all lane logits at once
            B = fused.shape[0]
            lane_logits = self.structure_head(fused)  # (B, future_len * max_lanes)
            lane_logits = lane_logits.view(B, self.future_len, self.max_lanes)
        else:
            # LSTM decoder: decode with hidden state tracking
            pred_future, hidden_states = self._decode_with_hidden(h_n, c_n, last_pos, last_vel)
            lane_logits = self.structure_head(hidden_states)  # (B, T, max_lanes)

        return {
            "pred_future": pred_future,
            "lane_logits": lane_logits,
        }

    def _decode_with_hidden(self, h_n, c_n, last_pos, last_vel=None):
        """Decode future trajectory with CV-residual and collect hidden states."""
        decoder = self.decoder
        h_list = [h_n[i] for i in range(len(decoder.lstm_cell_layers))]
        c_list = [c_n[i] for i in range(len(decoder.lstm_cell_layers))]

        predictions = []
        hidden_states = []
        current_input = last_pos

        for t in range(decoder.future_len):
            for i, cell in enumerate(decoder.lstm_cell_layers):
                h_list[i], c_list[i] = cell(
                    current_input if i == 0 else h_list[i - 1],
                    (h_list[i], c_list[i]),
                )

            residual = decoder.output_proj(h_list[-1])

            if last_vel is not None:
                cv_pos = last_pos + last_vel * (t + 1)
                next_pos = cv_pos + residual
            else:
                next_pos = current_input + residual

            predictions.append(next_pos)
            hidden_states.append(h_list[-1])
            current_input = next_pos

        pred_future = torch.stack(predictions, dim=1)
        hidden_states = torch.stack(hidden_states, dim=1)

        return pred_future, hidden_states

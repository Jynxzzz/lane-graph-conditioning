"""Dual-Supervised Transformer for trajectory prediction.

Extends TransformerLaneCond with a structure prediction head that
classifies which lane each future timestep belongs to.
"""

import torch
import torch.nn as nn

from models.transformer_lane_cond import TransformerLaneCond


class TransformerDualSup(TransformerLaneCond):
    """Transformer with lane conditioning + structure supervision.

    Adds a structure head that predicts per-timestep lane assignments.
    Loss = smooth_l1(trajectory) + lambda * cross_entropy(lane_logits).
    """

    def __init__(self, max_lanes=16, **kwargs):
        super().__init__(**kwargs)
        self.max_lanes = max_lanes

        self.structure_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.future_len * max_lanes),
        )

    def forward(self, batch):
        ego_repr = self._encode(batch)

        # Decode trajectory
        sdc_history = batch["sdc_history"]
        h_n = ego_repr.unsqueeze(0)
        last_pos = sdc_history[:, -1, :]
        last_vel = sdc_history[:, -1, :] - sdc_history[:, -2, :]
        pred_future = self.decoder(h_n, None, last_pos, last_vel)

        # Structure prediction
        B = ego_repr.shape[0]
        lane_logits = self.structure_head(ego_repr)
        lane_logits = lane_logits.view(B, self.future_len, self.max_lanes)

        return {
            "pred_future": pred_future,
            "lane_logits": lane_logits,
        }

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """Encode history trajectory into a hidden state."""

    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, history):
        """
        Args:
            history: (B, T_hist, 2) trajectory positions

        Returns:
            h_n: (num_layers, B, hidden_dim) final hidden state
            c_n: (num_layers, B, hidden_dim) final cell state
        """
        x = self.embed(history)  # (B, T, embed_dim)
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class NeighborEncoder(nn.Module):
    """Encode neighbor trajectories and pool into a single context vector."""

    def __init__(self, input_dim=2, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, neighbor_history, neighbor_mask):
        """
        Args:
            neighbor_history: (B, N, T_hist, 2)
            neighbor_mask: (B, N) float, 1.0 for valid neighbors

        Returns:
            context: (B, hidden_dim) max-pooled neighbor representation
        """
        B, N, T, D = neighbor_history.shape
        # Flatten batch and neighbor dims
        x = neighbor_history.reshape(B * N, T, D)
        x = self.embed(x)  # (B*N, T, embed_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (1, B*N, hidden_dim)
        h = h_n.squeeze(0).reshape(B, N, -1)  # (B, N, hidden_dim)

        # Mask invalid neighbors with very negative values before max-pool
        mask = neighbor_mask.unsqueeze(-1)  # (B, N, 1)
        h = h * mask + (1 - mask) * (-1e9)

        # Max-pool over neighbors
        context, _ = h.max(dim=1)  # (B, hidden_dim)

        # If no valid neighbors, zero out
        any_valid = (neighbor_mask.sum(dim=1, keepdim=True) > 0).float()  # (B, 1)
        context = context * any_valid

        return context


class TrajectoryDecoder(nn.Module):
    """Autoregressive LSTM decoder with CV-residual prediction.

    Predicts future = CV_baseline + learned_residual.
    This ensures at least constant-velocity performance as a floor,
    and the model only needs to learn corrections (turns, stops, etc.).
    """

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, future_len=30):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        self.lstm_cell_layers = nn.ModuleList([
            nn.LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, h_n, c_n, last_pos, last_vel=None):
        """
        Args:
            h_n: (num_layers, B, hidden_dim) from encoder
            c_n: (num_layers, B, hidden_dim) from encoder
            last_pos: (B, 2) last observed position
            last_vel: (B, 2) last observed velocity (for CV baseline)

        Returns:
            pred_future: (B, future_len, 2) predicted positions
        """
        h_list = [h_n[i] for i in range(len(self.lstm_cell_layers))]
        c_list = [c_n[i] for i in range(len(self.lstm_cell_layers))]

        predictions = []
        current_input = last_pos  # (B, 2)

        for t in range(self.future_len):
            for i, cell in enumerate(self.lstm_cell_layers):
                h_list[i], c_list[i] = cell(
                    current_input if i == 0 else h_list[i - 1],
                    (h_list[i], c_list[i]),
                )

            # Predict residual delta from LSTM hidden state
            residual = self.output_proj(h_list[-1])  # (B, 2)

            # CV baseline: constant velocity extrapolation
            if last_vel is not None:
                cv_pos = last_pos + last_vel * (t + 1)
                next_pos = cv_pos + residual
            else:
                # Fallback: pure autoregressive (backward compatible)
                next_pos = current_input + residual

            predictions.append(next_pos)
            current_input = next_pos

        return torch.stack(predictions, dim=1)  # (B, future_len, 2)


class MLPDecoder(nn.Module):
    """Non-autoregressive MLP decoder with CV-residual prediction.

    Predicts all future residuals at once from the hidden state.
    Eliminates error compounding from autoregressive decoding.
    Output = CV_baseline_trajectory + learned_residuals.
    """

    def __init__(self, input_dim=2, hidden_dim=128, future_len=30):
        super().__init__()
        self.future_len = future_len
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, future_len * input_dim),
        )

    def forward(self, h_n, c_n, last_pos, last_vel=None):
        """
        Args:
            h_n: (num_layers, B, hidden_dim) from encoder
            c_n: (num_layers, B, hidden_dim) — unused
            last_pos: (B, 2) last observed position
            last_vel: (B, 2) last observed velocity

        Returns:
            pred_future: (B, future_len, 2) predicted positions
        """
        B = last_pos.shape[0]
        hidden = h_n[-1]  # (B, hidden_dim) — top-layer hidden

        # Predict all residuals at once
        residuals = self.mlp(hidden)  # (B, future_len * 2)
        residuals = residuals.view(B, self.future_len, self.input_dim)

        # CV baseline trajectory
        if last_vel is not None:
            t_indices = torch.arange(1, self.future_len + 1, device=last_pos.device).float()
            cv_traj = last_pos.unsqueeze(1) + last_vel.unsqueeze(1) * t_indices.unsqueeze(-1)
        else:
            cv_traj = last_pos.unsqueeze(1).expand(B, self.future_len, self.input_dim)

        return cv_traj + residuals


class LSTMBaseline(nn.Module):
    """LSTM encoder-decoder for trajectory prediction.

    Encodes ego history (and optionally neighbor histories) to predict
    future ego trajectory.
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
        decoder_type="mlp",
    ):
        super().__init__()
        self.use_neighbors = use_neighbors

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
            # Project concatenated features back to hidden_dim
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim + neighbor_hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        if decoder_type == "mlp":
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

        Returns:
            dict with "pred_future": (B, future_len, 2)
        """
        sdc_history = batch["sdc_history"]  # (B, 11, 2)

        # Encode ego history
        h_n, c_n = self.ego_encoder(sdc_history)  # (layers, B, hidden)

        # Optionally encode neighbors and fuse
        if self.use_neighbors:
            neighbor_ctx = self.neighbor_encoder(
                batch["neighbor_history"], batch["neighbor_mask"]
            )  # (B, neighbor_hidden)

            # Fuse with top-layer ego hidden state
            fused = self.fusion(
                torch.cat([h_n[-1], neighbor_ctx], dim=-1)
            )  # (B, hidden)
            h_n = h_n.clone()
            h_n[-1] = fused

        # Decode future trajectory with CV-residual
        last_pos = sdc_history[:, -1, :]  # (B, 2)
        last_vel = sdc_history[:, -1, :] - sdc_history[:, -2, :]  # (B, 2)
        pred_future = self.decoder(h_n, c_n, last_pos, last_vel)  # (B, 30, 2)

        return {"pred_future": pred_future}

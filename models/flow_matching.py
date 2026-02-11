"""LaneFlowNet: Flow Matching trajectory prediction conditioned on lane direction fields.

Combines lane direction vector fields on a BEV grid with conditional flow matching
to predict future ego trajectories. The key insight is that lane geometry provides
a natural prior for the velocity field: the model predicts a residual on top of
the lane-aligned flow, making it easier to learn turns, stops, and lane changes.

Architecture:
    TrajectoryEncoder (ego history LSTM, reused from lstm_baseline)
    NeighborEncoder (reused from lstm_baseline)
    LaneDirectionField (renders lane tangent vectors on BEV grid)
    LaneFieldEncoder (CNN to compress the 2-channel direction field)
    FlowMatchingDenoiser (Transformer-based velocity field predictor)
    Residual formulation: v = alpha(t) * s_0 * F_lane(x) + delta_theta(x, t, cond)
    where alpha(t) = t (zero at noise, full at data) and s_0 scales unit vectors to velocity magnitude
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm_baseline import NeighborEncoder, TrajectoryEncoder


class LaneDirectionField(nn.Module):
    """Constructs a continuous lane direction vector field on a BEV grid.

    Given lane centerline polylines from the dataset, renders a [2, H, W] tensor
    where each pixel contains the unit tangent direction of the nearest lane centerline.
    Uses inverse-distance weighting for smooth interpolation between nearby lanes.

    The BEV coordinate system follows the dataset convention:
        X+ = forward (ego heading), Y+ = left
    """

    def __init__(self, bev_range=(-30, 30, -10, 50), resolution=128):
        """
        Args:
            bev_range: (x_min, x_max, y_min, y_max) in meters, BEV crop around ego
            resolution: grid size (H = W = resolution)
        """
        super().__init__()
        self.x_min, self.x_max = bev_range[0], bev_range[1]
        self.y_min, self.y_max = bev_range[2], bev_range[3]
        self.resolution = resolution

        # Pre-compute pixel center coordinates (not learnable, but needed every forward)
        # grid_x: (H, W) with values in [x_min, x_max]
        # grid_y: (H, W) with values in [y_min, y_max]
        xs = torch.linspace(self.x_min, self.x_max, resolution)
        ys = torch.linspace(self.y_min, self.y_max, resolution)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each
        # Stack as (H, W, 2) for distance computations
        grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        self.register_buffer("grid_coords", grid_coords)

        # IDW power parameter (higher = sharper, more nearest-neighbor-like)
        self.idw_power = 2.0
        # Minimum distance to avoid division by zero
        self.eps = 1e-6

    def forward(self, lane_centerlines_bev, lane_mask):
        """Render lane direction field on BEV grid.

        Args:
            lane_centerlines_bev: (B, L, P, 2) - L lanes, P points per lane, BEV coords
            lane_mask: (B, L) - 1.0 for valid lanes, 0.0 for padding

        Returns:
            direction_field: (B, 2, H, W) - unit tangent direction at each pixel
        """
        B, L, P, _ = lane_centerlines_bev.shape
        H = W = self.resolution
        device = lane_centerlines_bev.device

        # Compute per-segment tangent vectors: (B, L, P-1, 2)
        segments = lane_centerlines_bev[:, :, 1:, :] - lane_centerlines_bev[:, :, :-1, :]
        seg_lengths = segments.norm(dim=-1, keepdim=True).clamp(min=self.eps)  # (B, L, P-1, 1)
        seg_tangents = segments / seg_lengths  # (B, L, P-1, 2) unit tangents

        # Segment midpoints: (B, L, P-1, 2)
        seg_midpoints = 0.5 * (
            lane_centerlines_bev[:, :, :-1, :] + lane_centerlines_bev[:, :, 1:, :]
        )

        # Create segment validity mask: segment is valid only if its lane is valid
        # (B, L) -> (B, L, P-1)
        seg_mask = lane_mask.unsqueeze(-1).expand(B, L, P - 1)

        # Flatten segments: (B, S, 2) where S = L * (P-1)
        S = L * (P - 1)
        flat_midpoints = seg_midpoints.reshape(B, S, 2)       # (B, S, 2)
        flat_tangents = seg_tangents.reshape(B, S, 2)          # (B, S, 2)
        flat_mask = seg_mask.reshape(B, S)                      # (B, S)

        # Compute distances from every grid point to every segment midpoint
        # grid_coords: (H, W, 2) -> (1, H*W, 2)
        grid_flat = self.grid_coords.reshape(1, H * W, 2).expand(B, -1, -1)  # (B, HW, 2)

        # (B, HW, 1, 2) - (B, 1, S, 2) -> (B, HW, S, 2)
        diff = grid_flat.unsqueeze(2) - flat_midpoints.unsqueeze(1)  # (B, HW, S, 2)
        dist = diff.norm(dim=-1).clamp(min=self.eps)  # (B, HW, S)

        # IDW weights: w_i = 1 / d_i^p, masked for invalid segments
        weights = 1.0 / (dist ** self.idw_power)  # (B, HW, S)
        weights = weights * flat_mask.unsqueeze(1)  # zero out invalid segments

        # Normalize weights
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=self.eps)  # (B, HW, 1)
        weights_norm = weights / weight_sum  # (B, HW, S)

        # Weighted sum of tangent vectors
        # (B, HW, S) x (B, S, 2) -> use einsum
        direction = torch.einsum("bgs,bsd->bgd", weights_norm, flat_tangents)  # (B, HW, 2)

        # Normalize to unit vectors (some pixels may get near-zero if far from all lanes)
        dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        direction = direction / dir_norm

        # Handle pixels with no nearby valid lanes: check if all weights were zero
        any_valid = (flat_mask.sum(dim=-1, keepdim=True) > 0).float()  # (B, 1)
        # If a batch element has no valid lanes at all, zero the field
        direction = direction * any_valid.unsqueeze(1)  # (B, HW, 2)

        # Reshape to (B, 2, H, W)
        direction_field = direction.reshape(B, H, W, 2).permute(0, 3, 1, 2)

        return direction_field

    def sample_at_points(self, direction_field, points):
        """Sample the direction field at specific BEV coordinates.

        Uses bilinear interpolation via grid_sample.

        Args:
            direction_field: (B, 2, H, W) the rendered field
            points: (B, T, 2) trajectory points in BEV coordinates (x, y)

        Returns:
            sampled_directions: (B, T, 2) interpolated direction vectors
        """
        B, T, _ = points.shape

        # Normalize points to [-1, 1] for grid_sample
        # grid_sample expects (x_norm, y_norm) in [-1, 1] mapping to (W, H)
        x_norm = 2.0 * (points[:, :, 0] - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_norm = 2.0 * (points[:, :, 1] - self.y_min) / (self.y_max - self.y_min) - 1.0

        # grid_sample expects grid of shape (B, H_out, W_out, 2)
        # For point sampling, use (B, 1, T, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)  # (B, 1, T, 2)

        # Bilinear interpolation with zero padding for out-of-bounds
        sampled = F.grid_sample(
            direction_field, grid,
            mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (B, 2, 1, T)

        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, T, 2)
        return sampled


class LaneFieldEncoder(nn.Module):
    """CNN encoder that compresses a (2, H, W) lane direction field into a context vector.

    Uses a simple stack of strided convolutions followed by global average pooling.
    """

    def __init__(self, in_channels=2, hidden_channels=32, out_dim=64, resolution=128):
        super().__init__()
        # 4 downsampling blocks: resolution -> res/2 -> res/4 -> res/8 -> res/16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_channels * 4, out_dim)

    def forward(self, direction_field):
        """
        Args:
            direction_field: (B, 2, H, W) lane direction field

        Returns:
            lane_ctx: (B, out_dim) compressed lane representation
        """
        x = self.conv(direction_field)  # (B, C, H', W')
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        return self.proj(x)  # (B, out_dim)


def sinusoidal_time_embedding(t, dim):
    """Sinusoidal positional embedding for diffusion time.

    Args:
        t: (B,) float tensor in [0, 1]
        dim: embedding dimension (must be even)

    Returns:
        emb: (B, dim) time embedding
    """
    assert dim % 2 == 0, "dim must be even"
    half_dim = dim // 2
    # Frequencies: exp(-log(10000) * i / (half_dim - 1))
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
    )
    # (B, 1) * (1, half_dim) -> (B, half_dim)
    args = t.unsqueeze(-1) * freq.unsqueeze(0) * 1000.0  # scale to match standard diffusion
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class FlowMatchingDenoiser(nn.Module):
    """Predicts velocity field v_theta(x_t, t | conditions).

    Takes a noisy trajectory x_t, the diffusion time t, and conditioning signals
    (ego history encoding, neighbor context, lane field context) and predicts
    the velocity field for flow matching (i.e., the direction to move x_t toward x_1).

    Architecture:
        - Time embedding: sinusoidal -> MLP
        - Trajectory tokens: per-timestep MLP embedding of (x, y)
        - Cross-attention with condition tokens
        - Self-attention Transformer over trajectory tokens
        - Output projection to (T, 2) velocity
    """

    def __init__(
        self,
        traj_dim=2,
        future_len=30,
        hidden_dim=256,
        ego_hidden=128,
        neighbor_hidden=64,
        lane_field_dim=64,
        n_heads=4,
        n_layers=3,
        time_embed_dim=128,
    ):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embed_raw_dim = time_embed_dim

        # Noisy trajectory embedding: per-timestep
        self.traj_embed = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition projections (map each condition to hidden_dim tokens)
        self.ego_proj = nn.Linear(ego_hidden, hidden_dim)
        self.nbr_proj = nn.Linear(neighbor_hidden, hidden_dim)
        self.lane_proj = nn.Linear(lane_field_dim, hidden_dim)
        # Lane residual direction projection (per-timestep, 2D direction at each traj point)
        self.lane_dir_proj = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable positional encoding for future timesteps
        self.pos_enc = nn.Parameter(torch.randn(1, future_len, hidden_dim) * 0.02)

        # Transformer blocks (self-attention over trajectory tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output projection: hidden_dim -> 2 (velocity)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, traj_dim),
        )

    def forward(self, x_t, t, ego_ctx, neighbor_ctx, lane_field_ctx, lane_dir_at_points):
        """
        Args:
            x_t: (B, T, 2) noisy trajectory
            t: (B,) diffusion time in [0, 1]
            ego_ctx: (B, d_ego) ego history encoding
            neighbor_ctx: (B, d_nbr) neighbor context
            lane_field_ctx: (B, d_lane) global lane field features
            lane_dir_at_points: (B, T, 2) lane direction sampled at each trajectory point

        Returns:
            velocity: (B, T, 2) predicted velocity field
        """
        B, T, _ = x_t.shape

        # Time embedding: (B,) -> (B, hidden_dim)
        t_emb = sinusoidal_time_embedding(t, self.time_embed_raw_dim)
        t_emb = self.time_embed(t_emb)  # (B, hidden_dim)

        # Embed noisy trajectory tokens: (B, T, 2) -> (B, T, hidden_dim)
        traj_tokens = self.traj_embed(x_t) + self.pos_enc[:, :T, :]

        # Embed lane direction at trajectory points: (B, T, 2) -> (B, T, hidden_dim)
        lane_dir_tokens = self.lane_dir_proj(lane_dir_at_points)

        # Condition tokens: project each to (B, 1, hidden_dim) and concatenate
        cond_time = t_emb.unsqueeze(1)                           # (B, 1, H)
        cond_ego = self.ego_proj(ego_ctx).unsqueeze(1)           # (B, 1, H)
        cond_nbr = self.nbr_proj(neighbor_ctx).unsqueeze(1)      # (B, 1, H)
        cond_lane = self.lane_proj(lane_field_ctx).unsqueeze(1)  # (B, 1, H)

        # Combine: [cond_time, cond_ego, cond_nbr, cond_lane, traj_tokens + lane_dir]
        # The condition tokens are prepended so the Transformer can attend to them
        tokens = torch.cat([
            cond_time, cond_ego, cond_nbr, cond_lane,
            traj_tokens + lane_dir_tokens,
        ], dim=1)  # (B, 4 + T, hidden_dim)

        # Self-attention over all tokens
        tokens = self.transformer(tokens)  # (B, 4 + T, hidden_dim)

        # Extract trajectory tokens (skip the 4 condition tokens)
        traj_out = tokens[:, 4:, :]  # (B, T, hidden_dim)

        # Project to velocity
        velocity = self.output_proj(traj_out)  # (B, T, 2)

        return velocity


class LaneFlowNet(nn.Module):
    """Complete LaneFlowNet: flow matching trajectory prediction with lane direction fields.

    Combines:
    - TrajectoryEncoder (ego history LSTM, reused from lstm_baseline)
    - NeighborEncoder (reused from lstm_baseline)
    - LaneDirectionField (renders lane tangent vectors on BEV grid)
    - LaneFieldEncoder (CNN to compress the direction field)
    - FlowMatchingDenoiser (Transformer-based velocity predictor)
    - Residual formulation: v = alpha(t)*s_0*F_lane(x) + delta_theta(x, t, cond)
      where alpha(t)=t and s_0=lane_field_scale

    During training, the model learns to predict the velocity field that transports
    noise x_0 to ground truth x_1 along a straight-line OT path.

    During inference, the model integrates the learned velocity from noise to
    generate trajectory samples via Euler ODE integration.
    """

    def __init__(self, config):
        """
        Args:
            config: dict or OmegaConf with keys:
                input_dim (int): trajectory coordinate dim (default 2)
                embed_dim (int): ego encoder embedding dim (default 64)
                hidden_dim (int): ego encoder LSTM hidden dim (default 128)
                num_layers (int): ego encoder LSTM layers (default 2)
                future_len (int): prediction horizon in frames (default 30)
                use_neighbors (bool): whether to use neighbor encoder (default True)
                neighbor_hidden_dim (int): neighbor encoder hidden dim (default 64)
                denoiser_hidden_dim (int): Transformer hidden dim (default 256)
                denoiser_n_heads (int): Transformer attention heads (default 4)
                denoiser_n_layers (int): Transformer layers (default 3)
                lane_field_dim (int): lane field encoder output dim (default 64)
                lane_field_resolution (int): BEV grid resolution (default 128)
                bev_range (list): [x_min, x_max, y_min, y_max] (default [-30, 30, -10, 50])
                sigma (float): noise scale for flow matching (default 1.0)
                lane_field_scale (float): magnitude scale for lane field residual,
                    should approximate the average per-step displacement in meters
                    so that unit-vector lane directions are comparable to target
                    velocities. E.g., if vehicles move ~0.6 m/step at 10Hz,
                    lane_field_scale ≈ 50 (total displacement over 80 steps).
                    (default 50.0)
        """
        super().__init__()

        # Unpack config (support both dict and OmegaConf)
        def _get(key, default):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default) if hasattr(config, key) else default

        input_dim = _get("input_dim", 2)
        embed_dim = _get("embed_dim", 64)
        hidden_dim = _get("hidden_dim", 128)
        num_layers = _get("num_layers", 2)
        future_len = _get("future_len", 30)
        use_neighbors = _get("use_neighbors", True)
        neighbor_hidden_dim = _get("neighbor_hidden_dim", 64)
        denoiser_hidden_dim = _get("denoiser_hidden_dim", 256)
        denoiser_n_heads = _get("denoiser_n_heads", 4)
        denoiser_n_layers = _get("denoiser_n_layers", 3)
        lane_field_dim = _get("lane_field_dim", 64)
        lane_field_resolution = _get("lane_field_resolution", 128)
        bev_range = _get("bev_range", [-30, 30, -10, 50])
        sigma = _get("sigma", 1.0)
        lane_field_scale = _get("lane_field_scale", 50.0)
        use_lane_field = _get("use_lane_field", True)

        self.future_len = future_len
        self.use_neighbors = use_neighbors
        self.sigma = sigma
        self.lane_field_scale = lane_field_scale
        self.use_lane_field = use_lane_field
        self.input_dim = input_dim

        # --- Reused encoders from existing codebase ---
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

        # --- New: Lane direction field ---
        self.lane_direction_field = LaneDirectionField(
            bev_range=tuple(bev_range),
            resolution=lane_field_resolution,
        )
        self.lane_field_encoder = LaneFieldEncoder(
            in_channels=2,
            hidden_channels=32,
            out_dim=lane_field_dim,
            resolution=lane_field_resolution,
        )

        # --- New: Flow matching denoiser ---
        self.denoiser = FlowMatchingDenoiser(
            traj_dim=input_dim,
            future_len=future_len,
            hidden_dim=denoiser_hidden_dim,
            ego_hidden=hidden_dim,
            neighbor_hidden=neighbor_hidden_dim if use_neighbors else hidden_dim,
            lane_field_dim=lane_field_dim,
            n_heads=denoiser_n_heads,
            n_layers=denoiser_n_layers,
        )

        # Dummy neighbor context for when neighbors are disabled
        if not use_neighbors:
            self.dummy_neighbor_dim = neighbor_hidden_dim

    def _encode_conditions(self, batch):
        """Encode all conditioning signals from the batch.

        Returns:
            ego_ctx: (B, hidden_dim) ego history encoding (top LSTM layer)
            neighbor_ctx: (B, neighbor_hidden_dim) neighbor context
            lane_field: (B, 2, H, W) rendered lane direction field
            lane_field_ctx: (B, lane_field_dim) compressed lane context
        """
        sdc_history = batch["sdc_history"]  # (B, 11, 2)

        # Ego encoding
        h_n, _ = self.ego_encoder(sdc_history)
        ego_ctx = h_n[-1]  # (B, hidden_dim)

        # Neighbor encoding
        if self.use_neighbors:
            neighbor_ctx = self.neighbor_encoder(
                batch["neighbor_history"], batch["neighbor_mask"]
            )  # (B, neighbor_hidden_dim)
        else:
            B = sdc_history.shape[0]
            neighbor_ctx = torch.zeros(
                B, self.dummy_neighbor_dim, device=sdc_history.device
            )

        # Lane direction field (skip if disabled for vanilla FM ablation)
        if self.use_lane_field:
            lane_centerlines = batch["lane_centerlines_bev"]  # (B, L, P, 2)
            lane_mask = batch["lane_mask"]                     # (B, L)
            lane_field = self.lane_direction_field(lane_centerlines, lane_mask)  # (B, 2, H, W)
            lane_field_ctx = self.lane_field_encoder(lane_field)  # (B, lane_field_dim)
        else:
            B = sdc_history.shape[0]
            H = self.lane_direction_field.resolution
            lane_field = torch.zeros(B, 2, H, H, device=sdc_history.device)
            lane_field_ctx = torch.zeros(
                B, self.lane_field_encoder.proj.out_features, device=sdc_history.device
            )

        return ego_ctx, neighbor_ctx, lane_field, lane_field_ctx

    def forward(self, batch, t=None, x_t=None):
        """Forward pass for training: compute flow matching loss.

        During training, samples random t and x_0, computes x_t via OT interpolation,
        and returns the predicted velocity and target velocity for loss computation.

        Args:
            batch: dict from dataloader
            t: (B,) optional pre-sampled diffusion time (for external control)
            x_t: (B, T, 2) optional pre-computed noisy trajectory

        Returns:
            dict with:
                pred_velocity: (B, T, 2) predicted velocity field
                target_velocity: (B, T, 2) ground truth velocity (x_1 - x_0)
                x_t: (B, T, 2) the noisy trajectory used
                t: (B,) the sampled times
        """
        B = batch["sdc_history"].shape[0]
        device = batch["sdc_history"].device
        x_1 = batch["sdc_future"]  # (B, T, 2) ground truth future

        # Encode conditions
        ego_ctx, neighbor_ctx, lane_field, lane_field_ctx = self._encode_conditions(batch)

        # Sample diffusion time
        if t is None:
            t = torch.rand(B, device=device)  # U(0, 1)

        # Sample noise
        if x_t is None:
            x_0 = torch.randn_like(x_1) * self.sigma  # (B, T, 2)

            # OT straight-line interpolation: x_t = (1 - t) * x_0 + t * x_1
            t_expand = t[:, None, None]  # (B, 1, 1)
            x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        else:
            x_0 = None  # externally provided

        # Target velocity: u_t = x_1 - x_0 (constant along straight path)
        if x_0 is not None:
            target_velocity = x_1 - x_0  # (B, T, 2)
        else:
            target_velocity = None

        # Sample lane direction at trajectory points for residual formulation
        lane_dir_at_points = self.lane_direction_field.sample_at_points(
            lane_field, x_t
        )  # (B, T, 2)

        # Predict velocity: v_theta = delta_theta(x_t, t, cond)
        # The full velocity is: v = alpha(t) * s_0 * F_lane(x_t) + delta_theta
        pred_delta = self.denoiser(
            x_t, t, ego_ctx, neighbor_ctx, lane_field_ctx, lane_dir_at_points
        )  # (B, T, 2)

        # Time-dependent scaling: alpha(t) = t
        # At t=0 (noise), x_t is random -> lane field at random positions is meaningless -> weight = 0
        # At t=1 (data), x_t is near GT trajectory -> lane field is meaningful -> weight = s_0
        # s_0 = lane_field_scale matches the lane unit vectors to the velocity magnitude
        alpha_t = t[:, None, None]  # (B, 1, 1) for broadcasting over (B, T, 2)
        scaled_lane_dir = alpha_t * self.lane_field_scale * lane_dir_at_points

        # Residual formulation: total predicted velocity
        pred_velocity = scaled_lane_dir + pred_delta

        return {
            "pred_velocity": pred_velocity,
            "target_velocity": target_velocity,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample(self, batch, n_samples=6, n_steps=10):
        """Generate multiple trajectory samples via Euler ODE integration.

        Integrates from t=0 (noise) to t=1 (data) using the learned velocity field.
        Generates n_samples independent trajectories per batch element.

        Args:
            batch: dict from dataloader
            n_samples: number of trajectory samples per batch element
            n_steps: number of Euler integration steps

        Returns:
            trajectories: (B, n_samples, T, 2) sampled future trajectories
        """
        B = batch["sdc_history"].shape[0]
        device = batch["sdc_history"].device
        T = self.future_len

        # Encode conditions (shared across all samples)
        ego_ctx, neighbor_ctx, lane_field, lane_field_ctx = self._encode_conditions(batch)

        # Repeat conditions for n_samples
        ego_ctx_rep = ego_ctx.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        neighbor_ctx_rep = neighbor_ctx.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        lane_field_rep = lane_field.unsqueeze(1).expand(B, n_samples, -1, -1, -1).reshape(
            B * n_samples, *lane_field.shape[1:]
        )
        lane_field_ctx_rep = lane_field_ctx.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)

        # Start from noise
        x = torch.randn(B * n_samples, T, self.input_dim, device=device) * self.sigma

        # Euler ODE integration: t goes from 0 to 1
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step * dt
            t_tensor = torch.full((B * n_samples,), t_val, device=device)

            # Sample lane direction at current trajectory positions
            lane_dir = self.lane_direction_field.sample_at_points(lane_field_rep, x)

            # Predict velocity with time-dependent scaling
            pred_delta = self.denoiser(
                x, t_tensor, ego_ctx_rep, neighbor_ctx_rep,
                lane_field_ctx_rep, lane_dir,
            )
            alpha_t = t_val  # scalar, broadcasts naturally
            velocity = alpha_t * self.lane_field_scale * lane_dir + pred_delta

            # Euler step
            x = x + velocity * dt

        # Reshape: (B * n_samples, T, 2) -> (B, n_samples, T, 2)
        trajectories = x.reshape(B, n_samples, T, self.input_dim)
        return trajectories

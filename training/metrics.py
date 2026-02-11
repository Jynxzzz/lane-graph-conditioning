import torch


def compute_ade(pred, gt):
    """Average Displacement Error over all timesteps.

    Args:
        pred: (B, T, 2) predicted positions
        gt:   (B, T, 2) ground truth positions

    Returns:
        scalar tensor, mean L2 error over batch and time
    """
    return torch.norm(pred - gt, dim=-1).mean()


def compute_fde(pred, gt):
    """Final Displacement Error at last timestep.

    Args:
        pred: (B, T, 2) predicted positions
        gt:   (B, T, 2) ground truth positions

    Returns:
        scalar tensor, mean L2 error at final frame over batch
    """
    return torch.norm(pred[:, -1] - gt[:, -1], dim=-1).mean()


def compute_metrics(pred, gt):
    """Compute ADE and FDE."""
    return compute_ade(pred, gt), compute_fde(pred, gt)


def compute_metrics_at_horizons(pred, gt, horizons_frames=(10, 20, 30)):
    """Compute ADE and FDE at multiple prediction horizons.

    Args:
        pred: (B, T, 2)
        gt:   (B, T, 2)
        horizons_frames: tuple of frame counts (10=1s, 20=2s, 30=3s at 10Hz)

    Returns:
        dict with keys like "ade_1s", "fde_1s", etc.
    """
    results = {}
    horizon_names = {10: "1s", 20: "2s", 30: "3s"}

    for h in horizons_frames:
        if h > pred.shape[1]:
            continue
        name = horizon_names.get(h, f"{h}f")
        results[f"ade_{name}"] = compute_ade(pred[:, :h], gt[:, :h])
        results[f"fde_{name}"] = compute_fde(pred[:, :h], gt[:, :h])

    return results


def compute_min_ade(pred_multimodal, gt):
    """Compute minimum ADE across K predictions (for multi-modal models).

    Args:
        pred_multimodal: (B, K, T, 2) - K predictions per sample
        gt:   (B, T, 2) - ground truth

    Returns:
        scalar tensor, minimum ADE across modes
    """
    B, K, T, _ = pred_multimodal.shape
    gt_expanded = gt.unsqueeze(1).expand(B, K, T, 2)  # (B, K, T, 2)

    # Compute ADE for each mode
    errors = torch.norm(pred_multimodal - gt_expanded, dim=-1)  # (B, K, T)
    ade_per_mode = errors.mean(dim=2)  # (B, K)

    # Take minimum across modes
    min_ade = ade_per_mode.min(dim=1)[0].mean()  # scalar

    return min_ade


def compute_min_fde(pred_multimodal, gt):
    """Compute minimum FDE across K predictions.

    Args:
        pred_multimodal: (B, K, T, 2)
        gt:   (B, T, 2)

    Returns:
        scalar tensor
    """
    B, K, T, _ = pred_multimodal.shape
    gt_final = gt[:, -1:, :].unsqueeze(1).expand(B, K, 1, 2)  # (B, K, 1, 2)
    pred_final = pred_multimodal[:, :, -1:, :]  # (B, K, 1, 2)

    errors = torch.norm(pred_final - gt_final, dim=-1).squeeze(-1)  # (B, K)
    min_fde = errors.min(dim=1)[0].mean()

    return min_fde


def compute_multimodal_metrics(pred_multimodal, gt):
    """Compute minADE and minFDE for multi-modal predictions.

    Args:
        pred_multimodal: (B, K, T, 2)
        gt: (B, T, 2)

    Returns:
        tuple of (minADE, minFDE)
    """
    return compute_min_ade(pred_multimodal, gt), compute_min_fde(pred_multimodal, gt)

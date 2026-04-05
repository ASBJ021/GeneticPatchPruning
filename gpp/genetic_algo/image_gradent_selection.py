from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _sobel_gradient_magnitude(gray: torch.Tensor) -> torch.Tensor:
    """Return Sobel gradient magnitude map for a single-channel image tensor [B,1,H,W]."""
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=gray.dtype,
        device=gray.device,
    ).unsqueeze(0)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=gray.dtype,
        device=gray.device,
    ).unsqueeze(0)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def image_gradient_patch_selection(
    pixel_values: torch.Tensor,
    num_patches: int,
    keep: Optional[int] = None,
    min_keep: Optional[int] = None,
    max_keep: Optional[int] = None,
) -> Tuple[List[int], float]:
    """Select patch indices using average image-gradient magnitude per patch.

    Args:
        pixel_values: Preprocessed image tensor [B, C, H, W] (first sample is used).
        num_patches: Number of image patches/tokens (without CLS token).
        keep: Fixed keep count (if provided).
        min_keep: Lower bound for adaptive keep.
        max_keep: Upper bound for adaptive keep.

    Returns:
        selected_indices: List of kept patch indices.
        keep_ratio: len(selected_indices) / max(1, num_patches).
    """
    if pixel_values is None or pixel_values.ndim != 4:
        return [], 0.0
    if num_patches <= 0:
        return [], 0.0

    image = pixel_values[:1]
    _, c, h, w = image.shape

    if c >= 3:
        gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
    else:
        gray = image[:, :1]

    grad_mag = _sobel_gradient_magnitude(gray)

    grid = int(round(num_patches ** 0.5))
    if grid <= 0:
        return [], 0.0
    patch_h = max(1, h // grid)
    patch_w = max(1, w // grid)

    scores = F.avg_pool2d(grad_mag, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    scores_1d = scores.flatten()[:num_patches]
    n = int(scores_1d.numel())
    if n <= 0:
        return [], 0.0

    if keep is not None:
        k = max(1, min(int(keep), n))
    else:
        lo = max(1, min(int(min_keep if min_keep is not None else 1), n))
        hi = max(lo, min(int(max_keep if max_keep is not None else n), n))
        mean_score = float(scores_1d.mean().item())
        max_score = float(scores_1d.max().item())
        strength = mean_score / (max_score + 1e-8)
        strength = max(0.0, min(1.0, strength))
        k = int(round(lo + (hi - lo) * strength))
        k = max(lo, min(hi, k))

    topk_idx = torch.topk(scores_1d, k=k, dim=0).indices
    selected_indices = [int(i) for i in topk_idx.detach().cpu().tolist()]
    keep_ratio = len(selected_indices) / max(1, num_patches)
    return selected_indices, float(keep_ratio)


import torch
import numpy as np
# Histogram Parallelized GPU
import numba
from numba import prange

# Logger
import logging
logger = logging.getLogger(__name__)

def estimate_range(samples, debug=False):
    """
    Estimate binning range using 1st and 99th percentiles.
    Supports shape (N, H, W, 3).
    """
    if samples.size == 0:
        return 1e-8, 1.0

    flat = samples.reshape(-1, 3)
    positive = flat[flat > 1e-8]

    if len(positive) == 0:
        return 1e-8, 1.0

    min_val = np.percentile(positive, 1)
    max_val = np.percentile(positive, 99)

    if min_val == max_val:
        max_val *= 1.1

    if debug:
        logger.info(f"Linear estimated radiance range: [{min_val:.4e}, {max_val:.4e}]")
    return min_val, max_val


@numba.njit(parallel=True)
def accumulate_histogram_numba(hist, samples, bin_edges, num_bins):
    """
    Numba-accelerated histogram accumulation.

    hist: np.ndarray (H, W, 3, B) zero-initialized array to fill
    samples: np.ndarray (N, H, W, 3)
    bin_edges: np.ndarray (B+1,)

    This replaces vectorized calls with explicit parallel loops.
    """
    N, H, W, C = samples.shape

    for y in prange(H):
        for x in range(W):
            for c in range(C):
                # Initialize local histogram for this pixel & channel
                local_hist = np.zeros(num_bins, dtype=np.int32)
                for n in range(N):
                    val = samples[n, y, x, c]
                    # Find bin index via binary search on bin_edges
                    # Since bin_edges is sorted, use simple linear or binary search
                    # For small num_bins, linear search is fine
                    bin_idx = 0
                    for b in range(num_bins):
                        if val >= bin_edges[b] and val < bin_edges[b + 1]:
                            bin_idx = b
                            break
                        # Handle edge case val == max(bin_edges)
                        if val == bin_edges[-1]:
                            bin_idx = num_bins - 1
                    local_hist[bin_idx] += 1
                for b in range(num_bins):
                    hist[y, x, c, b] = local_hist[b]


def accumulate_histogram_gpu(samples, bin_edges, num_bins):
    """
    samples: (N, H, W, 3) float32
    bin_edges: (B+1,) float32
    Returns:
        hist: (H, W, 3, B) int32
    """
    device = samples.device
    N, H, W, C = samples.shape
    B = num_bins

    # Bucketize samples to bin indices
    bin_ids = torch.bucketize(samples, bin_edges) - 1               # (N, H, W, C)
    bin_ids = torch.clamp(bin_ids, 0, B - 1)

    # Prepare histogram tensor
    hist = torch.zeros((H, W, C, B), dtype=torch.int32, device=device)

    # Prepare indices
    h_idx = torch.arange(H, device=device).view(1, H, 1, 1).expand(N, H, W, C)
    w_idx = torch.arange(W, device=device).view(1, 1, W, 1).expand(N, H, W, C)
    c_idx = torch.arange(C, device=device).view(1, 1, 1, C).expand(N, H, W, C)
    b_idx = bin_ids  # already shape (N, H, W, C)

    # Flatten all index tensors
    h_idx = h_idx.reshape(-1)
    w_idx = w_idx.reshape(-1)
    c_idx = c_idx.reshape(-1)
    b_idx = b_idx.reshape(-1)

    # Flatten values to add (1 per sample)
    values = torch.ones_like(b_idx, dtype=torch.int32)
    # Accumulate into histogram
    hist.index_put_((h_idx, w_idx, c_idx, b_idx), values, accumulate=True)
    return hist


@torch.no_grad()
def accumulate_histogram_gpu_fixed_edges(samples, bin_edges, num_bins: int):
    """
    Build per-pixel, per-channel histograms over N samples using fixed edges.

    Args:
        samples:   (N, H, W, 3) or (N, 3, H, W), float32, linear HDR
        bin_edges: (3, B+1) per-channel edges  OR (B+1,) shared edges
        num_bins:  int B

    Returns:
        hist: (3, H, W, B) int32   # matches your downstream code
    """
    # Normalize input shape to (N, H, W, 3)
    if samples.dim() != 4:
        raise ValueError(f"samples must be 4D, got {samples.shape}")
    if samples.shape[-1] == 3:
        N, H, W, C = samples.shape
        x = samples
    elif samples.shape[1] == 3:
        # (N, 3, H, W) -> (N, H, W, 3)
        N, C, H, W = samples.shape
        x = samples.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError("samples must be (N,H,W,3) or (N,3,H,W)")

    assert C == 3, "Expecting 3 color channels"
    B = num_bins
    device = x.device

    # Prepare output: (3, H, W, B)
    hist = torch.zeros((3, H, W, B), dtype=torch.int32, device=device)

    # Ensure edges are on device and shaped per channel
    if bin_edges.dim() == 1:
        # shared edges for all channels
        edges = bin_edges.to(device)
        if edges.numel() != B + 1:
            raise ValueError("bin_edges length must be B+1")
        per_channel_edges = [edges, edges, edges]
    elif bin_edges.dim() == 2:
        # (3, B+1)
        if bin_edges.shape != (3, B + 1):
            raise ValueError(f"bin_edges expected shape (3,{B+1}), got {tuple(bin_edges.shape)}")
        per_channel_edges = [bin_edges[c].to(device) for c in range(3)]
    else:
        raise ValueError("bin_edges must be 1D (B+1) or 2D (3, B+1)")

    # Build index grids once (expanded across N later)
    h_idx = torch.arange(H, device=device).view(1, H, 1, 1).expand(N, H, W, 1)
    w_idx = torch.arange(W, device=device).view(1, 1, W, 1).expand(N, H, W, 1)

    # Accumulate channel by channel (each with its own edges)
    for c in range(3):
        edges_c = per_channel_edges[c]
        # Bucketize this channel's samples: (N, H, W)
        bin_ids = torch.bucketize(x[..., c].contiguous(), edges_c.contiguous()) - 1
        bin_ids = torch.clamp(bin_ids, 0, B - 1)

        # Flatten all indices for index_put_
        h_flat = h_idx.reshape(-1)
        w_flat = w_idx.reshape(-1)
        b_flat = bin_ids.reshape(-1)

        # Per channel accumulation target (H, W, B)
        # Note: index_put_ needs all index tensors to be same dtype (long)
        values = torch.ones_like(b_flat, dtype=torch.int32)
        hist_c = torch.zeros((H, W, B), dtype=torch.int32, device=device)
        hist_c.index_put_((h_flat.long(), w_flat.long(), b_flat.long()), values, accumulate=True)

        hist[c] = hist_c  # (H, W, B)

    return hist


def generate_histograms(samples, num_bins, device=None, debug=False, log_binning=False, min_val=None, max_val=None):
    """
    Generate histograms per pixel per channel from input samples using CPU (NumPy + Numba).

    Args:
        samples (np.array): Shape (N, H, W, 3), input radiance samples
        num_bins (int): Number of histogram bins
        debug (bool): Print debug info
        log_binning (bool): Use logarithmic binning if True
        min_val (float): Optional manual minimum radiance
        max_val (float): Optional manual maximum radiance

    Returns:
        hist (torch.Tensor): Shape (H, W, 3, B), histogram per pixel per channel
        bin_edges (torch.Tensor): Shape (B+1,), bin edges
    """
    _, H, W, _ = samples.shape
    if min_val is None or max_val is None:
        min_val, max_val = estimate_range(samples, debug=debug)
    if log_binning:
        bin_edges = np.logspace(np.log10(max(min_val, 1e-4)), np.log10(max_val), num_bins + 1)
    else:
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    hist = np.zeros((H, W, 3, num_bins), dtype=np.int32)
    samples = samples.transpose(0, 2, 3, 1)  # shape (N, W, H, 3)
    accumulate_histogram_numba(hist, samples, bin_edges, num_bins)

    hist = torch.from_numpy(hist).float()
    bin_edges = torch.from_numpy(bin_edges).float()
    return hist, bin_edges


def generate_histograms_torch(samples, num_bins, device=None, debug=False, log_binning=False, min_val=None, max_val=None, fixed_edges=None):
    """
    Generate histograms per pixel per channel from input samples using GPU (PyTorch).

    Args:
        samples (torch.Tensor): Shape (N, H, W, 3), input radiance samples
        num_bins (int): Number of histogram bins
        log_binning (bool): Use logarithmic binning if True
        min_val (float): Optional manual minimum radiance
        max_val (float): Optional manual maximum radiance

    Returns:
        hist (torch.Tensor): Shape (H, W, 3, B), histogram per pixel per channel
        bin_edges (torch.Tensor): Shape (B+1,), bin edges
    """
    _, H, W, _ = samples.shape
    if fixed_edges is not None: 
        samples = samples.contiguous()
        hist = accumulate_histogram_gpu_fixed_edges(
            samples, fixed_edges, num_bins
        )
        return hist, fixed_edges  # keep edges consistent
    else: 
        if min_val is None or max_val is None:
            min_val, max_val = estimate_range(samples, debug=debug)
        if log_binning:
            bin_edges = torch.logspace(torch.log10(torch.tensor(max(min_val, 1e-4))), torch.log10(torch.tensor(max_val)), steps=num_bins + 1, device=samples.device)
        else:
            bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=samples.device)
    
        samples = samples.contiguous()
        hist = accumulate_histogram_gpu(samples, bin_edges, num_bins)
        return hist, bin_edges


def generate_hist_statistics(samples, return_channels='luminance'):
    """
    Compute mean and relative luminance variance of radiance per pixel.

    Args:
        samples (torch.Tensor): Shape (N, 3, H, W), input radiance samples
        device (str or torch.device, optional): Compute device (CPU or CUDA)

    Returns:
        stats (dict): {
            'mean': torch.Tensor of shape (H, W, 3),
            'relative_variance': torch.Tensor of shape (H, W, 1),  # single channel: luminance
        }
    """
    epsilon = 1e-6
    mean = samples.mean(dim=0)                          # (3, H, W)
    mean_sq = (samples ** 2).mean(dim=0)                # (3, H, W)
    var = mean_sq - mean ** 2                           # (3, H, W)
    relative_var = var / (mean ** 2 + epsilon)          # (3, H, W)

    # Luminance conversion (standard Rec.709)
    if return_channels == 'luminance':
        # Luminance conversion (standard Rec.709)
        rel_var_lum = (
            0.2126 * relative_var[0, ...] +
            0.7152 * relative_var[1, ...] +
            0.0722 * relative_var[2, ...]
        ).unsqueeze(0)                                  # (1, H, W)
        relative_variance = rel_var_lum
    else:
        relative_variance = relative_var                # (3, H, W)

    return {'mean': mean, 'var': var, 'relative_variance': relative_variance}

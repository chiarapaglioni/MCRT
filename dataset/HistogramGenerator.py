
import torch
import numpy as np

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
        print(f"Linear estimated radiance range: [{min_val:.4e}, {max_val:.4e}]")
    return min_val, max_val


def accumulate_histogram(hist, samples, bin_edges, num_bins):
    """
    Vectorized histogram accumulation.

    samples: (N, H, W, 3)
    hist: (H, W, 3, B)
    """
    _, H, W, C = samples.shape
    bins = np.searchsorted(bin_edges, samples, side='right') - 1
    bins = np.clip(bins, 0, num_bins - 1)

    for c in range(C):
        bincounts = np.zeros((H, W, num_bins), dtype=np.int32)
        for b in range(num_bins):
            bincounts[:, :, b] = (bins[:, :, :, c] == b).sum(axis=0)
        hist[:, :, c, :] = bincounts  # replace not add to avoid accumulation


def accumulate_histogram_vectorized(hist, samples, bin_edges, num_bins):
    """
    Vectorized histogram accumulation without explicit loops.

    Paramters:
        hist (np.ndarray): (H, W, 3, B) zero-initialized histogram to fill.
        samples (np.ndarray): (N, H, W, 3) samples.
        bin_edges (np.ndarray): (B+1,) bin edges.
        num_bins (int): number of bins.
    """
    # samples: (N, H, W, 3)
    # Transpose samples to (H, W, 3, N) for easier manipulation
    samples_t = np.transpose(samples, (1, 2, 3, 0))
    
    # Digitize samples to bin indices (shape: H, W, 3, N)
    bins = np.digitize(samples_t, bin_edges) - 1
    bins = np.clip(bins, 0, num_bins - 1)
    
    H, W, C, N = bins.shape
    
    for c in range(C):
        # Flatten spatial dims and N: shape (H*W, N)
        flat_bins = bins[:, :, c, :].reshape(-1, N)
        # Count occurrences of each bin along axis 1
        bincounts = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_bins), axis=1, arr=flat_bins)
        # Reshape back to (H, W, B)
        hist[:, :, c, :] = bincounts.reshape(H, W, num_bins)


def accumulate_histogram_torch(samples, bin_edges, num_bins, device='cuda'):
    """
    samples: Tensor shape (N, H, W, 3), float32
    Returns: hist Tensor (H, W, 3, num_bins)
    """
    samples = samples.to(device)

    # Fix the bin_edges warning
    if not isinstance(bin_edges, torch.Tensor):
        bin_edges = torch.tensor(bin_edges, dtype=torch.float32, device=device)
    else:
        bin_edges = bin_edges.to(device=device, dtype=torch.float32)

    N, H, W, C = samples.shape
    hist = torch.zeros((H, W, C, num_bins), device=device, dtype=torch.int32)

    bins = torch.bucketize(samples, bin_edges) - 1
    bins = torch.clamp(bins, 0, num_bins - 1)

    for c in range(C):
        flat_bins = bins[:, :, :, c].reshape(N, -1)
        for idx in range(flat_bins.shape[1]):
            hist.view(H*W, C, num_bins)[idx, c].index_add_(
                0,
                flat_bins[:, idx],
                torch.ones(N, device=device, dtype=torch.int32)
            )
    return hist


def generate_histograms(samples, num_bins, device=None, debug=False):
    """
    Generate histograms per pixel per channel from input samples.
    Uses GPU acceleration via PyTorch if CUDA is available.
    
    Args:
        samples (np.ndarray): Shape (N, H, W, 3), input radiance samples
        num_bins (int): Number of histogram bins
        debug (bool): Print debug info
    
    Returns:
        hist (np.ndarray): Shape (H, W, 3, B), histogram per pixel per channel
        bin_edges (np.ndarray): Shape (B+1,), bin edges
    """
    _, H, W, _ = samples.shape
    min_val, max_val = estimate_range(samples, debug=debug)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    if device is not None and torch.cuda.is_available():
        with torch.no_grad():
            torch_samples = torch.tensor(samples, dtype=torch.float32, device=device)
            bin_edges_torch = torch.tensor(bin_edges, dtype=torch.float32, device=device)
            hist_torch = accumulate_histogram_torch(torch_samples, bin_edges_torch, num_bins, device=device)
            hist = hist_torch.cpu().numpy()
    else:
        hist = np.zeros((H, W, 3, num_bins), dtype=np.int32)
        accumulate_histogram_vectorized(hist, samples, bin_edges, num_bins)

    return hist, bin_edges

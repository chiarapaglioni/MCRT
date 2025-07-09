
import numpy as np

def estimate_range(samples):
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

    print(f"Linear estimated radiance range: [{min_val:.4e}, {max_val:.4e}]")
    return min_val, max_val

def accumulate_histogram(hist, samples, bin_edges, num_bins):
    """
    Vectorized histogram accumulation.

    samples: (N, H, W, 3)
    hist: (H, W, 3, B)
    """
    _, H, W, C = samples.shape
    bins = np.searchsorted(bin_edges, samples, side='right')
    bins = np.clip(bins, 0, num_bins - 1)

    for c in range(C):
        # (N, H, W)
        bincounts = np.zeros((H, W, num_bins), dtype=np.int32)
        for b in range(num_bins):
            bincounts[:, :, b] = (bins[:, :, :, c] == b).sum(axis=0)
        hist[:, :, c, :] += bincounts

def generate_histograms(samples, num_bins):
    """
    Returns:
        hist: (H, W, 3, B)
        bin_edges: (B,)
    """
    _, H, W, _ = samples.shape
    min_val, max_val = estimate_range(samples)
    bin_edges = np.linspace(min_val, max_val, num_bins)

    hist = np.zeros((H, W, 3, num_bins), dtype=np.int32)
    accumulate_histogram(hist, samples, bin_edges, num_bins)

    return hist, bin_edges
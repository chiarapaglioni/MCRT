
import torch
import numpy as np

# Logger
import logging
logger = logging.getLogger(__name__)


def generate_histograms_with_zero_bin(samples, num_bins, device=None, debug=False):
    """
    Generate histograms with a dedicated zero bin at index 0.

    Args:
        samples (np.ndarray): (N, H, W, 3), input radiance samples
        num_bins (int): Total number of bins including zero bin
        device: torch device or None
        debug: print debug info

    Returns:
        hist (np.ndarray): (H, W, 3, num_bins)
        bin_edges (np.ndarray): (num_bins + 1,), bin edges including zero bin
    """
    _, H, W, _ = samples.shape

    # Separate zero bin
    # Estimate positive range ignoring zeros
    positive_samples = samples[samples > 1e-8]

    if positive_samples.size == 0:
        # fallback range
        min_val, max_val = 1e-8, 1.0
    else:
        min_val = np.percentile(positive_samples, 1)
        max_val = np.percentile(positive_samples, 99)
        if min_val == max_val:
            max_val *= 1.1

    if debug:
        logger.info(f"Positive radiance range (excluding zero bin): [{min_val:.4e}, {max_val:.4e}]")

    # Create bin edges with a zero bin at front
    # zero bin covers [0, epsilon], then linspace for rest
    epsilon = 1e-8
    # +1 because zero bin is extra bin
    positive_bin_edges = np.linspace(min_val, max_val, num_bins)

    # Construct bin edges: start with 0, then epsilon, then positive bins shifted by epsilon
    bin_edges = np.zeros(num_bins + 1, dtype=np.float32)
    bin_edges[0] = 0.0
    bin_edges[1] = epsilon  # tiny bin for zero bin
    bin_edges[2:] = positive_bin_edges[1:]  # rest bins cover positive range

    if debug:
        logger.info(f"Bin edges with zero bin: {bin_edges}")

    # Now accumulate histogram, making sure zero values go to zero bin (bin index 0)
    if device is not None and torch.cuda.is_available():
        with torch.no_grad():
            torch_samples = torch.tensor(samples, dtype=torch.float32, device=device)
            bin_edges_torch = torch.tensor(bin_edges, dtype=torch.float32, device=device)
            hist_torch = accumulate_histogram_torch_with_zero_bin(torch_samples, bin_edges_torch, num_bins, device=device)
            hist = hist_torch.cpu().numpy()
    else:
        hist = np.zeros((H, W, 3, num_bins), dtype=np.int32)
        accumulate_histogram_vectorized_with_zero_bin(hist, samples, bin_edges, num_bins)

    return hist, bin_edges


def accumulate_histogram_vectorized_with_zero_bin(hist, samples, bin_edges, num_bins):
    """
    Vectorized histogram accumulation with zero bin at index 0.

    samples: (N, H, W, 3)
    hist: (H, W, 3, num_bins)
    """
    samples_t = np.transpose(samples, (1, 2, 3, 0))  # (H, W, 3, N)

    # Assign samples <= epsilon (zero bin upper edge) to bin 0
    epsilon = bin_edges[1]
    bins = np.digitize(samples_t, bin_edges) - 1
    bins = np.clip(bins, 0, num_bins - 1)

    # Force samples <= epsilon into zero bin (bin 0)
    bins[samples_t <= epsilon] = 0

    H, W, C, N = bins.shape
    for c in range(C):
        flat_bins = bins[:, :, c, :].reshape(-1, N)
        bincounts = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_bins), axis=1, arr=flat_bins)
        hist[:, :, c, :] = bincounts.reshape(H, W, num_bins)


def accumulate_histogram_torch_with_zero_bin(samples, bin_edges, num_bins, device='cuda'):
    """
    Torch version of accumulate_histogram with zero bin at index 0.

    samples: Tensor (N, H, W, 3)
    bin_edges: Tensor (num_bins+1,)
    """
    samples = samples.to(device)
    bin_edges = bin_edges.to(device=device, dtype=torch.float32)

    N, H, W, C = samples.shape
    hist = torch.zeros((H, W, C, num_bins), device=device, dtype=torch.int32)

    epsilon = bin_edges[1].item()

    # digitize samples
    bins = torch.bucketize(samples, bin_edges) - 1
    bins = torch.clamp(bins, 0, num_bins - 1)

    # Force samples <= epsilon to zero bin
    bins = torch.where(samples <= epsilon, torch.zeros_like(bins), bins)

    for c in range(C):
        flat_bins = bins[:, :, :, c].reshape(N, -1)
        for idx in range(flat_bins.shape[1]):
            hist.view(H*W, C, num_bins)[idx, c].index_add_(
                0,
                flat_bins[:, idx],
                torch.ones(N, device=device, dtype=torch.int32)
            )
    return hist


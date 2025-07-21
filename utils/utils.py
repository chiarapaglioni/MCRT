import os
import math
import torch
import logging
import tifffile
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Logger
import logging
logger = logging.getLogger(__name__)


def compute_psnr(pred, target):
    '''
    Computes PSNR over images with dynamic range
    '''
    # Convert to torch if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    # Ensure float32 for precision
    pred = pred.float()
    target = target.float()

    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return float('inf')

    # Compute dynamic range (min/max over the target)
    max_val = target.max().item()
    min_val = target.min().item()
    dynamic_range = max_val - min_val

    if dynamic_range == 0:
        return float('inf')  # Avoid division by zero

    return 20 * math.log10(dynamic_range) - 10 * math.log10(mse)


def save_tiff(data, file_name):
    """
    Saves data of shape (N, H, W, C, B) to TIFF file using BigTIFF if needed.

    Parameters:
    - data (np array): data to save
    - file_name (str): file name / scene name
    """
    tifffile.imwrite(file_name, data, compression='lzw', bigtiff=True)
    logger.info(f"Saved {file_name} with shape {data.shape} <3")


def plot_images(noisy, init_psnr, hist_pred, hist_psnr, img_pred, img_psnr, target, clean=None, save_path=None):
    """
    Plot denoised images generated from the noise2noise and hist2nosie next to the clean one.

    Parameters: 
    - noisy (torch tensor): noisy input, i.e. average or histogram of N-1 samples rendered with 1 spp
    - hist_pred (torch tensor): denoised prediction from hist2noise
    - noise_pred (torch tensor): denoised prediction from noise2noise
    - target (torch tensor): noisy target (1 sample)
    - clean (torch tensor): clean GT rendered with high spp
    """
    def to_img(t):
        if t.dim() == 4:  # [1, 3, H, W]
            t = t.squeeze(0)
        return t.detach().cpu().numpy().transpose(1, 2, 0)
    
    _, axes = plt.subplots(1, 5 if clean is not None else 4, figsize=(20, 4))
    axes[0].imshow(to_img(noisy));          axes[0].set_title(f"Noisy Input - PSNR:  {init_psnr:.2f} dB")
    axes[1].imshow(to_img(target));         axes[1].set_title("Target Sample - PSNR")
    axes[2].imshow(to_img(hist_pred));      axes[2].set_title(f"Hist2Noise Output - PSNR:  {hist_psnr:.2f} dB")
    axes[3].imshow(to_img(img_pred));       axes[3].set_title(f"Noise2Noise Output - PSNR:  {img_psnr:.2f} dB")
    if clean is not None:
        axes[4].imshow(to_img(clean));      axes[4].set_title("Clean (GT)")
    for ax in axes: ax.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    plt.show()


def hist_to_img(img_tensor, mean=None, std=None):
    """
    Convert a histogram tensor to an image (mean feature), with optional unstandardization.

    Args:
        img_tensor (torch.Tensor): Tensor of shape [C, bins, H, W] or [C, H, W]
        mean (optional): Per-channel mean for unstandardization
        std (optional): Per-channel std for unstandardization

    Returns:
        np.ndarray: Image of shape [H, W, C] or [H, W] for grayscale
    """
    if img_tensor.dim() == 4:
        mean_idx = -2  # Second-to-last bin
        img = img_tensor[:, mean_idx, :, :]
    elif img_tensor.dim() == 3:
        img = img_tensor
    else:
        raise ValueError(f"Unexpected tensor shape: {img_tensor.shape}")

    if mean is not None and std is not None:
        img = unstandardize_tensor(img, mean, std)

    img_np = img.permute(1, 2, 0).cpu().numpy()
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]

    return img_np


def unstandardize_tensor(tensor, mean=None, std=None):
    """
    Unstandardizes a tensor using per-channel mean and std.

    Args:
        tensor (torch.Tensor): Standardized tensor of shape [C, H, W]
        mean (list, np.ndarray, or torch.Tensor): Per-channel mean
        std (list, np.ndarray, or torch.Tensor): Per-channel std

    Returns:
        torch.Tensor: Unstandardized tensor of shape [C, H, W]
    """
    import torch

    if isinstance(mean, (list, tuple, np.ndarray)):
        mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    if isinstance(std, (list, tuple, np.ndarray)):
        std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    return tensor * std + mean


def plot_debug_images(batch, preds=None, epoch=None, batch_idx=None, image_mean=None, image_std=None):
    input_imgs = batch['input'].cpu()
    noisy_imgs = batch['noisy'].cpu()
    target_imgs = batch['target'].cpu()
    clean_imgs = batch.get('clean')
    if clean_imgs is not None:
        clean_imgs = clean_imgs.cpu()
    if preds is not None:
        preds = preds.detach().cpu()

    idx = 0

    _, axes = plt.subplots(1, 5 if clean_imgs is not None else 4, figsize=(15, 5))

    axes[0].imshow(hist_to_img(input_imgs[idx]))
    axes[0].set_title("Input (Standardised)")
    axes[1].imshow(hist_to_img(target_imgs[idx]))
    axes[1].set_title("Target (Standardised)")
    axes[2].imshow(hist_to_img(noisy_imgs[idx]))
    axes[2].set_title("Noisy")
    axes[3].imshow(hist_to_img(preds[idx], mean=image_mean[idx], std=image_std[idx]))
    axes[3].set_title("Predicted")
    if clean_imgs is not None:
        axes[4].imshow(hist_to_img(clean_imgs[idx]))
        axes[4].set_title("Clean")

    for ax in axes:
        ax.axis('off')
    plt.suptitle(f"Epoch {epoch} Batch {batch_idx}")
    plt.show()


def save_loss_plot(train_losses, val_losses, save_dir, filename="loss_plot.png", title="Training and Validation Loss"):
    """
    Plots and saves training and validation loss curves.

    Args:
        train_losses (list or array): List of training loss values per epoch.
        val_losses (list or array): List of validation loss values per epoch.
        save_dir (str or Path): Directory to save the plot.
        filename (str): Filename for the saved plot image.
        title (str): Title of the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Loss plot saved to {save_path}")


def save_psnr_plot(psnr_values, save_dir="plots", filename="psnr_plot.png"):
    """
    Saves the PSNR plot over epochs.
    
    Args:
        psnr_values (list): List of PSNR values (floats).
        save_dir (str): Directory to save the plot.
        filename (str): Name of the output PNG file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    plt.figure(figsize=(8, 6))
    plt.plot(psnr_values, marker='o', label='PSNR')
    plt.title("Validation PSNR over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    logger.info(f"Saved PSNR plot to {path}")


def setup_logger(logfile='run.log'):
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, logfile)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def decode_pred_logits(pred_probs):
    """
    pred_probs: Tensor (B, 3, n_bins, H, W), probabilities already softmaxed
    Returns:
      expected_rgb: Tensor (B, 3, H, W) with pixel values in [0, 1]
    """
    _, _, bins, _, _ = pred_probs.shape
    device = pred_probs.device

    # Create bin centers from 0 to 1
    bin_centers = torch.linspace(0, 1, bins, device=device).view(1, 1, bins, 1, 1)  # shape (1,1,bins,1,1)

    # Expected value: sum over bins of probability * bin_center
    expected_rgb = (pred_probs * bin_centers).sum(dim=2)  # sum over bins dimension -> (B,3,H,W)

    return expected_rgb


def print_histogram_at_pixel(hist_tensor, x, y, used_bins):
    """
    Prints the histogram values at a given pixel location (x, y) for each RGB channel.

    Args:
        hist_tensor (np.ndarray): Histogram tensor with shape [3, B, H, W]
        x (int): x-coordinate (width direction)
        y (int): y-coordinate (height direction)
        used_bins (int): Number of bins used in the histogram
    """
    print(f"\n--- Histogram values at pixel ({x}, {y}) ---")
    for c, color in enumerate(['R', 'G', 'B']):
        vals = hist_tensor[c, :used_bins, y, x]
        print(f"{color} channel: {vals}")


def compute_global_mean_std(root_dir, low_spp):
    all_pixels = []

    # Iterate scenes and load spp1 images
    for subdir in sorted(os.listdir(root_dir)):
        full_subdir = os.path.join(root_dir, subdir)
        if not os.path.isdir(full_subdir):
            continue
        for fname in os.listdir(full_subdir):
            if fname.endswith(f"spp1x{low_spp}.tiff"):
                path = os.path.join(full_subdir, fname)
                img = tifffile.imread(path)  # (low_spp, H, W, 3)
                # Reshape to (-1, 3) and append
                pixels = img.reshape(-1, 3)
                all_pixels.append(pixels)

    all_pixels = np.concatenate(all_pixels, axis=0)  # shape (N_pixels, 3)

    mean = np.mean(all_pixels, axis=0)  # (3,)
    std = np.std(all_pixels, axis=0)    # (3,)

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def standardize_per_image(tensor):
    """
    Standardizes N images of shape (C, H, W) rendered at 1spp and returns their channel mean and std

    Parameters: 
        tensor: (N, C, H, W)
    Returns:
        standardized tensor, means, std (shape (3, 1, 1))
    """
    # tensor: (N, C, H, W)
    mean = tensor.mean(dim=(0, 2, 3)).view(-1, 1, 1)  # (3, 1, 1)
    std = tensor.std(dim=(0, 2, 3)).view(-1, 1, 1)    # (3, 1, 1)
    standardized = (tensor - mean[None]) / (std[None] + 1e-8)
    return standardized, mean, std

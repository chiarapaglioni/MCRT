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


def compute_psnr(pred, target):
    '''
    Computes PSNR (Peak Signal-to-Noise Ratio) between two images,
    using the dynamic range of the target image.

    Parameters: 
    - pred (Tensor or ndarray): Predicted image (e.g., denoised output)
    - target (Tensor or ndarray): Reference clean image

    Returns: 
    - psnr (float): Peak Signal-to-Noise Ratio in decibels (dB)
    '''
    # Convert to torch if input is a numpy array
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    # Ensure type is float32 for accurate MSE computation
    pred = pred.float()
    target = target.float()

    # Compute Mean Squared Error
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return float('inf')  # Perfect match

    # Compute the dynamic range of the target (max - min)
    max_val = target.max().item()
    min_val = target.min().item()
    dynamic_range = max_val - min_val

    if dynamic_range == 0:
        return float('inf')  # Avoid log(0) if image is constant

    # Compute PSNR using dynamic range
    return 20 * math.log10(dynamic_range) - 10 * math.log10(mse)


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


def hist_to_img(img_tensor, mean=None, std=None, lamda=None):
    """
    Convert a tensor to an image (mean feature), with optional unstandardization and inverse Box-Cox transform.

    Args:
        img_tensor (torch.Tensor): Tensor of shape [C, bins, H, W] or [C, H, W]
        mean (optional): Per-channel mean (after Box-Cox transform)
        std (optional): Per-channel std (after Box-Cox transform)
        lmbda (float, optional): Lambda parameter for Box-Cox transform. If provided, applies inverse Box-Cox.

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

    if lamda is not None:
        img = boxcox_inverse(img, lmbda=lamda)               # Inverse boxcox
    
    # img = reinhard_inverse(img)
    img = torch.expm1(img)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]

    return img_np


def plot_debug_images(batch, preds=None, epoch=None, batch_idx=None, image_mean=None, image_std=None, lamda=None):
    """
    Plots a single row of debug images from a training batch for visual inspection.

    Displays:
    - Input (standardised, histogram-based)
    - Target (standardised)
    - Noisy image
    - Predicted image (optionally denormalised using mean/std/lambda)
    - Clean image (if available)

    Parameters:
    - batch: dictionary of input tensors (includes 'input', 'target', 'noisy', optional 'clean')
    - preds: model predictions (optional, tensor)
    - epoch: current epoch number (optional, used in title)
    - batch_idx: index of batch (optional, used in title)
    - image_mean: tensor of mean values for unnormalising predictions
    - image_std: tensor of std values for unnormalising predictions
    - lamda: tensor of Box-Cox lambda values for inverse transform
    """
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
    # axes[3].imshow(hist_to_img(preds[idx], mean=image_mean[idx], std=image_std[idx], lamda=lamda[idx].item()))
    axes[3].imshow(hist_to_img(preds[idx]))
    axes[3].set_title("Predicted")
    if clean_imgs is not None:
        axes[4].imshow(hist_to_img(clean_imgs[idx]))
        axes[4].set_title("Clean")

    for ax in axes:
        ax.axis('off')
    plt.suptitle(f"Epoch {epoch} Batch {batch_idx}")
    plt.show()


def save_tiff(data, file_name):
    """
    Saves data of shape (N, H, W, C, B) to TIFF file using BigTIFF if needed.

    Parameters:
    - data (np array): data to save
    - file_name (str): file name / scene name
    """
    tifffile.imwrite(file_name, data, compression='lzw', bigtiff=True)
    logger.info(f"Saved {file_name} with shape {data.shape} <3")


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


def compute_global_stats_from_spp32(root_dir, lambda_val=0.5, epsilon=1e-6):
    """
    Compute global mean and std from all 1x32spp TIFF images (MC rendered).
    Applies Box-Cox transform before computing stats.
    Returns: (mean, std) as torch tensors with shape (3, 1, 1)
    """
    sum_ = torch.zeros(3)
    sum_sq_ = torch.zeros(3)
    count = 0

    for subdir in sorted(os.listdir(root_dir)):
        full_subdir = os.path.join(root_dir, subdir)
        if not os.path.isdir(full_subdir):
            continue

        # Find 1x32spp image
        fname = next((f for f in os.listdir(full_subdir) if f.endswith("spp1x32.tiff")), None)
        if not fname:
            continue

        img_path = os.path.join(full_subdir, fname)
        img = tifffile.imread(img_path)  # shape: (32, H, W, 3)
        img = torch.from_numpy(img).permute(0, 3, 1, 2).float()  # (32, 3, H, W)

        img_bc = boxcox_transform(img, lmbda=lambda_val, epsilon=epsilon)  # Apply Box-Cox

        flat = img_bc.permute(0, 2, 3, 1).reshape(-1, 3)  # Flatten to (N*H*W, 3)
        sum_ += flat.sum(dim=0)
        sum_sq_ += (flat ** 2).sum(dim=0)
        count += flat.shape[0]

    mean = sum_ / count
    var = (sum_sq_ / count) - mean**2
    std = torch.sqrt(var + 1e-8)

    return mean.view(3, 1, 1), std.view(3, 1, 1)


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


def standardize_tensor(tensor, eps=1e-8):
    """
    Standardizes a tensor per channel by subtracting the mean and dividing by the standard deviation.

    Args:
        tensor (torch.Tensor): A tensor of shape [C, H, W] (e.g., an image).
        eps (float): A small value to avoid division by zero.

    Returns:
        tuple:
            - standardized (torch.Tensor): The standardized tensor.
            - mean (torch.Tensor): Per-channel mean used for standardization (shape [C, 1, 1]).
            - std (torch.Tensor): Per-channel std used for standardization (shape [C, 1, 1]).
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected tensor of shape [C, H, W], but got shape {tensor.shape}")

    # Compute per-channel mean and std
    mean = tensor.mean(dim=(1, 2), keepdim=True)         # shape: [C, 1, 1]
    std = tensor.std(dim=(1, 2), keepdim=True) + eps     # shape: [C, 1, 1]

    # Standardize
    standardized = (tensor - mean) / std

    return standardized, mean, std


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
    if isinstance(mean, (list, tuple, np.ndarray)):
        mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    if isinstance(std, (list, tuple, np.ndarray)):
        std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    return tensor * std + mean


def boxcox_transform(x, lmbda=0.2, epsilon=1e-6):
    """
    Applies the Box-Cox transformation to stabilize variance and make the data more Gaussian-like.

    Args:
        x (Tensor): Input tensor (should be non-negative or strictly positive).
        lmbda (float): The Box-Cox lambda parameter controlling the shape of the transformation.
                       - lmbda = 0: log transform
                       - lmbda != 0: power transform
        epsilon (float): A small value added to avoid numerical instability (e.g., log(0)).

    Returns:
        Tensor: Transformed tensor.
    """
    # Ensure all inputs are strictly positive to avoid invalid log or power operations
    x = x + epsilon

    # Apply log transform if lambda is zero
    if lmbda == 0:
        return torch.log(x)
    else:
        # Apply the general Box-Cox power transform formula
        return (x ** lmbda - 1) / lmbda


def boxcox_inverse(y, lmbda, epsilon=1e-6):
    """
    Inverts the Box-Cox transformation to recover the original input data.

    Args:
        y (Tensor): Transformed tensor from boxcox_transform().
        lmbda (float): The same lambda used during the forward Box-Cox transformation.
        epsilon (float): Small constant used in the forward transform (to be subtracted here).

    Returns:
        Tensor: Recovered original tensor.
    """
    if lmbda == 0:
        # Invert the log transform: x = exp(y)
        x = torch.exp(y)
    else:
        # Invert the power transform: x = (λy + 1)^(1/λ)
        x = (lmbda * y + 1) ** (1 / lmbda)

    # Subtract epsilon to restore original scale (reversing the epsilon added in the transform)
    return x - epsilon


def boxcox_and_standardize(tensor, dim=None, global_mean=None, global_std=None):
    """Applies Box-Cox and then standardizes. Falls back to global mean/std if provided."""
    transformed = boxcox_transform(tensor)

    if global_mean is not None and global_std is not None:
        mean = global_mean
        std = global_std + 1e-8
    else:
        mean = transformed.mean() if dim is None else transformed.mean(dim=dim)
        std = transformed.std() if dim is None else transformed.std(dim=dim) + 1e-8

    normalized = (transformed - mean) / std
    return normalized, mean, std


def reinhard_tonemap(x):
    return torch.pow(x / (1.0 + x), 1.0 / 2.2)

def reinhard_inverse(x):
    x = torch.clamp(x, min=0.0, max=1.0)
    x_pow = torch.pow(x, 2.2)
    return x_pow / (1.0 - x_pow + 1e-8)  # Add epsilon to avoid divide-by-zero
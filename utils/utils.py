import os
import time
import math
import torch
import logging
import tifffile
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Models
from model.UNet import GapUNet
from model.N2NUnet import N2Net
import torch.nn.functional as F

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


def load_model(model_config, model_path, mode='img', hist_bins=16, device='cpu'):
    # GAP
    if model_config['model_name']=='gap':
        model = GapUNet(
            in_channels=model_config['in_channels'],
            n_bins=hist_bins,
            out_mode=model_config['out_mode'],
            merge_mode=model_config['merge_mode'],
            depth=model_config['depth'],
            start_filters=model_config['start_filters'],
            mode=mode
        ).to(device)

    # Noise2Noise
    elif model_config['model_name']=='n2n':
        model = N2Net(
            in_channels=model_config['in_channels']
        ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def compute_psnr(pred, target):
    '''
    Computes PSNR (Peak Signal-to-Noise Ratio) between HDR images.

    Parameters: 
    - pred (torch.Tensor): Predicted image (e.g., denoised output)
    - target (torch.Tensor): Reference clean image

    Returns: 
    - psnr (float): Peak Signal-to-Noise Ratio in decibels (dB)
    '''
    # Ensure type is float32 for accurate MSE computation
    pred = pred.float()
    target = target.float()

    # Mean Squared Error MSE
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return float('inf')

    # max val of HDR image
    max_val = target.max().item()
    return 10 * math.log10((max_val ** 2) / mse)


def tonemap_gamma_correct(hdr_image, gamma=2.2):
    """
    Applies Reinhard tone mapping + gamma correction to a HDR image.

    Parameters:
        hdr_image (np.ndarray): HDR image in linear radiance (float32) shape (H, W, 3)
        gamma (float): Gamma correction factor (default = 2.2)
    """  
    tone_mapped = hdr_image / (1.0 + hdr_image)                     # Reinhard tone mapping
    display_img = np.power(np.clip(tone_mapped, 0, 1), 1.0 / gamma) # Gamma correction
    return display_img



def plot_images(noisy, init_psnr, hist_pred, hist_psnr, img_pred, img_psnr, target, clean=None, save_path=None, correct=False):
    """
    Plot denoised images generated from the noise2noise and hist2nosie next to the clean one.

    Parameters: 
    - noisy (torch tensor): noisy input, i.e. average or histogram of N-1 samples rendered with 1 spp
    - hist_pred (torch tensor): denoised prediction from hist2noise
    - noise_pred (torch tensor): denoised prediction from noise2noise
    - target (torch tensor): noisy target (1 sample)
    - clean (torch tensor): clean GT rendered with high spp
    """
    def to_img(t, correct):
        if t.dim() == 4:  # [1, 3, H, W]
            t = t.squeeze(0)
        image = t.detach().cpu().numpy().transpose(1, 2, 0)
        if correct:
            image = tonemap_gamma_correct(image)
        return image
    
    _, axes = plt.subplots(1, 5 if clean is not None else 4, figsize=(20, 4))
    axes[0].imshow(to_img(noisy, correct));          axes[0].set_title(f"Noisy Input - PSNR:  {init_psnr:.2f} dB")
    axes[1].imshow(to_img(target, correct));         axes[1].set_title("Target Sample - PSNR")
    axes[2].imshow(to_img(hist_pred, correct));      axes[2].set_title(f"Hist2Noise Output - PSNR:  {hist_psnr:.2f} dB")
    axes[3].imshow(to_img(img_pred, correct));       axes[3].set_title(f"Noise2Noise Output - PSNR:  {img_psnr:.2f} dB")
    if clean is not None:
        axes[4].imshow(to_img(clean, correct));      axes[4].set_title("Clean (GT)")
    for ax in axes: ax.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    plt.show()


def hist_to_img(img_tensor):
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

    # move image from 3, H, W, ---> H, W, 3
    img_np = img.permute(1, 2, 0).cpu().numpy()
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]
    return img_np



def plot_debug_images(batch, preds=None, epoch=None, batch_idx=None, correct=False, save_dir='debug_plots'):
    """
    Plots and saves a single debug image (idx=0) from the batch for visual inspection.

    Saves:
    - Input
    - Target
    - Noisy
    - Predicted
    - Clean (if available)

    All images are saved to `save_dir` with filename indicating epoch and batch.
    """
    os.makedirs(save_dir, exist_ok=True)

    input_imgs = batch['input'].cpu()
    noisy_imgs = batch['noisy'].cpu()
    target_imgs = batch['target'].cpu()
    clean_imgs = batch.get('clean', None)
    bin_edges = batch.get('bin_edges', None)
    if clean_imgs is not None:
        clean_imgs = clean_imgs.cpu()
    if preds is not None:
        preds = preds.detach().cpu()

    idx = 0  # only first image
    B, _, H, W = input_imgs.shape

    # Handle img and hist data
    if input_imgs.shape[1] <= 15:
        input_img = input_imgs[idx][:3]
    elif input_imgs.shape[1] > 15 and bin_edges is not None:
        bins = bin_edges.shape[1] - 1
        hist_size = bins * 3
        hist_decoded = input_imgs[:, :hist_size]                # (B, bins*3, H, W)
        hist_decoded = hist_decoded.view(B, 3, bins, H, W)      # (B, 3, bins, H, W)
        imgs_decoded = [
                decode_image_from_probs(
                probs=hist_decoded[i:i+1],         # (1, 3, bins, H, W)
                bin_edges=bin_edges[i:i+1]         # (1, bins+1)
            )[0]                                   # remove batch dim after decoding
            for i in range(B)
        ]
        input_img = imgs_decoded[idx]
    else: 
        input_img = input_imgs[idx]

    # recover images from mean
    clean_img = clean_imgs[idx]
    pred_img = preds[idx]
    target_img = target_imgs[idx]
    noisy_img = noisy_imgs[idx]

    # Compute PSNR
    inp_psnr = compute_psnr(input_img, clean_img) if input_imgs is not None else None
    pred_psnr = compute_psnr(pred_img, clean_img) if preds is not None else None

    # Plot
    _, axes = plt.subplots(1, 5 if clean_imgs is not None else 4, figsize=(15, 5))

    def show_img(ax, img_tensor, title):
        img = tonemap_gamma_correct(hist_to_img(img_tensor)) if correct else hist_to_img(img_tensor)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    inp_title = f"Input\nPSNR: {inp_psnr:.2f} dB" if inp_psnr else "Input"
    show_img(axes[0], input_img, inp_title)
    show_img(axes[1], target_img, "Target (Standardised)")
    show_img(axes[2], noisy_img, "Noisy")
    pred_title = f"Predicted\nPSNR: {pred_psnr:.2f} dB" if pred_psnr else "Predicted"
    show_img(axes[3], preds[idx], pred_title)
    if clean_imgs is not None:
        show_img(axes[4], clean_imgs[idx], "Clean")

    plt.suptitle(f"Epoch {epoch}, Batch {batch_idx}")
    filename = os.path.join(save_dir, f"epoch_{epoch:03d}_batch_{batch_idx:03d}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_tiff(data, file_name):
    """
    Saves data of shape (N, H, W, C, B) to TIFF file using BigTIFF if needed.

    Parameters:
    - data (np array): data to save
    - file_name (str): file name / scene name
    """
    tifffile.imwrite(file_name, data, bigtiff=True)
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


def decode_image_from_probs(probs, bin_edges):
    """
    Convert predicted bin probabilities to expected radiance, per batch.

    Args:
        probs: (B, C, bins, H, W), softmax output from logits
        bin_edges: (B, bins + 1), array of bin edges per batch

    Returns:
        predicted_radiance: (B, C, H, W)
    """
    # Compute bin centers per batch: shape (B, bins)
    bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])  # (B, bins)
    
    # Reshape to broadcast over C, H, W dimensions (B, 1, bins, 1, 1)
    bin_centers = bin_centers.view(probs.shape[0], 1, -1, 1, 1)
    
    # Multiply probabilities with bin centers and sum over bins dimension
    pred_radiance = (probs * bin_centers).sum(dim=2)  # (B, C, H, W)
    return pred_radiance
    


def decode_pred_logits_zero(probs, bin_edges):
    """
    Convert predicted bin probabilities to expected radiance, per batch.

    Args:
        probs: (B, C, bins, H, W), softmax output from logits
        bin_edges: (B, bins + 1), array of bin edges per batch

    Returns:
        predicted_radiance: (B, C, H, W)
    """
    # Compute bin centers per batch: shape (B, bins)
    bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])  # (B, bins)
    
    # Set first bin center explicitly to 0 for the zero bin
    bin_centers[:, 0] = 0.0
    
    # Reshape to broadcast over C, H, W dimensions
    # (B, 1, bins, 1, 1)
    bin_centers = bin_centers.view(probs.shape[0], 1, -1, 1, 1)
    
    # Multiply probabilities with bin centers and sum over bins dimension
    pred_radiance = (probs * bin_centers).sum(dim=2)  # (B, C, H, W)
    
    return pred_radiance


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


def compute_global_mean_std(root_dir):
    """
    Compute global mean and std from all 1x32spp TIFF images (MC rendered).
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

        flat = img.permute(0, 2, 3, 1).reshape(-1, 3)  # Flatten to (N*H*W, 3)
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


def reinhard_tonemap_gamma(x, gamma=2.2):
    """
    Applies gamma-compressed Reinhard global tone mapping to input.

    Args:
        x (torch.Tensor or np.ndarray): HDR image, values >= 0
        gamma (float): Gamma compression value

    Returns:
        Same type as input: tone-mapped image in [0, 1] range
    """
    eps = 1e-8
    return torch.pow(x / (1.0 + x + eps), 1.0 / gamma)


def reinhard_tonemap(x):
    """
    Applies standard Reinhard global tone mapping: T(v) = v / (1 + v)

    Args:
        x (torch.Tensor): HDR image, values >= 0

    Returns:
        Same type as input: tone-mapped image in [0, 1) range
    """
    eps = 1e-8
    return x / (1.0 + x + eps)


def input_tonemap(x):
    """
    Applies SBMC input tone mapping: T_i(v) = (1/10) * ln(1 + v)

    Args:
        x (torch.Tensor): HDR input image, values >= 0

    Returns:
        Same type as input: tone-mapped input
    """
    eps = 1e-8  # small epsilon to avoid log(0)
    return (1.0 / 10.0) * torch.log1p(x + eps)


def apply_tonemap(hdr_tensor, tonemap):
    '''
    Function called in the loss function to compare prediction of the network to target image.

    Parameters: 
        hdr_tensor (torch.Tensor): torch tensor in HDR linear space
        tonemap (str): tonemap type

    Returns: 
        tonemapped tensor
    '''
    if tonemap == 'log':
        return torch.log1p(hdr_tensor)

    elif tonemap == 'reinhard':
        return reinhard_tonemap(hdr_tensor)
        
    elif tonemap == 'reinhard_gamma': 
        return reinhard_tonemap_gamma(hdr_tensor)

    else: 
        return hdr_tensor # no tonemapped applied


def local_variance(image, window_size=16, save_path=None, cmap='viridis'):
    """
    Compute approximate local variance of an image using non-overlapping sliding windows.

    Args:
        image (Tensor): Input tensor of shape (3, H, W).
        window_size (int): Size of the sliding window (patch size).
        save_path (str, optional): If provided, saves a heatmap of the variance to this path.
        cmap (str): Colormap used for the heatmap visualization.

    Returns:
        Tensor: Variance heatmap of shape (H_patch, W_patch).
    """
    luminance = image.mean(dim=0, keepdim=True)  # (1, H, W)
    padding = (window_size - 1) // 2
    mean = F.avg_pool2d(luminance, kernel_size=window_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(luminance ** 2, kernel_size=window_size, stride=1, padding=padding)
    var = mean_sq - mean ** 2  # (1, H, W)
    var = var.squeeze(0)  # (H, W)

    if save_path is not None:
        var_norm = (var - var.min()) / (var.max() - var.min() + 1e-8)
        plt.figure(figsize=(6, 6))
        plt.imshow(var_norm.cpu(), cmap=cmap)
        plt.colorbar(label='Local Variance')
        plt.title("Local Variance Heatmap")
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    return var

def sample_crop_coords_from_variance(varmap, crop_size):
    H, W = varmap.shape
    crop_H, crop_W = crop_size, crop_size

    var_tensor = varmap.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    crop_var = F.avg_pool2d(var_tensor, kernel_size=(crop_H, crop_W), stride=1).squeeze(0).squeeze(0)  # (H-crop+1, W-crop+1)
    
    probs = crop_var.flatten()
    probs -= probs.min()
    probs /= probs.sum() + 1e-8

    idx = torch.multinomial(probs, 1).item()
    out_W = crop_var.shape[1]
    i = idx // out_W
    j = idx % out_W

    return int(i), int(j), crop_H, crop_W

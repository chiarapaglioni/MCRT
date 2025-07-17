import os
import logging
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# Logger
import logging
logger = logging.getLogger(__name__)


def save_tiff(data, file_name):
    """
    Saves data of shape (N, H, W, C, B) to TIFF file using BigTIFF if needed.

    Parameters:
    - data (np array): data to save
    - file_name (str): file name / scene name
    """
    tifffile.imwrite(file_name, data, compression='lzw', bigtiff=True)
    logger.info(f"Saved {file_name} with shape {data.shape} <3")


def plot_images(noisy, init_psnr, hist_pred, hist_psnr, img_pred, img_psnr, target, target_psnr, clean=None, save_path=None):
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
    axes[1].imshow(to_img(target));         axes[1].set_title(f"Target Sample - PSNR:  {target_psnr:.2f} dB")
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



def standardize_image(img: np.ndarray, per_channel: bool = True, epsilon: float = 1e-8) -> np.ndarray:
    """
    Standardizes an image to have zero mean and unit variance.

    Args:
        img (np.ndarray): Input image with shape (H, W, C) or (C, H, W)
        per_channel (bool): If True, normalize each channel independently. Otherwise, normalize globally.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Standardized image with same shape.
    """
    # Ensure shape is (H, W, C)
    if img.ndim == 3 and img.shape[0] in [1, 3] and img.shape[0] < img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    img = img.astype(np.float32)

    if per_channel:
        for c in range(img.shape[-1]):
            channel = img[..., c]
            mean = channel.mean()
            std = channel.std()
            img[..., c] = (channel - mean) / (std + epsilon)
    else:
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + epsilon)

    return img


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

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# Datasets
from dataset.HistImgDataset import HistogramDataset, HistogramBinomDataset

# Logger
import logging
logger = logging.getLogger(__name__)


def print_histogram_at_pixel(hist, x, y, used_bins):
    print(f"Histogram at pixel ({x}, {y}):")
    for c, channel in enumerate("RGB"):
        values = hist[c, :used_bins, y, x]
        print(f"  {channel}: {values}")


def plot_hist_bar(ax, hist, title, x, y, used_bins):
    r = hist[0, :used_bins, y, x]
    g = hist[1, :used_bins, y, x]
    b = hist[2, :used_bins, y, x]

    bar_width = 0.25
    bin_positions = np.arange(used_bins)

    print_histogram_at_pixel(hist, x, y, used_bins)

    mean = hist[:, -2, y, x]
    var = hist[:, -1, y, x]
    print(f"Pixel ({x}, {y}) - Mean: R={mean[0]:.4f}, G={mean[1]:.4f}, B={mean[2]:.4f}")
    print(f"Pixel ({x}, {y}) - Var:  R={var[0]:.4f}, G={var[1]:.4f}, B={var[2]:.4f}")

    ax.bar(bin_positions - bar_width, r, width=bar_width, color='red', label='R')
    ax.bar(bin_positions, g, width=bar_width, color='green', label='G')
    ax.bar(bin_positions + bar_width, b, width=bar_width, color='blue', label='B')

    ax.set_title(title)
    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.set_xticks(bin_positions)
    ax.legend()


def test_data_loader(config):
    dataset_cfg = config['dataset']
    out_mode = config.get("out_mode", "mean")
    mode = dataset_cfg.get("mode", "img")

    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    logger.info(f"Data Loader - Mode: {mode.upper()} - Out: {out_mode.upper()} !!")

    # Validate mode/out_mode combination
    if out_mode == "dist" and mode == "img":
        raise ValueError("dist out_mode with img mode is not allowed.")

    # Pick dataset
    if out_mode == "dist":
        dataset = HistogramBinomDataset(**dataset_cfg)
        input_key = 'input_hist'
        target_key = 'target_hist'
    elif out_mode == "mean":
        dataset = HistogramDataset(**dataset_cfg)
        input_key = 'input'
        target_key = 'target'
    else:
        raise ValueError(f"Unsupported out_mode: {out_mode}. Must be 'dist' or 'mean'.")

    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('num_workers', 0),
        shuffle=config.get('shuffle', False)
    )

    for batch in dataloader:
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Input shape:  {batch[input_key].shape}")
        logger.info(f"Target shape: {batch[target_key].shape}")
        if 'clean' in batch and batch['clean'] is not None:
            logger.info(f"Clean shape: {batch['clean'].shape}")
        logger.info(f"Scene: {batch['scene']}")
        if 'image_mean' in batch and 'image_std' in batch:
            logger.info(f"Image Mean: {batch['image_mean'].shape}")
            logger.info(f"Image Std: {batch['image_std'].shape}")

        input_tensor = batch[input_key][0]
        target_tensor = batch[target_key][0]
        x, y = input_tensor.shape[-2] // 2, input_tensor.shape[-1] // 2

        if out_mode == "mean":
            if mode == "img":
                input_img = input_tensor.numpy().transpose(1, 2, 0)
                target_img = target_tensor.numpy().transpose(1, 2, 0)

                _, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(np.clip(input_img, 0, 1))
                axs[0].set_title("Input Image")
                axs[0].axis("off")

                axs[1].imshow(np.clip(target_img, 0, 1))
                axs[1].set_title("Target Image")
                axs[1].axis("off")
                plt.tight_layout()
                plt.show()

            elif mode == "hist":
                input_hist = input_tensor.numpy()
                used_bins = input_hist.shape[1] - 2
                target_img = target_tensor.numpy().transpose(1, 2, 0)

                _, axs = plt.subplots(1, 2, figsize=(12, 6))
                plot_hist_bar(axs[0], input_hist, "Input Histogram", x, y, used_bins)
                axs[1].imshow(np.clip(target_img, 0, 1))
                axs[1].set_title("Target Image")
                axs[1].axis("off")
                plt.tight_layout()
                plt.show()

        elif out_mode == "dist":
            input_hist = input_tensor.numpy()
            target_hist = target_tensor.numpy()
            used_bins = input_hist.shape[1] - 2

            _, axs = plt.subplots(1, 2, figsize=(12, 6))
            plot_hist_bar(axs[0], input_hist, "Input Histogram", x, y, used_bins)
            plot_hist_bar(axs[1], target_hist, "Target Histogram", x, y, used_bins)
            plt.tight_layout()
            plt.show()

        break  # only one batch
